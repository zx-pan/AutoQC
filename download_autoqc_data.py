#!/usr/bin/env python3
"""
Download data from BioImage Archive study S-BIAD2133 over HTTPS (requests),
with optional selection via flags.

Flags (can be combined):
  --only_train    → only items under 'train/'
  --only_test     → only items under 'test/'
  --only_splits   → only items under 'splits/'

No flags → download ALL three (train, test, splits).

Output directory can be set with --out (default: ./AutoQC_data).
"""

import os
import json
import argparse
import urllib.request
import urllib.parse
from typing import List, Tuple, Optional
import threading
from queue import Queue
import requests
from html.parser import HTMLParser

ACCESSION = "S-BIAD2133"
OUT_DIR_DEFAULT = "./AutoQC_data"
BLOCKSIZE = 1024 * 1024   # 1 MiB chunks for streaming
TIMEOUT = 60
WORKERS = 3

# Optional: overall progress bar; script works without tqdm
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    class tqdm:
        def __init__(self, iterable=None, total=None, unit="", unit_scale=False, desc="", leave=True):
            self.total = total or 0
            self.n = 0
        def update(self, n=1): self.n += n
        def close(self): pass

# ------------------------- helpers: API + path building -------------------------

def get_ftp_link(accession: str) -> str:
    """Fetch ftpLink from BioStudies API (HTTPS)."""
    url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ftp_link = data.get("ftpLink")
    if not ftp_link:
        raise RuntimeError("No ftpLink found in API response.")
    return ftp_link

def normalize_to_https(ftp_link: str) -> str:
    """
    Convert the ftpLink path to the HTTPS mirror root.
    Examples:
      ftp_link path like /biostudies/pub/databases/biostudies/... → /pub/databases/biostudies/...
    """
    parsed = urllib.parse.urlparse(ftp_link)
    path = parsed.path if parsed.path.startswith("/") else f"/{parsed.path}"
    # BioStudies sometimes returns paths with '/biostudies/pub/databases/biostudies'
    fixed = path.replace("/biostudies/pub/databases/biostudies", "/pub/databases/biostudies")
    return f"https://ftp.ebi.ac.uk{fixed}"

# ------------------------- helpers: HTTPS "directory listing" -------------------------

class _DirIndexParser(HTMLParser):
    """Very small parser for Apache-style directory indexes."""
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)

def https_exists(session: requests.Session, url: str, timeout: int = TIMEOUT) -> bool:
    try:
        r = session.get(url, timeout=timeout)
        return 200 <= r.status_code < 300
    except Exception:
        return False

def https_listdir(session: requests.Session, url: str) -> Tuple[List[str], List[str]]:
    """
    Parse an Apache-style directory index at `url` and return (dirs, files) as names.
    Ignores sorting/query links like '?C=S;O=A'.
    """
    if not url.endswith("/"):
        url += "/"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text

    p = _DirIndexParser()
    p.feed(html)

    def _valid(href: str) -> bool:
        if not href:
            return False
        if href in ("../",):
            return False
        # ignore Apache sorting / query / anchors
        if href.startswith("?") or "?" in href or href.startswith("#"):
            return False
        # (optional) be strict: skip weird entries that look like scheme or absolute urls
        if "://" in href:
            return False
        return True

    dirs: List[str] = []
    files: List[str] = []
    for href in p.links:
        if not _valid(href):
            continue
        if href.endswith("/"):
            dirs.append(href.rstrip("/"))
        else:
            files.append(href)
    return dirs, files


def https_walk(session: requests.Session, base_url: str):
    """Recursive walk starting at base_url (must be a directory URL)."""
    stack = [base_url.rstrip("/")]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            dirs, files = https_listdir(session, cur)
        except Exception:
            continue
        yield cur, dirs, files
        for d in dirs:
            stack.append(f"{cur}/{d}")

# ------------------------- size probing + download -------------------------

def head_content_length(session: requests.Session, url: str) -> Optional[int]:
    """Try to get Content-Length via HEAD; return None if unknown."""
    try:
        h = session.head(url, timeout=TIMEOUT, allow_redirects=True)
        if "Content-Length" in h.headers:
            try:
                return int(h.headers["Content-Length"])
            except Exception:
                return None
    except Exception:
        return None
    return None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_https(session: requests.Session, remote_url: str, local_path: str, expected_size: Optional[int]) -> str:
    """
    Stream a file over HTTPS to disk (atomic via .part).
    Skip when existing file size matches expected_size (if known).
    """
    ensure_dir(os.path.dirname(local_path))
    # If expected size known, skip when equal
    if expected_size and expected_size > 0 and os.path.exists(local_path):
        try:
            if os.path.getsize(local_path) == expected_size:
                return "skip"
        except Exception:
            pass

    tmp = local_path + ".part"
    with session.get(remote_url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(chunk_size=BLOCKSIZE):
                if not chunk:
                    continue
                out.write(chunk)

    # Validate size if known
    if expected_size and expected_size > 0:
        try:
            if os.path.getsize(tmp) != expected_size:
                raise RuntimeError(f"Incomplete HTTPS download for {remote_url}")
        except Exception as e:
            # Clean temp on error
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    os.replace(tmp, local_path)
    return "ok"

# ------------------------- worker (requests.Session per worker) -------------------------

def worker_loop(job_q: Queue, done_q: Queue, stop_event: threading.Event):
    """
    Consume (remote_url, local, expected_size) jobs.
    Keep a requests.Session per thread for connection pooling.
    """
    session = requests.Session()
    try:
        while not stop_event.is_set():
            item = job_q.get()
            if item is None:
                job_q.task_done()
                break
            remote_url, local, expected_size = item

            try:
                # If we don't know expected_size yet, try a HEAD on demand
                size = expected_size
                if size is None:
                    size = head_content_length(session, remote_url)

                status = download_https(session, remote_url, local, size)
                done_q.put(("ok" if status == "ok" else "skip", remote_url))
            except Exception as e:
                done_q.put(("err", remote_url, str(e)))
            finally:
                job_q.task_done()
    finally:
        session.close()

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Download train/test/splits from S-BIAD2133 via HTTPS with optional selection flags.")
    ap.add_argument("--only_train",  action="store_true", help="Download only items under 'train/'.")
    ap.add_argument("--only_test",   action="store_true", help="Download only items under 'test/'.")
    ap.add_argument("--only_splits", action="store_true", help="Download only items under 'splits/'.")
    ap.add_argument("--out", type=str, default=OUT_DIR_DEFAULT, help=f"Output directory (default: {OUT_DIR_DEFAULT})")
    args = ap.parse_args()

    print(f"[1/4] Fetching ftpLink for {ACCESSION} ...")
    ftp_link = get_ftp_link(ACCESSION)
    https_root_candidate = normalize_to_https(ftp_link)  # e.g. https://ftp.ebi.ac.uk/pub/databases/biostudies/.../S-BIAD2133
    # Prefer ".../Files" if present
    candidate_roots = [f"{https_root_candidate.rstrip('/')}/Files", https_root_candidate]

    # Build list of files (HTTPS)
    session_probe = requests.Session()
    try:
        chosen_root = None
        for cand in candidate_roots:
            url = cand if cand.endswith("/") else cand + "/"
            if https_exists(session_probe, url):
                chosen_root = cand.rstrip("/")
                break
        if not chosen_root:
            print("[Error] Could not find a valid HTTPS root.")
            return

        print("[2/4] Preparing file list ...")
        # Which top-level dirs to traverse?
        top_names = (["train","test","splits"] if not (args.only_train or args.only_test or args.only_splits)
                     else [d for d,flag in (("train",args.only_train),("test",args.only_test),("splits",args.only_splits)) if flag])

        top_dirs = []
        for name in top_names:
            url = f"{chosen_root}/{name}/"
            if https_exists(session_probe, url):
                top_dirs.append(url.rstrip("/"))

        if not top_dirs:
            print("[Notice] No requested top-level dirs present (train/test/splits).")
            return

        # Walk only selected top-level dirs; build job list
        all_jobs: List[Tuple[str, str, Optional[int]]] = []  # (remote_url, local_path, expected_size or None)
        for td in top_dirs:
            for url_abs, dirs, files in https_walk(session_probe, td):
                for fname in files:
                    remote_url = f"{url_abs.rstrip('/')}/{fname}"
                    # Relative path under chosen_root (for output structure)
                    rel = remote_url.split(f"{chosen_root}/", 1)[-1]
                    local = os.path.join(args.out, rel)
                    # We'll lazily probe size via HEAD in the worker; set None here
                    all_jobs.append((remote_url, local, None))
    finally:
        session_probe.close()

    if not all_jobs:
        print("[Notice] No matching files found.")
        return

    print(f"[Info] Selected {len(all_jobs)} files")
    ensure_dir(args.out)

    print(f"[3/4] Starting downloads with {WORKERS} HTTPS workers ...")
    job_q: Queue = Queue()
    done_q: Queue = Queue()
    stop_event = threading.Event()

    threads = []
    for _ in range(WORKERS):
        t = threading.Thread(target=worker_loop, args=(job_q, done_q, stop_event), daemon=True)
        t.start()
        threads.append(t)

    for job in all_jobs:
        job_q.put(job)

    downloaded = skipped = errors = 0
    completed = 0
    bar = tqdm(total=len(all_jobs), desc="Progress", unit="file")
    try:
        while completed < len(all_jobs):
            msg = done_q.get()
            if not msg:
                continue
            kind = msg[0]
            if kind == "ok":
                downloaded += 1
            elif kind == "skip":
                skipped += 1
            else:
                errors += 1
                remote = msg[1]
                emsg = msg[2] if len(msg) > 2 else ""
                print(f"ERR {remote} :: {emsg}")
            completed += 1
            bar.update(1)
    except KeyboardInterrupt:
        print("\n[ABORTED] Ctrl+C detected – stopping downloads...")
        stop_event.set()
        # Drain the queue so workers can exit
        while not job_q.empty():
            try:
                job_q.get_nowait(); job_q.task_done()
            except Exception:
                break
        for _ in range(WORKERS):
            job_q.put(None)
        for t in threads:
            t.join(timeout=2)
        bar.close()
        raise SystemExit(1)
    finally:
        for _ in range(WORKERS):
            job_q.put(None)
        for t in threads:
            t.join()
        bar.close()

    print("[4/4] Done.")
    print(f"   Downloaded: {downloaded}")
    print(f"   Skipped:    {skipped}")
    print(f"   Errors:     {errors}")

if __name__ == "__main__":
    main()
