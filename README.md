<div align="center">
<h2>AutoQC</h2>

[Zixuan Pan](https://scholar.google.com/citations?user=3VuW2gcAAAAJ&hl=en), [Justin Sonneck](https://scholar.google.com/citations?user=ReDjyuAAAAAJ&hl=en&oi=ao), [Dennis Nagel](https://scholar.google.com/citations?user=bW4feA4AAAAJ&hl=en&oi=ao), Anja Hasenberg, [Matthias Gunzer](https://scholar.google.com/citations?user=1uh0hw4AAAAJ&hl=en&oi=ao), [Yiyu Shi](https://scholar.google.com/citations?hl=en&user=LrjbEkIAAAAJ&view_op=list_works), [Jianxu Chen](https://scholar.google.com/citations?hl=en&user=HdolpOgAAAAJ)

</div>

## Introduction
AutoQC is a benchmarking framework for automatic quality control of high-throughput microscopy images.
It provides a benchmark dataset and a set of baseline methods for evaluating the performance of quality control algorithms.

<p align="center">
  <img src="figs/dataset.png" width="70%">
</p>
<p align="center">Fig. 1: Overview of the benchmark dataset.</p>

## Benchmark Dataset Preparation
All datasets curated in this paper and split files are available at [BioStudies]( https://doi.org/10.6019/S-BIAD2133).
We also provide the pre-trained models for all baselines at [Google Drive](https://drive.google.com/drive/folders/1d8Fa2PZ3z7egrrjQfFMa5ohG7ThlAXIb?usp=share_link).
Once downloaded, please unzip the files and place them in the `data` folder of this repository.

## Installation
Before starting, we recommend to create a new conda environment, and install the required packages in [requirements.txt](requirements.txt). We test our
methods on Python 3.9.7 and cuda 11.8.

## Quick Start
We provide a [jupyter notebook](example.ipynb) in the root directory of this repository, which can be used to run inference on the example data.

To download the benchmark dataset from [BioStudies](https://doi.org/10.6019/S-BIAD2133):

```bash
python download_autoqc_data.py --out ./data
```

Using the flags `--only_train`, `--only_test`, or `--only_splits`, you can download individual subsets.

## Training and Evaluation
Before training, please change the config.yaml, datamodule/DNA.yaml and the method yaml files in the `configs` folder according to your needs.
Then, you can run the training script as follows:
```bash
bash train.sh
```
The test set will be evaluated automatically after training.
