import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader

from fanogan.test_anomaly_detection import fANOGAN

from cell_complexeye.dataloader.Datamodules_eval import DNA_ANO
from types import SimpleNamespace
import hydra
from omegaconf import DictConfig
import argparse


# Define your argument parser for `opt`
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=192,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    opt = parser.parse_args()
    return opt


@hydra.main(config_path="./", config_name="config.yaml")
def main(cfg: DictConfig):
    cfg = cfg.cfg
    opt = get_opt()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for fold in range(0, 5):
        # Load dataset
        CellDataset = DNA_ANO(cfg, fold=fold)
        test_dataloader = CellDataset.test_dataloader()
        val_dataloader = CellDataset.val_dataloader()

        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from mvtec_ad.model import Generator, Discriminator, Encoder

        generator = Generator(opt)
        discriminator = Discriminator(opt)
        encoder = Encoder(opt)

        fanogan = fANOGAN(cfg)
        fanogan.test_anomaly_detection(opt, generator, discriminator, encoder,
                                       val_dataloader, device, fold, True)
        fanogan.test_anomaly_detection(opt, generator, discriminator, encoder,
                                       test_dataloader, device, fold, False)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    main()
