import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch.utils.data import DataLoader

from fanogan.train_encoder_izif import train_encoder_izif

from cell_complexeye.dataloader.Datamodules_train import DNA
from types import SimpleNamespace
import hydra
from omegaconf import DictConfig
import argparse


# Define your argument parser for `opt`
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=192,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=8000,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--fold", type=int, default=0,
                        help="fold number")
    return parser.parse_args()


@hydra.main(config_path="./", config_name="config.yaml")
def main(cfg: DictConfig):
    cfg = cfg.cfg
    opt = get_opt()

    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    CellDataset = DNA(cfg, fold=opt.fold)
    train_dataloader = CellDataset.train_dataloader()

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator, Encoder

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder_izif(opt, generator, discriminator, encoder,
                       train_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    main()
