import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchgeometry as tgm
import torchio as tio
torch.multiprocessing.set_sharing_strategy('file_system')

"""
These codes are:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


def train_encoder_izif(opt, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("/afs/crc.nd.edu/user/z/zpan3/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/generator.pth"))
    discriminator.load_state_dict(torch.load("/afs/crc.nd.edu/user/z/zpan3/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/discriminator.pth"))
    print("Loaded pre-trained models")

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()
    ssim = tgm.losses.SSIM(window_size=5, reduction='mean')

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("/afs/crc.nd.edu/user/z/zpan3/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/images_e", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataloader):
            imgs = batch['vol'][tio.DATA].squeeze(-1)

            # Configure input
            real_imgs = imgs.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            real_features = discriminator.forward_features(real_imgs)
            # Fake features
            fake_features = discriminator.forward_features(fake_imgs)

            # izif architecture
            loss_imgs = ssim(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            # Output training log every n_critic steps
            if i % opt.n_critic == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

                if batches_done % opt.sample_interval == 0:
                    fake_z = encoder(fake_imgs)
                    reconfiguration_imgs = generator(fake_z)
                    save_image(reconfiguration_imgs.data[:25],
                               f"/afs/crc.nd.edu/user/z/zpan3/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/images_e/{batches_done:06}.png",
                               nrow=5, normalize=True)

                batches_done += opt.n_critic
                torch.save(encoder.state_dict(), "/afs/crc.nd.edu/user/z/zpan3/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/encoder.pth")
    torch.save(encoder.state_dict(), "/afs/crc.nd.edu/user/z/zpan3/Models/f-AnoGAN/cell_complexeye/outputs/2025-02-23/21-08-35/results/encoder.pth")
    print("Training finished")
    print("Saved encoder model")
