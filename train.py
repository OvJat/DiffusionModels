#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Support Python 3.8
    @author: Lou Xiao(louxiao@i32n.com)
    @maintainer: Lou Xiao(louxiao@i32n.com)
    @copyright: Copyright 2018~2023
    @created time: 2023-04-05 18:19:12 CST
    @updated time: 2023-04-05 18:19:12 CST
"""

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch.optim import AdamW
import torch.utils.data as tud

from diffusers.models.embeddings import get_timestep_embedding
from diffusers.schedulers import DDPMScheduler

from models import AutoEncoderKL
from models import AutoEncoderVQ
from models import PatchGANDiscriminator
from models import ConditionalUNet


# Fake Dataset, just for demo.
class FakeDataset(tud.Dataset):

    def __init__(self, src_shape=(17, 128, 128), dst_shape=(1, 128, 128)):
        self.src_shape = src_shape
        self.dst_shape = dst_shape
        self.sample_count = 10000

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index: int):
        xx = torch.rand(self.src_shape, dtype=torch.float32)
        yy = torch.rand(self.dst_shape, dtype=torch.float32)
        return xx, yy


def train_vae():
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32

    # init model VAE or VQ-VAE or VQ-GAN
    vae = AutoEncoderKL(17, latent_dim=128)
    # vae = AutoEncoderVQ(17, latent_dim=128)
    # init weight (optional)
    vae.to(device=default_device, dtype=default_type)
    optimizer = AdamW(vae.parameters(), lr=1e-4, weight_decay=0.05)

    discriminator = PatchGANDiscriminator(17)
    discriminator.to(device=default_device, dtype=default_type)
    discriminator_optimizer = AdamW(vae.parameters(), lr=2e-4, weight_decay=0.05)

    # init dataset
    ds = FakeDataset()
    dl = tud.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    vae.train()
    discriminator.train()
    for batch, (xx, yy) in enumerate(dl):
        xx = xx.to(device=default_device, dtype=default_type)
        # yy = yy.to(device=default_device, dtype=default_type)

        # train discriminator
        discriminator_optimizer.zero_grad()
        with torch.no_grad():
            z = vae.encode(xx)
            fake = vae.decode(z)
        d_fake = discriminator(fake, xx).reshape(-1, 1)
        d_fake_loss = tnf.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        d_real = discriminator(xx, xx).reshape(-1, 1)
        d_real_loss = tnf.binary_cross_entropy(d_real, torch.ones_like(d_real))
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        discriminator_optimizer.step()

        # train step
        optimizer.zero_grad()
        z, kl_loss = vae.encode(xx, return_loss=True)
        # print("kl_loss:",kl_loss)
        x_hat = vae.decode(z)
        d_real = discriminator(x_hat, xx)
        gan_loss = tnf.binary_cross_entropy(d_real, torch.ones_like(d_real))
        mse_loss = tnf.mse_loss(x_hat, xx)
        mse_loss = mse_loss / (mse_loss.detach() + 1e-6)
        loss = mse_loss + gan_loss + kl_loss
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        print("Batch {:10d} | Loss: {:8.4f}".format(batch, loss))


def make_conditions(timesteps: torch.Tensor, images: torch.Tensor = None, embedding_dim: int = 128) -> torch.Tensor:
    assert timesteps.ndim == 1

    timestep_embedding = get_timestep_embedding(timesteps, embedding_dim, max_period=10000)  # [B, C]
    timestep_embedding = timestep_embedding[:, None, :]  # [B, 1, C]

    if images is not None:
        assert images.shape[1] == embedding_dim
        img_embed = torch.flatten(images, 2)  # [B, C, H*W]
        img_embed = torch.swapdims(img_embed, 1, 2)  # [B, H*W, C]
        condition_embedding = torch.cat([timestep_embedding, img_embed], dim=1)  # [B, 1+L, C]
    else:
        condition_embedding = timestep_embedding

    length = condition_embedding.shape[1]
    positions = torch.arange(0, length, device=condition_embedding.device, dtype=condition_embedding.dtype)
    position_embedding = get_timestep_embedding(positions, embedding_dim, max_period=10000)  # [1+L, C]
    position_embedding = position_embedding[None, ...]  # [1, L, C]
    condition_embedding += position_embedding

    return condition_embedding


def train_diffusion():
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32

    # init model
    unet = ConditionalUNet(128, condition_dim=512)
    unet.to(device=default_device, dtype=default_type)
    optimizer = AdamW(unet.parameters(), lr=1e-4, weight_decay=0.05)

    # load encoder
    # Encoder Source domain
    src_vae = AutoEncoderVQ(17, latent_dim=512)  # as condition
    src_vae.to(device=default_device, dtype=default_type)
    # load from checkpoint
    src_vae.requires_grad_(False)
    src_vae.eval()

    # Encoder Target domain
    tgt_vae = AutoEncoderVQ(1, latent_dim=128)
    tgt_vae.to(device=default_device, dtype=default_type)
    # load from checkpoint
    tgt_vae.requires_grad_(False)
    tgt_vae.eval()

    # init dataset
    ds = FakeDataset()
    dl = tud.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    # init noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=False,
    )

    unet.train()
    for batch, (xx, yy) in enumerate(dl):
        xx = xx.to(device=default_device, dtype=default_type)
        yy = yy.to(device=default_device, dtype=default_type)

        # train step
        optimizer.zero_grad()
        with torch.no_grad():
            src_latent = src_vae.encode(xx)
            tgt_latent = tgt_vae.encode(yy)  # x_0
            # make condition
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (tgt_latent.shape[0],), device=default_device, dtype=torch.long)
            if batch % 2 == 0:
                conditions = make_conditions(timesteps, src_latent, embedding_dim=512)
            else:
                conditions = make_conditions(timesteps, None, embedding_dim=512)
            # add noise
            noise = torch.randn_like(tgt_latent)
            tgt_latent = noise_scheduler.add_noise(tgt_latent, noise, timesteps)
        # learning noise
        outputs = unet(tgt_latent, conditions)
        loss = tnf.mse_loss(outputs, noise)
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        print("Batch {:10d} | Loss: {:8.4f}".format(batch, loss))


def sampling_diffusion():
    default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    default_type = torch.float32

    # loading model
    unet = ConditionalUNet(128, condition_dim=512)
    unet.to(device=default_device, dtype=default_type)
    # load from checkpoint
    unet.requires_grad_(False)
    unet.eval()

    # Encoder Source domain
    src_vae = AutoEncoderVQ(17, latent_dim=512)  # as condition
    src_vae.to(device=default_device, dtype=default_type)
    # load from checkpoint
    src_vae.requires_grad_(False)
    src_vae.eval()

    # Encoder Target domain
    tgt_vae = AutoEncoderVQ(1, latent_dim=128)
    tgt_vae.to(device=default_device, dtype=default_type)
    # load from checkpoint
    tgt_vae.requires_grad_(False)
    tgt_vae.eval()

    # init dataset
    ds = FakeDataset()
    dl = tud.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    # init noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=False,
    )

    guidance_scale = 7.2
    noise_scheduler.set_timesteps(100)
    timesteps = noise_scheduler.timesteps
    print("sampling timesteps:", timesteps)
    with torch.no_grad():
        for batch, (xx, _) in enumerate(dl):
            xx = xx.to(device=default_device, dtype=default_type)
            src_latent = src_vae.encode(xx)
            tgt_latent = torch.randn([xx.shape[0], 128, 16, 16], device=default_device, dtype=default_type)
            # sampling steps
            for t in timesteps.tolist():
                ts = torch.full([xx.shape[0]], t, device=default_device, dtype=default_type)
                # conditional sampling
                conditions = make_conditions(ts, src_latent, embedding_dim=512)
                conditional_noise = unet(tgt_latent, conditions)
                # unconditional sampling
                conditions = make_conditions(ts, None, embedding_dim=512)
                unconditional_noise = unet(tgt_latent, conditions)
                # Classifier-Free Guidance
                noise = guidance_scale * conditional_noise + (1 - guidance_scale) * unconditional_noise
                tgt_latent = noise_scheduler.step(noise, t, tgt_latent).prev_sample
            # decode, get target sample.
            tgt = tgt_vae.decode(tgt_latent)
            print("target sample:", tgt.shape)


def main():
    # train_vae()
    # train_diffusion()
    sampling_diffusion()


if __name__ == '__main__':
    main()
