#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Support Python 3.8
    @author: Lou Xiao(louxiao@i32n.com)
    @maintainer: Lou Xiao(louxiao@i32n.com)
    @copyright: Copyright 2018~2023
    @created time: 2023-04-05 15:05:52 CST
    @updated time: 2023-04-05 15:05:52 CST
"""

import torch
import torch.nn as nn

from diffusers.models.vae import DiagonalGaussianDistribution
from diffusers.models.vae import VectorQuantizer


class ConvBlock(nn.Module):

    def __init__(self, num_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.layers(inputs)
        return h


class ResBlock(nn.Module):

    def __init__(self, num_channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = inputs + self.residual(inputs)
        return h


class AutoEncoder(nn.Module):

    def __init__(self, num_channels: int, base_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.base_channels = base_channels

        self.conv_in = nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            # stage 1
            nn.Sequential(
                ConvBlock(base_channels),
                ConvBlock(base_channels),
            ),
            # stage 2
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(2 * base_channels),
                ConvBlock(2 * base_channels),
            ),
            # stage 3
            nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(4 * base_channels),
                ConvBlock(4 * base_channels),
            ),
            # stage 4
            nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(8 * base_channels),
                ConvBlock(8 * base_channels),
            ),
        )
        self.decoder = nn.Sequential(
            # stage 4
            nn.Sequential(
                ConvBlock(8 * base_channels),
                ConvBlock(8 * base_channels),
            ),
            # stage 3
            nn.ConvTranspose2d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(4 * base_channels),
                ConvBlock(4 * base_channels),
            ),
            # stage 2
            nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(2 * base_channels),
                ConvBlock(2 * base_channels),
            ),
            # stage 1
            nn.ConvTranspose2d(2 * base_channels, 1 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(1 * base_channels),
                ConvBlock(1 * base_channels),
            ),
        )
        self.conv_out = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(inputs)
        h = self.encoder(h)
        h = self.decoder(h)
        h = self.conv_out(h)
        return h


class AutoEncoderKL(nn.Module):

    def __init__(
            self,
            num_channels: int,
            latent_dim: int,
            base_channels: int = 64
    ):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.conv_in = nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            # stage 1
            nn.Sequential(
                ConvBlock(base_channels),
                ConvBlock(base_channels),
            ),
            # stage 2
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(2 * base_channels),
                ConvBlock(2 * base_channels),
            ),
            # stage 3
            nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(4 * base_channels),
                ConvBlock(4 * base_channels),
            ),
            # stage 4
            nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(8 * base_channels),
                ConvBlock(8 * base_channels),
            ),
        )

        self.encode_latent = nn.Conv2d(8 * base_channels, 2 * latent_dim, kernel_size=1, stride=1, padding=0)
        # KL sampling
        self.decode_latent = nn.Conv2d(latent_dim, 8 * base_channels, kernel_size=1, stride=1, padding=0)

        self.decoder = nn.Sequential(
            # stage 4
            nn.Sequential(
                ConvBlock(8 * base_channels),
                ConvBlock(8 * base_channels),
            ),
            # stage 3
            nn.ConvTranspose2d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(4 * base_channels),
                ConvBlock(4 * base_channels),
            ),
            # stage 2
            nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(2 * base_channels),
                ConvBlock(2 * base_channels),
            ),
            # stage 1
            nn.ConvTranspose2d(2 * base_channels, 1 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(1 * base_channels),
                ConvBlock(1 * base_channels),
            ),
        )
        self.conv_out = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, inputs: torch.Tensor, sampling: bool = False, return_loss: bool = False):
        h = self.conv_in(inputs)
        h = self.encoder(h)
        h = self.encode_latent(h)  # avg, std
        dist = DiagonalGaussianDistribution(h)
        if sampling:
            return dist.sample()
        elif return_loss:
            kl_loss = dist.kl().mean()
            return dist.sample(), kl_loss
        else:
            return dist.mode()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.decode_latent(latent)
        h = self.decoder(h)
        h = self.conv_out(h)
        return h


class AutoEncoderVQ(nn.Module):

    def __init__(
            self,
            num_channels: int,
            latent_dim: int,
            base_channels: int = 64,
            num_vq_embeddings: int = 8192,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_vq_embeddings = num_vq_embeddings

        self.conv_in = nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            # stage 1
            nn.Sequential(
                ConvBlock(base_channels),
                ConvBlock(base_channels),
            ),
            # stage 2
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(2 * base_channels),
                ConvBlock(2 * base_channels),
            ),
            # stage 3
            nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(4 * base_channels),
                ConvBlock(4 * base_channels),
            ),
            # stage 4
            nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(8 * base_channels),
                ConvBlock(8 * base_channels),
            ),
        )

        self.encode_latent = nn.Conv2d(8 * base_channels, latent_dim, kernel_size=1, stride=1, padding=0)
        #  VQ
        self.vq = VectorQuantizer(num_vq_embeddings, latent_dim, beta=0.25, sane_index_shape=True, legacy=False)
        self.decode_latent = nn.Conv2d(latent_dim, 8 * base_channels, kernel_size=1, stride=1, padding=0)

        self.decoder = nn.Sequential(
            # stage 4
            nn.Sequential(
                ConvBlock(8 * base_channels),
                ConvBlock(8 * base_channels),
            ),
            # stage 3
            nn.ConvTranspose2d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(4 * base_channels),
                ConvBlock(4 * base_channels),
            ),
            # stage 2
            nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(2 * base_channels),
                ConvBlock(2 * base_channels),
            ),
            # stage 1
            nn.ConvTranspose2d(2 * base_channels, 1 * base_channels, kernel_size=2, stride=2, padding=0),
            nn.Sequential(
                ConvBlock(1 * base_channels),
                ConvBlock(1 * base_channels),
            ),
        )
        self.conv_out = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, inputs: torch.Tensor, sampling: bool = False, return_loss: bool = False):
        h = self.conv_in(inputs)
        h = self.encoder(h)
        h = self.encode_latent(h)  # avg, std
        z_q, loss, _ = self.vq(h)
        if sampling:
            return z_q
        elif return_loss:  # train
            return z_q, loss
        else:
            return h

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.decode_latent(latent)
        h = self.decoder(h)
        h = self.conv_out(h)
        return h


class PatchGANDiscriminator(nn.Module):

    def __init__(self, in_channels: int, num_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(2 * in_channels, num_channels, kernel_size=4, stride=2, padding=1),
                # nn.BatchNorm2d(num_channels),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(1 * num_channels, 2 * num_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(2 * num_channels),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(2 * num_channels, 4 * num_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(4 * num_channels),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(4 * num_channels, 8 * num_channels, kernel_size=4, stride=1, padding=1),
                nn.BatchNorm2d(8 * num_channels),
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(8 * num_channels, 1, kernel_size=4, stride=1, padding=1),
                nn.Sigmoid(),
            ),
        )

    # forward method
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        h = torch.cat([outputs, targets], 1)
        h = self.layers(h)
        return h


class UNet(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            base_channels: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.encoder_list = nn.ModuleList([
            # stage 1
            nn.Sequential(
                ResBlock(1 * base_channels),
                ResBlock(1 * base_channels),
            ),
            # stage 2
            nn.Sequential(
                nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock(2 * base_channels),
                ResBlock(2 * base_channels),
            ),
            # stage 3
            nn.Sequential(
                nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock(4 * base_channels),
                ResBlock(4 * base_channels),
            ),
            # stage 4
            nn.Sequential(
                nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock(8 * base_channels),
                ResBlock(8 * base_channels),
            ),
        ])
        self.middle = nn.Sequential(
            nn.Conv2d(8 * base_channels, 32 * base_channels, kernel_size=2, stride=2, padding=0),
            ResBlock(32 * base_channels),
            ResBlock(32 * base_channels),
            nn.ConvTranspose2d(32 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
        )
        self.decoder_list = nn.Sequential(
            # stage 4
            nn.Sequential(
                nn.Conv2d(2 * 8 * base_channels, 8 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(8 * base_channels),
                ResBlock(8 * base_channels),
                nn.ConvTranspose2d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 3
            nn.Sequential(
                nn.Conv2d(2 * 4 * base_channels, 4 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(4 * base_channels),
                ResBlock(4 * base_channels),
                nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 2
            nn.Sequential(
                nn.Conv2d(2 * 2 * base_channels, 2 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(2 * base_channels),
                ResBlock(2 * base_channels),
                nn.ConvTranspose2d(2 * base_channels, 1 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 1
            nn.Sequential(
                nn.Conv2d(2 * 1 * base_channels, 1 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(1 * base_channels),
                ResBlock(1 * base_channels),
            ),
        )
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(inputs)
        skip_list = []
        for m in self.encoder_list:
            h = m(h)
            skip_list.insert(0, h)
        h = self.middle(h)
        for m, skip in zip(self.decoder_list, skip_list):
            h = torch.concat([skip, h], dim=1)
            h = m(h)
        h = self.conv_out(h)
        return h


class TwoWaysModule(object):
    pass


class TwoWaysSequential(nn.Module):

    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        h = inputs
        for m in self.module_list:
            if isinstance(m, TwoWaysModule):
                h = m(h, conditions)
            else:
                h = m(h)
        return h


class CrossAttentionBlock(nn.Module, TwoWaysModule):

    def __init__(
            self,
            num_channels: int,
            condition_dim: int,
            num_heads: int = 8,
            layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.layer_norm = nn.GroupNorm(1, num_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels,
            kdim=condition_dim,
            vdim=condition_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_scale = nn.Parameter(torch.full([num_channels, 1, 1], layer_scale_init, dtype=torch.float))

    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        # inputs: shape[B, C, H, W]
        # conditions: shape[B, L, C']
        h = self.layer_norm(inputs)
        # cross attention
        bb, cc, hh, ww = h.shape
        h = h.reshape([bb, cc, hh * ww])  # [B, C, L]
        h = torch.swapdims(h, 1, 2)  # [B, L, C]
        h, _ = self.attention(h, conditions, conditions)  # Q, K, V
        h = torch.swapdims(h, 1, 2)
        h = h.reshape([bb, cc, hh, ww])
        h = inputs + self.layer_scale * h  # residual
        return h


class ConditionalUNet(nn.Module):

    def __init__(
            self,
            num_channels: int,
            condition_dim: int = 512,
            base_channels: int = 64,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.base_channels = base_channels
        self.condition_dim = condition_dim

        self.conv_in = nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.encoder_list = nn.ModuleList([
            # stage 1
            TwoWaysSequential(
                ResBlock(1 * base_channels),
                CrossAttentionBlock(1 * base_channels, condition_dim),
                ResBlock(1 * base_channels),
                CrossAttentionBlock(1 * base_channels, condition_dim),
            ),
            # stage 2
            TwoWaysSequential(
                nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock(2 * base_channels),
                CrossAttentionBlock(2 * base_channels, condition_dim),
                ResBlock(2 * base_channels),
                CrossAttentionBlock(2 * base_channels, condition_dim),
            ),
            # stage 3
            TwoWaysSequential(
                nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock(4 * base_channels),
                CrossAttentionBlock(4 * base_channels, condition_dim),
                ResBlock(4 * base_channels),
                CrossAttentionBlock(4 * base_channels, condition_dim),
            ),
            # stage 4
            TwoWaysSequential(
                nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock(8 * base_channels),
                ResBlock(8 * base_channels),
            ),
        ])
        self.middle = TwoWaysSequential(
            nn.Conv2d(8 * base_channels, 32 * base_channels, kernel_size=2, stride=2, padding=0),
            ResBlock(32 * base_channels),
            CrossAttentionBlock(32 * base_channels, condition_dim),
            ResBlock(32 * base_channels),
            CrossAttentionBlock(32 * base_channels, condition_dim),
            nn.ConvTranspose2d(32 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
        )
        self.decoder_list = nn.ModuleList([
            # stage 4
            TwoWaysSequential(
                nn.Conv2d(2 * 8 * base_channels, 8 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(8 * base_channels),
                CrossAttentionBlock(8 * base_channels, condition_dim),
                ResBlock(8 * base_channels),
                CrossAttentionBlock(8 * base_channels, condition_dim),
                nn.ConvTranspose2d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 3
            TwoWaysSequential(
                nn.Conv2d(2 * 4 * base_channels, 4 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(4 * base_channels),
                CrossAttentionBlock(4 * base_channels, condition_dim),
                ResBlock(4 * base_channels),
                CrossAttentionBlock(4 * base_channels, condition_dim),
                nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 2
            TwoWaysSequential(
                nn.Conv2d(2 * 2 * base_channels, 2 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(2 * base_channels),
                CrossAttentionBlock(2 * base_channels, condition_dim),
                ResBlock(2 * base_channels),
                CrossAttentionBlock(2 * base_channels, condition_dim),
                nn.ConvTranspose2d(2 * base_channels, 1 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 1
            TwoWaysSequential(
                nn.Conv2d(2 * 1 * base_channels, 1 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock(1 * base_channels),
                CrossAttentionBlock(1 * base_channels, condition_dim),
                ResBlock(1 * base_channels),
                CrossAttentionBlock(1 * base_channels, condition_dim),
            ),
        ])
        self.conv_out = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(inputs)
        skip_list = []
        for m in self.encoder_list:
            h = m(h, conditions)
            skip_list.insert(0, h)
        h = self.middle(h, conditions)
        for m, skip in zip(self.decoder_list, skip_list):
            h = torch.concat([skip, h], dim=1)
            h = m(h, conditions)
        h = self.conv_out(h)
        return h


def debug():
    net = ConditionalUNet(3, condition_dim=512)
    xx = torch.rand([4, 3, 128, 128])
    cc = torch.rand([4, 128, 512])
    yy = net(xx, cc)
    print(yy.shape)


if __name__ == '__main__':
    debug()
