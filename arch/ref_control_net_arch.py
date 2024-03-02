import numpy as np
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from utils.logging_tool import get_logger

from .fema_vqgan_arch import FaceCoderNet, ResBlock


class RefControlNet(nn.Module):

    def __init__(self, vq_config_path, vq_state_dict, bilevel=False):
        super().__init__()
        logger = get_logger()
        # load face vq_model
        if vq_config_path is not None:
            vq_config = OmegaConf.load(vq_config_path)
            self.vqgan = FaceCoderNet(**vq_config.g_model)
            if os.path.exists(vq_state_dict):
                self.vqgan.load_state_dict(torch.load(vq_state_dict, map_location='cpu'))
                logger.info(f"Load vq model weight from {vq_state_dict}")
            else:
                raise FileNotFoundError(f"vq_model weight {vq_state_dict} not found")

        # freeze vqgan
        for p in self.vqgan.parameters():
            p.requires_grad = False

        self.bilevel = bilevel
        # we only use encoder to encode reference image
        self.controller = nn.ModuleList()
        for ch in self.vqgan.multiscale_encoder.latent_out_ch[::-1]:
            # self.controller.append(nn.Conv2d(ch, ch, 1, 1, 0), )
            self.controller.append(AdaConvBlock(ch, ch, self.bilevel))

        self._zero_init()

    def _zero_init(self):
        for m in self.controller.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            else:
                NotImplementedError(f"{m} does not support yet.")

    def forward(self, x, ref):
        # encode original image
        z = self.vqgan.encode(x)
        z_q, _, _ = self.vqgan.quantize(z)

        # generate control signal
        control_latent = self.control_signal(ref)

        # decode z_q image with control signals
        y = self.vqgan.decode(z_q=z_q, control_latent=control_latent)
        return y

    def control_signal(self, ref):
        # encode reference image
        multiscale_latent = self.vqgan.multiscale_encoder(ref, return_latent=True)[::-1]
        # generate control signal
        control_latent = []
        for m, latent in zip(self.controller, multiscale_latent):
            control_latent.append(partial(m, enc_feat=latent))
        # control_latent = [m(latent) for m, latent in zip(self.controller, multiscale_latent)]

        return control_latent

    def __str__(self):
        return self.__class__.__name__.lower()


class AdaConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, bilevel=False):
        super().__init__()
        self.bilevel = bilevel
        self.fuse_encoder = nn.Sequential(
            ResBlock(2 * in_channels, 2 * in_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.mean_var = nn.Sequential(
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * out_channels, 2 * out_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
        )

        if self.bilevel:
            self.up_mean_var = nn.Sequential(
                nn.Upsample(2),
                ResBlock(out_channels, out_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, 2 * out_channels, kernel_size=1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            )

    def forward(self, enc_feat, dec_feat):
        assert enc_feat.size() == dec_feat.size()
        fused_feat = self.fuse_encoder(torch.cat([enc_feat, dec_feat], dim=1))
        shift, scale = torch.chunk(self.mean_var(fused_feat), chunks=2, dim=1)

        if self.bilevel:
            up_shift, up_scale = torch.chunk(self.up_mean_var(fused_feat), chunks=2, dim=1)
            return dec_feat + (dec_feat + shift) * scale, (up_shift, up_scale)
        else:
            return dec_feat + (dec_feat + shift) * scale
