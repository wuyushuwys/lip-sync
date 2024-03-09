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

    def __init__(self, vq_config_path, vq_state_dict, modulate_type='ada_gated_modulate', zero_init=False):
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

        # we only use encoder to encode reference image
        self.controller = nn.ModuleList()
        for ch in self.vqgan.multiscale_encoder.latent_out_ch[::-1]:
            self.controller.append(AdaConvBlock(ch, ch, modulate_type=modulate_type))

        if zero_init:
            self._zero_init()

    def _zero_init(self):
        for m in self.controller.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

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


def ada_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def ada_gated_modulate(x, shift, scale, gate):
    return x + gate * (x * (1 + scale) + shift)


def ada_residual_modulate(x, shift, scale):
    return x + (x + shift) * scale


class AdaConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, modulate_type='ada_modulate'):
        super().__init__()
        self.fuse_encoder = nn.Sequential(
            ResBlock(2 * in_channels, 2 * in_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1)
        )

        self.modulate_type = modulate_type

        assert modulate_type in ['ada_modulate',
                                 'ada_gated_modulate',
                                 'ada_residual_modulate'], f"Not supported {modulate_type}"

        if modulate_type == 'ada_gated_modulate':
            self.num_split = 3
        else:
            self.num_split = 2

        self.ada_modulation = nn.Sequential(
            nn.Conv2d(out_channels, self.num_split * out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.num_split * out_channels, self.num_split * out_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, enc_feat, dec_feat):
        assert enc_feat.size() == dec_feat.size()
        fused_feat = self.fuse_encoder(torch.cat([enc_feat, dec_feat], dim=1))

        if self.modulate_type == 'ada_modulate':
            shift, scale = torch.chunk(self.ada_modulation(fused_feat), chunks=self.num_split, dim=1)
            return ada_modulate(dec_feat, shift, scale)
        elif self.modulate_type == 'ada_gated_modulate':
            shift, scale, gated = torch.chunk(self.ada_modulation(fused_feat), chunks=self.num_split, dim=1)
            return ada_gated_modulate(dec_feat, shift, scale, gated)
        elif self.modulate_type == 'ada_residual_modulate':
            shift, scale = torch.chunk(self.ada_modulation(fused_feat), chunks=self.num_split, dim=1)
            return ada_residual_modulate(dec_feat, shift, scale)
        else:
            NotImplementedError(f'{self.modulate_type} not implemented')
