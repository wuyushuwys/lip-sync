import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from utils.logging_tool import get_logger

from .fema_vqgan_arch import FaceCoderNet
from .auxiliary_arch import AudioNet


class RefControlNet(nn.Module):

    def __init__(self, vq_config_path, vq_state_dict, ):
        super().__init__()
        logger = get_logger()
        # load face vq_model
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
            self.controller.append(nn.Conv2d(ch, ch, 1, 1, 0))

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

        # encode reference image
        multiscale_latent = self.vqgan.multiscale_encoder(ref, return_latent=True)[::-1]
        # generate control signal
        control_latent = [m(latent) for m, latent in zip(self.controller, multiscale_latent)]

        # decode z_q image with control signals
        y = self.vqgan.decode(z_q=z_q, control_latent=control_latent)
        return y

    def __str__(self):
        return self.__class__.__name__.lower()
