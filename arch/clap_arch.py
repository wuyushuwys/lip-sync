import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from .fema_vqgan_arch import FaceCoderNet
from .auxiliary_arch import AudioNet


class ClAP(nn.Module):

    def __init__(self, embed_dim,
                 vq_config_path, vq_state_dict,
                 # audio_config_path, audio_ckpt,
                 quick_gelu: bool = False,
                 init_logit_scale=np.log(1 / 0.07),
                 init_logit_bias=None):
        super().__init__()

        # load face vq_model
        vq_config = OmegaConf.load(vq_config_path)
        self.vision = FaceCoderNet(**vq_config.g_model)
        if os.path.exists(vq_state_dict):
            self.vision.load_state_dict(torch.load(vq_state_dict, map_location='cpu'))
            logger.info(f"Load vq model weight from {vq_state_dict}")
        else:
            raise FileNotFoundError(f"vq_model weight {vq_state_dict} not found")

        # freeze face vq_model
        for p in visual_encoder.parameters():
            p.requires_grad = False

        # create audio encoder
        self.audio_encoder = AudioNet(emb_dim=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None