from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from .auxiliary_arch import AudioAttNet, AudioNet
from .conditioned_mage_arch import DoubleConditionedMAGE


class SyncMage(DoubleConditionedMAGE):

    def __init__(self, num_embed=1024, **kwargs):
        super(SyncMage, self).__init__(**kwargs)
        for p in self.parameters():
            p.requires_grad = False

        self.img_probe = nn.Linear(self.encoder_embed_dim, num_embed)

        self.audio_encoder = AudioAttNet(seq_len=16, dim_aud=80)
        self.audio_probe = nn.Linear(80, num_embed)

    @override
    def forward(self, imgs, gt=None, ref=None, audio=None, generate=False, return_loss=True):
        # encode image with pre-trained Transformer encoder
        img_latent, _, _, _ = self.forward_encoder(imgs, gt)
        img_embed = self.img_probe(img_latent[:, 0]).squeeze(1)

        # encode audio with audio_encoder, audio input should be bsz, 1, 80, 16 (only for consistency with wav2lip)
        audio_latent = self.audio_encoder(audio)
        audio_embed = self.audio_probe(audio_latent).squeeze(1)

        assert audio_embed.size() == img_embed.size(), f"{audio_embed.size()}, {img_embed.size()}"

        return audio_embed, img_embed


