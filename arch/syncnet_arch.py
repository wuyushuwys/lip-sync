from functools import partial, partialmethod

import torch

from timm.models.vision_transformer import PatchEmbed, Block

from torch import nn
from torch.nn import functional as F

from .ops import Conv2d, get_2d_sincos_pos_embed


def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class SyncNet(nn.Module):
    def __init__(self, norm='bn', dropout=0.1):
        super(SyncNet, self).__init__()

        block = partialclass(Conv2d, norm=norm, dropout=dropout)

        self.face_encoder = nn.Sequential(
            block(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            block(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            block(64, 128, kernel_size=3, stride=2, padding=1),
            block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            block(128, 256, kernel_size=3, stride=2, padding=1),
            block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            block(256, 512, kernel_size=3, stride=2, padding=1),
            block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            block(512, 1024, kernel_size=3, stride=2, padding=1),
            block(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            block(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),

            block(1024, 1024, kernel_size=3, stride=2, padding=1),
            block(1024, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
            block(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'),

            nn.AdaptiveAvgPool2d(1)
        )

        self.audio_encoder = nn.Sequential(
            block(1, 32, kernel_size=3, stride=1, padding=1),
            block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            block(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            block(64, 128, kernel_size=3, stride=3, padding=1),
            block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            block(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            block(256, 512, kernel_size=3, stride=1, padding=1),
            block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            block(512, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
            block(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'), )

        self._init_weights()

    def forward(self, audio_sequences, face_sequences):  # audio_sequences := (B, dim, T)
        # print(audio_sequences.shape, face_sequences.shape)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)

    def __str__(self):
        return self.__class__.__name__.lower()
