from torch import nn
from torch.nn import functional as F

from .ops import Conv2d, ResBlock, Shape
from utils.evaluation import evaluate_sync


class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),

            nn.AdaptiveAvgPool2d(1)
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0), )

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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return "SyncNet"

    @staticmethod
    def evaluate(*args, **kwargs):
        return evaluate_sync.evaluation(*args, **kwargs)


class SyncNet_Color(nn.Module):
    def __init__(self):
        super(SyncNet_Color, self).__init__()

        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(32, 32, kernel_size=5, stride=1),

            ResBlock(32, 64, kernel_size=5, stride=(1, 2)),
            ResBlock(64, 64, kernel_size=3, stride=1),

            ResBlock(64, 128, kernel_size=3, stride=2),
            ResBlock(128, 128, kernel_size=3, stride=1),

            ResBlock(128, 256, kernel_size=3, stride=2),
            ResBlock(256, 256, kernel_size=3, stride=1),
            ResBlock(256, 256, kernel_size=3, stride=1),

            ResBlock(256, 512, kernel_size=3, stride=2),
            ResBlock(512, 512, kernel_size=3, stride=1),
            ResBlock(512, 512, kernel_size=3, stride=1),

            ResBlock(512, 512, kernel_size=3, stride=2),
            ResBlock(512, 512, kernel_size=3, stride=1),
            ResBlock(512, 512, kernel_size=3, stride=1),

            ResBlock(512, 1024, kernel_size=3, stride=2),
            ResBlock(1024, 1024, kernel_size=3, stride=1),
            ResBlock(1024, 1024, kernel_size=3, stride=1, act='relu'),

            nn.AdaptiveAvgPool2d(1),

        )

        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            ResBlock(32, 32, kernel_size=3, stride=1),
            ResBlock(32, 32, kernel_size=3, stride=1),

            ResBlock(32, 64, kernel_size=3, stride=(3, 1)),
            ResBlock(64, 64, kernel_size=3, stride=1),

            ResBlock(64, 128, kernel_size=3, stride=3),
            ResBlock(128, 128, kernel_size=3, stride=1),
            ResBlock(128, 128, kernel_size=3, stride=1),

            ResBlock(128, 256, kernel_size=3, stride=(3, 2)),
            ResBlock(256, 256, kernel_size=3, stride=1),
            ResBlock(256, 256, kernel_size=3, stride=1),

            ResBlock(256, 512, kernel_size=3, stride=2),
            ResBlock(512, 512, kernel_size=3, stride=1),
            ResBlock(512, 512, kernel_size=3, stride=1),

            ResBlock(512, 1024, kernel_size=3, stride=2),
            ResBlock(1024, 1024, kernel_size=3, stride=1),
            ResBlock(1024, 1024, kernel_size=3, stride=1, act='relu'),

            nn.AdaptiveAvgPool2d(1),

        )

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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return "SyncNet"

    @staticmethod
    def evaluate(*args, **kwargs):
        return evaluate_sync.evaluation(*args, **kwargs)
