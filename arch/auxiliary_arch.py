import torch

from torch import nn

from utils.logging_tool import get_logger


class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=76, seq_len=8):
        super(AudioAttNet, self).__init__()
        logger = get_logger()

        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len,
                      out_features=self.seq_len, bias=True),
            nn.Softmax(dim=2)
        )

        logger.info(f"build {self.__str__()}")

    def forward(self, x):
        x = x.transpose(-1, -2).squeeze(1)  # bsz, 1, seq, emb
        y = x.permute(0, 2, 1)  # bsz * 1 * 80 * 16 -> bsz * 80 * 16 -> bsz * 16 * 80
        y = self.attentionConvNet(y)
        y = self.attentionNet(y)
        return torch.matmul(y, x).squeeze(1)

    def __str__(self):
        return self.__class__.__name__.lower()


class AudioNet(nn.Module):
    """
    AudioNet feed-in mel-spectrogram and generate audio latent for cross attention

    Input: mel-spectrogram bsz * 1 * 80 * 16 ( 1 * 80 * mel_step_size)
    Output: audio latent bsz * seq_len * num_embed (we map mel_step_size with seq_len)

    """

    def __init__(self, emb_dim=256, seq_len=16, downsample=False, out_seq=1):
        super(AudioNet, self).__init__()
        logger = get_logger()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(80, 32,
                      kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32,
                      kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64,
                      kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64,
                      kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, emb_dim),
        )

        self.downsample = downsample
        if self.downsample:
            self.squeeze_encode = nn.Conv1d(seq_len, out_seq, kernel_size=3, stride=1, padding=1)

        logger.info(f"build {self.__str__()}")

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: mel-spectrogram bsz * 1 * 80 * 16 ( bsz * 1 * 80 * mel_step_size)

        Returns: bsz * seq_len * num_embed

        """
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # bsz * 1 * 80 * 16 -> bsz * 80 * 16
        x = self.encoder_conv(x).permute(0, 2, 1)
        if self.downsample:
            x = self.squeeze_encode(x)
        x = self.encoder_fc1(x)  # bsz * seq_len * num_embed
        return x

    def __str__(self):
        return self.__class__.__name__.lower()


class LipNet(nn.Module):

    def __init__(self, emb_dim=256):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 4, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        # x = x.reshape(x.shape[0], -1)
        x = self.encoder_fc1(x).squeeze()
        return x


class LipAttn(nn.Module):

    def __init__(self, emb_dim=256):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2,
                      padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, emb_dim, kernel_size=3, stride=2,
                      padding=1, bias=True),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # x: bsz, seq, emb_dim
        assert x.dim() == 3
        attn = x.
        x = self.encoder_conv(x)

        return x
