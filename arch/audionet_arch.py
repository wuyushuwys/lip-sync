import torch

from torch import nn

from utils.logging_tool import get_logger


class AudioNet(nn.Module):
    """
    AudioNet feed-in mel-spectrogram and generate audio latent for cross attention

    Input: mel-spectrogram bsz * 1 * 80 * 16 ( 1 * 80 * mel_step_size)
    Output: audio latent bsz * seq_len * num_embed (we map mel_step_size with seq_len)

    """

    def __init__(self, emb_dim=256):
        super(AudioNet, self).__init__()
        logger = get_logger()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(80, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, emb_dim),
        )

        logger.info(f"Create {self.__str__()}")

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: mel-spectrogram bsz * 1 * 80 * 16 ( bsz * 1 * 80 * mel_step_size)

        Returns: bsz * seq_len * num_embed

        """

        x = x.squeeze(1)  # bsz * 1 * 80 * 16 -> bsz * 80 * 16 ->
        x = self.encoder_conv(x).permute(0, 2, 1)
        x = self.encoder_fc1(x)  # bsz * seq_len * num_embed
        return x

    def __str__(self):
        return self.__class__.__name__.lower()
