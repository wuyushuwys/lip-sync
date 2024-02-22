import torch

from torch import nn
from torch.nn import functional as F

from utils.logging_tool import get_logger
from .ops import ResBlock


class MaskNet(nn.Module):
    """
    MaskNet feed-in masked image and generate mask logits that indicates the masked feature in latent domain

    Input: masked image bsz * 3 * H * W
    Output: Masked logits bsz * 1 * h * w

    We tend to use CONV ResBlock as vq-encoder adopted. (consistency for local information)

    """

    def __init__(self, gt_resolution, latent_resolution, in_ch, nf, ch_mult, num_embed, num_res_blocks=1):
        super(MaskNet, self).__init__()
        logger = get_logger()
        self.gt_resolution = gt_resolution
        self.nf = nf
        self.latent_resolution = latent_resolution
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = [nn.Conv2d(in_channels=in_ch, out_channels=nf, kernel_size=3, padding=1, stride=1)]

        for i in range(self.num_resolutions):

            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            blocks.append(nn.Conv2d(block_in_ch, block_out_ch, kernel_size=3, stride=2, padding=1),)
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_out_ch, block_out_ch))
            block_in_ch = block_out_ch

        # blocks.append(nn.Conv2d(block_in_ch, num_embed, kernel_size=3, stride=1, padding=1))
        self.last_layer = nn.Conv2d(block_in_ch, 1, kernel_size=3, stride=1, padding=1)

        logger.info(f"Create Mask Predictor with width {[nf * ch for ch in in_ch_mult]}\t"
                    f"gt_res:{self.gt_resolution}\t"
                    f"latent_res:{self.latent_resolution}")

        self.blocks = nn.ModuleList(blocks)

        self._init_weights()

    def forward(self, x) -> torch.Tensor:
        assert x.size(-1) == self.gt_resolution, f"got {x.size()}"

        for block in self.blocks:
            x = block(x)

        assert x.size(-1) == self.latent_resolution, f"got {x.size()}"
        # x_emb = x
        # flatten the output to match shape of min_encoding_indices in vqgan
        # x = torch.sigmoid(self.last_layer(x).flatten(1))
        x = self.last_layer(x).flatten(1)

        return x

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
