import numpy as np

from collections import namedtuple

import torch
from torch import nn as nn

from torchvision.utils import make_grid
from einops import rearrange
from arch.fema_vqgan_arch import ActLayer, NormLayer, FaceCoderNet, DecoderBlock, ResBlock

VQInfo = namedtuple('VQInfo', ['z', 'z_q', 'codebook_loss', 'semantic_loss', 'quantizer_info'])


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


class TemporalResBlock(ResBlock):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(TemporalResBlock, self).__init__(in_channel, out_channel, norm_type, act_type)

        self.temporal_conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv3d(in_channel, out_channel, (3, 1, 1), stride=1, padding=(1, 0, 0)),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv3d(out_channel, out_channel, (3, 1, 1), stride=1, padding=(1, 0, 0)),
        )

    def forward(self, x: torch.FloatTensor, batch_size: int = 5):
        res = x
        x = self.conv(x)

        batch_frames, channels, height, width = x.shape
        num_frames = batch_frames // batch_size

        x = rearrange(x, '(b f) c h w -> b c f h w', f=num_frames)
        x = self.temporal_conv(x)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        out = x + res

        return out


class TemporalDecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            TemporalResBlock(out_channel, out_channel, norm_type, act_type),
            TemporalResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class FaceCoderTemporalNet(FaceCoderNet):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_scale=32, codebook_size=1024, emb_dim=512,
                 quantizer_type="nearest",
                 beta=0.25,
                 gumbel_kl_weight=1e-8,
                 gumbel_straight_through=False,
                 gt_resolution=256,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 **ignore_kwargs):
        super(FaceCoderTemporalNet, self).__init__(
            in_channel=in_channel,
            codebook_scale=codebook_scale, codebook_size=codebook_size, emb_dim=emb_dim,
            quantizer_type=quantizer_type,
            beta=beta,
            gumbel_kl_weight=gumbel_kl_weight,
            gumbel_straight_through=gumbel_straight_through,
            gt_resolution=gt_resolution,
            norm_type=norm_type,
            act_type=act_type,
            use_quantize=use_quantize,
        )

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale))

        self.latent_resolution = self.multiscale_encoder.latent_resolution

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(TemporalDecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

    def freeze_spatial(self):
        self.requires_grad_(False)
        for n, p in self.named_parameters():
            if 'temporal_conv' in n:
                p.requires_grad = True

    def decode(self, z_q, num_batch=1, control_latent=None):
        x = self.after_quant(z_q)
        for idx, decoder_layer in enumerate(self.decoder_group):
            x = decoder_layer(x, num_batch=num_batch)
        x = self.out_conv(x)
        return x

    def forward(self, x, num_batch=1):

        z = self.encode(x)
        z_q, codebook_loss, quantizer_info = self.quantize(z)
        output = self.decode(z_q if self.use_quantize else z, num_batch=num_batch)

        return output, VQInfo(z=z, z_q=z_q, codebook_loss=codebook_loss, semantic_loss=None,
                              quantizer_info=quantizer_info)

    def __str__(self):
        return self.__class__.__name__.lower()
