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
            nn.Conv3d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv3d(out_channel, out_channel, 3, stride=1, padding=1),
        )
        for m in self.temporal_conv:
            if isinstance(m, nn.Conv3d):
                nn.init.zeros_(m.weight.data)

    def forward(self, x: torch.FloatTensor, num_batch: int = 1):
        # spatial
        res = x
        x = self.conv(x)
        x = x + res
        # temporal
        batch_frames, channels, height, width = x.shape
        res = x
        num_frames = batch_frames // num_batch
        x = rearrange(x, '(b f) c h w -> b c f h w', f=num_frames)
        x = self.temporal_conv(x)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        out = x + res

        return out


class TemporalMultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        self.num_resblock = 2
        self.latent_out_ch = []
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ksz, stride=2, padding=1), ]
            tmp_down_block.extend(TemporalResBlock(in_channel=out_ch,
                                                   out_channel=out_ch,
                                                   norm_type=norm_type,
                                                   act_type=act_type) for _ in range(self.num_resblock))
            self.blocks.append(nn.Sequential(*tmp_down_block))
            self.latent_out_ch.append(out_ch)
            res = res // 2
        self.latent_resolution = res

    def forward(self, x, return_latent=False):
        if return_latent:
            outputs = []
        x = self.in_conv(x)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            if return_latent:
                outputs.append(x)

        return outputs if return_latent else x


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

        self.block = nn.ModuleList(self.block)

    def forward(self, x, num_batch=1):
        for module in self.block:
            if isinstance(module, TemporalResBlock):
                x = module(x, num_batch)
            else:
                x = module(x)
        return x


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
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = TemporalMultiScaleEncoder(
            in_channel,
            encode_depth,
            self.gt_res,
            channel_query_dict,
            norm_type, act_type
        )

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
