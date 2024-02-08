import numpy as np
import math

from collections import namedtuple

import torch
from torch.nn import functional as F
from torch import nn as nn

from torchvision.utils import make_grid

from utils.logging_tool import get_logger

from arch.vgg_arch import PerceptualVGG
from arch.quantizer_arch import VectorQuantizer, GumbelQuantizer, VectorQuantizer2

VQInfo = namedtuple('VQInfo', ['z', 'z_q', 'codebook_loss', 'semantic_loss', 'quantizer_info'])


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            raise NotImplementedError('Norm type {} not support.'.format(norm_type))

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """

    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        elif relu_type == 'swish':
            self.norm = lambda x: swish(x)
        else:
            assert 1 == 0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ResBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, x):
        res = x
        x = self.conv(x)
        out = x + res
        return out


class CombineQuantBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out


class MultiScaleEncoder(nn.Module):
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
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
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


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class FeMaSRNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_residual=True,
                 use_semantic_loss=False,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]

        self.codebook_size = int(codebook_params[0, 1])
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.use_residual = use_residual

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
        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            encode_depth,
            self.gt_res,
            channel_query_dict,
            norm_type, act_type
        )

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(codebook_emb_dim[-1], 512, 1, 1, 0),
                nn.ReLU(),
            )
            self.vgg_feat_layer = '26'  # relu4_4
            self.vgg_feat_extractor = PerceptualVGG([self.vgg_feat_layer])

    def encode_and_decode(self, input, gt_indices=None):
        enc_feats = self.multiscale_encoder(input.detach(), return_latent=True)

        enc_feats = enc_feats[::-1]

        if self.use_semantic_loss and self.training:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        quantizer_info_list = []
        semantic_loss_list = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        x = enc_feats[0]

        for i in range(self.max_depth):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** i

            if cur_res in self.codebook_scale:  # needs to perform quantize
                if prev_dec_feat is not None:
                    before_quant_feat = torch.cat((enc_feats[i], prev_dec_feat), dim=1)
                else:
                    before_quant_feat = enc_feats[i]
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                if gt_indices is not None:
                    z_quant, codebook_loss, quantizer_info = self.quantize_group[quant_idx](feat_to_quant,
                                                                                            gt_indices[quant_idx])
                else:
                    z_quant, codebook_loss, quantizer_info = self.quantize_group[quant_idx](feat_to_quant)

                if self.use_semantic_loss and self.training:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                quantizer_info_list.append(quantizer_info)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            x = self.decoder_group[i](x)
            prev_dec_feat = x

        out_img = self.out_conv(x)

        codebook_loss = sum(codebook_loss_list)
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0
        if self.use_semantic_loss and self.training:
            return out_img, VQInfo(z=enc_feats[0], z_q=z_quant, codebook_loss=codebook_loss,
                                   semantic_loss=semantic_loss, quantizer_info=quantizer_info_list)
            # return out_img, codebook_loss, semantic_loss, indices_list
        else:
            return out_img, VQInfo(z=enc_feats[0], z_q=z_quant, codebook_loss=codebook_loss,
                                   semantic_loss=None, quantizer_info=quantizer_info_list)
            # return out_img, codebook_loss, indices_list

    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, input):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 8 // self.scale_factor * 8
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        dec, _, _ = self.encode_and_decode(input)

        output = dec
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]

        self.use_semantic_loss = org_use_semantic_loss
        return output

    def forward(self, input, gt_indices=None):
        if gt_indices is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, vq_info = self.encode_and_decode(input, gt_indices)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, vq_info = self.encode_and_decode(input)
        return dec, vq_info

    @torch.no_grad()
    def vis_codebook(self, up_factor=2, norm=True):
        code_idx = torch.arange(self.codebook_size).reshape(self.codebook_size, 1, 1, 1)
        code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
        output_img = self.decode_indices(code_idx)
        output_img = make_grid(output_img, nrow=int(np.sqrt(self.codebook_size)), normalize=norm)
        return output_img[None, ...], self.codebook_size

    def __str__(self):
        return self.__class__.__name__.lower()


class FaceCoderNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_scale=32, codebook_size=1024, emb_dim=512,
                 quantizer="nearest",
                 beta=0.25,
                 gumbel_kl_weight=1e-8,
                 gumbel_straight_through=False,
                 gt_resolution=256,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,

                 **ignore_kwargs):
        super(FaceCoderNet, self).__init__()
        logger = get_logger()

        self.codebook_scale = codebook_scale
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.quantizer_type = quantizer

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution

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
        encode_depth = int(np.log2(gt_resolution // self.codebook_scale))
        self.multiscale_encoder = MultiScaleEncoder(
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
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        if self.quantizer_type == "nearest":
            self.beta = beta  # 0.25
            self.quantizer = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "nearest2":
            self.beta = beta  # 0.25
            self.quantizer = VectorQuantizer2(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantizer = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        logger.info(f'VQAutoEncoder quantizer: {self.quantizer_type} '
                    f'codebook_size: {self.codebook_size} embed_dim: {self.embed_dim}')
        # build vector quantizer
        # self.quantizer = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)

        scale_in_ch = channel_query_dict[self.codebook_scale]

        self.before_quant = nn.Conv2d(scale_in_ch, self.embed_dim, 1)
        self.after_quant = nn.Conv2d(self.embed_dim, scale_in_ch, 3, 1, 1)

    def encode(self, x):
        x = self.multiscale_encoder(x)
        x = self.before_quant(x)
        return x

    def quantize(self, z):
        z_q, codebook_loss, quantizer_info = self.quantizer(z)
        return z_q, codebook_loss, quantizer_info

    def decode(self, z_q):
        x = self.after_quant(z_q)
        for decoder_layer in self.decoder_group:
            x = decoder_layer(x)
        x = self.out_conv(x)
        return x

    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_q = self.quantizer.get_codebook_entry(indices)
        out_img = self.decode(z_q)
        return out_img

    def forward(self, x):

        z = self.encode(x)
        z_q, codebook_loss, quantizer_info = self.quantize(z)
        output = self.decode(z_q if self.use_quantize else z)

        return output, VQInfo(z=z, z_q=z_q, codebook_loss=codebook_loss, semantic_loss=None,
                              quantizer_info=quantizer_info)

    @torch.no_grad()
    def vis_codebook(self, up_factor=2, norm=True):
        code_idx = torch.arange(self.codebook_size).reshape(self.codebook_size, 1, 1, 1)
        code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
        output_img = self.decode_indices(code_idx)
        output_img = make_grid(output_img, nrow=int(np.sqrt(self.codebook_size)), normalize=norm)
        return output_img[None, ...], self.codebook_size

    def __str__(self):
        return self.__class__.__name__.lower()
