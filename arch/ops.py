import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


def conv(in_planes: int,
         out_planes: int,
         kernel_size: int = 3,
         stride: int = 1,
         groups: int = 1,
         dilation: int = 1) -> nn.Conv2d:
    """convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                     groups=groups, bias=False, dilation=dilation, )


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation, )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


#
# class ResBlock(nn.Module):
#
#     def __init__(self, inplanes, outplanes, kernel_size, stride, norm_layer=None, act='leaky'):
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#
#         if act == 'relu':
#             self.activation = nn.ReLU(True)
#         elif act == 'leaky':
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         else:
#             raise NotImplementedError()
#
#         self.conv1 = conv(inplanes, outplanes, kernel_size, stride)
#         self.bn1 = norm_layer(outplanes)
#         self.conv2 = conv3x3(outplanes, outplanes)
#         self.bn2 = norm_layer(outplanes)
#         if stride != 1 or (inplanes != outplanes):
#             self.downsample = conv1x1(inplanes, outplanes, stride)
#         else:
#             self.downsample = nn.Identity()
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += self.downsample(identity)
#         out = self.activation(out)
#         return out


class Shape(nn.Module):

    def forward(self, x):
        print(x.size())
        return x


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, batch_norm=True, residual=False, act='leaky'):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout) if batch_norm else nn.Identity()
        )
        self.residual = residual
        self.act = nn.ReLU(True)
        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise NotImplementedError()

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dNew(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, batch_norm=True, residual=False, act='relu',
                 bias=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(cout) if batch_norm else nn.Identity()
        )
        self.residual = residual
        self.act = nn.ReLU(True)
        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
