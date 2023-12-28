import torch
from torch import nn
from torch.nn import functional as F


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


class ResBlock(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size, stride,
                 norm_layer=None, act='leaky'):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if act == 'relu':
            self.activation = nn.ReLU(True)
        elif act == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise NotImplementedError()

        self.conv1 = conv(inplanes, outplanes, kernel_size, stride)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        if stride != 1 or (inplanes != outplanes):
            self.downsample = conv1x1(inplanes, outplanes, stride)
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.activation(out)
        return out


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
