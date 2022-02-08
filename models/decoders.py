import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import math
from models.encoders import NConv


class Up(torch.nn.Module):
    r"""Up: Upscale + Crop/Concat + forward path
    Upscaling part of U-Net and crop/concat step

    args:
        n: number of conv+batchnorm+relu combos to execute (passed to NConv)
        in_channels: number of channels the path starts with (passed to NConv)
        out_channels: number of channels the path ends with (passed to NConv)
    """
    def __init__(self, n: int, in_channels: int, out_channels: int):
        super(Up, self).__init__()
        # upscale, crop/concat, then NConv
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
        self.conv = NConv(n, in_channels, out_channels)
    
    # x1 comes from prev step, x2 makes up the crop/concat step
    def forward(self, x1, x2):
        x1 = self.up(x1) # upscale tensor

        # crop incoming tensor from across U
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x2 = x2[:, :, diffX//2:diffX//2+x1.size()[2], diffY//2:diffY//2+x1.size()[3]]
        x = torch.cat([x2, x1], dim=1) # concat them together

        return self.conv(x) # conv forward path


class OutConv(torch.nn.Module):
    r"""OutConv
    Final 1x1 convolution layer to predict class

    args:
        in_channels: number of channels the path starts with (passed to NConv)
        out_channels: number of classes we predict
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.feature = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.feature(x)


class UNetDecoder(torch.nn.Module):
    r"""UNet Upscaling Decoder
    """
    def __init__(self, level_length:int, n_classes:int) -> None:
        super(UNetDecoder, self).__init__()
        self.up3 = Up(level_length, 512, 256)
        self.up2 = Up(level_length, 256, 128)
        self.up1 = Up(level_length, 128, 64)

        self.outc = OutConv(64, n_classes)
    
    def forward(self, xs):
        x= self.up3(xs[3], xs[2])
        x = self.up2(x, xs[1])
        x = self.up1(x, xs[0])
        return self.outc(x)
