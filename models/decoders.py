import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import math
from models.encoders import NConv

# from https://github.com/yassouali/CCT/blob/master/models/decoders.py
def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x

def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsampling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)

class UpscalingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(UpscalingDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x

## Dropout
class UpscalingDropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(UpscalingDropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(self.dropout(x))
        return x
## FeatureDrop

## FeatureNoise

## Do I want to use cutout decoders?

def centerCrop(x, size: tuple):
            diffX = x.size()[2] - size[2]
            diffY = x.size()[3] - size[3]

            return x[:, :, diffX//2:diffX//2+size[2], diffY//2:diffY//2+size[3]]


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
        
        # dY = x2.size()[2] - x1.size()[2]
        # dX = x2.size()[3] - x1.size()[3]

        # x1 = torch.nn.functional.pad(x1, [dX // 2, dX - dX // 2,
        #                                   dY // 2, dY - dY // 2])
        
        x2 = centerCrop(x2, x1.size()) # crop incoming tensor from across U
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

class UNetDropoutDecoder(torch.nn.Module):
    r"""UNet Upscaling Decoder with Dropout
    """
    def __init__(self, level_length:int, n_classes:int, drop_rate=0.3, spatial_dropout=True) -> None:
        super(UNetDropoutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.up3 = Up(level_length, 512, 256)
        self.up2 = Up(level_length, 256, 128)
        self.up1 = Up(level_length, 128, 64)

        self.outc = OutConv(64, n_classes)
    
    def forward(self, xs):
        x= self.up3(self.dropout(xs[3]), self.dropout(xs[2]))
        x = self.up2(x, self.dropout(xs[1]))
        x = self.up1(x, self.dropout(xs[0]))
        return self.outc(x)

class UNetFeatureDropoutDecoder(torch.nn.Module):
    r"""UNet Upscaling Decoder with Feature Dropout:
        Drops lambda ~ (0.7, 0.9) least active regions
    """
    def __init__(self, level_length:int, n_classes:int) -> None:
        super(UNetFeatureDropoutDecoder, self).__init__()
        self.up3 = Up(level_length, 512, 256)
        self.up2 = Up(level_length, 256, 128)
        self.up1 = Up(level_length, 128, 64)
        self.outc = OutConv(64, n_classes)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, xs):
        x= self.up3(self.feature_dropout(xs[3]), self.feature_dropout(xs[2]))
        x = self.up2(x, self.feature_dropout(xs[1]))
        x = self.up1(x, self.feature_dropout(xs[0]))
        return self.outc(x)

class UNetNoisyDecoder(torch.nn.Module):
    r"""UNet Upscaling Decoder with Uniform Random Noise"""
    def __init__(self, level_length:int, n_classes:int, uniform_range:float=0.3) -> None:
        super(UNetNoisyDecoder, self).__init__()
        self.uniform = Uniform(-uniform_range, uniform_range)

        self.up3 = Up(level_length, 512, 256)
        self.up2 = Up(level_length, 256, 128)
        self.up1 = Up(level_length, 128, 64)
        self.outc = OutConv(64, n_classes)
    
    def add_noise(self, x):
        noise = self.uniform.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noisy = x.mul(noise) + x
        return x_noisy
    
    def forward(self, xs):
        x= self.up3(self.add_noise(xs[3]), self.add_noise(xs[2]))
        x = self.up2(x, self.add_noise(xs[1]))
        x = self.up1(x, self.add_noise(xs[0]))
        return self.outc(x)
