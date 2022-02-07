import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.backbones.resnet_backbone import ResNetBackbone

resnet50 = {
    "path": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
}

class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        if not os.path.isfile(resnet50["path"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_pretrained_model.sh')


        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=True) # download pre-trained backbone
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x = self.base(x)
        x = self.psp(x)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()

### UNET ARCHITECTURE

class NConv(torch.nn.Module):
    r"""NConv: 3x3 Convolution Layer + Dropout + ReLU, N times. 
    Makes up the "forward" paths of the U-Net

    args:
        n: number of conv+batchnorm+relu combos to execute
        in_channels: number of channels the path starts with
        out_channels: number of channels the path ends with
    """
    def __init__(self, n: int, in_channels: int, out_channels: int):
        super(NConv, self).__init__()
        self.bodylayers = torch.nn.ModuleList([])
        # 3x 3x3 conv + relu, add channels
        self.initLayer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace = True)
        )

        # nx 3x3 conv + relu, add channels
        for _ in range(n-1):
            self.bodylayers.append(torch.nn.Sequential(
                torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace = True)
            ))

    def forward(self, x):
        x = self.initLayer(x)
        for body in self.bodylayers:
            x = body(x)
        return x


class Down(torch.nn.Module):
    r"""Down: MaxPool Downscale + forward path. 
    The downscaling part of U-Net

    args:
        n: number of conv+batchnorm+relu combos to execute (passed to NConv)
        in_channels: number of channels the path starts with (passed to NConv)
        out_channels: number of channels the path ends with (passed to NConv)
    """
    def __init__(self, n: int, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        # downscale then NConv
        self.feature = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            NConv(n, in_channels, out_channels)
        )

    def forward(self, x):
        return self.feature(x)

class UNetEncoder(torch.nn.Module):
    r"""Full UNet Downscale Decoder
    """
    def __init__(self, level_length:int , n_channels: int) -> None:
        super(UNetEncoder, self).__init__()
        self.inc = NConv(level_length, n_channels, 64)

        self.down1 = Down(level_length, 64, 128)
        self.down2 = Down(level_length, 128, 256)
        self.down3 = Down(level_length, 256, 512)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        return(x0, x1, x2, x3)
        