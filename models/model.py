import torch
import torchvision
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from models.decoders import *
from models.encoders import *


class UNet(torch.nn.Module):
    r"""UNet - full U-Net architecture
        args:
            sup_loss: supervised loss function
            unsup_loss: unsupervised loss function
    """
    """
            level_length: number of convolutional/relu layers to include on each layer (normally 2 or 3)
            u_depth: how many downscale/upscale layers to do (normally 4 or 5)
            n_channels: number of channels the input is, usually 1 (grayscale) or 3 (RGB)
            n_clases: number of classes we predict
    """


    def __init__(self) -> None:
        super(UNet, self).__init__()
        """
        level_length: number of convolutional/relu layers to include on each layer (normally 2 or 3)
        u_depth: how many downscale/upscale layers to do (normally 4 or 5)
        n_channels: number of channels the input is, usually 1 (grayscale) or 3 (RGB)
        n_clases: number of classes we predict
        """
        level_length = 4
        n_channels = 3
        n_classes = 2

        self.encoder = UNetEncoder(level_length, n_channels)
        self.main_decoder = UNetDecoder(level_length, n_classes)


    def _initialize_weights(self):
        filter_size = 3

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                k = np.sqrt(2 / (filter_size**2 * m.in_channels))
                torch.nn.init.uniform_(m.weight, -k, k)
                # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.ConvTranspose2d):
                k = np.sqrt(2 / (filter_size**2 * m.in_channels))
                torch.nn.init.uniform_(m.weight, -k, k)
                # torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    

    def forward(self, x):
        x = self.encoder(x)
        x = self.main_decoder(x)
        return x 
            