import torch
import torchvision
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from models.decoders import *
from models.encoders import *

class ResNetSegmentationNet(torch.nn.Module):
    def __init__(self, sup_loss, unsup_loss):
        super(ResNetSegmentationNet, self).__init__()
        self.encoder = ResNetEncoder()

        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4 
        self.main_decoder = UpscalingDecoder(upscale, decoder_in_ch, num_classes=2)
        self.aux_decoder = UpscalingDropOutDecoder(upscale, decoder_in_ch, num_classes=2)

        self.sup_loss = sup_loss
        self.unsup_loss = unsup_loss

    def forward(self, x_l, target_l, x_ul = None):
        if not self.training:
            out_l = self.main_decoder(self.encoder(x_l))
            loss_l = self.sup_loss(out_l, target_l.squeeze(1).long())

            return out_l, loss_l, None, None

        # pass labeled example through main decoder and calculate supervised loss
        out_l = self.main_decoder(self.encoder(x_l))
        loss_l = self.sup_loss(out_l, target_l.squeeze(1).long())

        # pass unlabeled example through network
        x_ul = self.encoder(x_ul)
        main_ul = self.main_decoder(x_ul)

        # # Get target from result
        target_ul = F.softmax(main_ul.detach(), dim=1)

        # fig = plt.figure(figsize=(4, 4))
        # ax1 = plt.subplot(1, 1, 1)

        # ax1.imshow(target_ul[0][0].cpu().numpy())
        # ax1.set_axis_off()
        # ax1.set_title('Predicted Mask')

        # plt.savefig(f'./out/test.png')
        # plt.close()

        # calculate loss
        out_ul = self.aux_decoder(x_ul)
        loss_ul = self.unsup_loss(out_ul, target_ul)

        return out_l, loss_l, out_ul, loss_ul
        # return out_l, loss_l, None, None

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

    def __init__(self, sup_loss, unsup_loss, mode = 'semi') -> None:
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
        self.aux_decoders = nn.ModuleList([
            UNetDropoutDecoder(level_length, n_classes, drop_rate=0.75),
            UNetNoisyDecoder(level_length, n_classes, uniform_range=0.9),
            UNetFeatureDropoutDecoder(level_length, n_classes)
        ])

        self._initialize_weights()
        if sup_loss == "crossentropy":
            self.sup_loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
        if unsup_loss == "mse":
            self.unsup_loss = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss()] # TODO: scale with num unsup branches
        else:
            raise NotImplementedError

        self.mode = mode
        print(self.mode)

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
    

    def forward(self, x_l, target_l, x_ul = None):
        # Supervised Section

        xs = self.encoder(x_l)
        out_l = self.main_decoder(xs)

        loss_l = self.sup_loss(out_l, target_l.squeeze(1).long())

        if self.training and self.mode == 'semi':
            # Unsupervised Section
            xs = self.encoder(x_ul)
            main_ul = self.main_decoder(xs)
            target_ul = F.softmax(main_ul.detach(), dim=1)

            out_ul, loss_ul = [None]*len(self.aux_decoders), [None]*len(self.aux_decoders)

            out_ul  = [aux_decoder(xs) for aux_decoder in self.aux_decoders]
            loss_ul = [unsup_loss(out, target_ul) for unsup_loss, out in zip(self.unsup_loss, out_ul)]

            loss_vals = {
                "main": loss_l.item(),
                "dropout": loss_ul[0].item(),
                "noisy": loss_ul[1].item()
            }

            loss_ul = sum(loss_ul) / len(loss_ul)   # calculate average unsupervised loss
            loss = loss_l + loss_ul                 # sum to get total loss
            out_ul.insert(0, out_l)                 # add supervised output
            out = out_ul                            # rename
            
        else:  
            loss = loss_l
            out = [out_l]
            loss_vals = {
                "main": loss_l.item()
            }
            
        # TODO: Returning the loss values breaks, fix it!
        return out, loss# , loss_vals 
            