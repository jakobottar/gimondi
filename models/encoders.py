import torch


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
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

        # nx 3x3 conv + relu, add channels
        for _ in range(n - 1):
            self.bodylayers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, padding=1
                    ),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True),
                )
            )

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
            torch.nn.MaxPool2d(2), NConv(n, in_channels, out_channels)
        )

    def forward(self, x):
        return self.feature(x)


class UNetEncoder(torch.nn.Module):
    r"""Full UNet Downscale Decoder"""

    def __init__(self, level_length: int, n_channels: int) -> None:
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

        return (x0, x1, x2, x3)
