import torch
import torch.nn as nn


class EntropyBranch(nn.Module):
    """
    Input : [B, 1, H, W]
    Output: [B, 256, H/2, W/2]

    Designed to align with a future injection at ScaleLSD DPT path_1.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == '__main__':
    model = EntropyBranch()
    x = torch.randn(2, 1, 576, 1024)
    y = model(x)
    print('input :', tuple(x.shape))
    print('output:', tuple(y.shape))  # expected [2, 256, 288, 512]
