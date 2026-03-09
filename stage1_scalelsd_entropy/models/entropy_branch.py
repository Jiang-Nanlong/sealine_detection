import torch
import torch.nn as nn

# ============================================================
# Global config for PyCharm / server-side direct execution
# Edit these variables directly before running this file.
# ============================================================
TEST_BATCH = 2
TEST_H = 576
TEST_W = 1024


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


def main():
    model = EntropyBranch()
    x = torch.randn(TEST_BATCH, 1, TEST_H, TEST_W)
    y = model(x)
    print('input :', tuple(x.shape))
    print('output:', tuple(y.shape))  # expected [B, 256, H/2, W/2]


if __name__ == '__main__':
    main()
