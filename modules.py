import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    条纹池化模块 (Strip Pooling)

    适用场景：
      - 海天线 / 车道线 / 细长目标边界等长条形结构
      - 通过“沿单一方向池化”捕捉长程上下文，再广播回原分辨率

    说明：
      这里实现的是一个轻量、实用的 StripPooling（带方向卷积），
      比你原先“只池化后相加”更稳定、更有效。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（默认等于 in_channels）
        norm_layer: 归一化层（默认 BatchNorm2d）
        reduction: 通道压缩比例（默认 4）
    """

    def __init__(self, in_channels, out_channels=None, norm_layer=nn.BatchNorm2d, reduction: int = 4):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        mid = max(in_channels // reduction, 16)

        # 1) reduce: C -> mid
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            norm_layer(mid),
            nn.ReLU(inplace=True),
        )

        # 2) horizontal / vertical strip convs on pooled strips
        self.conv_h = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
            norm_layer(mid),
            nn.ReLU(inplace=True),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
            norm_layer(mid),
            nn.ReLU(inplace=True),
        )

        # 3) expand: mid -> out_channels
        self.conv_expand = nn.Sequential(
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

        # 若 out_channels != in_channels，用 1x1 把残差支路投影到同维度
        self.skip_proj = None
        if out_channels != in_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.conv_reduce(x)  # [B, mid, H, W]

        # 水平条纹池化：沿宽度池化 -> [B, mid, H, 1]
        y_h = F.adaptive_avg_pool2d(y, output_size=(h, 1))
        y_h = self.conv_h(y_h)
        y_h = F.interpolate(y_h, size=(h, w), mode="bilinear", align_corners=False)

        # 垂直条纹池化：沿高度池化 -> [B, mid, 1, W]
        y_w = F.adaptive_avg_pool2d(y, output_size=(1, w))
        y_w = self.conv_w(y_w)
        y_w = F.interpolate(y_w, size=(h, w), mode="bilinear", align_corners=False)

        y_out = self.conv_expand(y_h + y_w)

        skip = x if self.skip_proj is None else self.skip_proj(x)
        return self.act(skip + y_out)  # 残差连接 + ReLU


class DoubleConv(nn.Module):
    """基础卷积块 (Conv-BN-ReLU x 2)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
