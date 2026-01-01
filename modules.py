import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    条纹池化模块 (Strip Pooling)
    """

    def __init__(self, in_channels, out_channels=None, norm_layer=nn.BatchNorm2d, reduction=4):
        super(StripPooling, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        mid = max(in_channels // reduction, 16)

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            norm_layer(mid),
            nn.ReLU(inplace=True)
        )

        self.conv_h = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0), bias=False),
            norm_layer(mid),
            nn.ReLU(inplace=True)
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(1, 3), padding=(0, 1), bias=False),
            norm_layer(mid),
            nn.ReLU(inplace=True)
        )

        self.conv_expand = nn.Sequential(
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, h, w = x.size()
        y = self.conv_reduce(x)

        y_h = F.adaptive_avg_pool2d(y, (h, 1))
        y_h = self.conv_h(y_h)
        y_h = F.interpolate(y_h, (h, w), mode="bilinear", align_corners=False)

        y_w = F.adaptive_avg_pool2d(y, (1, w))
        y_w = self.conv_w(y_w)
        y_w = F.interpolate(y_w, (h, w), mode="bilinear", align_corners=False)

        out = self.conv_expand(y_h + y_w)
        return self.relu(self.shortcut(x) + out)


class DoubleConv(nn.Module):
    """
    支持切换 Norm 类型的卷积块
    norm_type: 'bn' (BatchNorm), 'gn' (GroupNorm), 'in' (InstanceNorm), 'none'
    复原任务推荐 'gn'，分割任务推荐 'bn'
    """

    def __init__(self, in_channels, out_channels, norm_type='bn', num_groups=32):
        super().__init__()

        # 辅助函数：获取 Norm 层
        def get_norm(channels):
            if norm_type == 'bn':
                return nn.BatchNorm2d(channels)
            elif norm_type == 'gn':
                # GroupNorm 需要 num_groups，通常设为 32 或 16
                # 如果通道数少于 group 数，group 设为通道数的一半
                groups = num_groups if channels >= num_groups else max(1, channels // 2)
                return nn.GroupNorm(num_groups=groups, num_channels=channels)
            elif norm_type == 'in':
                return nn.InstanceNorm2d(channels, affine=True)
            else:
                return nn.Identity()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)