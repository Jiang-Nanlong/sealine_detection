import torch
import torch.nn as nn
import torch.nn.functional as F


class CSDN_Tem(nn.Module):
    """
    Depthwise Separable Conv block:
      - depth_conv (groups=in_channels)
      - point_conv (1x1)
    参数名严格对齐你权重里的 key：
      e_convX.depth_conv.weight / bias
      e_convX.point_conv.weight / bias
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1,
            groups=in_channels, bias=True
        )
        self.point_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            bias=True
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class C_DCE_Net(nn.Module):
    """
    Zero-DCE++ (轻量版) 结构（与你下载的 repo / Epoch99.pth 对齐）：
      - 基础通道 n=32
      - e_conv1..e_conv4: 3/32/32/32/32
      - e_conv5..e_conv6: cat 后 64 -> 32
      - e_conv7: cat 后 64 -> 3  (这就是你权重里 e_conv7.point_conv.weight = [3,64,1,1])
    forward 默认只返回增强后的图像（按你的需求：只做亮度增强）。
    """
    def __init__(self, scale_factor: int = 1, n: int = 32, enhance_iters: int = 8):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.enhance_iters = int(enhance_iters)

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = CSDN_Tem(3, n)
        self.e_conv2 = CSDN_Tem(n, n)
        self.e_conv3 = CSDN_Tem(n, n)
        self.e_conv4 = CSDN_Tem(n, n)
        self.e_conv5 = CSDN_Tem(n * 2, n)
        self.e_conv6 = CSDN_Tem(n * 2, n)
        self.e_conv7 = CSDN_Tem(n * 2, 3)

    @staticmethod
    def _enhance_curve(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # LE-curve: I + r * I(1-I)  （常见实现写作：x + r*(x*x - x) 等价）
        return x + r * (x * x - x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] in [0,1]
        return: enhanced image [B,3,H,W] in [0,1]
        """
        input_size = x.shape[-2:]

        # 可选下采样（你 repo 里常见有 scale_factor，用于加速）
        if self.scale_factor != 1:
            # 例如 scale_factor=12：先缩小到 1/12，再预测曲线，再插回原尺寸
            x_small = F.interpolate(
                x, scale_factor=1.0 / self.scale_factor,
                mode="bilinear", align_corners=False
            )
        else:
            x_small = x

        x1 = self.relu(self.e_conv1(x_small))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))

        # 关键：Zero-DCE++ 输出 3 通道曲线参数
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # 插回原图尺寸
        if x_r.shape[-2:] != input_size:
            x_r = F.interpolate(x_r, size=input_size, mode="bilinear", align_corners=False)

        # 迭代应用曲线（常用 8 次）
        out = x
        for _ in range(self.enhance_iters):
            out = self._enhance_curve(out, x_r)

        return torch.clamp(out, 0.0, 1.0)
