import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetV3Encoder(nn.Module):
    """
    使用 torchvision 的 MobileNetV3-Large 作为编码器.
    这个类封装了特征提取的逻辑，并返回用于跳跃连接的中间特征图.
    """

    def __init__(self, pretrained=True):
        super(MobileNetV3Encoder, self).__init__()
        # 加载预训练的 MobileNetV3 Large 模型的特征提取部分
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.features = mobilenet.features

        # 定义需要提取特征的层的索引
        # 这些索引对应于 MobileNetV3 特征提取器中下采样发生的位置
        # 对应输出特征图尺寸为原图的 1/2, 1/4, 1/8, 1/16
        # 原始 MobileNetV3-Large 的特征提取层结构：
        # 0: Conv2d (input) -> 16 (stride 2) - output 1/2
        # 1: InvertedResidual (16->16, stride 1)
        # 2: InvertedResidual (16->24, stride 2) - output 1/4
        # 3: InvertedResidual (24->24, stride 1)
        # 4: InvertedResidual (24->40, stride 2) - output 1/8
        # 5: InvertedResidual (40->40, stride 1)
        # 6: InvertedResidual (40->40, stride 1)
        # 7: InvertedResidual (40->80, stride 2) - output 1/16
        # 8-12: ...
        # 13: InvertedResidual (112->112, stride 1)
        # 14: Conv2d (112->960, stride 1) - bottleneck
        self.skip_connection_indices = [1, 3, 6, 12]  # 对应于 1/2, 1/4, 1/8, 1/16 尺寸的特征图
        # 实际通道数 (从浅到深):
        # skips[0] (index 1): 16 (after layer 1)
        # skips[1] (index 3): 24 (after layer 3)
        # skips[2] (index 6): 40 (after layer 6)
        # skips[3] (index 12): 112 (after layer 12)
        # bottleneck (after layer 14): 960

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.skip_connection_indices:
                skip_connections.append(x)
        return x, skip_connections


class DecoderBlock(nn.Module):
    """
    U-Net 解码器中的一个基本块.
    包含上采样、与跳跃连接的特征融合、卷积.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 使用双线性插值进行上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 卷积层，输入通道为上采样后的通道 + 跳跃连接的通道
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # 拼接特征图
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)
        return x


class DenoiseUNet(nn.Module):
    """
    使用 MobileNetV3 作为编码器的 U-Net 结构.
    """

    def __init__(self, pretrained_encoder=True):
        super(DenoiseUNet, self).__init__()
        self.encoder = MobileNetV3Encoder(pretrained=pretrained_encoder)

        # MobileNetV3-Large 特征图的通道数
        # bottleneck: 960
        # skips (从深到浅，对应 encoder.skip_connection_indices 的逆序):
        # skip_channels_list[0] 对应 encoder.features[12] -> 112
        # skip_channels_list[1] 对应 encoder.features[6] -> 40
        # skip_channels_list[2] 对应 encoder.features[3] -> 24
        # skip_channels_list[3] 对应 encoder.features[1] -> 16
        bottleneck_channels = 960
        skip_channels_list = [112, 40, 24, 16]  # 对应于 MobileNetV3Encoder 中 skip_connection_indices 的逆序输出通道

        # 解码器块
        self.decoder1 = DecoderBlock(bottleneck_channels, skip_channels_list[0], 256)  # 960 + 112 -> 256
        self.decoder2 = DecoderBlock(256, skip_channels_list[1], 128)  # 256 + 40 -> 128
        self.decoder3 = DecoderBlock(128, skip_channels_list[2], 64)  # 128 + 24 -> 64
        self.decoder4 = DecoderBlock(64, skip_channels_list[3], 32)  # 64 + 16 -> 32

        # 最终的上采样和输出层
        # 最后一个解码器块输出的特征图尺寸是输入图像的 1/2
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        # 编码
        bottleneck, skips = self.encoder(x)
        # 为了方便解码，将跳跃连接的顺序反转 (从深到浅，即从最小尺寸的特征图到最大尺寸的特征图)
        skips = skips[::-1]

        # 解码
        d = self.decoder1(bottleneck, skips[0])
        d = self.decoder2(d, skips[1])
        d = self.decoder3(d, skips[2])
        d = self.decoder4(d, skips[3])

        # 生成最终图像
        out = self.final_upsample(d)
        out = self.final_conv(out)

        return out