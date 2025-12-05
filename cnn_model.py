import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    SE-Block (通道注意力模块)
    作用：显式地建模通道之间的相互依赖关系，自适应地重新校准通道式的特征响应。
    通俗解释：让网络学会“看哪个通道更准”。
    """

    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        # 全局平均池化：把 HxW 的图压缩成一个数，代表这个通道的全局信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层：学习通道间的权重关系
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction if in_channels // reduction > 0 else 1),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction if in_channels // reduction > 0 else 1, in_channels),
            nn.Sigmoid()  # 输出 0~1 之间的权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 原特征图 x 乘以 权重 y
        return x * y


class HorizonDetNet(nn.Module):
    def __init__(self, in_channels=4, img_h=362, img_w=180):
        super(HorizonDetNet, self).__init__()

        # 1. 注意力融合层
        self.attention = ChannelAttention(in_channels)

        # 2. 特征提取主干 (Backbone)
        # 这是一个类似 VGG 的简单卷积堆叠
        self.features = nn.Sequential(
            # Block 1: 提取浅层纹理特征
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 尺寸减半

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 尺寸再减半

            # Block 3: 提取深层语义特征
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 这里的池化很关键：无论输入多大，强制将特征图压缩为 4x4 大小
            # 这样后面的全连接层参数量就固定了，且大大减小了模型体积
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 3. 回归头 (Regression Head)
        self.regressor = nn.Sequential(
            nn.Flatten(),  # 展平: 128通道 * 4 * 4 = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 最终输出 2 个数: [rho, theta]
        )

    def forward(self, x):
        # x shape: [Batch, 4, 362, 180]

        # step 1: 融合
        x = self.attention(x)

        # step 2: 提取特征
        x = self.features(x)

        # step 3: 回归坐标
        out = self.regressor(x)

        return out