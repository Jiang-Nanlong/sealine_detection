import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CBAM(nn.Module):
    """ 空间+通道注意力模块，适合显卡资源充足时使用 """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = x * self.sigmoid_channel(out)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        x_spatial = self.conv_spatial(x_spatial)
        return x * self.sigmoid_spatial(x_spatial)


class HorizonResNet(nn.Module):
    # 3传统特征+1语义分割特征
    def __init__(self, in_channels=4, block=BasicBlock, num_blocks=[3, 4, 6, 3], img_h=2240, img_w=180):
        super(HorizonResNet, self).__init__()
        self.in_planes = 64
        self.img_h = img_h
        self.img_w = img_w

        # ... (ResNet Backbone 部分保持不变: conv1 到 layer4 + cbam) ...
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.cbam = CBAM(512)

        # === 修改部分：新的 Head ===
        # 1. 通道降维：从 512 降到 1，得到一张特征热力图
        self.conv_end = nn.Conv2d(512 * block.expansion, 1, kernel_size=1)

        # 2. 不需要全连接层了！
        # 我们将在 forward 里通过数学运算求坐标

        # 3. 生成坐标网格 (Buffer) 用于计算期望
        # 我们生成 0, 1, 2... 的序列，注册为 buffer (不会被更新，但在 GPU 上)
        # 注意：因为经过了 ResNet，特征图尺寸缩小了 32 倍
        # 输入 2240 -> 特征图高 70
        # 输入 180  -> 特征图宽 6 (180/32 = 5.625，ResNet最后几层可能padding处理不一样，我们动态计算)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def _make_layer(self, block, planes, num_blocks, stride):
        # ... (保持不变) ...
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 3, 2240, 180]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)  # Feature Map: [B, 512, H_feat, W_feat]

        # 1. 降维成单通道热力图
        x = self.conv_end(x)  # [B, 1, 70, 6] (尺寸取决于输入和 padding)

        # 2. 分离 Rho 和 Theta 的特征分布
        # Rho 分布: 在宽度方向求和 (挤压掉 Theta 维度) -> [B, 1, 70, 1] -> [B, 70]
        # Theta 分布: 在高度方向求和 (挤压掉 Rho 维度) -> [B, 1, 1, 6] -> [B, 6]

        # 注意：这里我们假设特征图的响应值代表 logit (未归一化的概率)
        feat_h = x.shape[2]
        feat_w = x.shape[3]

        # 将特征图展平计算概率太粗糙，我们使用 "Integral Regression"
        # 更好的做法：
        # 对 x 进行 Global Average Pool 确实会丢位置，
        # 但我们现在保留了 [70, 6] 的空间结构。

        # 方法 B 的变体：直接预测 Rho 和 Theta 的两个独立向量比较难，因为它们纠缠在一起。
        # 最简单的 Integral Regression 是直接在 2D 热力图上做 Soft-Argmax。

        # --- 2D Soft-Argmax ---
        batch_size = x.size(0)

        # 展平为 [B, H*W]
        x_flat = x.view(batch_size, -1)

        # Softmax 归一化为概率 (Temperature 可以控制峰的尖锐程度)
        prob = F.softmax(x_flat / torch.clamp(self.temperature, min=1e-3), dim=1)
        prob = prob.view(batch_size, feat_h, feat_w)  # [B, H, W]

        # 计算期望 (Expectation)
        # 生成网格
        device = x.device
        pos_y = torch.linspace(0, 1, feat_h, device=device).view(1, feat_h, 1)  # 归一化的 0~1 坐标
        pos_x = torch.linspace(0, 1, feat_w, device=device).view(1, 1, feat_w)

        # 期望值 = sum(prob * pos)
        # sum over W and H
        # expected_rho = sum(prob * pos_y)
        expected_rho = torch.sum(prob * pos_y, dim=(1, 2))  # [B]
        expected_theta = torch.sum(prob * pos_x, dim=(1, 2))  # [B]

        # 拼接输出 [B, 2]
        out = torch.stack([expected_rho, expected_theta], dim=1)

        return out


def get_resnet34_model():
    # 注意：这里我们不再需要全连接层，输入尺寸对结构有影响但 Soft-Argmax 自动适应
    return HorizonResNet(in_channels=4, num_blocks=[3, 4, 6, 3])