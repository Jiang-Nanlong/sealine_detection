# ZERO-DCE++

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCE_net(nn.Module):
    """
    Zero-DCE的核心网络结构.
    它是一个小型的7层CNN，用于预测光照增强曲线的参数A.
    """

    def __init__(self, n_iter=8):
        super(DCE_net, self).__init__()
        self.n_iter = n_iter

        # 定义网络层
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        # 输出通道数为 3 * n_iter，因为每个通道(R,G,B)在每次迭代中都需要一个参数A
        self.conv5 = nn.Conv2d(32 * 2, 32, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(32 * 2, 32, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(32 * 2, 3 * n_iter, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 前向传播并使用跳跃连接
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        # 对称的跳跃连接
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))

        # A是曲线参数图
        A = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))

        return A


class ZeroDCE(nn.Module):
    """
    完整的Zero-DCE模块，包含DCE_net和曲线应用逻辑.
    """

    def __init__(self, n_iter=8):
        super(ZeroDCE, self).__init__()
        self.n_iter = n_iter
        self.dce_net = DCE_net(n_iter=self.n_iter)

    def forward(self, low_light_img):
        # 1. 从DCE-Net获取曲线参数A
        A_maps = self.dce_net(low_light_img)

        # 2. 迭代应用光照增强曲线
        enhanced_img = low_light_img
        for i in range(self.n_iter):
            # 从参数图中获取本次迭代的参数
            A = A_maps[:, (i * 3):(i * 3 + 3), :, :]
            enhanced_img = enhanced_img + A * (torch.pow(enhanced_img, 2) - enhanced_img)

    #     return enhanced_img
        return enhanced_img, A_maps    #返回曲线参数A，后续微调使用