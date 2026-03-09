#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scalelsd_entropy_wrapper.py — Stage-1 模型: 基于图像的 horizon 回归 + entropy 注入

两种模式:
  1. baseline:  普通的 image-based ResNet horizon 回归（不使用 entropy）
  2. entropy:   在 backbone layer2 输出后注入 entropy feature

骨干网络:
  - 复用项目中 cnn_model.py 的 BasicBlock / CBAM 结构
  - 输入从 sinogram 改为 RGB image [B, 3, 576, 1024]
  - 同样使用 Soft-Argmax 回归 (rho_norm, theta_norm)

注意:
  - 这里不经过 Radon 变换，直接在图像域回归
  - 中间特征和原始 HorizonResNet (sinogram 域) 不共享权重
  - 后续阶段可以在此基础上扩展
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 将项目根目录加入 path，以便 import cnn_model
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from cnn_model import BasicBlock, CBAM
from stage1_scalelsd_entropy.models.entropy_branch import EntropyBranch
from stage1_scalelsd_entropy.models.fusion_module import FeatureFusion


class ImageHorizonBackbone(nn.Module):
    """
    图像域 ResNet-34 backbone + Soft-Argmax 回归 head。

    与 cnn_model.HorizonResNet 结构相同，但:
      - 输入是 RGB image [B, 3, H, W] 而非 sinogram
      - 不使用 FiLM 调制
      - 提供 layer2 输出的 hook 接口用于 entropy fusion
    """

    def __init__(self, in_channels: int = 3, num_blocks=(3, 4, 6, 3)):
        super().__init__()
        block = BasicBlock
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.cbam = CBAM(512)

        # Soft-Argmax regression head
        self.conv_end = nn.Conv2d(512 * block.expansion, 1, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        提取特征，返回 (layer2_output, final_heatmap)。

        这是一个内部方法，用于 wrapper 在 layer2 后插入 entropy fusion。
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat_l2 = self.layer2(x)
        return feat_l2

    def forward_head(self, feat_l2: torch.Tensor, return_conf: bool = False):
        """从 layer2 输出继续前向传播到最终回归输出。"""
        x = self.layer3(feat_l2)
        x = self.layer4(x)
        x = self.cbam(x)

        x = self.conv_end(x)  # [B, 1, Hf, Wf]

        batch_size = x.size(0)
        feat_h = x.shape[2]
        feat_w = x.shape[3]

        x_flat = x.view(batch_size, -1)
        prob = F.softmax(x_flat / torch.clamp(self.temperature, min=1e-3), dim=1)
        prob = prob.view(batch_size, feat_h, feat_w)

        device = x.device
        pos_y = torch.linspace(0, 1, feat_h, device=device).view(1, feat_h, 1)
        pos_x = torch.linspace(0, 1, feat_w, device=device).view(1, 1, feat_w)

        expected_rho = torch.sum(prob * pos_y, dim=(1, 2))
        expected_theta = torch.sum(prob * pos_x, dim=(1, 2))

        out = torch.stack([expected_rho, expected_theta], dim=1)

        if not return_conf:
            return out

        pflat = prob.view(batch_size, -1)
        peak_prob = torch.max(pflat, dim=1).values
        entropy = -(pflat * torch.log(pflat + 1e-12)).sum(dim=1) / math.log(pflat.shape[1])
        conf = 0.5 * peak_prob + 0.5 * (1.0 - entropy)
        conf = torch.clamp(conf, 0.0, 1.0)
        return out, conf

    def forward(self, x: torch.Tensor, return_conf: bool = False):
        feat_l2 = self.forward_features(x)
        return self.forward_head(feat_l2, return_conf=return_conf)


# ==================================================================
# Stage-1 Wrapper: baseline / entropy-injection
# ==================================================================
class ScaleLSDEntropyWrapper(nn.Module):
    """
    Stage-1 模型封装。

    mode="baseline":
      - 只用 ImageHorizonBackbone(RGB) 直接回归
      - entropy_map 被忽略

    mode="entropy":
      - EntropyBranch 提取 entropy 特征
      - 在 backbone layer2 输出后 concat + 1×1 conv 注入
      - 然后继续 layer3/layer4/CBAM/head

    Parameters:
        mode:             "baseline" 或 "entropy"
        entropy_branch_ch: entropy branch 输出通道数 (默认 128)
        backbone_l2_ch:   backbone layer2 输出通道数 (默认 128, ResNet-34)
    """

    def __init__(
        self,
        mode: str = "entropy",
        entropy_branch_ch: int = 128,
        backbone_l2_ch: int = 128,
    ):
        super().__init__()
        assert mode in ("baseline", "entropy"), f"Unknown mode: {mode}"
        self.mode = mode

        # Backbone: image-based ResNet-34
        self.backbone = ImageHorizonBackbone(in_channels=3)

        # Entropy branch + fusion (只在 entropy 模式下使用)
        if self.mode == "entropy":
            self.entropy_branch = EntropyBranch(out_channels=entropy_branch_ch)
            self.fusion = FeatureFusion(
                backbone_channels=backbone_l2_ch,
                entropy_channels=entropy_branch_ch,
            )

    def forward(
        self,
        image: torch.Tensor,
        entropy_map: torch.Tensor | None = None,
        return_conf: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            image:       [B, 3, H, W] RGB image, float32 [0,1]
            entropy_map: [B, 1, H, W] entropy map, float32 [0,1]
                         None 或 baseline 模式下忽略
            return_conf: 是否同时返回 confidence

        Returns:
            baseline:  [B, 2]  (rho_norm, theta_norm)
            entropy:   [B, 2]  (rho_norm, theta_norm)
            若 return_conf=True: ([B,2], [B])
        """
        # Backbone: 提取到 layer2
        feat_l2 = self.backbone.forward_features(image)

        if self.mode == "entropy" and entropy_map is not None:
            # Entropy branch
            ent_feat = self.entropy_branch(entropy_map)
            # Fusion: inject into layer2 output
            feat_l2 = self.fusion(feat_l2, ent_feat)

        # Continue backbone head
        return self.backbone.forward_head(feat_l2, return_conf=return_conf)
