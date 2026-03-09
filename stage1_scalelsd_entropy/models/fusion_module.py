#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_module.py — 特征级融合模块

将 entropy branch 特征注入到 backbone 中层特征中。

策略: concat + 1×1 conv 对齐通道
  - backbone_feat: [B, C_backbone, H, W]
  - entropy_feat:  [B, C_entropy, H', W']  (可能尺寸不同，先 resize)
  - 输出:          [B, C_backbone, H, W]    (保持 backbone 通道数不变)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    特征融合: concat → 1×1 conv → BN → ReLU

    将 entropy feature 对齐到 backbone feature 的空间尺寸，
    concat 后通过 1×1 卷积恢复 backbone 的通道数。

    Parameters:
        backbone_channels: backbone 中层特征的通道数
        entropy_channels:  entropy branch 输出的通道数
    """

    def __init__(self, backbone_channels: int, entropy_channels: int):
        super().__init__()
        in_ch = backbone_channels + entropy_channels
        self.align = nn.Sequential(
            nn.Conv2d(in_ch, backbone_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        backbone_feat: torch.Tensor,
        entropy_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            backbone_feat: [B, C_b, Hb, Wb]
            entropy_feat:  [B, C_e, He, We]
        Returns:
            fused: [B, C_b, Hb, Wb]
        """
        # 空间对齐: resize entropy feature to backbone spatial size
        if entropy_feat.shape[2:] != backbone_feat.shape[2:]:
            entropy_feat = F.interpolate(
                entropy_feat,
                size=backbone_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # concat + 1x1 conv
        cat = torch.cat([backbone_feat, entropy_feat], dim=1)
        return self.align(cat)
