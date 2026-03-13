"""
linea_with_entropy.py — 带局部熵注入的 LINEA 模型

提供两种模型:
  1. LINEAWithEntropy   — 原始单层加性注入 (LINEA_ENTROPY)
  2. LINEAWithEntropyA  — 多层门控 FiLM 注入 (LINEA_ENTROPY_A)
"""

import torch
from torch import nn

from LINEA.models.linea.linea import LINEA, PostProcess
from LINEA.models.linea.hgnetv2 import build_hgnetv2
from LINEA.models.linea.decoder import build_decoder

from stage1_linea_entropy.models.encoder_with_entropy import (
    build_hybrid_encoder_with_entropy,
    build_hybrid_encoder_with_entropy_a,
)


class LINEAWithEntropy(LINEA):
    """
    LINEA 子类，encoder 替换为 HybridEncoderWithEntropy。

    构造参数与原始 LINEA 完全相同（backbone, encoder, decoder），
    唯一要求是 encoder 必须是 HybridEncoderWithEntropy 实例。

    forward 签名新增 entropy_map 参数：
      forward(samples, targets=None, entropy_map=None)
    """

    def forward(self, samples, targets=None, entropy_map=None):
        """
        覆写父类 forward，将 entropy_map 传给 encoder。

        Args:
            samples : Tensor [B, 3, H, W] — 输入图像（RGB）
            targets : list of dict or None — LINEA 训练 target
            entropy_map : Tensor [B, 1, H, W] or None — 局部熵图
                          为 None 时 encoder 退化为原始行为

        Returns:
            out : dict — 与原始 LINEA 输出格式完全一致
                  包含 pred_logits, pred_lines, aux_outputs 等
        """
        features = self.backbone(samples)

        features = self.encoder(features, entropy_map=entropy_map)

        out = self.decoder(features, targets)

        return out


def build_linea_with_entropy(args):
    """
    构建 LINEAWithEntropy + PostProcess。

    与原始 build_linea 的唯一区别是 encoder 使用
    build_hybrid_encoder_with_entropy（带 entropy_branch + alpha）。

    Args:
        args: 配置对象，需包含原始 LINEA 所需的全部字段

    Returns:
        model         : LINEAWithEntropy
        postprocessors: PostProcess
    """
    backbone = build_hgnetv2(args)
    encoder = build_hybrid_encoder_with_entropy(args)
    decoder = build_decoder(args)

    model = LINEAWithEntropy(
        backbone,
        encoder,
        decoder,
    )

    postprocessors = PostProcess()

    return model, postprocessors


# ============================================================
# 主线 A: 多层门控 FiLM 注入模型
# ============================================================
class LINEAWithEntropyA(LINEA):
    """
    LINEA 子类，encoder 替换为 HybridEncoderWithEntropyA。
    支持多层门控式熵注入。
    """

    def forward(self, samples, targets=None, entropy_map=None):
        features = self.backbone(samples)
        features = self.encoder(features, entropy_map=entropy_map)
        out = self.decoder(features, targets)
        return out


def build_linea_with_entropy_a(args):
    """
    构建 LINEAWithEntropyA + PostProcess。
    encoder 使用 build_hybrid_encoder_with_entropy_a（多层门控注入）。
    """
    backbone = build_hgnetv2(args)
    encoder = build_hybrid_encoder_with_entropy_a(args)
    decoder = build_decoder(args)

    model = LINEAWithEntropyA(
        backbone,
        encoder,
        decoder,
    )

    postprocessors = PostProcess()

    return model, postprocessors
