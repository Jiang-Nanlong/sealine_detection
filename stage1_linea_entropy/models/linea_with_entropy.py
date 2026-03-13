"""
linea_with_entropy.py — 带局部熵注入的 LINEA 模型

提供三种模型:
  1. LINEAWithEntropy   — 原始单层加性注入 (LINEA_ENTROPY)
  2. LINEAWithEntropyA  — 多层门控 FiLM 注入 (LINEA_ENTROPY_A)
  3. LINEAWithEntropyB  — detector 内置 horizon-aware scoring head (LINEA_ENTROPY_B)
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


# ============================================================
# 主线 B: detector 内置 horizon-aware scoring head
# ============================================================
class LINEAWithEntropyB(LINEA):
    """
    LINEA 子类，encoder 使用 HybridEncoderWithEntropy，
    decoder 输出经 HorizonScoringHead 增强。

    当 enable_horizon_head=True 时：
      combined_logit = raw_det_logit + score_scale * horizon_logit
    当 enable_horizon_head=False 时：
      退化为 LINEAWithEntropy 行为
    """

    def __init__(self, backbone, encoder, decoder,
                 horizon_head=None, enable_horizon_head=True):
        super().__init__(backbone, encoder, decoder)
        self.enable_horizon_head = enable_horizon_head
        self.horizon_head = horizon_head

    def forward(self, samples, targets=None, entropy_map=None):
        features = self.backbone(samples)
        features = self.encoder(features, entropy_map=entropy_map)
        out = self.decoder(features, targets)

        # 移除 hs_last（不需要时清理；需要时由 horizon head 消费）
        hs_last = out.pop('hs_last', None)

        if self.enable_horizon_head and self.horizon_head is not None and hs_last is not None:
            img_h, img_w = samples.shape[2], samples.shape[3]

            horizon_logit, fusion_gate, score_scale = self.horizon_head(
                hs_last=hs_last,
                pred_lines=out['pred_lines'],
                encoder_feat=features[0],
                images=samples,
                entropy_map=entropy_map,
                img_h=img_h,
                img_w=img_w,
            )

            raw_logits = out['pred_logits']

            if fusion_gate is not None:
                # 候选线自适应融合
                combined = raw_logits + score_scale * fusion_gate * horizon_logit
                out['pred_fusion_gate'] = fusion_gate
            else:
                # 回退：全局标量融合
                combined = raw_logits + score_scale * horizon_logit

            out['pred_logits_raw_det'] = raw_logits
            out['pred_logits_horizon'] = horizon_logit
            out['pred_logits_combined'] = combined
            out['pred_logits'] = combined

        return out


def build_linea_with_entropy_b(args):
    """
    构建 LINEAWithEntropyB + PostProcess。
    encoder: HybridEncoderWithEntropy（单层熵注入）
    decoder: 标准 LINEA decoder
    horizon_head: HorizonScoringHead（可配置）
    """
    from stage1_linea_entropy.models.horizon_scoring_head import HorizonScoringHead

    backbone = build_hgnetv2(args)
    encoder = build_hybrid_encoder_with_entropy(args)
    decoder = build_decoder(args)

    enable_hh = getattr(args, 'enable_horizon_head', True)
    horizon_head = None
    if enable_hh:
        # 支持新的 band_widths 列表，同时兼容旧的单 band_width
        band_widths = getattr(args, 'horizon_band_widths', None)
        if band_widths is None:
            bw = getattr(args, 'horizon_band_width', 3.0)
            band_widths = [bw]

        horizon_head = HorizonScoringHead(
            d_model=args.hidden_dim,
            hidden_dim=getattr(args, 'horizon_hidden_dim', 256),
            num_classes=args.num_classes,
            num_sample_points=getattr(args, 'horizon_num_sample_points', 16),
            band_widths=band_widths,
            use_feat_context=getattr(args, 'horizon_use_feat_context', True),
            use_entropy_context=getattr(args, 'horizon_use_entropy_context', True),
            use_gradient_context=getattr(args, 'horizon_use_gradient_context', True),
            use_geometry=getattr(args, 'horizon_use_geometry', True),
            use_adaptive_fusion_gate=getattr(args, 'horizon_use_adaptive_fusion_gate', True),
            score_init_scale=getattr(args, 'horizon_score_init_scale', 0.1),
        )

    model = LINEAWithEntropyB(
        backbone, encoder, decoder,
        horizon_head=horizon_head,
        enable_horizon_head=enable_hh,
    )

    postprocessors = PostProcess()

    return model, postprocessors
