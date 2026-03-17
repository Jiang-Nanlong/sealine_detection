"""
linea_with_entropy.py — 带局部熵注入的 LINEA 模型

提供四种模型:
  1. LINEAWithEntropy          — 原始单层加性注入 (LINEA_ENTROPY)
  2. LINEAWithEntropyA         — 多层门控 FiLM 注入 (LINEA_ENTROPY_A)
  3. LINEAWithEntropyB         — detector 内置 horizon-aware scoring head (LINEA_ENTROPY_B)
  4. LINEAWithEntropyBEnhanced — MSLEP + SAI + EGAB + HASH 增强版 (LINEA_ENTROPY_B_ENHANCED)
"""

import torch
from torch import nn
import torch.nn.functional as F

from LINEA.models.linea.linea import LINEA, PostProcess
from LINEA.models.linea.hgnetv2 import build_hgnetv2
from LINEA.models.linea.decoder import build_decoder

from stage1_linea_entropy.models.encoder_with_entropy import (
    build_hybrid_encoder_with_entropy,
    build_hybrid_encoder_with_entropy_a,
    build_hybrid_encoder_with_entropy_enhanced,
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

    融合公式（取决于 use_raw_calibration）：
      use_raw_calibration=True :
        base = RawCalibrationHead(query, raw_logits)
      use_raw_calibration=False:
        base = raw_logits  (无快捷路径，迫使梯度经过 horizon 分支)

      combined = base + score_scale * [gate *] horizon_delta
    """

    def __init__(self, backbone, encoder, decoder,
                 horizon_head=None, enable_horizon_head=True,
                 use_raw_calibration=True):
        super().__init__(backbone, encoder, decoder)
        self.enable_horizon_head = enable_horizon_head
        self.horizon_head = horizon_head
        self.use_raw_calibration = use_raw_calibration

    def forward(self, samples, targets=None, entropy_map=None):
        features = self.backbone(samples)
        features = self.encoder(features, entropy_map=entropy_map)
        out = self.decoder(features, targets)

        # 取出 hs_stack（decoder 暴露的多层隐状态）
        hs_stack = out.pop('hs_stack', None)

        if self.enable_horizon_head and self.horizon_head is not None and hs_stack is not None:
            img_h, img_w = samples.shape[2], samples.shape[3]
            raw_logits = out['pred_logits']

            horizon_delta, fusion_gate, raw_calibrated, score_scale = self.horizon_head(
                hs_stack=hs_stack,
                pred_lines=out['pred_lines'],
                raw_logits=raw_logits,
                encoder_feat=features[0],
                images=samples,
                entropy_map=entropy_map,
                img_h=img_h,
                img_w=img_w,
            )

            # 选择基础 logit：有校准头则用校准后的，否则直接用 raw
            base_logits = raw_calibrated if self.use_raw_calibration else raw_logits

            if fusion_gate is not None:
                combined = base_logits + score_scale * fusion_gate * horizon_delta
                out['pred_fusion_gate'] = fusion_gate
            else:
                combined = base_logits + score_scale * horizon_delta

            out['pred_logits_raw_det'] = raw_logits
            out['pred_logits_raw_calibrated'] = raw_calibrated
            out['pred_logits_horizon'] = horizon_delta
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
            use_multilayer_query=getattr(args, 'horizon_use_multilayer_query', True),
            num_query_layers=getattr(args, 'horizon_num_query_layers', 3),
            use_scale_attention=getattr(args, 'horizon_use_scale_attention', True),
            score_init_scale=getattr(args, 'horizon_score_init_scale', 0.1),
            clamp_delta=getattr(args, 'horizon_clamp_delta', False),
            entropy_channels=getattr(args, 'entropy_in_channels', 1),
        )

    model = LINEAWithEntropyB(
        backbone, encoder, decoder,
        horizon_head=horizon_head,
        enable_horizon_head=enable_hh,
        use_raw_calibration=getattr(args, 'horizon_use_raw_calibration', True),
    )

    postprocessors = PostProcess()

    return model, postprocessors


# ============================================================
# 增强版主线 B: MSLEP + SAI + EGAB + HASH
# ============================================================
class LINEAWithEntropyBEnhanced(LINEA):
    """
    全量增强版 LINEA_ENTROPY_B, 集成四个创新模块:

    1. MSLEP (Multi-Scale Local Entropy Prior)
       — 多尺度熵先验提取, 在 encoder 内完成

    2. SAI (Spatial Adaptive Injection)
       — 空间自适应注入, 在 encoder 内完成

    3. EGAB (Entropy-Guided Attention Bias)
       — 在 decoder 各层 cross-attention 后, 用熵图引导残差修正

    4. HASH (Horizon-Aware Scoring Head)
       — 在 decoder 输出后, 对候选线做内部重评分

    encoder 使用 HybridEncoderWithEntropyEnhanced (已含 MSLEP + SAI).
    decoder 使用标准 LINEA decoder, EGAB 在本类 forward 中实现.
    """

    def __init__(self, backbone, encoder, decoder,
                 horizon_head=None, enable_horizon_head=True,
                 use_raw_calibration=False,
                 egab=None, enable_egab=True):
        super().__init__(backbone, encoder, decoder)
        self.enable_horizon_head = enable_horizon_head
        self.horizon_head = horizon_head
        self.use_raw_calibration = use_raw_calibration
        self.egab = egab
        self.enable_egab = enable_egab

    def _sample_entropy_at_refpoints(self, entropy_map, ref_points):
        """
        在参考线中点处采样熵值, 用于 EGAB post_attn 模式.

        Args:
            entropy_map : [B, 1, H, W]
            ref_points  : [nq, B, 4] — 归一化参考线 [x1, y1, x2, y2]

        Returns:
            [nq, B, 1]
        """
        nq, B, _ = ref_points.shape
        mid_x = 0.5 * (ref_points[:, :, 0] + ref_points[:, :, 2])
        mid_y = 0.5 * (ref_points[:, :, 1] + ref_points[:, :, 3])

        grid_x = mid_x.permute(1, 0) * 2 - 1  # [B, nq]
        grid_y = mid_y.permute(1, 0) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # [B, nq, 1, 2]

        sampled = F.grid_sample(
            entropy_map, grid, mode='bilinear',
            padding_mode='border', align_corners=False,
        )  # [B, C, nq, 1]  (C=1 or 3)

        return sampled.mean(dim=1).squeeze(-1).permute(1, 0).unsqueeze(-1)  # [nq, B, 1]

    def forward(self, samples, targets=None, entropy_map=None):
        features = self.backbone(samples)
        features = self.encoder(features, entropy_map=entropy_map)

        # --- EGAB: memory_bias 模式 ---
        if (self.enable_egab and self.egab is not None
                and entropy_map is not None
                and self.egab.mode == 'memory_bias'):
            entropy_flat_parts = []
            for feat in features:
                _, _, fh, fw = feat.shape
                ent_level = F.interpolate(
                    entropy_map, size=(fh, fw),
                    mode='bilinear', align_corners=False,
                )
                entropy_flat_parts.append(ent_level.flatten(2).permute(0, 2, 1))

            entropy_flat = torch.cat(entropy_flat_parts, dim=1)
            memory_bias = self.egab.compute_memory_bias(entropy_flat)

            enhanced_features = []
            offset = 0
            for feat in features:
                _, c, fh, fw = feat.shape
                n = fh * fw
                bias_level = memory_bias[:, offset:offset + n, :].permute(0, 2, 1).reshape(-1, c, fh, fw)
                enhanced_features.append(feat + bias_level)
                offset += n
            features = enhanced_features

        out = self.decoder(features, targets)

        # --- EGAB: post_attn 模式 ---
        hs_stack = out.pop('hs_stack', None)

        if (self.enable_egab and self.egab is not None
                and entropy_map is not None
                and self.egab.mode == 'post_attn'
                and hs_stack is not None):
            pred_lines = out['pred_lines']  # [B, nq, 4]
            ref_points = pred_lines.detach().permute(1, 0, 2)  # [nq, B, 4]
            entropy_vals = self._sample_entropy_at_refpoints(
                entropy_map, ref_points,
            )  # [nq, B, 1]
            last_hs = hs_stack[-1].permute(1, 0, 2)  # [nq, B, d_model]
            delta = self.egab.compute_post_attn_residual(last_hs, entropy_vals)
            hs_stack = hs_stack.clone()
            hs_stack[-1] = (last_hs + delta).permute(1, 0, 2)

        # --- HASH: Horizon-Aware Scoring Head ---
        if self.enable_horizon_head and self.horizon_head is not None and hs_stack is not None:
            img_h, img_w = samples.shape[2], samples.shape[3]
            raw_logits = out['pred_logits']

            horizon_delta, fusion_gate, raw_calibrated, score_scale = self.horizon_head(
                hs_stack=hs_stack,
                pred_lines=out['pred_lines'],
                raw_logits=raw_logits,
                encoder_feat=features[0],
                images=samples,
                entropy_map=entropy_map,
                img_h=img_h,
                img_w=img_w,
            )

            base_logits = raw_calibrated if self.use_raw_calibration else raw_logits

            if fusion_gate is not None:
                combined = base_logits + score_scale * fusion_gate * horizon_delta
                out['pred_fusion_gate'] = fusion_gate
            else:
                combined = base_logits + score_scale * horizon_delta

            out['pred_logits_raw_det'] = raw_logits
            out['pred_logits_raw_calibrated'] = raw_calibrated
            out['pred_logits_horizon'] = horizon_delta
            out['pred_logits_combined'] = combined
            out['pred_logits'] = combined

        return out


def build_linea_with_entropy_b_enhanced(args):
    """
    构建 LINEAWithEntropyBEnhanced + PostProcess.
    """
    from stage1_linea_entropy.models.horizon_scoring_head import HorizonScoringHead
    from stage1_linea_entropy.models.entropy_branch import EntropyGuidedAttentionBias

    backbone = build_hgnetv2(args)
    encoder = build_hybrid_encoder_with_entropy_enhanced(args)
    decoder = build_decoder(args)

    # --- EGAB ---
    enable_egab = getattr(args, 'enable_egab', True)
    egab = None
    if enable_egab:
        egab = EntropyGuidedAttentionBias(
            d_model=args.hidden_dim,
            mode=getattr(args, 'egab_mode', 'post_attn'),
        )

    # --- Horizon Scoring Head ---
    enable_hh = getattr(args, 'enable_horizon_head', True)
    horizon_head = None
    if enable_hh:
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
            use_adaptive_fusion_gate=getattr(args, 'horizon_use_adaptive_fusion_gate', False),
            use_multilayer_query=getattr(args, 'horizon_use_multilayer_query', True),
            num_query_layers=getattr(args, 'horizon_num_query_layers', 3),
            use_scale_attention=getattr(args, 'horizon_use_scale_attention', True),
            score_init_scale=getattr(args, 'horizon_score_init_scale', 1.0),
            clamp_delta=getattr(args, 'horizon_clamp_delta', False),
            entropy_channels=getattr(args, 'entropy_in_channels', 1),
        )

    model = LINEAWithEntropyBEnhanced(
        backbone, encoder, decoder,
        horizon_head=horizon_head,
        enable_horizon_head=enable_hh,
        use_raw_calibration=getattr(args, 'horizon_use_raw_calibration', False),
        egab=egab,
        enable_egab=enable_egab,
    )

    postprocessors = PostProcess()

    return model, postprocessors
