"""
encoder_with_entropy.py — 带局部熵注入的 LINEA Hybrid Encoder

提供两种 encoder:
  1. HybridEncoderWithEntropy   — 原始单层加性注入 (LINEA_ENTROPY)
  2. HybridEncoderWithEntropyA  — 多层门控 FiLM 注入 (LINEA_ENTROPY_A)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from LINEA.models.linea.hybrid_encoder import HybridEncoderAsymConv
from stage1_linea_entropy.models.entropy_branch import EntropyBranch, MultiScaleEntropyBranch


class HybridEncoderWithEntropy(HybridEncoderAsymConv):
    """
    HybridEncoderAsymConv 子类，仅新增 entropy_branch + alpha。

    构造方式与父类完全相同，额外参数为零。
    所有父类参数（in_channels, hidden_dim, nhead, ...）原样透传。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ---- 新增：局部熵注入组件 ----
        self.entropy_branch = EntropyBranch()           # [B, 1, H, W] → [B, 256, H/2, W/2]
        self.alpha = nn.Parameter(torch.tensor(0.0))    # 可学习缩放因子，初始 0

    def forward(self, feats, entropy_map=None):
        """
        覆写父类 forward，在 input_proj 之后注入局部熵特征。

        Args:
            feats: list of Tensor — backbone 多尺度输出
                   feats[i] shape: [B, C_i, H_i, W_i]
            entropy_map: Tensor or None — [B, 1, H, W] 局部熵图
                         为 None 时行为与父类完全一致

        Returns:
            outs: list of Tensor — 增强后的多尺度特征
                  outs[i] shape: [B, hidden_dim, H_i, W_i]
        """
        # ---- 1. input_proj：将 backbone 各层投影到 hidden_dim ----
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # ---- 2. 局部熵注入（仅 proj_feats[0]，最高分辨率层） ----
        if entropy_map is not None:
            entropy_feat = self.entropy_branch(entropy_map)     # [B, 256, H/2, W/2]
            target_h, target_w = proj_feats[0].shape[2], proj_feats[0].shape[3]
            if entropy_feat.shape[2] != target_h or entropy_feat.shape[3] != target_w:
                entropy_feat = F.interpolate(
                    entropy_feat,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )
            proj_feats[0] = proj_feats[0] + self.alpha * entropy_feat

        # ---- 3. Transformer encoder（与父类相同） ----
        for i, enc_idx in enumerate(self.use_encoder_idx):
            N_, C_, H_, W_ = proj_feats[enc_idx].shape
            src_flatten = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.create_sinehw_position_embedding(
                    H_, W_, self.hidden_dim // 2, device=src_flatten.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_idx}', None).to(src_flatten.device)

            proj_feats[enc_idx] = self.encoder[i](
                src_flatten,
                pos_embed=pos_embed,
            ).permute(0, 2, 1).reshape(N_, C_, H_, W_).contiguous()

        # ---- 4. FPN top-down（与父类相同） ----
        inner_outs = [proj_feats[-1]]
        for idx in range(self.n_levels - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[self.n_levels - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[self.n_levels - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # ---- 5. PAN bottom-up（与父类相同） ----
        outs = [inner_outs[0]]
        for idx in range(self.n_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs


def build_hybrid_encoder_with_entropy(args):
    """
    构建 HybridEncoderWithEntropy，参数与官方 build_hybrid_encoder 完全一致。
    """
    return HybridEncoderWithEntropy(
        in_channels=args.in_channels_encoder,
        feat_strides=args.feat_strides,
        n_levels=args.num_feature_levels,
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enc_act='gelu',
        expansion=args.expansion,
        depth_mult=args.depth_mult,
        act='silu',
        temperatureH=args.pe_temperatureH,
        temperatureW=args.pe_temperatureW,
        eval_spatial_size=args.eval_spatial_size,
    )


# ============================================================
# 主线 A: 多层门控 FiLM 注入 Encoder
# ============================================================
class HybridEncoderWithEntropyA(HybridEncoderAsymConv):
    """
    多层门控式熵注入 Encoder — 用于 LINEA_ENTROPY_A。

    相比 HybridEncoderWithEntropy 的区别：
      - 支持对 proj_feats[0]/[1]/[2] 的任意子集做注入
      - 每层独立的 GatedFiLM 调制 (空间门 + 通道调制 + 可学习 alpha)
      - 不兼容旧 checkpoint，从头训练

    配置项 (通过 args 传入):
      enable_multiscale_entropy  : bool, True 则启用多层门控注入
      entropy_inject_levels      : list[int], e.g. [0, 1, 2]
      entropy_use_spatial_gate   : bool
      entropy_use_channel_modulation : bool
    """

    def __init__(self, *args,
                 enable_multiscale_entropy=True,
                 entropy_inject_levels=(0, 1, 2),
                 entropy_use_spatial_gate=True,
                 entropy_use_channel_modulation=True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.enable_multiscale_entropy = enable_multiscale_entropy
        self.entropy_inject_levels = list(entropy_inject_levels)

        if enable_multiscale_entropy:
            self.entropy_ms_branch = MultiScaleEntropyBranch(
                hidden_dim=self.hidden_dim,
                inject_levels=entropy_inject_levels,
                feat_strides=self.feat_strides[:self.n_levels],
                use_spatial_gate=entropy_use_spatial_gate,
                use_channel_mod=entropy_use_channel_modulation,
            )
        else:
            # 回退到单层加性注入 (类似旧版)
            self.entropy_branch = EntropyBranch()
            self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, feats, entropy_map=None):
        # ---- 1. input_proj ----
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # ---- 2. 熵注入 ----
        if entropy_map is not None:
            if self.enable_multiscale_entropy:
                ent_feats = self.entropy_ms_branch.extract_entropy_features(entropy_map)
                for lvl in self.entropy_inject_levels:
                    if lvl < len(proj_feats):
                        proj_feats[lvl] = self.entropy_ms_branch.modulate(
                            lvl, proj_feats[lvl], ent_feats[lvl],
                        )
            else:
                # 回退: 单层加性注入 (与旧版行为一致)
                entropy_feat = self.entropy_branch(entropy_map)
                target_h, target_w = proj_feats[0].shape[2], proj_feats[0].shape[3]
                if entropy_feat.shape[2] != target_h or entropy_feat.shape[3] != target_w:
                    entropy_feat = F.interpolate(
                        entropy_feat, size=(target_h, target_w),
                        mode='bilinear', align_corners=False,
                    )
                proj_feats[0] = proj_feats[0] + self.alpha * entropy_feat

        # ---- 3. Transformer encoder ----
        for i, enc_idx in enumerate(self.use_encoder_idx):
            N_, C_, H_, W_ = proj_feats[enc_idx].shape
            src_flatten = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_spatial_size is None:
                pos_embed = self.create_sinehw_position_embedding(
                    H_, W_, self.hidden_dim // 2, device=src_flatten.device)
            else:
                pos_embed = getattr(self, f'pos_embed{enc_idx}', None).to(src_flatten.device)

            proj_feats[enc_idx] = self.encoder[i](
                src_flatten,
                pos_embed=pos_embed,
            ).permute(0, 2, 1).reshape(N_, C_, H_, W_).contiguous()

        # ---- 4. FPN top-down ----
        inner_outs = [proj_feats[-1]]
        for idx in range(self.n_levels - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[self.n_levels - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[self.n_levels - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # ---- 5. PAN bottom-up ----
        outs = [inner_outs[0]]
        for idx in range(self.n_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs


def build_hybrid_encoder_with_entropy_a(args):
    """
    构建 HybridEncoderWithEntropyA，带多层门控注入配置。
    """
    enable_ms = getattr(args, 'enable_multiscale_entropy', True)
    inject_levels = getattr(args, 'entropy_inject_levels', [0, 1, 2])
    use_sg = getattr(args, 'entropy_use_spatial_gate', True)
    use_cm = getattr(args, 'entropy_use_channel_modulation', True)

    return HybridEncoderWithEntropyA(
        in_channels=args.in_channels_encoder,
        feat_strides=args.feat_strides,
        n_levels=args.num_feature_levels,
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enc_act='gelu',
        expansion=args.expansion,
        depth_mult=args.depth_mult,
        act='silu',
        temperatureH=args.pe_temperatureH,
        temperatureW=args.pe_temperatureW,
        eval_spatial_size=args.eval_spatial_size,
        enable_multiscale_entropy=enable_ms,
        entropy_inject_levels=inject_levels,
        entropy_use_spatial_gate=use_sg,
        entropy_use_channel_modulation=use_cm,
    )
