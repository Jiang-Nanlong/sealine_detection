"""
encoder_with_entropy.py — 带局部熵注入的 LINEA Hybrid Encoder

核心思路：
  继承 LINEA 官方 HybridEncoderAsymConv，仅覆写 forward()，
  在 input_proj 之后、Transformer encoder 之前，对最高分辨率层
  proj_feats[0]（stride=8）做一次加性熵特征注入。

注入公式：
  proj_feats[0] = proj_feats[0] + alpha * entropy_feat

新增参数：
  entropy_branch  — 轻量 3 层卷积，[B,1,H,W] → [B,256,H/2,W/2]
  alpha           — 可学习标量，初始化为 0.0

alpha 的作用：
  alpha=0 时注入项为零，forward 行为与原始 HybridEncoderAsymConv 完全一致，
  因此加载官方预训练权重后模型输出不变。
  微调阶段 alpha 自动学习到合适的注入强度。

输入说明：
  feats       — backbone 输出的多尺度特征列表，长度 = n_levels（通常 3）
                feats[0]: [B, C0, H/8,  W/8 ]  stride=8，最高分辨率
                feats[1]: [B, C1, H/16, W/16]  stride=16
                feats[2]: [B, C2, H/32, W/32]  stride=32
                其中 C0, C1, C2 取决于 backbone 型号（例如 B4 = 512, 1024, 2048）

  entropy_map — 局部熵图 [B, 1, H, W]，与输入图像同空间尺寸
                归一化方式：clip(raw_entropy / 8.0, 0, 1)
                为 None 时退化为原始 encoder 行为

注入位置说明：
  input_proj 将 feats 各层统一投影到 hidden_dim=256 通道：
    proj_feats[i]: [B, 256, H_i, W_i]

  entropy_branch 输出: [B, 256, H/2, W/2]（stride=2）
  proj_feats[0] 尺寸:  [B, 256, H/8, W/8]（stride=8）

  两者空间尺寸不一致，使用 F.interpolate(bilinear) 将 entropy_feat
  下采样对齐到 proj_feats[0] 的空间尺寸后再做加性注入。

state_dict 兼容性：
  原始 LINEA checkpoint 可通过 strict=False 加载。
  缺失 key 仅为 entropy_branch.* 和 alpha。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from LINEA.models.linea.hybrid_encoder import HybridEncoderAsymConv
from stage1_linea_entropy.models.entropy_branch import EntropyBranch


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
