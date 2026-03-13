"""
entropy_branch.py — 局部熵特征提取分支

提供两种分支：
  1. EntropyBranch        — 原始单尺度分支 (向后兼容 LINEA_ENTROPY)
  2. MultiScaleEntropyBranch — 多尺度门控分支 (用于 LINEA_ENTROPY_A)

直接运行本文件可进行 shape 自测。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Global config for direct execution shape test
# ============================================================
TEST_BATCH = 2
TEST_H = 576
TEST_W = 1024


# ============================================================
# 原始单尺度分支 (保持不变，用于 LINEA_ENTROPY)
# ============================================================
class EntropyBranch(nn.Module):
    """
    Input : [B, 1, H, W]
    Output: [B, 256, H/2, W/2]

    Designed to align with injection at LINEA HybridEncoder proj_feats[0].
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 单层门控调制模块 (GatedFiLM)
# ============================================================
class GatedFiLMLayer(nn.Module):
    """
    对单个尺度 proj_feat 进行门控 FiLM 调制。

    给定 entropy feature (与 feat 同尺寸同通道)，产生：
      - spatial_gate : [B, 1, H, W]   — 空间位置门
      - gamma        : [B, C, 1, 1]   — 通道乘性调制
      - beta         : [B, C, 1, 1]   — 通道加性调制
      - ent_feat_out : [B, C, H, W]   — 空间熵特征（加性分量）

    融合公式：
      modulated = feat * (1 + alpha * tanh(gamma) * spatial_gate)
                + alpha * ent_feat * spatial_gate
                + alpha * beta

    alpha 初始化为 0.1 使模块从训练起即有小幅作用。
    """

    def __init__(self, channels, use_spatial_gate=True, use_channel_mod=True):
        super().__init__()
        self.channels = channels
        self.use_spatial_gate = use_spatial_gate
        self.use_channel_mod = use_channel_mod

        # 空间门：entropy_feat → 1-channel sigmoid gate
        if use_spatial_gate:
            self.spatial_gate_conv = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1, bias=True),
            )

        # 通道调制：全局池化 → gamma/beta
        if use_channel_mod:
            self.channel_mod = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels * 2),  # gamma + beta
            )

        # 可学习缩放因子，初始化 0.1 使模块起步便有作用
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat, ent_feat):
        """
        Args:
            feat     : [B, C, H, W] — proj_feats[i]
            ent_feat : [B, C, H, W] — 该层对应的熵特征

        Returns:
            modulated: [B, C, H, W] — 调制后的特征
        """
        alpha = self.alpha

        # spatial gate
        if self.use_spatial_gate:
            sg = torch.sigmoid(self.spatial_gate_conv(ent_feat))  # [B, 1, H, W]
        else:
            sg = torch.ones(feat.shape[0], 1, feat.shape[2], feat.shape[3],
                            device=feat.device, dtype=feat.dtype)

        # channel modulation
        if self.use_channel_mod:
            cm = self.channel_mod(ent_feat)  # [B, C*2]
            gamma, beta = cm.chunk(2, dim=1)  # each [B, C]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        else:
            gamma = torch.zeros(feat.shape[0], self.channels, 1, 1,
                                device=feat.device, dtype=feat.dtype)
            beta = torch.zeros_like(gamma)

        # 门控 FiLM 融合
        modulated = (
            feat * (1.0 + alpha * torch.tanh(gamma) * sg)
            + alpha * ent_feat * sg
            + alpha * beta
        )
        return modulated


# ============================================================
# 多尺度熵特征提取器
# ============================================================
class MultiScaleEntropyBranch(nn.Module):
    """
    多尺度门控式熵注入分支 — 用于 LINEA_ENTROPY_A。

    输入:
      entropy_map [B, 1, H, W]

    输出:
      list of per-level 调制信息, 长度 = len(inject_levels)
      每项为 dict: {'ent_feat': Tensor [B, hidden_dim, H_i, W_i]}

    内部结构:
      共享浅层特征提取 stem (stride=4) → 每级独立的 head 投影到 hidden_dim
      + 按需 interpolate 对齐到 proj_feats[i] 的空间尺寸

    参数:
      hidden_dim      : 目标通道数 (与 HybridEncoder 的 hidden_dim 一致, 通常 256)
      inject_levels   : 要注入的层索引列表, e.g. [0, 1, 2]
      feat_strides    : 各层相对于输入图像的 stride, e.g. [8, 16, 32]
      use_spatial_gate : 是否用空间门
      use_channel_mod  : 是否用通道调制
    """

    def __init__(self, hidden_dim=256, inject_levels=(0, 1, 2),
                 feat_strides=(8, 16, 32),
                 use_spatial_gate=True, use_channel_mod=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inject_levels = list(inject_levels)
        self.feat_strides = list(feat_strides)
        self.use_spatial_gate = use_spatial_gate
        self.use_channel_mod = use_channel_mod

        # 共享 stem: [B,1,H,W] → [B, stem_ch, H/4, W/4]  (stride=4)
        stem_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # 每层独立的投影头 + GatedFiLM 模块
        self.level_heads = nn.ModuleDict()
        self.level_films = nn.ModuleDict()
        for lvl in self.inject_levels:
            # stem output stride=4, target stride=feat_strides[lvl]
            # 如果 target stride > 4, 需要进一步下采样
            extra_stride = self.feat_strides[lvl] // 4
            layers = []
            in_ch = stem_ch
            while extra_stride > 1:
                out_ch = min(in_ch * 2, hidden_dim)
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])
                in_ch = out_ch
                extra_stride //= 2
            # 最终投影到 hidden_dim
            if in_ch != hidden_dim:
                layers.extend([
                    nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ])
            self.level_heads[str(lvl)] = nn.Sequential(*layers) if layers else nn.Identity()
            self.level_films[str(lvl)] = GatedFiLMLayer(
                hidden_dim,
                use_spatial_gate=use_spatial_gate,
                use_channel_mod=use_channel_mod,
            )

    def extract_entropy_features(self, entropy_map):
        """
        提取各级熵特征 (不做 FiLM 调制本身)。

        Returns:
            dict: {level_idx: ent_feat_tensor}
        """
        stem_feat = self.stem(entropy_map)  # [B, stem_ch, H/4, W/4]
        ent_feats = {}
        for lvl in self.inject_levels:
            ent_feats[lvl] = self.level_heads[str(lvl)](stem_feat)
        return ent_feats

    def modulate(self, lvl, feat, ent_feat):
        """
        对 proj_feats[lvl] 做门控 FiLM 调制。

        Args:
            lvl      : 层索引
            feat     : [B, C, H, W] — proj_feats[lvl]
            ent_feat : [B, C, H', W'] — 熵特征 (如尺寸不匹配会自动 interpolate)

        Returns:
            modulated: [B, C, H, W]
        """
        # 对齐空间尺寸
        if ent_feat.shape[2:] != feat.shape[2:]:
            ent_feat = F.interpolate(
                ent_feat, size=feat.shape[2:], mode='bilinear', align_corners=False,
            )
        return self.level_films[str(lvl)](feat, ent_feat)


def main():
    print("=== EntropyBranch (original single-scale) ===")
    model = EntropyBranch()
    x = torch.randn(TEST_BATCH, 1, TEST_H, TEST_W)
    y = model(x)
    print('input :', tuple(x.shape))
    print('output:', tuple(y.shape))  # expected [B, 256, H/2, W/2]

    print("\n=== MultiScaleEntropyBranch (gated FiLM) ===")
    ms_branch = MultiScaleEntropyBranch(
        hidden_dim=256,
        inject_levels=[0, 1, 2],
        feat_strides=[8, 16, 32],
        use_spatial_gate=True,
        use_channel_mod=True,
    )
    ent_map = torch.randn(TEST_BATCH, 1, TEST_H, TEST_W)
    ent_feats = ms_branch.extract_entropy_features(ent_map)
    print(f'entropy_map: {tuple(ent_map.shape)}')
    for lvl, ef in ent_feats.items():
        print(f'  level {lvl} ent_feat: {tuple(ef.shape)}')

    # simulate proj_feats
    proj_feats = [
        torch.randn(TEST_BATCH, 256, TEST_H // 8, TEST_W // 8),
        torch.randn(TEST_BATCH, 256, TEST_H // 16, TEST_W // 16),
        torch.randn(TEST_BATCH, 256, TEST_H // 32, TEST_W // 32),
    ]
    for lvl in [0, 1, 2]:
        out = ms_branch.modulate(lvl, proj_feats[lvl], ent_feats[lvl])
        print(f'  level {lvl} modulated: {tuple(out.shape)} (was {tuple(proj_feats[lvl].shape)})')

    n_params = sum(p.numel() for p in ms_branch.parameters())
    print(f'  total params: {n_params:,}')


if __name__ == '__main__':
    main()
