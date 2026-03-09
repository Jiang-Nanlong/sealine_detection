"""
dpt_with_entropy.py — 基于已有 DPTFieldModel 实例的 entropy 注入包装器

功能：
  接收一个由 ScaleLSD 真实创建的 DPTFieldModel 实例（base_backbone），
  直接复用其 pretrained / scratch 等全部子模块，仅新增 entropy_branch + alpha。
  在 DPT.forward() 的 path_1 处做加性注入：
      path_1 = path_1 + alpha * entropy_feat

  不重新构造 backbone，不硬编码任何 DPT 构造参数。

注入公式：
  alpha (nn.Parameter, scalar, 初始 0.0) 控制 entropy 特征的强度。
  alpha=0 时加性项为零，前向行为与原始 DPT 完全一致。

state_dict 兼容性：
  原始 ScaleLSD checkpoint 可通过 strict=False 加载。
  缺失 key 仅为 entropy_branch.* 和 alpha（全部为新增参数）。
  其余 key（pretrained.*, scratch.*）与原始 backbone 一一对应。

导入前提：
  PyCharm 中需将项目根目录和 scalelsd/ 目录均设为 Source Root，
  或通过其他方式确保 scalelsd.ssl.* 和 stage1_scalelsd_entropy.* 可正常导入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scalelsd.ssl.backbones.dpt.blocks import forward_vit

from stage1_scalelsd_entropy.models.entropy_branch import EntropyBranch


class DPTFieldModelWithEntropy(nn.Module):
    """
    基于已有 DPTFieldModel 实例的 entropy 注入包装器。

    构造方式：
        original = build_backbone(...)           # 由 ScaleLSD.__init__ 创建
        wrapped  = DPTFieldModelWithEntropy(original)

    直接复用 original 的 pretrained / scratch / channels_last / stride，
    参数 key 路径不变，只额外新增 entropy_branch.* 和 alpha。

    forward(x, entropy_map=None):
        x           : [B, 1, H, W]  灰度图像（内部 1ch→3ch）
        entropy_map : [B, 1, H, W]  局部熵图，为 None 时退化为原始行为
        → 返回 (out, None)
        out         : [B, 9, H/2, W/2]  HAFM 9 通道预测（与原始 DPT 一致）

    alpha 的作用：
        可学习标量缩放因子。初始化为 0.0，保证：
        - 加载原始预训练权重后，模型行为与原始 DPTFieldModel 完全一致
        - 微调阶段 alpha 自动学习到合适的 entropy 注入强度
    """

    def __init__(self, base_backbone):
        """
        Args:
            base_backbone: 由 ScaleLSD 真实创建的 DPTFieldModel 实例。
                           其 pretrained / scratch 子模块将被直接复用（零拷贝）。
        """
        super().__init__()

        # ---- 直接复用原始 backbone 的所有子模块（不重新构造）----
        self.pretrained = base_backbone.pretrained
        self.scratch = base_backbone.scratch
        self.channels_last = base_backbone.channels_last
        self.stride = base_backbone.stride

        # ---- 新增：entropy 注入组件 ----
        self.entropy_branch = EntropyBranch()   # [B,1,H,W] → [B,256,H/2,W/2]
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, entropy_map=None):
        """
        覆写 DPT 前向，在 path_1 处注入 entropy 特征。

        流程：
          1. 灰度 1ch → 3ch（与原 DPTFieldModel 相同）
          2. forward_vit → layerX_rn → refinenet cascade → path_1
             [B, 256, H/2, W/2]
          3. 若 entropy_map 不为 None：
               entropy_feat = entropy_branch(entropy_map)  → [B, 256, H/2, W/2]
               path_1 = path_1 + alpha * entropy_feat
             否则跳过，行为与原 DPT 完全一致
          4. output_conv(path_1) → out [B, 9, H/2, W/2]

        Returns:
            (out, None)  — 与原 DPTFieldModel.forward 返回格式一致
        """
        # ---- step 1: 灰度 → 3ch ----
        if x.shape[1] == 1:
            x = torch.cat([x, x, x], dim=1)

        # ---- step 2: DPT 前向（复用 self.pretrained / self.scratch）----
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # ---- step 3: entropy injection ----
        if entropy_map is not None:
            # 对齐 device / dtype，防止 entropy_map 与 path_1 不一致
            entropy_map = entropy_map.to(dtype=path_1.dtype, device=path_1.device)
            entropy_feat = self.entropy_branch(entropy_map)  # [B,256,H/2,W/2]
            # 空间尺寸安全对齐（正常情况下两者一致，此处防御性检查）
            if entropy_feat.shape[2:] != path_1.shape[2:]:
                entropy_feat = F.interpolate(
                    entropy_feat, size=path_1.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            path_1 = path_1 + self.alpha * entropy_feat

        # ---- step 4: output conv（head）→ [B, 9, H/2, W/2] ----
        out = self.scratch.output_conv(path_1)

        return out, None
