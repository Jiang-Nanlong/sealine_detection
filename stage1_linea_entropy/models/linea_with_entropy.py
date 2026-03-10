"""
linea_with_entropy.py — 带局部熵注入的 LINEA 模型

核心思路：
  继承原始 LINEA，仅覆写 forward()，将 entropy_map 传给
  HybridEncoderWithEntropy。backbone / decoder / postprocessor 全部
  复用原始 LINEA 实现，零改动。

组件复用说明：
  - backbone   : 原始 HGNetv2，不改动
  - encoder    : 替换为 HybridEncoderWithEntropy（唯一变化）
  - decoder    : 原始 LINEATransformer，不改动
  - PostProcess: 原始 PostProcess，不改动
  - criterion  : 由外部 build_criterion 构建，与本文件无关

entropy_map 传递路径：
  LINEAWithEntropy.forward(samples, targets, entropy_map)
    → self.encoder(features, entropy_map=entropy_map)
      → HybridEncoderWithEntropy.forward(feats, entropy_map)
        → proj_feats[0] = proj_feats[0] + alpha * entropy_feat

  原始 LINEA.forward 仅 3 行逻辑：
    features = self.backbone(samples)
    features = self.encoder(features)
    out = self.decoder(features, targets)
  本文件覆写 forward，唯一区别是给 encoder 多传一个 entropy_map 参数。

预训练权重兼容说明：
  模型结构为 model.backbone.* / model.encoder.* / model.decoder.*。
  加载原始 LINEA checkpoint（strict=False）时：
    - backbone.* / decoder.* 完全匹配
    - encoder.* 中除以下两类 key 外完全匹配：
      missing keys  : encoder.entropy_branch.*, encoder.alpha
      unexpected keys: 无
  alpha 初始为 0.0，加载后模型行为与原始 LINEA 完全一致。
"""

import torch
from torch import nn

from LINEA.models.linea.linea import LINEA, PostProcess
from LINEA.models.linea.hgnetv2 import build_hgnetv2
from LINEA.models.linea.decoder import build_decoder

from stage1_linea_entropy.models.encoder_with_entropy import build_hybrid_encoder_with_entropy


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
