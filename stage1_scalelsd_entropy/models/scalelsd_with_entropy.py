"""
scalelsd_with_entropy.py — 带局部熵注入的 ScaleLSD 检测器

核心思路：
  先调用原始 ScaleLSD.__init__() 得到真实 backbone（DPTFieldModel），
  再用 DPTFieldModelWithEntropy 包装该 backbone（零拷贝复用 pretrained / scratch），
  仅额外新增 entropy_branch 和 alpha。

  不再通过硬编码工厂函数重新构造 backbone 配置。

新增参数（仅存在于 backbone 中）：
  backbone.entropy_branch.*   — 轻量 3 层卷积（全新，需训练）
  backbone.alpha              — 可学习标量，初始 0.0

兼容性：
  原始 ScaleLSD 预训练权重可通过 strict=False 加载，仅缺失
  backbone.entropy_branch.* 和 backbone.alpha。
  alpha=0 保证初始行为与原模型完全一致。

entropy_map 传递机制：
  采用 attribute-stashing 模式：forward / forward_train / forward_test 将
  entropy_map 暂存到 self._entropy_map，覆写的 forward_backbone 读取后传给
  backbone.forward(x, entropy_map=...)。无需复制父类大段训练/测试逻辑。

导入前提：
  PyCharm 中需将项目根目录和 scalelsd/ 目录均设为 Source Root。
"""

import torch
from torch import nn

from scalelsd.ssl.models.detector import ScaleLSD
from stage1_scalelsd_entropy.models.dpt_with_entropy import DPTFieldModelWithEntropy


class ScaleLSDWithEntropy(ScaleLSD):
    """
    ScaleLSD 子类，backbone 替换为 DPTFieldModelWithEntropy。

    构造流程：
      1. super().__init__() → 创建真实 DPTFieldModel backbone
      2. DPTFieldModelWithEntropy(self.backbone) → 包装真实 backbone，
         直接复用其 pretrained / scratch，仅新增 entropy_branch + alpha
      3. 替换 self.backbone
    """

    def __init__(self, gray_scale=False, use_layer_scale=False, enable_attention_hooks=False):
        # ---- 调用原始 ScaleLSD.__init__，创建真实 backbone ----
        super().__init__(
            gray_scale=gray_scale,
            use_layer_scale=use_layer_scale,
            enable_attention_hooks=enable_attention_hooks,
        )
        # ---- 基于真实 backbone 包装，仅新增 entropy 组件 ----
        self.backbone = DPTFieldModelWithEntropy(self.backbone)
        self.stride = self.backbone.stride
        self._entropy_map = None  # 临时存储，供 forward_backbone 读取

    # ------------------------------------------------------------------
    #  forward_backbone: 覆写以传递 entropy_map
    # ------------------------------------------------------------------

    def forward_backbone(self, images, entropy_map=None):
        """
        覆写父类 forward_backbone，传递 entropy_map 给 backbone。

        entropy_map 来源（按优先级）：
          1. 显式参数（直接调用 forward_backbone 时）
          2. self._entropy_map（由 forward / forward_train / forward_test stash）

        返回值与原始 forward_backbone 完全相同：
            (outputs, features, auxputs)
        """
        if entropy_map is None:
            entropy_map = self._entropy_map
        outputs, features = self.backbone(images, entropy_map=entropy_map)
        if isinstance(outputs, list):
            auxputs = outputs[1:]
            outputs = outputs[0]
        else:
            auxputs = []
        return outputs, features, auxputs

    # ------------------------------------------------------------------
    #  forward / forward_train / forward_test: stash entropy_map 后委托父类
    # ------------------------------------------------------------------

    def forward(self, images, annotations=None, targets=None, entropy_map=None):
        """stash entropy_map，然后委托给父类 forward。"""
        self._entropy_map = entropy_map
        try:
            return super().forward(images, annotations=annotations, targets=targets)
        finally:
            self._entropy_map = None

    def forward_train(self, images, annotations=None, entropy_map=None):
        """允许直接调用时传入 entropy_map。"""
        if entropy_map is not None:
            self._entropy_map = entropy_map
        try:
            return super().forward_train(images, annotations=annotations)
        finally:
            self._entropy_map = None

    @torch.no_grad()
    def forward_test(self, images, annotations=None, merge=False, entropy_map=None):
        """允许直接调用时传入 entropy_map。"""
        if entropy_map is not None:
            self._entropy_map = entropy_map
        try:
            return super().forward_test(images, annotations=annotations, merge=merge)
        finally:
            self._entropy_map = None


def load_scalelsd_with_entropy(ckpt_path=None, gray_scale=True, use_layer_scale=False):
    """
    创建 ScaleLSDWithEntropy 并加载原始 ScaleLSD 预训练权重。

    Args:
        ckpt_path: 预训练权重路径，None 时不加载权重。
        gray_scale: 是否灰度模式（MU-SID 为 True）。
        use_layer_scale: 是否使用 layer scale。

    Returns:
        ScaleLSDWithEntropy 模型实例。
        missing_keys / unexpected_keys 在加载时打印。

    checkpoint 格式兼容：
        支持 {"model_state": ...}、{"model": ...}、以及直接 state_dict。
    """
    model = ScaleLSDWithEntropy(gray_scale=gray_scale, use_layer_scale=use_layer_scale)

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[load_scalelsd_with_entropy] missing keys  : {missing}")
        print(f"[load_scalelsd_with_entropy] unexpected keys: {unexpected}")

    return model
