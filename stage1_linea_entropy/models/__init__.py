"""
stage1_linea_entropy/models/__init__.py — 模型构建入口

提供两种使用方式：

方式 1（推荐，直接调用）：
    from stage1_linea_entropy.models import build_linea_with_entropy
    model, postprocessors = build_linea_with_entropy(args)

方式 2（通过 LINEA 官方 registry）：
    import stage1_linea_entropy.models  # 触发注册
    # 之后在 config 中设置 modelname = 'LINEA_ENTROPY' 即可被 create(args, 'modelname') 找到

预训练权重兼容：
    原始 LINEA checkpoint 可通过 strict=False 加载到 LINEAWithEntropy。
    missing keys 仅来自 encoder.entropy_branch.* 和 encoder.alpha。
    其余 backbone.* / encoder.* / decoder.* 完全匹配。
    示例（注释，不要直接运行）：
        state = torch.load(ckpt_path, map_location='cpu')['model']
        missing, unexpected = model.load_state_dict(state, strict=False)
        # missing 应仅包含 encoder.entropy_branch.* 和 encoder.alpha
"""

from stage1_linea_entropy.models.entropy_branch import EntropyBranch
from stage1_linea_entropy.models.linea_with_entropy import (
    LINEAWithEntropy,
    build_linea_with_entropy,
)
from stage1_linea_entropy.models.encoder_with_entropy import (
    HybridEncoderWithEntropy,
    build_hybrid_encoder_with_entropy,
)

# ---- 注册到 LINEA 官方 registry，使 modelname='LINEA_ENTROPY' 可用 ----
from LINEA.models.registry import MODULE_BUILD_FUNCS

if 'LINEA_ENTROPY' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY')(build_linea_with_entropy)