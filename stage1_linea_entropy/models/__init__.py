"""
stage1_linea_entropy/models/__init__.py — 模型构建入口

提供三种模型:

  LINEA_ENTROPY   — 原始单层加性熵注入 (build_linea_with_entropy)
  LINEA_ENTROPY_A — 多层门控 FiLM 熵注入 (build_linea_with_entropy_a)

使用方式:
  方式 1 — 直接调用:
      from stage1_linea_entropy.models import build_linea_with_entropy
      from stage1_linea_entropy.models import build_linea_with_entropy_a

  方式 2 — 通过 LINEA 官方 registry:
      import stage1_linea_entropy.models  # 触发注册
      # config 中 modelname = 'LINEA_ENTROPY' 或 'LINEA_ENTROPY_A'
"""

from stage1_linea_entropy.models.entropy_branch import (
    EntropyBranch,
    MultiScaleEntropyBranch,
    GatedFiLMLayer,
)
from stage1_linea_entropy.models.linea_with_entropy import (
    LINEAWithEntropy,
    LINEAWithEntropyA,
    build_linea_with_entropy,
    build_linea_with_entropy_a,
)
from stage1_linea_entropy.models.encoder_with_entropy import (
    HybridEncoderWithEntropy,
    HybridEncoderWithEntropyA,
    build_hybrid_encoder_with_entropy,
    build_hybrid_encoder_with_entropy_a,
)

# ---- 注册到 LINEA 官方 registry ----
from LINEA.models.registry import MODULE_BUILD_FUNCS

if 'LINEA_ENTROPY' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY')(build_linea_with_entropy)

if 'LINEA_ENTROPY_A' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY_A')(build_linea_with_entropy_a)