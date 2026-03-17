"""
stage1_linea_entropy/models/__init__.py — 模型构建入口

提供三种模型:

  LINEA_ENTROPY   — 原始单层加性熵注入 (build_linea_with_entropy)
  LINEA_ENTROPY_A — 多层门控 FiLM 熵注入 (build_linea_with_entropy_a)
  LINEA_ENTROPY_B — detector 内置 horizon-aware scoring head (build_linea_with_entropy_b)
"""

from stage1_linea_entropy.models.entropy_branch import (
    EntropyBranch,
    MultiScaleEntropyBranch,
    GatedFiLMLayer,
)
from stage1_linea_entropy.models.linea_with_entropy import (
    LINEAWithEntropy,
    LINEAWithEntropyA,
    LINEAWithEntropyB,
    LINEAWithEntropyBEnhanced,
    build_linea_with_entropy,
    build_linea_with_entropy_a,
    build_linea_with_entropy_b,
    build_linea_with_entropy_b_enhanced,
)
from stage1_linea_entropy.models.encoder_with_entropy import (
    HybridEncoderWithEntropy,
    HybridEncoderWithEntropyA,
    HybridEncoderWithEntropyEnhanced,
    build_hybrid_encoder_with_entropy,
    build_hybrid_encoder_with_entropy_a,
    build_hybrid_encoder_with_entropy_enhanced,
)
from stage1_linea_entropy.models.horizon_scoring_head import (
    HorizonScoringHead,
)

# ---- 注册到 LINEA 官方 registry ----
from models.registry import MODULE_BUILD_FUNCS

if 'LINEA_ENTROPY' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY')(build_linea_with_entropy)

if 'LINEA_ENTROPY_A' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY_A')(build_linea_with_entropy_a)

if 'LINEA_ENTROPY_B' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY_B')(build_linea_with_entropy_b)

if 'LINEA_ENTROPY_B_ENHANCED' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY_B_ENHANCED')(build_linea_with_entropy_b_enhanced)