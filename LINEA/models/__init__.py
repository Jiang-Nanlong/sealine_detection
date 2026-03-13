# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .linea import build_linea

# ---- 延迟注册 LINEA_ENTROPY / LINEA_ENTROPY_A / LINEA_ENTROPY_B，避免循环导入 ----
from .registry import MODULE_BUILD_FUNCS

def _build_linea_with_entropy_lazy(args):
    from stage1_linea_entropy.models.linea_with_entropy import build_linea_with_entropy
    return build_linea_with_entropy(args)

def _build_linea_with_entropy_a_lazy(args):
    from stage1_linea_entropy.models.linea_with_entropy import build_linea_with_entropy_a
    return build_linea_with_entropy_a(args)

def _build_linea_with_entropy_b_lazy(args):
    from stage1_linea_entropy.models.linea_with_entropy import build_linea_with_entropy_b
    return build_linea_with_entropy_b(args)

if 'LINEA_ENTROPY' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY')(_build_linea_with_entropy_lazy)

if 'LINEA_ENTROPY_A' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY_A')(_build_linea_with_entropy_a_lazy)

if 'LINEA_ENTROPY_B' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY_B')(_build_linea_with_entropy_b_lazy)

