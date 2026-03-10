# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .linea import build_linea

# ---- 注册 LINEA_ENTROPY 到本地 registry 实例 ----
# main.py 通过 "from models.registry" 获取 MODULE_BUILD_FUNCS，
# 必须用同一个实例注册，不能经由 LINEA.models.registry（Python 视为不同模块）
from .registry import MODULE_BUILD_FUNCS
from stage1_linea_entropy.models.linea_with_entropy import build_linea_with_entropy

if 'LINEA_ENTROPY' not in MODULE_BUILD_FUNCS._module_dict:
    MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEA_ENTROPY')(build_linea_with_entropy)

