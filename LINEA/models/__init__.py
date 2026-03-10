# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .linea import build_linea

# ---- 注册 LINEA_ENTROPY 到 MODULE_BUILD_FUNCS ----
# 导入触发 stage1_linea_entropy.models.__init__.py 中的注册逻辑
import stage1_linea_entropy.models  # noqa: F401

