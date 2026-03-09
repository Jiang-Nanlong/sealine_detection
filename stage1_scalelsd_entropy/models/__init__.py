"""
models — Stage-1 模型组件

包含：
  entropy_branch.py          — 轻量卷积分支，将 1ch entropy map 提取为 256ch 中层特征
  fusion_module.py           — 特征级融合模块，将 entropy 特征注入 backbone 中层
  scalelsd_entropy_wrapper.py — 图像域 ResNet backbone + entropy 注入的完整模型
"""