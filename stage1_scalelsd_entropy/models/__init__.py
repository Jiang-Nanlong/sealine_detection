"""
models — Stage-1 模型组件

包含：
  entropy_branch.py           — 轻量卷积分支，将 1ch entropy map 提取为 256ch 中层特征
  dpt_with_entropy.py         — DPTFieldModel 子类，在 path_1 处注入 entropy 特征
  scalelsd_with_entropy.py    — ScaleLSD 子类，backbone 替换为 DPTFieldModelWithEntropy
  smoke_test_injection.py     — path_1 注入冒烟测试
  fusion_module.py            — (旧) 特征级融合模块
  scalelsd_entropy_wrapper.py — (旧) 图像域 ResNet backbone + entropy 注入的完整模型
"""