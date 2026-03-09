"""
data — 数据加载与熵图预计算模块

包含：
  entropy_precompute.py      — MU-SID 图像局部熵离线预计算（blue/gray 模式）
  musid_dataset.py           — MU-SID Dataset（轻量版，返回 image + entropy + 原始标注）
  musid_entropy_dataset.py   — MU-SID Dataset（完整版，含 resize/endpoint 缩放）
  entropy_dataset_wrapper.py — 通用 Dataset 包装器，为已有 Dataset 额外加载 entropy map
"""