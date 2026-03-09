"""
stage1_scalelsd_entropy — 第一阶段最小可行版本

功能：MU-SID 数据集 + 局部熵图 + 特征级注入

子包结构：
  data/     — 数据加载与熵图预计算
  models/   — entropy branch、fusion module、ScaleLSD wrapper
  utils/    — CSV 检查、熵图可视化
  configs/  — YAML 配置文件

入口脚本：
  train_stage1.py — 训练
  test_stage1.py  — 测试
"""
