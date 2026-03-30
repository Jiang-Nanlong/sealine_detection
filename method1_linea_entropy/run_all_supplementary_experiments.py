"""
run_all_supplementary_experiments.py — 一键运行全部补充实验

在 PyCharm 中右键运行本文件即可依次执行：
  P0: 排序质量分析（需要 GPU 推理）
  P1: 模型开销分析（需要 GPU 推理）
  P2: 收敛曲线绘制（仅读取 log 文件，无需 GPU）

所有输出保存在 method1_linea_entropy/ 下对应子目录。
"""

import os
import sys

# 确保工作目录为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print("=" * 70)
print("  工作目录:", os.getcwd())
print("=" * 70)


# ============================================================
# P2: 收敛曲线（最快，先跑）
# ============================================================
print("\n\n" + "#" * 70)
print("#  [1/3] P2: 收敛曲线绘制")
print("#" * 70)
from method1_linea_entropy.plot_convergence_curves import main as run_convergence
run_convergence()


# ============================================================
# P1: 模型开销分析
# ============================================================
print("\n\n" + "#" * 70)
print("#  [2/3] P1: 模型开销分析")
print("#" * 70)
from method1_linea_entropy.analyze_model_overhead import main as run_overhead
run_overhead()


# ============================================================
# P0: 排序质量分析（最慢，最后跑）
# ============================================================
print("\n\n" + "#" * 70)
print("#  [3/3] P0: 排序质量分析")
print("#" * 70)
from method1_linea_entropy.analyze_ranking_quality import main as run_ranking
run_ranking()


print("\n\n" + "=" * 70)
print("  全部补充实验完成！")
print("  输出目录：")
print("    - method1_linea_entropy/convergence_analysis/")
print("    - method1_linea_entropy/overhead_analysis/")
print("    - method1_linea_entropy/ranking_analysis/")
print("=" * 70)
