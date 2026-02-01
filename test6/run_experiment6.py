# -*- coding: utf-8 -*-
"""
Experiment 6: In-Domain Training on SMD and Buoy Datasets.

目的: 分别在 SMD 和 Buoy 数据集上训练，测试在各自数据集上的效果

执行流程:
  1. 准备 SMD 数据集划分 (prepare_smd_trainset.py)
  2. 准备 Buoy 数据集划分 (prepare_buoy_trainset.py)
  3. 生成 SMD 训练缓存 (make_fusion_cache_smd_train.py)
  4. 生成 Buoy 训练缓存 (make_fusion_cache_buoy_train.py)
  5. 在 SMD 上训练 (train_fusion_cnn_smd.py)
  6. 在 Buoy 上训练 (train_fusion_cnn_buoy.py)
  7. 评估 SMD 模型 (evaluate_smd_indomain.py)
  8. 评估 Buoy 模型 (evaluate_buoy_indomain.py)

PyCharm: 直接运行此文件

可选: 修改下面的 RUN_* 变量来控制执行哪些步骤
"""

import sys
import subprocess
from pathlib import Path

# ============================
# PyCharm 配置区 - 控制执行步骤
# ============================
# SMD 相关
RUN_PREPARE_SMD = True        # 1. 准备 SMD 数据集划分
RUN_CACHE_SMD = True          # 2. 生成 SMD 缓存
RUN_TRAIN_SMD = True          # 3. 训练 SMD 模型
RUN_EVAL_SMD = True           # 4. 评估 SMD 模型

# Buoy 相关
RUN_PREPARE_BUOY = True       # 5. 准备 Buoy 数据集划分
RUN_CACHE_BUOY = True         # 6. 生成 Buoy 缓存
RUN_TRAIN_BUOY = True         # 7. 训练 Buoy 模型
RUN_EVAL_BUOY = True          # 8. 评估 Buoy 模型
# ============================

TEST6_DIR = Path(__file__).resolve().parent

def run_script(name: str, script_path: Path):
    """Run a Python script and print status."""
    print("\n" + "=" * 70)
    print(f"[Running] {name}")
    print(f"[Script]  {script_path}")
    print("=" * 70)

    if not script_path.exists():
        print(f"[Error] Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent)
    )
    
    if result.returncode != 0:
        print(f"[Error] {name} failed with code {result.returncode}")
        return False
    
    print(f"[Done] {name}")
    return True


def main():
    print("=" * 70)
    print("Experiment 6: In-Domain Training")
    print("=" * 70)

    scripts = []

    # SMD pipeline
    if RUN_PREPARE_SMD:
        scripts.append(("1. Prepare SMD trainset", TEST6_DIR / "prepare_smd_trainset.py"))
    if RUN_CACHE_SMD:
        scripts.append(("2. Generate SMD cache", TEST6_DIR / "make_fusion_cache_smd_train.py"))
    if RUN_TRAIN_SMD:
        scripts.append(("3. Train on SMD", TEST6_DIR / "train_fusion_cnn_smd.py"))
    if RUN_EVAL_SMD:
        scripts.append(("4. Evaluate SMD model", TEST6_DIR / "evaluate_smd_indomain.py"))

    # Buoy pipeline
    if RUN_PREPARE_BUOY:
        scripts.append(("5. Prepare Buoy trainset", TEST6_DIR / "prepare_buoy_trainset.py"))
    if RUN_CACHE_BUOY:
        scripts.append(("6. Generate Buoy cache", TEST6_DIR / "make_fusion_cache_buoy_train.py"))
    if RUN_TRAIN_BUOY:
        scripts.append(("7. Train on Buoy", TEST6_DIR / "train_fusion_cnn_buoy.py"))
    if RUN_EVAL_BUOY:
        scripts.append(("8. Evaluate Buoy model", TEST6_DIR / "evaluate_buoy_indomain.py"))

    if not scripts:
        print("[Info] No steps selected. Modify RUN_* variables to enable steps.")
        return 0

    print(f"\n[Plan] Will run {len(scripts)} steps:")
    for name, _ in scripts:
        print(f"  - {name}")

    failed = []
    for name, path in scripts:
        if not run_script(name, path):
            failed.append(name)
            print(f"\n[Warning] {name} failed, continuing...")

    # Summary
    print("\n" + "=" * 70)
    print("Experiment 6 Complete")
    print("=" * 70)
    
    if failed:
        print(f"[Warning] {len(failed)} step(s) failed:")
        for f in failed:
            print(f"  - {f}")
    else:
        print("[Success] All steps completed!")

    print("\n[Outputs]")
    outputs = [
        TEST6_DIR / "weights" / "best_fusion_cnn_smd.pth",
        TEST6_DIR / "weights" / "best_fusion_cnn_buoy.pth",
        TEST6_DIR / "eval_smd_indomain.csv",
        TEST6_DIR / "eval_buoy_indomain.csv",
    ]
    for o in outputs:
        status = "✓" if o.exists() else "✗"
        print(f"  {status} {o}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
