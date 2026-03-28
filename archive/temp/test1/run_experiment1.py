# -*- coding: utf-8 -*-
"""
Experiment 1: In-Domain Training on SMD and Buoy Datasets.

目的: 分别�?SMD �?Buoy 数据集上训练 UNet + CNN，测试在各自数据集上的效�?

完整执行流程:
  SMD:
    1. 准备 SMD 数据集划�?(prepare_smd_trainset.py)
    2. 训练 SMD UNet - 5阶段自动运行: A→B→C1→B2→C2
    3. 生成 SMD 训练缓存 (make_fusion_cache_smd_train.py)
    4. 训练 SMD CNN (train_fusion_cnn_smd.py)
    5. 评估 SMD 模型 (evaluate_smd_full.py) - 完整指标�?evaluate_full_pipeline.py 对齐
  
  Buoy:
    6. 准备 Buoy 数据集划�?(prepare_buoy_trainset.py)
    7. 训练 Buoy UNet - 5阶段自动运行: A→B→C1→B2→C2
    8. 生成 Buoy 训练缓存 (make_fusion_cache_buoy_train.py)
    9. 训练 Buoy CNN (train_fusion_cnn_buoy.py)
    10. 评估 Buoy 模型 (evaluate_buoy_full.py) - 完整指标�?evaluate_full_pipeline.py 对齐

PyCharm: 直接运行此文�?
"""

import sys
import subprocess
from pathlib import Path

# ============================
# PyCharm 配置�?- 控制执行步骤
# ============================
# SMD 完整流程
RUN_SMD_PIPELINE = True       # 运行完整 SMD 流程

# Buoy 完整流程
RUN_BUOY_PIPELINE = False     # 运行完整 Buoy 流程

# 细粒度控制（仅当上面对应�?PIPELINE �?True 时生效）
SKIP_PREPARE = False           # 跳过数据准备（如果已运行过）
SKIP_UNET = False             # 跳过 UNet 训练（如果已训练完成�?
SKIP_CACHE = False            # 跳过缓存生成
SKIP_CNN = False              # 跳过 CNN 训练
SKIP_EVAL = False             # 跳过评估
# ============================

test1_DIR = Path(__file__).resolve().parent
UNET_STAGES = ["A", "B", "C1", "B2", "C2"]


def run_script(name: str, script_path: Path, extra_args: list = None):
    """Run a Python script and print status."""
    print("\n" + "=" * 70)
    print(f"[Running] {name}")
    print(f"[Script]  {script_path}")
    print("=" * 70)

    if not script_path.exists():
        print(f"[Error] Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=str(script_path.parent))
    
    if result.returncode != 0:
        print(f"[Error] {name} failed with code {result.returncode}")
        return False
    
    print(f"[Done] {name}")
    return True


def run_unet_all_stages(dataset: str):
    """Run UNet training for all 5 stages."""
    script_name = f"train_unet_{dataset}.py"
    script_path = test1_DIR / script_name
    
    print("\n" + "=" * 70)
    print(f"[UNet Training] {dataset.upper()} - 5 Stages")
    print("=" * 70)
    
    for stage in UNET_STAGES:
        name = f"UNet {dataset.upper()} Stage {stage}"
        print(f"\n>>> Starting {name} <<<")
        
        if not run_script(name, script_path, ["--stage", stage]):
            print(f"[Error] {name} failed!")
            return False
    
    print(f"\n[Done] UNet {dataset.upper()} all 5 stages completed!")
    return True


def run_pipeline(dataset: str):
    """Run complete pipeline for a dataset (SMD or Buoy)."""
    print("\n" + "#" * 70)
    print(f"# Pipeline: {dataset.upper()}")
    print("#" * 70)
    
    failed = []
    
    # 1. Prepare dataset split
    if not SKIP_PREPARE:
        if not run_script(
            f"Prepare {dataset.upper()} trainset",
            test1_DIR / f"prepare_{dataset}_trainset.py"
        ):
            failed.append(f"Prepare {dataset}")
    
    # 2. Train UNet (5 stages)
    if not SKIP_UNET:
        if not run_unet_all_stages(dataset):
            failed.append(f"UNet {dataset}")
            print(f"[Warning] UNet training failed, but continuing...")
    
    # 3. Generate fusion cache
    if not SKIP_CACHE:
        if not run_script(
            f"Generate {dataset.upper()} cache",
            test1_DIR / f"make_fusion_cache_{dataset}_train.py"
        ):
            failed.append(f"Cache {dataset}")
    
    # 4. Train CNN
    if not SKIP_CNN:
        if not run_script(
            f"Train {dataset.upper()} CNN",
            test1_DIR / f"train_fusion_cnn_{dataset}.py"
        ):
            failed.append(f"CNN {dataset}")
    
    # 5. Evaluate (full metrics aligned with evaluate_full_pipeline.py)
    if not SKIP_EVAL:
        if not run_script(
            f"Evaluate {dataset.upper()} model (full metrics)",
            test1_DIR / f"evaluate_{dataset}_full.py"
        ):
            failed.append(f"Eval {dataset}")
    
    return failed


def main():
    print("=" * 70)
    print("Experiment 1: In-Domain Training (UNet + CNN)")
    print("=" * 70)
    
    all_failed = []
    
    # SMD pipeline
    if RUN_SMD_PIPELINE:
        failed = run_pipeline("smd")
        all_failed.extend(failed)
    
    # Buoy pipeline
    if RUN_BUOY_PIPELINE:
        failed = run_pipeline("buoy")
        all_failed.extend(failed)
    
    if not RUN_SMD_PIPELINE and not RUN_BUOY_PIPELINE:
        print("[Info] No pipeline selected. Set RUN_SMD_PIPELINE or RUN_BUOY_PIPELINE to True.")
        return 0

    # Summary
    print("\n" + "=" * 70)
    print("Experiment 1 Complete")
    print("=" * 70)
    
    if all_failed:
        print(f"[Warning] {len(all_failed)} step(s) failed:")
        for f in all_failed:
            print(f"  - {f}")
    else:
        print("[Success] All steps completed!")

    print("\n[Outputs]")
    outputs = [
        # SMD outputs
        test1_DIR / "weights_smd" / "smd_rghnet_best_seg_c2.pth",
        test1_DIR / "weights" / "best_fusion_cnn_smd.pth",
        test1_DIR / "eval_smd_full_outputs" / "full_eval_smd_test.csv",
        test1_DIR / "eval_smd_full_outputs" / "eval_summary_smd.csv",
        # Buoy outputs
        test1_DIR / "weights_buoy" / "buoy_rghnet_best_seg_c2.pth",
        test1_DIR / "weights" / "best_fusion_cnn_buoy.pth",
        test1_DIR / "eval_buoy_full_outputs" / "full_eval_buoy_test.csv",
        test1_DIR / "eval_buoy_full_outputs" / "eval_summary_buoy.csv",
    ]
    for o in outputs:
        status = "�? if o.exists() else "�?
        print(f"  {status} {o}")

    return 0 if not all_failed else 1


if __name__ == "__main__":
    sys.exit(main())
