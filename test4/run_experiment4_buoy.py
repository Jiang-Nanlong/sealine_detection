# -*- coding: utf-8 -*-
"""
Master script to run the complete Experiment 4: Cross-Dataset Generalization on Buoy.

This script orchestrates the entire pipeline:
  1. Data preparation: prepare_buoy_testset.py
  2. Feature cache generation: make_fusion_cache_buoy.py
  3. Evaluation: evaluate_fusion_cnn_buoy.py
  4. Summary generation: summarize_buoy_results.py

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import subprocess
import sys
from pathlib import Path


# ============================
# PyCharm 配置区 (在这里修改)
# ============================
SKIP_PREP = False       # 跳过数据准备（已有 buoy_frames 和 CSV）
SKIP_CACHE = False      # 跳过 FusionCache 生成（已有缓存）
SKIP_EVAL = False       # 跳过评估（只生成汇总报告）
SKIP_SUMMARY = False    # 跳过汇总报告生成
# ============================


# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST4_DIR = SCRIPT_DIR


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'=' * 60}")
    print(f"[Step] {description}")
    print(f"{'=' * 60}")
    print(f"Running: {script_path}")

    if not script_path.exists():
        print(f"[Error] Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"[Error] Script failed with return code {result.returncode}")
        return False

    return True


def main():
    """
    PyCharm: 修改文件顶部的配置区即可:
        SKIP_PREP = False       # 跳过数据准备
        SKIP_CACHE = False      # 跳过 FusionCache 生成
        SKIP_EVAL = False       # 跳过评估
        SKIP_SUMMARY = False    # 跳过汇总报告
    """
    print("=" * 60)
    print("Experiment 4: Cross-Dataset Generalization on Buoy")
    print("=" * 60)
    print(f"配置: SKIP_PREP={SKIP_PREP}, SKIP_CACHE={SKIP_CACHE}, SKIP_EVAL={SKIP_EVAL}, SKIP_SUMMARY={SKIP_SUMMARY}")

    # Check Buoy directory exists
    buoy_dir = PROJECT_ROOT / "Buoy"
    if not buoy_dir.exists():
        print(f"[Error] Buoy directory not found: {buoy_dir}")
        print("  Please ensure the Buoy dataset is placed in the correct location.")
        return 1

    # Step 1: Data preparation
    gt_csv = TEST4_DIR / "Buoy_GroundTruth.csv"
    frames_dir = TEST4_DIR / "buoy_frames"

    if SKIP_PREP and gt_csv.exists() and frames_dir.exists():
        print("\n[Skip] Data already prepared (Buoy_GroundTruth.csv and buoy_frames/ exist)")
    elif SKIP_PREP:
        print("\n[Skip] Data preparation (SKIP_PREP=True)")
    else:
        script = TEST4_DIR / "prepare_buoy_testset.py"
        if not run_script(script, "Prepare Buoy test set"):
            return 1

    # Step 2: Feature cache generation
    cache_dir = TEST4_DIR / "FusionCache_Buoy" / "test"

    if SKIP_CACHE and cache_dir.exists() and any(cache_dir.glob("*.npy")):
        print(f"\n[Skip] FusionCache already exists: {cache_dir}")
        print(f"  Found {len(list(cache_dir.glob('*.npy')))} cached files")
    elif SKIP_CACHE:
        print("\n[Skip] FusionCache generation (SKIP_CACHE=True)")
    else:
        # Check weights exist
        weights_dir = PROJECT_ROOT / "weights"
        required_weights = [
            "rghnet_best_c2.pth",
            "Epoch99.pth",
            "best_fusion_cnn_1024x576.pth",
        ]
        missing = [w for w in required_weights if not (weights_dir / w).exists()]
        if missing:
            print(f"\n[Error] Missing weight files in {weights_dir}:")
            for w in missing:
                print(f"  - {w}")
            return 1

        script = TEST4_DIR / "make_fusion_cache_buoy.py"
        if not run_script(script, "Generate fusion cache for Buoy"):
            return 1

    # Step 3: Evaluation
    eval_csv = TEST4_DIR / "eval_buoy_test_per_sample.csv"

    if SKIP_EVAL and eval_csv.exists():
        print(f"\n[Skip] Evaluation CSV already exists: {eval_csv}")
    elif SKIP_EVAL:
        print("\n[Skip] Evaluation (SKIP_EVAL=True)")
    else:
        script = TEST4_DIR / "evaluate_fusion_cnn_buoy.py"
        if not run_script(script, "Evaluate Fusion-CNN on Buoy"):
            return 1

    # Step 4: Summary generation
    if SKIP_SUMMARY:
        print("\n[Skip] Summary generation (SKIP_SUMMARY=True)")
    else:
        script = TEST4_DIR / "summarize_buoy_results.py"
        if not run_script(script, "Generate summary report"):
            return 1

    print("\n" + "=" * 60)
    print("Experiment 4 (Buoy) COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {TEST4_DIR / 'experiment4_results'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
