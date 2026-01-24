# -*- coding: utf-8 -*-
"""
run_experiment4.py

Master script to run the complete Experiment 4: Cross-Dataset Generalization on SMD.

This script orchestrates:
  1. Data preparation (if needed): prepare_smd_testset.py
  2. Feature cache generation: make_fusion_cache_smd.py
  3. Evaluation: evaluate_fusion_cnn_smd.py
  4. Summary generation: summarize_smd_results.py
  5. Visualization: visualize_smd_predictions.py

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import subprocess
import sys
from pathlib import Path

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST4_DIR = PROJECT_ROOT / "test4"

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
SKIP_PREPARE = True    # True: 跳过数据准备步骤
SKIP_CACHE = True      # True: 跳过缓存生成步骤（如果已生成）
SKIP_VIS = False        # True: 跳过可视化步骤
N_VIS_SAMPLES = 20      # 可视化样本数量
# ============================


def run_script(script_path, description, extra_args=None):
    """Run a Python script and check for errors."""
    print("\n" + "=" * 60)
    print(f"[Step] {description}")
    print(f"[Script] {script_path}")
    print("=" * 60 + "\n")
    
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\n[Error] Script failed with return code {result.returncode}")
        return False
    return True


def check_prerequisites():
    """Check that required files exist."""
    issues = []
    
    # Check weights
    weights_path = PROJECT_ROOT / "weights" / "best_fusion_cnn_1024x576.pth"
    if not weights_path.exists():
        issues.append(f"Fusion CNN weights not found: {weights_path}")
    
    rghnet_path = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"
    if not rghnet_path.exists():
        issues.append(f"RGHNet weights not found: {rghnet_path}")
    
    dce_path = PROJECT_ROOT / "weights" / "Epoch99.pth"
    if not dce_path.exists():
        issues.append(f"DCE weights not found: {dce_path}")
    
    return issues


def main():
    print("=" * 60)
    print("Experiment 4: Cross-Dataset Generalization on SMD")
    print("=" * 60)
    print(f"\n[Config]")
    print(f"  SKIP_PREPARE = {SKIP_PREPARE}")
    print(f"  SKIP_CACHE   = {SKIP_CACHE}")
    print(f"  SKIP_VIS     = {SKIP_VIS}")
    print(f"  N_VIS_SAMPLES = {N_VIS_SAMPLES}")
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("\n[Error] Missing prerequisites:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease ensure all weights are in the weights/ directory.")
        sys.exit(1)
    
    # Step 1: Data preparation
    gt_csv = TEST4_DIR / "SMD_GroundTruth.csv"
    frames_dir = TEST4_DIR / "smd_frames"
    
    if SKIP_PREPARE:
        print("\n[Skip] Data preparation (SKIP_PREPARE=True)")
    elif gt_csv.exists() and frames_dir.exists():
        print("\n[Skip] Data already prepared (SMD_GroundTruth.csv and smd_frames/ exist)")
    else:
        script = TEST4_DIR / "prepare_smd_testset.py"
        if not run_script(script, "Prepare SMD test set"):
            sys.exit(1)
    
    # Step 2: Generate fusion cache
    cache_dir = TEST4_DIR / "FusionCache_SMD_1024x576" / "test"
    
    if SKIP_CACHE:
        print("\n[Skip] Cache generation (SKIP_CACHE=True)")
    elif cache_dir.exists() and any(cache_dir.glob("*.npy")):
        print(f"\n[Skip] Cache already exists: {cache_dir}")
        n_cached = len(list(cache_dir.glob("*.npy")))
        print(f"  -> {n_cached} cached files found")
    else:
        script = TEST4_DIR / "make_fusion_cache_smd.py"
        if not run_script(script, "Generate fusion cache for SMD"):
            sys.exit(1)
    
    # Step 3: Evaluate
    eval_csv = TEST4_DIR / "eval_smd_test_per_sample.csv"
    script = TEST4_DIR / "evaluate_fusion_cnn_smd.py"
    if not run_script(script, "Evaluate Fusion-CNN on SMD"):
        sys.exit(1)
    
    # Step 4: Generate summary
    script = TEST4_DIR / "summarize_smd_results.py"
    if not run_script(script, "Generate summary tables"):
        sys.exit(1)
    
    # Step 5: Visualization
    if SKIP_VIS:
        print("\n[Skip] Visualization (SKIP_VIS=True)")
    else:
        script = TEST4_DIR / "visualize_smd_predictions.py"
        
        # Random samples
        if not run_script(script, "Visualize random samples", 
                         ["--mode", "random", "--n_samples", str(N_VIS_SAMPLES)]):
            print("[Warning] Visualization failed, continuing...")
        
        # Worst cases
        if not run_script(script, "Visualize worst cases",
                         ["--mode", "worst", "--n_samples", str(min(10, N_VIS_SAMPLES))]):
            print("[Warning] Visualization failed, continuing...")
        
        # Best cases
        if not run_script(script, "Visualize best cases",
                         ["--mode", "best", "--n_samples", str(min(10, N_VIS_SAMPLES))]):
            print("[Warning] Visualization failed, continuing...")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Experiment 4 Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Per-sample results: {eval_csv}")
    print(f"  - Summary tables: {TEST4_DIR / 'experiment4_results'}")
    print(f"  - Visualizations: {TEST4_DIR / 'visualization'}")
    print("\nUse these results for Section 4.5 of your thesis (Cross-Dataset Generalization).")


if __name__ == "__main__":
    main()
