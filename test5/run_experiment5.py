# -*- coding: utf-8 -*-
"""
run_experiment5.py

Master script to run the complete Experiment 5: Degradation Robustness.

This script orchestrates:
  1. Generate degraded images: generate_degraded_images.py
  2. Build FusionCache: make_fusion_cache_degraded.py
  3. Evaluate: evaluate_degraded.py
  4. Generate summary: summarize_degraded_results.py
  5. Visualize: visualize_degraded.py

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
TEST5_DIR = PROJECT_ROOT / "test5"

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
SKIP_GENERATE = False   # True: 跳过生成退化图像步骤
SKIP_CACHE = False      # True: 跳过缓存生成步骤
SKIP_VIS = False        # True: 跳过可视化步骤
# ============================


def run_script(script_path, description):
    """Run a Python script and check for errors."""
    print("\n" + "=" * 60)
    print(f"[Step] {description}")
    print(f"[Script] {script_path}")
    print("=" * 60 + "\n")
    
    cmd = [sys.executable, str(script_path)]
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
    
    # Check MU-SID data
    musid_dir = PROJECT_ROOT / "Hashmani's Dataset" / "clear"
    if not musid_dir.exists():
        issues.append(f"MU-SID images not found: {musid_dir}")
    
    splits_dir = PROJECT_ROOT / "splits_musid"
    if not splits_dir.exists():
        issues.append(f"Splits directory not found: {splits_dir}")
    
    return issues


def main():
    print("=" * 60)
    print("Experiment 5: Degradation Robustness")
    print("=" * 60)
    print(f"\n[Config]")
    print(f"  SKIP_GENERATE = {SKIP_GENERATE}")
    print(f"  SKIP_CACHE    = {SKIP_CACHE}")
    print(f"  SKIP_VIS      = {SKIP_VIS}")
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("\n[Error] Missing prerequisites:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease ensure all required files are in place.")
        sys.exit(1)
    
    # Step 1: Generate degraded images
    degraded_dir = TEST5_DIR / "degraded_images"
    
    if SKIP_GENERATE:
        print("\n[Skip] Generate degraded images (SKIP_GENERATE=True)")
    elif degraded_dir.exists() and any(degraded_dir.iterdir()):
        n_types = len([d for d in degraded_dir.iterdir() if d.is_dir()])
        print(f"\n[Skip] Degraded images already exist: {n_types} types found")
    else:
        script = TEST5_DIR / "generate_degraded_images.py"
        if not run_script(script, "Generate degraded images"):
            sys.exit(1)
    
    # Step 2: Build FusionCache
    cache_dir = TEST5_DIR / "FusionCache_Degraded"
    
    if SKIP_CACHE:
        print("\n[Skip] Cache generation (SKIP_CACHE=True)")
    elif cache_dir.exists() and any(cache_dir.iterdir()):
        n_types = len([d for d in cache_dir.iterdir() if d.is_dir()])
        print(f"\n[Skip] Cache already exists: {n_types} types found")
    else:
        script = TEST5_DIR / "make_fusion_cache_degraded.py"
        if not run_script(script, "Build FusionCache for degraded images"):
            sys.exit(1)
    
    # Step 3: Evaluate
    script = TEST5_DIR / "evaluate_degraded.py"
    if not run_script(script, "Evaluate degraded images"):
        sys.exit(1)
    
    # Step 4: Generate summary
    script = TEST5_DIR / "summarize_degraded_results.py"
    if not run_script(script, "Generate summary tables"):
        sys.exit(1)
    
    # Step 5: Visualization
    if SKIP_VIS:
        print("\n[Skip] Visualization (SKIP_VIS=True)")
    else:
        script = TEST5_DIR / "visualize_degraded.py"
        if not run_script(script, "Visualize degraded predictions"):
            print("[Warning] Visualization failed, continuing...")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Experiment 5 Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Degraded images: {degraded_dir}")
    print(f"  - Evaluation results: {TEST5_DIR / 'eval_results'}")
    print(f"  - Summary tables: {TEST5_DIR / 'experiment5_results'}")
    print(f"  - Visualizations: {TEST5_DIR / 'visualization'}")
    print("\nUse these results for Section 4.X of your thesis (Robustness Analysis).")


if __name__ == "__main__":
    main()
