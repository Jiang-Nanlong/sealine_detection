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

注意：运行前请确�?generate_degraded_images.py、make_fusion_cache_degraded.py�?
      evaluate_degraded.py、summarize_degraded_results.py 中的 DATASET 变量
      与本文件中的 DATASET 一致！
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
test1_DIR = PROJECT_ROOT / "test1"
TEST4_DIR = PROJECT_ROOT / "test4"

# ============================
# PyCharm 配置�?(在这里修�?
# ============================
SKIP_GENERATE = False   # True: 跳过生成退化图像步�?
SKIP_CACHE = False      # True: 跳过缓存生成步骤
SKIP_VIS = False        # True: 跳过可视化步�?
# 选择数据�? "musid", "smd", "buoy"
DATASET = "musid"
GLOBAL_SEED = 42        # 全局种子（传递给 generate_degraded_images.py�?
# ============================

# 命令行参数覆�?
if "--dataset" in sys.argv:
    _idx = sys.argv.index("--dataset")
    if _idx + 1 < len(sys.argv):
        DATASET = sys.argv[_idx + 1]

if "--seed" in sys.argv:
    _idx = sys.argv.index("--seed")
    if _idx + 1 < len(sys.argv):
        GLOBAL_SEED = int(sys.argv[_idx + 1])

# 数据集配�?
DATASET_CONFIGS = {
    "musid": {
        "cnn_weights": PROJECT_ROOT / "weights" / "best_fusion_cnn_1024x576.pth",
        "unet_weights": PROJECT_ROOT / "weights" / "rghnet_best_c2.pth",
        "img_dir": PROJECT_ROOT / "Hashmani's Dataset" / "MU-SID",
        "splits_dir": PROJECT_ROOT / "splits_musid",
        "degraded_dir": TEST5_DIR / "degraded_images",
        "cache_dir": TEST5_DIR / "FusionCache_Degraded",
        "eval_dir": TEST5_DIR / "eval_results",
        "results_dir": TEST5_DIR / "experiment5_results",
    },
    "smd": {
        "cnn_weights": test1_DIR / "weights" / "best_fusion_cnn_smd.pth",
        "unet_weights": test1_DIR / "weights_smd" / "smd_rghnet_best_seg_c2.pth",
        "img_dir": TEST4_DIR / "manual_review" / "kept_frames",
        "splits_dir": test1_DIR / "splits_smd",
        "degraded_dir": TEST5_DIR / "degraded_images_smd",
        "cache_dir": TEST5_DIR / "FusionCache_Degraded_SMD",
        "eval_dir": TEST5_DIR / "eval_results_smd",
        "results_dir": TEST5_DIR / "experiment5_results_smd",
    },
    "buoy": {
        "cnn_weights": test1_DIR / "weights" / "best_fusion_cnn_buoy.pth",
        "unet_weights": test1_DIR / "weights_buoy" / "buoy_rghnet_best_seg_c2.pth",
        "img_dir": TEST4_DIR / "buoy_frames",
        "splits_dir": test1_DIR / "splits_buoy",
        "degraded_dir": TEST5_DIR / "degraded_images_buoy",
        "cache_dir": TEST5_DIR / "FusionCache_Degraded_Buoy",
        "eval_dir": TEST5_DIR / "eval_results_buoy",
        "results_dir": TEST5_DIR / "experiment5_results_buoy",
    },
}


def run_script(script_path, description, pass_dataset=True, pass_seed=False):
    """Run a Python script and check for errors."""
    print("\n" + "=" * 60)
    print(f"[Step] {description}")
    print(f"[Script] {script_path}")
    print("=" * 60 + "\n")
    
    cmd = [sys.executable, str(script_path)]
    if pass_dataset:
        cmd.extend(["--dataset", DATASET])
    if pass_seed:
        cmd.extend(["--seed", str(GLOBAL_SEED)])
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\n[Error] Script failed with return code {result.returncode}")
        return False
    return True


def check_prerequisites():
    """Check that required files exist for the selected dataset."""
    cfg = DATASET_CONFIGS[DATASET]
    issues = []
    
    # Check CNN weights
    if not cfg["cnn_weights"].exists():
        issues.append(f"Fusion CNN weights not found: {cfg['cnn_weights']}")
    
    # Check UNet weights
    if not cfg["unet_weights"].exists():
        issues.append(f"RGHNet weights not found: {cfg['unet_weights']}")
    
    # Check DCE weights
    dce_path = PROJECT_ROOT / "weights" / "Epoch99.pth"
    if not dce_path.exists():
        issues.append(f"DCE weights not found: {dce_path}")
    
    # Check image directory
    if not cfg["img_dir"].exists():
        issues.append(f"Image directory not found: {cfg['img_dir']}")
    
    # Check splits directory
    if not cfg["splits_dir"].exists():
        issues.append(f"Splits directory not found: {cfg['splits_dir']}")
    
    return issues


def main():
    cfg = DATASET_CONFIGS[DATASET]
    
    print("=" * 60)
    print("Experiment 5: Degradation Robustness")
    print(f"Dataset: {DATASET.upper()}")
    print(f"[Seed] global_seed={GLOBAL_SEED}")
    print("=" * 60)
    print(f"\n[Config]")
    print(f"  DATASET       = {DATASET}")
    print(f"  GLOBAL_SEED   = {GLOBAL_SEED}")
    print(f"  SKIP_GENERATE = {SKIP_GENERATE}")
    print(f"  SKIP_CACHE    = {SKIP_CACHE}")
    print(f"  SKIP_VIS      = {SKIP_VIS}")
    print(f"\n[Info] 将自动传�?--dataset {DATASET} --seed {GLOBAL_SEED} �?generate_degraded_images.py")
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("\n[Error] Missing prerequisites:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease ensure all required files are in place.")
        sys.exit(1)
    
    # Step 1: Generate degraded images
    degraded_dir = cfg["degraded_dir"]
    
    if SKIP_GENERATE:
        print("\n[Skip] Generate degraded images (SKIP_GENERATE=True)")
    elif degraded_dir.exists() and any(degraded_dir.iterdir()):
        n_types = len([d for d in degraded_dir.iterdir() if d.is_dir()])
        print(f"\n[Skip] Degraded images already exist: {n_types} types found in {degraded_dir}")
    else:
        script = TEST5_DIR / "generate_degraded_images.py"
        if not run_script(script, f"Generate degraded images for {DATASET.upper()}", pass_seed=True):
            sys.exit(1)
    
    # Step 2: Build FusionCache
    cache_dir = cfg["cache_dir"]
    
    if SKIP_CACHE:
        print("\n[Skip] Cache generation (SKIP_CACHE=True)")
    elif cache_dir.exists() and any(cache_dir.iterdir()):
        n_types = len([d for d in cache_dir.iterdir() if d.is_dir()])
        print(f"\n[Skip] Cache already exists: {n_types} types found in {cache_dir}")
    else:
        script = TEST5_DIR / "make_fusion_cache_degraded.py"
        if not run_script(script, f"Build FusionCache for degraded images ({DATASET.upper()})"):
            sys.exit(1)
    
    # Step 3: Evaluate
    script = TEST5_DIR / "evaluate_degraded.py"
    if not run_script(script, f"Evaluate degraded images ({DATASET.upper()})"):
        sys.exit(1)
    
    # Step 4: Generate summary
    script = TEST5_DIR / "summarize_degraded_results.py"
    if not run_script(script, f"Generate summary tables ({DATASET.upper()})"):
        sys.exit(1)
    
    # Step 5: Visualization
    if SKIP_VIS:
        print("\n[Skip] Visualization (SKIP_VIS=True)")
    else:
        script = TEST5_DIR / "visualize_degraded.py"
        if not run_script(script, "Visualize degraded predictions", pass_dataset=True):
            print("[Warning] Visualization failed, continuing...")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"Experiment 5 Complete! ({DATASET.upper()})")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Degraded images: {degraded_dir}")
    print(f"  - Evaluation results: {cfg['eval_dir']}")
    print(f"  - Summary tables: {cfg['results_dir']}")
    print(f"\n如需运行其他数据集，请修改本文件及各子脚本中�?DATASET 变量�?)


if __name__ == "__main__":
    main()
