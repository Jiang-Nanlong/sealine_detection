# -*- coding: utf-8 -*-
"""
Prepare Buoy dataset for Experiment 6: In-Domain Training.

This script creates train/val/test splits for Buoy dataset.
Split ratio: 70% train, 15% val, 15% test

Inputs:
  - test4/Buoy_GroundTruth.csv   (from prepare_buoy_testset.py)
  - test4/buoy_frames/           (from prepare_buoy_testset.py)

Outputs:
  - test6/splits_buoy/train_indices.npy
  - test6/splits_buoy/val_indices.npy
  - test6/splits_buoy/test_indices.npy

Usage:
  python test6/prepare_buoy_trainset.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================
# PyCharm 配置区
# ============================
RANDOM_SEED = 2026
# Buoy 数据集较小(~996张)，使用 80/10/10 划分以保证测试集统计可靠性
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
# ============================

# Paths
TEST4_DIR = PROJECT_ROOT / "test4"
TEST6_DIR = PROJECT_ROOT / "test6"

BUOY_CSV = TEST4_DIR / "Buoy_GroundTruth.csv"
BUOY_FRAMES = TEST4_DIR / "buoy_frames"
SPLIT_DIR = TEST6_DIR / "splits_buoy"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    print("=" * 60)
    print("Prepare Buoy Train/Val/Test Splits for Experiment 6")
    print("=" * 60)

    # Check if Buoy data exists
    if not BUOY_CSV.exists():
        print(f"[Error] Buoy CSV not found: {BUOY_CSV}")
        print("  Please run test4/prepare_buoy_testset.py first.")
        return 1

    if not BUOY_FRAMES.exists():
        print(f"[Error] Buoy frames not found: {BUOY_FRAMES}")
        print("  Please run test4/prepare_buoy_testset.py first.")
        return 1

    # Load CSV
    df = pd.read_csv(BUOY_CSV)
    n_total = len(df)
    print(f"[Load] {n_total} samples from {BUOY_CSV}")

    # Shuffle indices
    np.random.seed(RANDOM_SEED)
    indices = np.arange(n_total, dtype=np.int64)
    np.random.shuffle(indices)

    # Split
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    print(f"[Split] train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Save splits
    ensure_dir(SPLIT_DIR)
    np.save(SPLIT_DIR / "train_indices.npy", train_indices)
    np.save(SPLIT_DIR / "val_indices.npy", val_indices)
    np.save(SPLIT_DIR / "test_indices.npy", test_indices)

    print(f"[Saved] Splits to {SPLIT_DIR}")

    # Print video distribution
    if "video" in df.columns:
        print("\n[Video distribution]")
        for split_name, split_idx in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            split_df = df.iloc[split_idx]
            video_counts = split_df["video"].value_counts()
            print(f"  {split_name}: {len(video_counts)} videos, {len(split_df)} frames")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
