# -*- coding: utf-8 -*-
"""
Prepare SMD dataset for Experiment 6: In-Domain Training.

This script creates train/val/test splits for SMD dataset.
Split ratio: 70% train, 15% val, 15% test

Inputs:
  - test4/SMD_GroundTruth.csv    (from prepare_smd_testset.py)
  - test4/smd_frames/            (from prepare_smd_testset.py)

Outputs:
  - test6/splits_smd/train_indices.npy
  - test6/splits_smd/val_indices.npy
  - test6/splits_smd/test_indices.npy

Usage:
  python test6/prepare_smd_trainset.py
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
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# ============================

# Paths
TEST4_DIR = PROJECT_ROOT / "test4"
TEST6_DIR = PROJECT_ROOT / "test6"

# 使用手动筛选后的数据 (manual_review 目录)
SMD_CSV = TEST4_DIR / "manual_review" / "SMD_GroundTruth_filtered.csv"
SMD_FRAMES = TEST4_DIR / "manual_review" / "kept_frames"
SPLIT_DIR = TEST6_DIR / "splits_smd"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    print("=" * 60)
    print("Prepare SMD Train/Val/Test Splits for Experiment 6")
    print("=" * 60)

    # Check if SMD data exists
    if not SMD_CSV.exists():
        print(f"[Error] SMD CSV not found: {SMD_CSV}")
        print("  Please run test4/manual_filter_smd_by_gt.py first to filter SMD data.")
        return 1

    if not SMD_FRAMES.exists():
        print(f"[Error] SMD frames not found: {SMD_FRAMES}")
        print("  Please run test4/manual_filter_smd_by_gt.py first to filter SMD data.")
        return 1

    # Load CSV
    df = pd.read_csv(SMD_CSV)
    n_total = len(df)
    print(f"[Load] {n_total} samples from {SMD_CSV}")

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

    # Print domain distribution
    if "domain" in df.columns:
        print("\n[Domain distribution]")
        for split_name, split_idx in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            split_df = df.iloc[split_idx]
            domain_counts = split_df["domain"].value_counts()
            print(f"  {split_name}:")
            for domain, count in domain_counts.items():
                print(f"    {domain}: {count}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
