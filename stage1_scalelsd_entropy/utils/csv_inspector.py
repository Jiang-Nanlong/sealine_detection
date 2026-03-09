#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_inspector.py — MU-SID GroundTruth.csv 检查工具

功能：
  1. 打印 CSV 的列名（或列索引）、前 N 行、总行数
  2. 检查图像文件名与 CSV 行是否能对应
  3. 输出统计信息：坐标范围、角度范围等

用法:
  python -m stage1_scalelsd_entropy.utils.csv_inspector \
      --csv  "D:/dataset/Hashmani's Dataset/GroundTruth.csv" \
      --img_dir "D:/dataset/Hashmani's Dataset/MU-SID" \
      --head 5
"""

import os
import argparse
import pandas as pd


# ================================================================
# Helper: 尝试根据 stem 找图像文件
# ================================================================
COMMON_EXTS = ("", ".JPG", ".jpg", ".jpeg", ".png", ".JPEG", ".PNG")


def resolve_image(img_dir: str, stem_or_name: str) -> str | None:
    """给定 CSV 第一列的值，尝试在 img_dir 下找到对应图像。"""
    name = str(stem_or_name).strip()
    for ext in COMMON_EXTS:
        p = os.path.join(img_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None


# ================================================================
# Main inspection
# ================================================================
def inspect(csv_path: str, img_dir: str | None = None, head: int = 5):
    print("=" * 60)
    print(f"CSV path : {csv_path}")
    print(f"Image dir: {img_dir}")
    print("=" * 60)

    # ---- 1. 基本信息 ----
    df = pd.read_csv(csv_path, header=None)
    n_rows, n_cols = df.shape
    print(f"\n[1] Shape: {n_rows} rows × {n_cols} columns")
    print(f"    Column indices: {list(df.columns)}")
    print(f"    Dtypes:\n{df.dtypes.to_string()}")

    # ---- 2. 前 N 行 ----
    print(f"\n[2] First {head} rows:")
    print(df.head(head).to_string(index=True))

    # ---- 3. 列级别统计 ----
    print(f"\n[3] Column-wise analysis:")
    print(f"    Col 0 (image stem): {df[0].nunique()} unique values, "
          f"sample = {df[0].iloc[0]!r}")

    for c in range(1, min(n_cols, 8)):
        col = pd.to_numeric(df[c], errors="coerce")
        valid = col.dropna()
        if len(valid) > 0:
            print(f"    Col {c}: min={valid.min():.3f}, max={valid.max():.3f}, "
                  f"mean={valid.mean():.3f}, NaN={col.isna().sum()}")
        else:
            print(f"    Col {c}: all non-numeric or NaN")

    # ---- 4. 解读 ----
    print(f"\n[4] Interpretation (based on 8-column no-header format):")
    print("    Col 0 : image stem  (e.g. DSC_0051_9)")
    print("    Col 1 : x1          (horizon left endpoint x)")
    print("    Col 2 : y1          (horizon left endpoint y)")
    print("    Col 3 : x2          (horizon right endpoint x)")
    print("    Col 4 : y2          (horizon right endpoint y)")
    print("    Col 5 : x_mid       (midpoint x, likely derived)")
    print("    Col 6 : y_mid       (midpoint y, likely derived)")
    print("    Col 7 : angle       (degrees)")

    # ---- 5. 图像匹配检查 ----
    if img_dir and os.path.isdir(img_dir):
        print(f"\n[5] Image matching check (img_dir has "
              f"{len(os.listdir(img_dir))} entries):")
        matched = 0
        missing = []
        for i in range(n_rows):
            stem = str(df.iloc[i, 0]).strip()
            if resolve_image(img_dir, stem) is not None:
                matched += 1
            else:
                missing.append(stem)

        print(f"    Matched  : {matched} / {n_rows}")
        print(f"    Missing  : {len(missing)}")
        if missing:
            show = missing[:10]
            print(f"    First missing: {show}")
    else:
        print("\n[5] Image matching check: SKIPPED (img_dir not provided or invalid)")

    print("\n" + "=" * 60)
    print("Inspection complete.")
    return df


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="MU-SID GroundTruth.csv inspector")
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument("--csv", type=str,
                        default=os.path.join(_project_root, "Hashmani's Dataset", "GroundTruth.csv"),
                        help="Path to GroundTruth.csv")
    parser.add_argument("--img_dir", type=str,
                        default=os.path.join(_project_root, "Hashmani's Dataset", "MU-SID"),
                        help="Path to MU-SID image directory")
    parser.add_argument("--head", type=int, default=5,
                        help="Number of rows to display")
    args = parser.parse_args()
    inspect(args.csv, args.img_dir, args.head)


if __name__ == "__main__":
    main()
