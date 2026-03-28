# -*- coding: utf-8 -*-
"""select_topk_by_edgey.py

Select Top-K hardest samples by edge-Y error (original scale) from the CSV saved by
`evaluate_full_pipeline.py` (default path: eval_full_outputs/full_eval_test.csv).

Why this script?
- You already have per-sample metrics in `full_eval_test.csv`, including:
    edgey_px_orig_cnn   (CNN-only)
    edgey_px_orig_final (after refine)
- For Top-K analysis experiments, you typically want to rank by CNN-only error.

Example (Windows):
  python select_topk_by_edgey.py --csv eval_full_outputs\\full_eval_test.csv --k 50 --which cnn

Outputs (written to --out_dir):
  - topk_edgey_<which>_<k>.csv : sorted table
  - topk_edgey_<which>_<k>.txt : just idx + img_name list (easy to feed other scripts)
"""

import argparse
import os

import pandas as pd
K = 50
WHICH = "cnn"
CSV_PATH = r"eval_full_outputs\full_eval_test.csv"
OUT_DIR = r"eval_full_outputs"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=os.path.join("eval_full_outputs", "full_eval_test.csv"))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--which", type=str, choices=["cnn", "final"], default="cnn",
                    help="Rank by CNN-only edgeY or final edgeY.")
    ap.add_argument("--out_dir", type=str, default="eval_full_outputs")
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run evaluate_full_pipeline.py first.")

    df = pd.read_csv(csv_path)

    col = "edgey_px_orig_cnn" if args.which == "cnn" else "edgey_px_orig_final"
    needed = ["idx", "img_name", "conf", "used_ref", col]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not in CSV. Available columns: {list(df.columns)}")

    df2 = df[needed].copy()
    df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=[col])

    df2 = df2.sort_values(col, ascending=False).head(int(args.k)).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"topk_edgey_{args.which}_{args.k}.csv")
    out_txt = os.path.join(args.out_dir, f"topk_edgey_{args.which}_{args.k}.txt")

    df2.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_txt, "w", encoding="utf-8") as f:
        for _, r in df2.iterrows():
            f.write(f"{int(r['idx'])}\t{r['img_name']}\n")

    print("N (rows in CSV) =", len(df))
    print("Selected Top-K =", len(df2), "| which =", args.which)
    print("Saved:")
    print(" ", out_csv)
    print(" ", out_txt)
    print("\nTop-5 preview:")
    print(df2.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
