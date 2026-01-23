# -*- coding: utf-8 -*-
"""
plot_refine_gain_two_weights.py

Inputs:
  - eval_full_outputs/full_eval_test_strong.csv
  - eval_full_outputs/full_eval_test_weak.csv

Outputs (to test_2/figs_refine_gain):
  1) fig_delta_edgey_boxplot.png   (paper-friendly)
  2) fig_delta_edgey_scatter.png
  3) worst_worsen_strong.csv       (pick a "mis-correction" example)
  4) best_improve_weak.csv         (pick a "successful correction" example)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os

# 脚本所在目录：.../sealine_detection/test_3
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 工程根目录：.../sealine_detection
PROJ_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

CSV_STRONG = os.path.join(PROJ_DIR, "eval_full_outputs", "full_eval_test_strong.csv")
CSV_WEAK   = os.path.join(PROJ_DIR, "eval_full_outputs", "full_eval_test_weak.csv")

OUT_DIR    = os.path.join(PROJ_DIR, "test_2", "figs_refine_gain")


TOPN_SAVE = 30   # number of examples to export


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def summarize(df: pd.DataFrame, name: str):
    d = df["delta_edgey_orig"].to_numpy()
    base = df["edgey_px_orig_cnn"].to_numpy()
    print(f"\n===== {name} =====")
    print(f"N={len(df)}")
    print(f"EdgeY(CNN) mean={base.mean():.4f} | p95={np.percentile(base,95):.4f}")
    print(f"ΔEdgeY(Final-CNN) mean={d.mean():.4f} | median={np.median(d):.4f} | p95={np.percentile(d,95):.4f}")
    print(f"Improve rate (Δ<0)={np.mean(d<0)*100:.2f}% | Worsen rate (Δ>0)={np.mean(d>0)*100:.2f}%")


def main():
    ensure_dir(OUT_DIR)
    print("[INFO] CSV_STRONG =", CSV_STRONG)
    print("[INFO] CSV_WEAK   =", CSV_WEAK)

    df_s = pd.read_csv(CSV_STRONG)
    df_w = pd.read_csv(CSV_WEAK)

    # Compute deltas (orig-scale approx)
    for df in (df_s, df_w):
        df["delta_edgey_orig"] = df["edgey_px_orig_final"] - df["edgey_px_orig_cnn"]
        df["delta_rho_orig"]   = df["rho_abs_px_orig_final"] - df["rho_abs_px_orig_cnn"]
        df["delta_line_orig"]  = df["line_dist_px_orig_final"] - df["line_dist_px_orig_cnn"]
        df["delta_theta_deg"]  = df["theta_abs_deg_final"] - df["theta_abs_deg_cnn"]

    summarize(df_w, "WEAK weight")
    summarize(df_s, "STRONG weight")

    # --------- Figure 1: Boxplot of ΔEdgeY ----------
    plt.figure()
    data = [df_w["delta_edgey_orig"].to_numpy(), df_s["delta_edgey_orig"].to_numpy()]
    plt.boxplot(data, labels=["weak", "strong"], showfliers=True)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.ylabel(r"$\Delta$EdgeY (Final - CNN) [px, orig-scale approx]")
    plt.title("Refine gain distribution (ΔEdgeY)")
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "fig_delta_edgey_boxplot.png")
    plt.savefig(out1, dpi=250)
    plt.close()

    # --------- Figure 2: Scatter baseline vs gain ----------
    plt.figure()
    plt.scatter(df_w["edgey_px_orig_cnn"], df_w["delta_edgey_orig"], s=10, alpha=0.45, label="weak")
    plt.scatter(df_s["edgey_px_orig_cnn"], df_s["delta_edgey_orig"], s=10, alpha=0.45, label="strong")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("EdgeY(CNN) [px, orig-scale approx]")
    plt.ylabel(r"$\Delta$EdgeY (Final - CNN) [px]")
    plt.title("When refine helps/hurts (ΔEdgeY vs baseline)")
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "fig_delta_edgey_scatter.png")
    plt.savefig(out2, dpi=250)
    plt.close()

    # --------- Export examples for qualitative figure ----------
    # Strong: worst mis-corrections (largest positive delta)
    worst_strong = df_s.sort_values("delta_edgey_orig", ascending=False).head(TOPN_SAVE)
    worst_path = os.path.join(OUT_DIR, "worst_worsen_strong.csv")
    worst_strong.to_csv(worst_path, index=False, encoding="utf-8-sig")

    # Weak: best corrections (most negative delta)
    best_weak = df_w.sort_values("delta_edgey_orig", ascending=True).head(TOPN_SAVE)
    best_path = os.path.join(OUT_DIR, "best_improve_weak.csv")
    best_weak.to_csv(best_path, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(out1)
    print(out2)
    print(worst_path)
    print(best_path)
    print("\nTip: Use worst_worsen_strong.csv to pick 1-2 samples for a qualitative 'mis-correction' figure.")


if __name__ == "__main__":
    main()
