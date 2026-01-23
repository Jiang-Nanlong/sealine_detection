# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

CSV_STRONG = os.path.join(PROJ_DIR, "eval_full_outputs", "full_eval_test_strong.csv")
CSV_WEAK   = os.path.join(PROJ_DIR, "eval_full_outputs", "full_eval_test_weak.csv")
OUT_DIR    = os.path.join(PROJ_DIR, "test_2", "figs_refine_gain")

EPS = 1e-12

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def outcome_rates(df: pd.DataFrame):
    # ΔEdgeY = Final - CNN
    d = df["edgey_px_orig_final"].to_numpy() - df["edgey_px_orig_cnn"].to_numpy()
    improve = np.mean(d < -EPS)
    worsen  = np.mean(d >  EPS)
    same    = 1.0 - improve - worsen
    return improve, same, worsen

def main():
    ensure_dir(OUT_DIR)
    df_s = pd.read_csv(CSV_STRONG)
    df_w = pd.read_csv(CSV_WEAK)

    w = outcome_rates(df_w)
    s = outcome_rates(df_s)

    labels = ["weak", "strong"]
    improve = [w[0], s[0]]
    same    = [w[1], s[1]]
    worsen  = [w[2], s[2]]

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, improve, label="improve (Δ<0)")
    plt.bar(x, same, bottom=improve, label="no-change (Δ=0)")
    plt.bar(x, worsen, bottom=np.array(improve)+np.array(same), label="worsen (Δ>0)")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Proportion")
    plt.title("Refine outcome rates on MU-SID test")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "fig_refine_outcome_rates.png")
    plt.savefig(out, dpi=250)
    plt.close()
    print("[SAVED]", out)

if __name__ == "__main__":
    main()
