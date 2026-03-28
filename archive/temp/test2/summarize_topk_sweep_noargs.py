import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("../eval_sweep_topk")  # 你的 sweep 输出根目录


def stats_series(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return dict(mean=np.nan, median=np.nan, p90=np.nan, p95=np.nan, max=np.nan)
    return dict(
        mean=float(x.mean()),
        median=float(x.median()),
        p90=float(x.quantile(0.90)),
        p95=float(x.quantile(0.95)),
        max=float(x.max()),
    )


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}"):
    # 简单 Markdown 表（不依赖 tabulate）
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append(floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Not found: {ROOT.resolve()}")

    # 找 k010, k020 ... k100
    k_dirs = []
    for p in ROOT.iterdir():
        if p.is_dir():
            m = re.match(r"^k(\d+)$", p.name)
            if m:
                k_dirs.append((int(m.group(1)), p))
    k_dirs.sort(key=lambda t: t[0])

    if not k_dirs:
        raise RuntimeError(f"No k*** folders found under: {ROOT.resolve()}")

    rows = []
    for k, d in k_dirs:
        csv_path = d / "full_eval_test.csv"
        gate_path = d / "gate_stats.csv"
        if not csv_path.exists():
            print(f"[Skip] missing {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        n = len(df)

        # 这些列是你当前 evaluate_full_pipeline 输出里稳定存在的
        col_edgey_cnn = "edgey_px_orig_cnn"
        col_edgey_final = "edgey_px_orig_final"
        col_lined_cnn = "line_dist_px_orig_cnn"
        col_lined_final = "line_dist_px_orig_final"
        col_rho_cnn = "rho_abs_px_orig_cnn"
        col_rho_final = "rho_abs_px_orig_final"

        # overall
        s_edgey_cnn = stats_series(df.get(col_edgey_cnn, pd.Series(dtype=float)))
        s_edgey_final = stats_series(df.get(col_edgey_final, pd.Series(dtype=float)))

        s_lined_cnn = stats_series(df.get(col_lined_cnn, pd.Series(dtype=float)))
        s_lined_final = stats_series(df.get(col_lined_final, pd.Series(dtype=float)))

        s_rho_cnn = stats_series(df.get(col_rho_cnn, pd.Series(dtype=float)))
        s_rho_final = stats_series(df.get(col_rho_final, pd.Series(dtype=float)))

        used_ref = int(pd.to_numeric(df.get("used_ref", 0), errors="coerce").fillna(0).sum())
        refine_rate = used_ref / n if n > 0 else np.nan

        # topk subset（如果有）
        if "in_topk" in df.columns:
            df_topk = df[df["in_topk"] == 1].copy()
        else:
            df_topk = df.iloc[0:0].copy()

        topk_n = len(df_topk)
        topk_used_ref = int(pd.to_numeric(df_topk.get("used_ref", 0), errors="coerce").fillna(0).sum()) if topk_n > 0 else 0

        topk_edgey_cnn = stats_series(df_topk.get(col_edgey_cnn, pd.Series(dtype=float)))
        topk_edgey_final = stats_series(df_topk.get(col_edgey_final, pd.Series(dtype=float)))

        rows.append(
            dict(
                K=k,
                N=n,
                used_ref=used_ref,
                refine_rate=refine_rate,

                edgey_mean_cnn=s_edgey_cnn["mean"],
                edgey_mean_final=s_edgey_final["mean"],
                edgey_mean_delta=(s_edgey_final["mean"] - s_edgey_cnn["mean"]),

                lined_mean_cnn=s_lined_cnn["mean"],
                lined_mean_final=s_lined_final["mean"],
                lined_mean_delta=(s_lined_final["mean"] - s_lined_cnn["mean"]),

                rho_mean_cnn=s_rho_cnn["mean"],
                rho_mean_final=s_rho_final["mean"],
                rho_mean_delta=(s_rho_final["mean"] - s_rho_cnn["mean"]),

                edgey_p95_cnn=s_edgey_cnn["p95"],
                edgey_p95_final=s_edgey_final["p95"],

                topk_n=topk_n,
                topk_used_ref=topk_used_ref,
                topk_edgey_mean_cnn=topk_edgey_cnn["mean"],
                topk_edgey_mean_final=topk_edgey_final["mean"],
                topk_edgey_mean_delta=(topk_edgey_final["mean"] - topk_edgey_cnn["mean"]),
            )
        )

        # gate_stats.csv 里如果你还想加 fail 统计，也可以在这里读
        if gate_path.exists():
            pass

    df_sum = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

    out_csv = ROOT / "summary_topk_sweep.csv"
    out_md = ROOT / "summary_topk_sweep.md"
    df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 写一个 markdown（不依赖 tabulate）
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Top-K sweep summary\n\n")
        f.write(md_table(df_sum))

    # 图1：K vs overall EdgeY mean（CNN vs Final）
    fig1 = plt.figure()
    plt.plot(df_sum["K"], df_sum["edgey_mean_cnn"], marker="o")
    plt.plot(df_sum["K"], df_sum["edgey_mean_final"], marker="o")
    plt.xlabel("K")
    plt.ylabel("EdgeY mean (px, orig)")
    plt.title("K sweep: EdgeY mean (overall)")
    plt.legend(["CNN", "Final"])
    fig1_path = ROOT / "k_sweep_plot.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 图2：K vs refine_rate
    fig2 = plt.figure()
    plt.plot(df_sum["K"], df_sum["refine_rate"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Refine rate (used_ref / N)")
    plt.title("K sweep: refinement trigger rate")
    fig2_path = ROOT / "k_sweep_refine_rate.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_md}")
    print(f"[Saved] {fig1_path}")
    print(f"[Saved] {fig2_path}")


if __name__ == "__main__":
    main()
