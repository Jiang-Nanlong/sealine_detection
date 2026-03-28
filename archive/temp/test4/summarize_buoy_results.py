# -*- coding: utf-8 -*-
"""
Summarize Buoy evaluation results for Experiment 4.

This script reads the per-sample evaluation CSV from evaluate_fusion_cnn_buoy.py
and generates summary tables and statistics.

Usage:
  python test4/summarize_buoy_results.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST4_DIR = SCRIPT_DIR

BUOY_EVAL_CSV = TEST4_DIR / "eval_buoy_test_per_sample.csv"


def pct_le(arr: np.ndarray, thr: float) -> float:
    """Percentage of values <= threshold."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return 100.0 * float(np.mean(arr <= thr))


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute summary metrics from dataframe."""
    rho = df["rho_err_px_orig"].values
    theta = df["theta_err_deg"].values
    line = df["line_dist_px_unet"].values

    return {
        "N": len(df),
        "rho_mean": float(np.mean(rho)),
        "rho_median": float(np.median(rho)),
        "rho_p95": float(np.percentile(rho, 95)),
        "rho_le5": pct_le(rho, 5),
        "rho_le10": pct_le(rho, 10),
        "rho_le20": pct_le(rho, 20),
        "theta_mean": float(np.mean(theta)),
        "theta_median": float(np.median(theta)),
        "theta_p95": float(np.percentile(theta, 95)),
        "theta_le1": pct_le(theta, 1),
        "theta_le2": pct_le(theta, 2),
        "theta_le5": pct_le(theta, 5),
        "line_mean": float(np.mean(line)),
        "line_median": float(np.median(line)),
        "line_le10": pct_le(line, 10),
    }


def generate_overall_table_md(buoy_metrics: dict, musid_metrics: dict = None) -> str:
    """Generate markdown table comparing Buoy and MU-SID performance."""
    lines = [
        "## Overall Performance Comparison",
        "",
        "| Dataset | N | ρ Mean (px) | ρ Median | ρ P95 | ρ≤10px | ρ≤20px | θ Mean (°) | θ Median | θ≤1° | θ≤2° |",
        "|---------|---|-------------|----------|-------|--------|--------|------------|----------|------|------|",
    ]

    # Buoy row
    b = buoy_metrics
    buoy_row = f"| Buoy (Zero-shot) | {b['N']} | {b['rho_mean']:.2f} | {b['rho_median']:.2f} | {b['rho_p95']:.2f} | {b['rho_le10']:.1f} | {b['rho_le20']:.1f} | {b['theta_mean']:.3f} | {b['theta_median']:.3f} | {b['theta_le1']:.1f} | {b['theta_le2']:.1f} |"
    lines.append(buoy_row)

    if musid_metrics:
        m = musid_metrics
        musid_row = f"| MU-SID (In-domain) | {m['N']} | {m['rho_mean']:.2f} | {m['rho_median']:.2f} | {m['rho_p95']:.2f} | {m['rho_le10']:.1f} | {m['rho_le20']:.1f} | {m['theta_mean']:.3f} | {m['theta_median']:.3f} | {m['theta_le1']:.1f} | {m['theta_le2']:.1f} |"
        lines.append(musid_row)

    lines.append("")
    lines.append("**Note**: Buoy evaluation uses MU-SID trained weights without any fine-tuning (zero-shot transfer).")
    return "\n".join(lines)


def generate_per_video_table_md(df: pd.DataFrame) -> str:
    """Generate per-video performance table."""
    lines = [
        "## Per-Video Performance on Buoy",
        "",
        "| Video | N | ρ Mean (px) | ρ Median | θ Mean (°) | θ Median | θ≤2° (%) | ρ≤10px (%) |",
        "|-------|---|-------------|----------|------------|----------|----------|------------|",
    ]

    for video in sorted(df["video"].unique()):
        vdf = df[df["video"] == video]
        m = compute_metrics(vdf)
        row = f"| {video} | {m['N']} | {m['rho_mean']:.2f} | {m['rho_median']:.2f} | {m['theta_mean']:.3f} | {m['theta_median']:.3f} | {m['theta_le2']:.1f} | {m['rho_le10']:.1f} |"
        lines.append(row)

    return "\n".join(lines)


def generate_latex_table(buoy_metrics: dict, video_metrics: dict = None) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-Dataset Generalization Performance on Buoy Dataset}",
        r"\label{tab:buoy_generalization}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Dataset & N & $\rho$ Mean (px) & $\rho$ Median & $\rho\leq$10px & $\theta$ Mean (°) & $\theta\leq$2° \\",
        r"\midrule",
    ]

    b = buoy_metrics
    lines.append(f"Buoy (Zero-shot) & {b['N']} & {b['rho_mean']:.2f} & {b['rho_median']:.2f} & {b['rho_le10']:.1f}\\% & {b['theta_mean']:.3f}° & {b['theta_le2']:.1f}\\% \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    if video_metrics:
        lines.extend([
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-Video Performance on Buoy Dataset}",
            r"\label{tab:buoy_per_video}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Video & N & $\rho$ Mean & $\rho$ Med & $\theta$ Mean & $\theta\leq$2° \\",
            r"\midrule",
        ])

        for video, m in sorted(video_metrics.items()):
            # Shorten video name for table
            short_name = video.replace("buoyGT_", "").replace("_", "-")
            lines.append(f"{short_name} & {m['N']} & {m['rho_mean']:.2f} & {m['rho_median']:.2f} & {m['theta_mean']:.3f} & {m['theta_le2']:.1f}\\% \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Summarize Buoy Results (Experiment 4)")
    print("=" * 60)

    if not BUOY_EVAL_CSV.exists():
        print(f"[Error] Evaluation CSV not found: {BUOY_EVAL_CSV}")
        print("  Please run evaluate_fusion_cnn_buoy.py first.")
        return

    # Load data
    df = pd.read_csv(BUOY_EVAL_CSV)
    print(f"[Load] {len(df)} samples from {BUOY_EVAL_CSV}")

    # Compute overall metrics
    buoy_metrics = compute_metrics(df)

    # Compute per-video metrics
    video_metrics = {}
    for video in df["video"].unique():
        vdf = df[df["video"] == video]
        video_metrics[video] = compute_metrics(vdf)

    # Generate markdown report
    md_lines = [
        "# Experiment 4: Cross-Dataset Generalization on Buoy",
        "",
        f"Total samples: {len(df)}",
        f"Number of videos: {len(video_metrics)}",
        "",
    ]

    md_lines.append(generate_overall_table_md(buoy_metrics))
    md_lines.append("")
    md_lines.append(generate_per_video_table_md(df))
    md_lines.append("")
    md_lines.append("## LaTeX Tables")
    md_lines.append("")
    md_lines.append("```latex")
    md_lines.append(generate_latex_table(buoy_metrics, video_metrics))
    md_lines.append("```")

    # Save markdown report
    out_dir = TEST4_DIR / "experiment4_results"
    out_dir.mkdir(exist_ok=True)

    md_path = out_dir / "buoy_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Saved] Markdown summary -> {md_path}")

    # Save metrics as CSV
    metrics_csv = out_dir / "buoy_metrics_summary.csv"
    metrics_df = pd.DataFrame([{
        "dataset": "Buoy",
        "type": "zero-shot",
        **buoy_metrics,
    }])
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[Saved] Metrics CSV -> {metrics_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Buoy (Zero-shot): N={buoy_metrics['N']}")
    print(f"  ρ: mean={buoy_metrics['rho_mean']:.2f}px, median={buoy_metrics['rho_median']:.2f}px, ≤10px={buoy_metrics['rho_le10']:.1f}%")
    print(f"  θ: mean={buoy_metrics['theta_mean']:.3f}°, median={buoy_metrics['theta_median']:.3f}°, ≤2°={buoy_metrics['theta_le2']:.1f}%")


if __name__ == "__main__":
    main()
