# -*- coding: utf-8 -*-
"""
summarize_smd_results.py

Generate summary tables and statistics for Experiment 4: Cross-dataset Generalization on SMD.

This script reads the per-sample evaluation CSV from evaluate_fusion_cnn_smd.py
and produces:
  1. Overall metrics table (for thesis Table X.X)
  2. Per-domain breakdown table (NIR / VIS_Onboard / VIS_Onshore)
  3. Comparison with MU-SID performance (if provided)
  4. Markdown and LaTeX formatted outputs

Run (from project root):
  python test4/summarize_smd_results.py
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
TEST4_DIR = PROJECT_ROOT / "test4"

# Input files
SMD_EVAL_CSV = TEST4_DIR / "eval_smd_test_per_sample.csv"
MUSID_EVAL_CSV = PROJECT_ROOT / "eval_outputs" / "eval_test.csv"  # MU-SID results for comparison

# Output files
OUT_DIR = TEST4_DIR / "experiment4_results"
SUMMARY_MD = OUT_DIR / "summary_table.md"
SUMMARY_LATEX = OUT_DIR / "summary_table.tex"
DOMAIN_MD = OUT_DIR / "domain_breakdown.md"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def compute_metrics(df: pd.DataFrame, prefix: str = "") -> dict:
    """Compute standard metrics from evaluation dataframe."""
    rho_col = "rho_err_px_orig" if "rho_err_px_orig" in df.columns else "rho_err_px_unet"
    theta_col = "theta_err_deg"
    line_col = "line_dist_px_unet"
    
    rho = df[rho_col].values
    theta = df[theta_col].values
    line = df[line_col].values if line_col in df.columns else np.zeros(len(df))
    
    n = len(df)
    
    return {
        f"{prefix}N": n,
        f"{prefix}rho_mean": np.mean(rho),
        f"{prefix}rho_median": np.median(rho),
        f"{prefix}rho_p95": np.percentile(rho, 95),
        f"{prefix}rho_le10": 100.0 * np.mean(rho <= 10),
        f"{prefix}rho_le20": 100.0 * np.mean(rho <= 20),
        f"{prefix}theta_mean": np.mean(theta),
        f"{prefix}theta_median": np.median(theta),
        f"{prefix}theta_p95": np.percentile(theta, 95),
        f"{prefix}theta_le1": 100.0 * np.mean(theta <= 1),
        f"{prefix}theta_le2": 100.0 * np.mean(theta <= 2),
        f"{prefix}line_mean": np.mean(line),
        f"{prefix}line_median": np.median(line),
        f"{prefix}line_le10": 100.0 * np.mean(line <= 10),
    }


def generate_overall_table_md(smd_metrics: dict, musid_metrics: dict = None) -> str:
    """Generate markdown table comparing SMD and MU-SID performance."""
    lines = [
        "# Experiment 4: Cross-Dataset Generalization Results",
        "",
        "## Overall Performance Comparison",
        "",
        "| Dataset | N | ρ Mean (px) | ρ Median | ρ P95 | ρ≤10px (%) | ρ≤20px (%) | θ Mean (°) | θ Median | θ≤1° (%) | θ≤2° (%) |",
        "|---------|---|-------------|----------|-------|------------|------------|------------|----------|----------|----------|",
    ]
    
    # SMD row
    s = smd_metrics
    smd_row = f"| SMD (Zero-shot) | {s['N']} | {s['rho_mean']:.2f} | {s['rho_median']:.2f} | {s['rho_p95']:.2f} | {s['rho_le10']:.1f} | {s['rho_le20']:.1f} | {s['theta_mean']:.3f} | {s['theta_median']:.3f} | {s['theta_le1']:.1f} | {s['theta_le2']:.1f} |"
    lines.append(smd_row)
    
    # MU-SID row (if available)
    if musid_metrics:
        m = musid_metrics
        musid_row = f"| MU-SID (In-domain) | {m['N']} | {m['rho_mean']:.2f} | {m['rho_median']:.2f} | {m['rho_p95']:.2f} | {m['rho_le10']:.1f} | {m['rho_le20']:.1f} | {m['theta_mean']:.3f} | {m['theta_median']:.3f} | {m['theta_le1']:.1f} | {m['theta_le2']:.1f} |"
        lines.append(musid_row)
    
    lines.append("")
    lines.append("**Note**: SMD evaluation uses MU-SID trained weights without any fine-tuning (zero-shot transfer).")
    lines.append("")
    
    return "\n".join(lines)


def generate_domain_table_md(domain_metrics: dict) -> str:
    """Generate markdown table for per-domain breakdown."""
    lines = [
        "## Per-Domain Performance on SMD",
        "",
        "| Domain | N | ρ Mean (px) | ρ Median | ρ≤10px (%) | θ Mean (°) | θ≤2° (%) | Line Dist Mean |",
        "|--------|---|-------------|----------|------------|------------|----------|----------------|",
    ]
    
    for domain in ["NIR", "VIS_Onboard", "VIS_Onshore"]:
        if domain in domain_metrics:
            d = domain_metrics[domain]
            row = f"| {domain} | {d['N']} | {d['rho_mean']:.2f} | {d['rho_median']:.2f} | {d['rho_le10']:.1f} | {d['theta_mean']:.3f} | {d['theta_le2']:.1f} | {d['line_mean']:.2f} |"
            lines.append(row)
    
    lines.append("")
    lines.append("**Domain descriptions**:")
    lines.append("- **NIR**: Near-infrared camera footage")
    lines.append("- **VIS_Onboard**: Visible spectrum camera mounted on vessel")
    lines.append("- **VIS_Onshore**: Visible spectrum camera from shore-based station")
    lines.append("")
    
    return "\n".join(lines)


def generate_latex_table(smd_metrics: dict, musid_metrics: dict = None, domain_metrics: dict = None) -> str:
    """Generate LaTeX table for thesis."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-Dataset Generalization Performance on SMD}",
        r"\label{tab:smd_generalization}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Dataset & N & $\rho$ Mean & $\rho$ Median & $\rho \leq 10$px & $\theta$ Mean & $\theta \leq 2°$ \\",
        r"\midrule",
    ]
    
    s = smd_metrics
    lines.append(f"SMD (Zero-shot) & {s['N']} & {s['rho_mean']:.2f} & {s['rho_median']:.2f} & {s['rho_le10']:.1f}\\% & {s['theta_mean']:.3f}° & {s['theta_le2']:.1f}\\% \\\\")
    
    if musid_metrics:
        m = musid_metrics
        lines.append(f"MU-SID (In-domain) & {m['N']} & {m['rho_mean']:.2f} & {m['rho_median']:.2f} & {m['rho_le10']:.1f}\\% & {m['theta_mean']:.3f}° & {m['theta_le2']:.1f}\\% \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    
    # Per-domain table
    if domain_metrics:
        lines.extend([
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-Domain Performance on SMD Dataset}",
            r"\label{tab:smd_domain}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Domain & N & $\rho$ Mean & $\rho \leq 10$px & $\theta$ Mean & $\theta \leq 2°$ \\",
            r"\midrule",
        ])
        
        for domain in ["NIR", "VIS_Onboard", "VIS_Onshore"]:
            if domain in domain_metrics:
                d = domain_metrics[domain]
                domain_name = domain.replace("_", " ")
                lines.append(f"{domain_name} & {d['N']} & {d['rho_mean']:.2f} & {d['rho_le10']:.1f}\\% & {d['theta_mean']:.3f}° & {d['theta_le2']:.1f}\\% \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
    
    return "\n".join(lines)


def main():
    ensure_dir(OUT_DIR)
    
    # Check if SMD evaluation exists
    if not SMD_EVAL_CSV.exists():
        print(f"[Error] SMD evaluation CSV not found: {SMD_EVAL_CSV}")
        print("Please run: python test4/evaluate_fusion_cnn_smd.py first")
        sys.exit(1)
    
    print(f"[Load] SMD results: {SMD_EVAL_CSV}")
    smd_df = pd.read_csv(SMD_EVAL_CSV)
    
    # Compute overall SMD metrics
    smd_metrics = compute_metrics(smd_df)
    print(f"  -> {smd_metrics['N']} samples loaded")
    
    # Compute per-domain metrics
    domain_metrics = {}
    for domain in ["NIR", "VIS_Onboard", "VIS_Onshore"]:
        domain_df = smd_df[smd_df["domain"] == domain]
        if len(domain_df) > 0:
            domain_metrics[domain] = compute_metrics(domain_df)
            print(f"  -> {domain}: {domain_metrics[domain]['N']} samples")
    
    # Try to load MU-SID results for comparison
    musid_metrics = None
    if MUSID_EVAL_CSV.exists():
        print(f"[Load] MU-SID results: {MUSID_EVAL_CSV}")
        musid_df = pd.read_csv(MUSID_EVAL_CSV)
        # MU-SID CSV may have different column names, adapt as needed
        if "rho_err" in musid_df.columns:
            musid_df = musid_df.rename(columns={"rho_err": "rho_err_px_orig"})
        if "theta_err" in musid_df.columns:
            musid_df = musid_df.rename(columns={"theta_err": "theta_err_deg"})
        if "line_dist" in musid_df.columns:
            musid_df = musid_df.rename(columns={"line_dist": "line_dist_px_unet"})
        musid_metrics = compute_metrics(musid_df)
        print(f"  -> {musid_metrics['N']} samples loaded")
    else:
        print(f"[Skip] MU-SID results not found (optional)")
    
    # Generate tables
    print("\n[Generate] Summary tables...")
    
    # Markdown
    md_content = generate_overall_table_md(smd_metrics, musid_metrics)
    md_content += "\n" + generate_domain_table_md(domain_metrics)
    
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  -> Markdown: {SUMMARY_MD}")
    
    # LaTeX
    latex_content = generate_latex_table(smd_metrics, musid_metrics, domain_metrics)
    with open(SUMMARY_LATEX, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"  -> LaTeX: {SUMMARY_LATEX}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print(md_content)
    print("=" * 60)
    
    # Analysis insights
    print("\n[Analysis] Key findings:")
    print(f"  - SMD zero-shot ρ accuracy (≤10px): {smd_metrics['rho_le10']:.1f}%")
    print(f"  - SMD zero-shot θ accuracy (≤2°):  {smd_metrics['theta_le2']:.1f}%")
    
    if musid_metrics:
        rho_drop = musid_metrics['rho_le10'] - smd_metrics['rho_le10']
        theta_drop = musid_metrics['theta_le2'] - smd_metrics['theta_le2']
        print(f"  - Performance drop from MU-SID to SMD:")
        print(f"      ρ≤10px: {rho_drop:+.1f}% (MU-SID: {musid_metrics['rho_le10']:.1f}%)")
        print(f"      θ≤2°:  {theta_drop:+.1f}% (MU-SID: {musid_metrics['theta_le2']:.1f}%)")
    
    # Find best/worst domain
    if domain_metrics:
        best_domain = max(domain_metrics.keys(), key=lambda d: domain_metrics[d]['rho_le10'])
        worst_domain = min(domain_metrics.keys(), key=lambda d: domain_metrics[d]['rho_le10'])
        print(f"  - Best performing domain:  {best_domain} ({domain_metrics[best_domain]['rho_le10']:.1f}% ρ≤10px)")
        print(f"  - Worst performing domain: {worst_domain} ({domain_metrics[worst_domain]['rho_le10']:.1f}% ρ≤10px)")
    
    print(f"\n[Done] Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
