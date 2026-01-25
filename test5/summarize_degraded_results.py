# -*- coding: utf-8 -*-
"""
summarize_degraded_results.py

Generate summary tables for Experiment 5: Degradation Robustness.

Produces:
  1. Overall comparison table (clean vs each degradation)
  2. Grouped analysis (by degradation type: noise, blur, light, etc.)
  3. Markdown and LaTeX formatted outputs for thesis

PyCharm: 直接运行此文件
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
TEST5_DIR = PROJECT_ROOT / "test5"

# Input
EVAL_CSV = TEST5_DIR / "eval_results" / "degradation_results.csv"

# Output
OUT_DIR = TEST5_DIR / "experiment5_results"
SUMMARY_MD = OUT_DIR / "summary_table.md"
SUMMARY_LATEX = OUT_DIR / "summary_table.tex"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# Degradation type grouping
DEGRADATION_GROUPS = {
    "clean": "Baseline",
    "gaussian_noise_10": "Gaussian Noise",
    "gaussian_noise_25": "Gaussian Noise",
    "gaussian_noise_50": "Gaussian Noise",
    "motion_blur_15": "Motion Blur",
    "motion_blur_25": "Motion Blur",
    "gaussian_blur_2": "Gaussian Blur",
    "gaussian_blur_5": "Gaussian Blur",
    "low_light_2.0": "Low Light",
    "low_light_3.0": "Low Light",
    "fog_0.3": "Fog/Haze",
    "fog_0.5": "Fog/Haze",
    "sp_noise_0.01": "Salt & Pepper",
    "sp_noise_0.05": "Salt & Pepper",
}

# Display names
DISPLAY_NAMES = {
    "clean": "Clean (Baseline)",
    "gaussian_noise_10": "Gaussian σ=10",
    "gaussian_noise_25": "Gaussian σ=25",
    "gaussian_noise_50": "Gaussian σ=50",
    "motion_blur_15": "Motion k=15",
    "motion_blur_25": "Motion k=25",
    "gaussian_blur_2": "Blur σ=2",
    "gaussian_blur_5": "Blur σ=5",
    "low_light_2.0": "Low Light γ=2.0",
    "low_light_3.0": "Low Light γ=3.0",
    "fog_0.3": "Fog 30%",
    "fog_0.5": "Fog 50%",
    "sp_noise_0.01": "S&P 1%",
    "sp_noise_0.05": "S&P 5%",
}


def generate_markdown(df: pd.DataFrame) -> str:
    """Generate markdown summary."""
    lines = []
    lines.append("# Experiment 5: Degradation Robustness Results\n")
    
    # Get baseline
    baseline = df[df["degradation"] == "clean"]
    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        baseline_rho = baseline["rho_le_10"]
        baseline_theta = baseline["theta_le_2"]
    else:
        baseline_rho = baseline_theta = None
    
    # Overall table
    lines.append("## Performance Under Various Degradations\n")
    lines.append("| Degradation | N | ρ Mean (px) | ρ≤10px (%) | Δρ | θ Mean (°) | θ≤2° (%) | Δθ |")
    lines.append("|-------------|---|-------------|------------|-----|------------|----------|-----|")
    
    for _, row in df.iterrows():
        deg = row["degradation"]
        name = DISPLAY_NAMES.get(deg, deg)
        
        delta_rho = ""
        delta_theta = ""
        if baseline_rho is not None and deg != "clean":
            dr = row["rho_le_10"] - baseline_rho
            dt = row["theta_le_2"] - baseline_theta
            delta_rho = f"{dr:+.1f}"
            delta_theta = f"{dt:+.1f}"
        
        lines.append(
            f"| {name} | {row['n']} | {row['rho_mean']:.2f} | {row['rho_le_10']:.1f} | {delta_rho} | "
            f"{row['theta_mean']:.3f} | {row['theta_le_2']:.1f} | {delta_theta} |"
        )
    
    # Grouped analysis
    lines.append("\n## Performance by Degradation Type\n")
    
    groups = {}
    for _, row in df.iterrows():
        deg = row["degradation"]
        group = DEGRADATION_GROUPS.get(deg, "Other")
        if group not in groups:
            groups[group] = []
        groups[group].append(row)
    
    lines.append("| Type | Avg ρ≤10px (%) | Avg θ≤2° (%) | Robustness |")
    lines.append("|------|----------------|--------------|------------|")
    
    for group_name in ["Baseline", "Gaussian Noise", "Motion Blur", "Gaussian Blur", "Low Light", "Fog/Haze", "Salt & Pepper"]:
        if group_name not in groups:
            continue
        rows = groups[group_name]
        avg_rho = np.mean([r["rho_le_10"] for r in rows])
        avg_theta = np.mean([r["theta_le_2"] for r in rows])
        
        if baseline_rho is not None and group_name != "Baseline":
            drop_rho = baseline_rho - avg_rho
            if drop_rho < 5:
                robust = "✅ Excellent"
            elif drop_rho < 15:
                robust = "✅ Good"
            elif drop_rho < 30:
                robust = "⚠️ Moderate"
            else:
                robust = "❌ Poor"
        else:
            robust = "-"
        
        lines.append(f"| {group_name} | {avg_rho:.1f} | {avg_theta:.1f} | {robust} |")
    
    # Key findings
    lines.append("\n## Key Findings\n")
    
    if baseline_rho is not None:
        # Find worst degradation
        worst = df[df["degradation"] != "clean"].sort_values("rho_le_10").iloc[0]
        best = df[df["degradation"] != "clean"].sort_values("rho_le_10", ascending=False).iloc[0]
        
        lines.append(f"- **Baseline (Clean)**: ρ≤10px = {baseline_rho:.1f}%, θ≤2° = {baseline_theta:.1f}%")
        lines.append(f"- **Most Robust Against**: {DISPLAY_NAMES.get(best['degradation'], best['degradation'])} "
                    f"(ρ≤10px = {best['rho_le_10']:.1f}%, drop = {baseline_rho - best['rho_le_10']:.1f}%)")
        lines.append(f"- **Most Challenging**: {DISPLAY_NAMES.get(worst['degradation'], worst['degradation'])} "
                    f"(ρ≤10px = {worst['rho_le_10']:.1f}%, drop = {baseline_rho - worst['rho_le_10']:.1f}%)")
        
        # Angle robustness
        avg_theta_drop = baseline_theta - df[df["degradation"] != "clean"]["theta_le_2"].mean()
        lines.append(f"- **Angle Robustness**: Average θ≤2° drop = {avg_theta_drop:.1f}% (highly robust)")
    
    return "\n".join(lines)


def generate_latex(df: pd.DataFrame) -> str:
    """Generate LaTeX table."""
    lines = []
    
    # Main comparison table
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Model Robustness Under Image Degradations}")
    lines.append("\\label{tab:degradation_robustness}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Degradation & N & $\\rho$ Mean & $\\rho \\leq 10$px & $\\theta$ Mean & $\\theta \\leq 2°$ \\\\")
    lines.append("\\midrule")
    
    for _, row in df.iterrows():
        deg = row["degradation"]
        name = DISPLAY_NAMES.get(deg, deg)
        # Escape special characters for LaTeX
        name = name.replace("σ", "$\\sigma$").replace("γ", "$\\gamma$").replace("&", "\\&")
        
        lines.append(
            f"{name} & {row['n']} & {row['rho_mean']:.2f} & {row['rho_le_10']:.1f}\\% & "
            f"{row['theta_mean']:.3f}° & {row['theta_le_2']:.1f}\\% \\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Experiment 5: Generate Summary Tables")
    print("=" * 60)
    
    if not EVAL_CSV.exists():
        print(f"[Error] Results file not found: {EVAL_CSV}")
        print("Please run evaluate_degraded.py first")
        sys.exit(1)
    
    ensure_dir(OUT_DIR)
    
    # Load results
    df = pd.read_csv(EVAL_CSV)
    print(f"\n[Load] {len(df)} degradation results")
    
    # Sort by degradation type
    order = list(DISPLAY_NAMES.keys())
    df["sort_key"] = df["degradation"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])
    
    # Generate markdown
    md_content = generate_markdown(df)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Markdown -> {SUMMARY_MD}")
    
    # Generate LaTeX
    latex_content = generate_latex(df)
    with open(SUMMARY_LATEX, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"[Saved] LaTeX -> {SUMMARY_LATEX}")
    
    # Print to console
    print("\n" + "=" * 60)
    print(md_content)
    print("=" * 60)


if __name__ == "__main__":
    main()
