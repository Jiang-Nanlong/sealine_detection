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

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
# 选择数据集: "musid", "smd", "buoy"
DATASET = "musid"
# ============================

# 数据集配置
DATASET_CONFIGS = {
    "musid": {
        "eval_csv": TEST5_DIR / "eval_results" / "degradation_results.csv",
        "out_dir": TEST5_DIR / "experiment5_results",
    },
    "smd": {
        "eval_csv": TEST5_DIR / "eval_results_smd" / "degradation_results.csv",
        "out_dir": TEST5_DIR / "experiment5_results_smd",
    },
    "buoy": {
        "eval_csv": TEST5_DIR / "eval_results_buoy" / "degradation_results.csv",
        "out_dir": TEST5_DIR / "experiment5_results_buoy",
    },
}

# Legacy compatibility
EVAL_CSV = TEST5_DIR / "eval_results" / "degradation_results.csv"
OUT_DIR = TEST5_DIR / "experiment5_results"
SUMMARY_MD = OUT_DIR / "summary_table.md"
SUMMARY_LATEX = OUT_DIR / "summary_table.tex"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# Degradation type grouping - 海洋场景分组
DEGRADATION_GROUPS = {
    "clean": "Baseline",
    # 基础退化
    "gaussian_noise_15": "Sensor Noise",
    "gaussian_noise_30": "Sensor Noise",
    "motion_blur_15": "Motion Blur",
    "motion_blur_25": "Motion Blur",
    "low_light_2.0": "Low Light",
    "low_light_2.5": "Low Light",
    "fog_0.3": "Fog/Haze",
    "fog_0.5": "Fog/Haze",
    # 海洋特有退化
    "rain_light": "Rain",
    "rain_medium": "Rain",
    "rain_heavy": "Rain",
    "glare_light": "Sun Glare",
    "glare_heavy": "Sun Glare",
    "jpeg_q20": "Compression",
    "jpeg_q10": "Compression",
    "lowres_0.5x": "Low Resolution",
    "lowres_0.25x": "Low Resolution",
}

# Display names - 中英文对照
DISPLAY_NAMES = {
    "clean": "Clean (基准)",
    # 基础退化
    "gaussian_noise_15": "噪声 σ=15",
    "gaussian_noise_30": "噪声 σ=30",
    "motion_blur_15": "运动模糊 k=15",
    "motion_blur_25": "运动模糊 k=25",
    "low_light_2.0": "低光照 γ=2.0",
    "low_light_2.5": "低光照 γ=2.5",
    "fog_0.3": "海雾 30%",
    "fog_0.5": "海雾 50%",
    # 海洋特有退化
    "rain_light": "小雨",
    "rain_medium": "中雨",
    "rain_heavy": "大雨",
    "glare_light": "轻度反光",
    "glare_heavy": "强反光",
    "jpeg_q20": "压缩 Q=20",
    "jpeg_q10": "压缩 Q=10",
    "lowres_0.5x": "低清 0.5x",
    "lowres_0.25x": "低清 0.25x",
}

# English display names for LaTeX
DISPLAY_NAMES_EN = {
    "clean": "Clean (Baseline)",
    "gaussian_noise_15": "Noise $\\sigma$=15",
    "gaussian_noise_30": "Noise $\\sigma$=30",
    "motion_blur_15": "Motion Blur k=15",
    "motion_blur_25": "Motion Blur k=25",
    "low_light_2.0": "Low Light $\\gamma$=2.0",
    "low_light_2.5": "Low Light $\\gamma$=2.5",
    "fog_0.3": "Fog 30\\%",
    "fog_0.5": "Fog 50\\%",
    "rain_light": "Rain (Light)",
    "rain_medium": "Rain (Medium)",
    "rain_heavy": "Rain (Heavy)",
    "glare_light": "Sun Glare (Light)",
    "glare_heavy": "Sun Glare (Heavy)",
    "jpeg_q20": "JPEG Q=20",
    "jpeg_q10": "JPEG Q=10",
    "lowres_0.5x": "Low-Res 0.5x",
    "lowres_0.25x": "Low-Res 0.25x",
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
    cfg = DATASET_CONFIGS[DATASET]
    eval_csv = cfg["eval_csv"]
    out_dir = cfg["out_dir"]
    summary_md = out_dir / "summary_table.md"
    summary_latex = out_dir / "summary_table.tex"
    
    print("=" * 60)
    print("Experiment 5: Generate Summary Tables")
    print(f"Dataset: {DATASET.upper()}")
    print("=" * 60)
    
    if not eval_csv.exists():
        print(f"[Error] Results file not found: {eval_csv}")
        print("Please run evaluate_degraded.py first")
        sys.exit(1)
    
    ensure_dir(out_dir)
    
    # Load results
    df = pd.read_csv(eval_csv)
    print(f"\n[Load] {len(df)} degradation results")
    
    # Sort by degradation type
    order = list(DISPLAY_NAMES.keys())
    df["sort_key"] = df["degradation"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])
    
    # Generate markdown
    md_content = generate_markdown(df)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Markdown -> {summary_md}")
    
    # Generate LaTeX
    latex_content = generate_latex(df)
    with open(summary_latex, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"[Saved] LaTeX -> {summary_latex}")
    
    # Print to console
    print("\n" + "=" * 60)
    print(md_content)
    print("=" * 60)


if __name__ == "__main__":
    main()
