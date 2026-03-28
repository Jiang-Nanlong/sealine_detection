import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = THIS_DIR / "evaluate_full_pipeline.py"

# 你要扫的 K
K_LIST = list(range(10, 101, 10))

# 总输出目录（会在里面生成 k010, k020 ... 子目录）
BASE_OUT = THIS_DIR / "eval_sweep_topk"
BASE_OUT.mkdir(parents=True, exist_ok=True)

def pct_le(arr, t):
    arr = np.asarray(arr, dtype=np.float64)
    return 100.0 * float(np.mean(arr <= t))

def stats6(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}\nAvailable columns:\n{list(df.columns)}")

def summarize_one_run(csv_path: Path, k: int):
    df = pd.read_csv(csv_path)

    # 尽量兼容你的列名（你现在 full_eval_test.csv 里基本就是这些）
    col_rho_cnn   = pick_col(df, ["rho_abs_px_orig_cnn"])
    col_ld_cnn    = pick_col(df, ["line_dist_px_orig_cnn"])
    col_ey_cnn    = pick_col(df, ["edgey_px_orig_cnn", "edgey_px_orig_cnn"])  # 兼容
    col_th_cnn    = pick_col(df, ["theta_abs_deg_cnn"])

    col_rho_final = pick_col(df, ["rho_abs_px_orig_final"])
    col_ld_final  = pick_col(df, ["line_dist_px_orig_final"])
    col_ey_final  = pick_col(df, ["edgey_px_orig_final"])
    col_th_final  = pick_col(df, ["theta_abs_deg_final"])

    used_ref = int(df["used_ref"].sum()) if "used_ref" in df.columns else None

    # 整体统计（Final）
    s_rho_f = stats6(df[col_rho_final].values)
    s_ld_f  = stats6(df[col_ld_final].values)
    s_ey_f  = stats6(df[col_ey_final].values)
    s_th_f  = stats6(df[col_th_final].values)

    # 整体统计（CNN）
    s_rho_c = stats6(df[col_rho_cnn].values)
    s_ld_c  = stats6(df[col_ld_cnn].values)
    s_ey_c  = stats6(df[col_ey_cnn].values)
    s_th_c  = stats6(df[col_th_cnn].values)

    row = {
        "K": k,
        "N": len(df),
        "used_ref": used_ref,

        # Final - mean (主看这几个就够写表/画图)
        "rho_mean_final": s_rho_f["mean"],
        "ld_mean_final":  s_ld_f["mean"],
        "ey_mean_final":  s_ey_f["mean"],
        "th_mean_final":  s_th_f["mean"],

        # CNN - mean（baseline，不随K变，但保留一列方便对比）
        "rho_mean_cnn": s_rho_c["mean"],
        "ld_mean_cnn":  s_ld_c["mean"],
        "ey_mean_cnn":  s_ey_c["mean"],
        "th_mean_cnn":  s_th_c["mean"],

        # Threshold 命中率（Final）
        "rho<=5_final(%)":  pct_le(df[col_rho_final].values, 5.0),
        "rho<=10_final(%)": pct_le(df[col_rho_final].values, 10.0),
        "ld<=5_final(%)":   pct_le(df[col_ld_final].values, 5.0),
        "ld<=10_final(%)":  pct_le(df[col_ld_final].values, 10.0),
        "ey<=5_final(%)":   pct_le(df[col_ey_final].values, 5.0),
        "ey<=10_final(%)":  pct_le(df[col_ey_final].values, 10.0),
        "th<=1_final(%)":   pct_le(df[col_th_final].values, 1.0),
        "th<=2_final(%)":   pct_le(df[col_th_final].values, 2.0),
    }

    return row

def run_one_k(k: int):
    out_dir = BASE_OUT / f"k{k:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OUT_DIR"] = str(out_dir)                 # 关键：避免覆盖
    env["TOPK_AUTO_K"] = str(k)                   # 关键：本次K
    env["TOPK_IGNORE_CONF_GATE"] = "1"            # 你论文主结果用 ignore_conf=True（你已经验证过更有效）

    cmd = [sys.executable, str(EVAL_SCRIPT)]
    print(f"\n===== Running K={k} =====")
    subprocess.run(cmd, env=env, check=True)

    csv_path = out_dir / "full_eval_test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected: {csv_path}")

    return csv_path

def main():
    rows = []
    for k in K_LIST:
        csv_path = run_one_k(k)
        rows.append(summarize_one_run(csv_path, k))

    df_sum = pd.DataFrame(rows).sort_values("K")
    out_csv = BASE_OUT / "sweep_topk_summary.csv"
    df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 同时输出一个 markdown 方便你直接贴到论文草稿里
    out_md = BASE_OUT / "sweep_topk_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df_sum.to_markdown(index=False))

    print("\n[SAVED]")
    print(out_csv)
    print(out_md)
    print("\nDone.")

if __name__ == "__main__":
    main()
