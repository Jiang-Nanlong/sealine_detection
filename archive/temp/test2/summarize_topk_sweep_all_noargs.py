import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) 固定路径（无参版本）
# =========================
ROOT = Path("../eval_sweep_topk")  # 你的 sweep 输出根目录（截图里就是这个）


# =========================
# 1) 这里写死 gate 阈值（与 evaluate_full_pipeline.py 保持一致）
#    如果你以后改了阈值，只改这里即可
# =========================
GATE_LEAK_RATIO_MAX = 0.05
GATE_BOUNDARY_RMSE_MAX = 4.0
RANSAC_MIN_INLIER_RATIO = 0.83
GATE_REQUIRE_RMSE_IMPROVEMENT = True
RMSE_IMPROVE_MARGIN = 0.05


# =========================
# 2) 工具函数
# =========================
def _to_float_series(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def stats_series(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return dict(mean=np.nan, median=np.nan, p90=np.nan, p95=np.nan, max=np.nan)
    return dict(
        mean=float(x.mean()),
        median=float(x.median()),
        p90=float(np.percentile(x, 90)),
        p95=float(np.percentile(x, 95)),
        max=float(x.max()),
    )


def pct_le(x: pd.Series, thr: float):
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return np.nan
    return 100.0 * float((x <= thr).mean())


def markdown_table(df: pd.DataFrame, floatfmt="{:.4f}"):
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


def find_k_folders(root: Path):
    out = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^k(\d+)$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def infer_attempted(gate_df: pd.DataFrame):
    """
    gate_stats.csv 通常是 per-sample 表。
    attempted 的定义：只要 leak_ratio/boundary_rmse/inlier_ratio 任意一个有数，就说明 UNet+边界+RANSAC 流程跑过（attempted）。
    """
    cols = [c for c in ["leak_ratio", "boundary_rmse", "inlier_ratio"] if c in gate_df.columns]
    if not cols:
        # 兜底：如果存在 refine_ok/used_ref 就认为 attempted=全部
        return pd.Series([True] * len(gate_df))
    mask = None
    for c in cols:
        m = pd.to_numeric(gate_df[c], errors="coerce").notna()
        mask = m if mask is None else (mask | m)
    return mask


def compute_rejections(gate_df: pd.DataFrame):
    """
    在 attempted==1 且 used_ref==0 的样本中，按阈值规则推断拒绝原因。
    返回 dict 统计计数。
    """
    n = len(gate_df)
    if n == 0:
        return dict(
            attempted=0, not_attempted=0,
            used_ref=0, rejected_after_attempt=0,
            fail_leak=0, fail_rmse=0, fail_inlier=0, fail_improve=0, other_reject=0
        )

    attempted = infer_attempted(gate_df)
    attempted_n = int(attempted.sum())
    not_attempted_n = int((~attempted).sum())

    used_ref = int(pd.to_numeric(gate_df.get("used_ref", 0), errors="coerce").fillna(0).astype(int).sum())

    # rejected after attempt: attempted but not used_ref
    df_att = gate_df[attempted].copy()
    if "used_ref" in df_att.columns:
        rej_mask = (pd.to_numeric(df_att["used_ref"], errors="coerce").fillna(0).astype(int) == 0)
    else:
        # 如果没有 used_ref，就无法精确判断是否最终使用 refine，这里当做 0
        rej_mask = pd.Series([True] * len(df_att))

    df_rej = df_att[rej_mask].copy()
    rejected_after_attempt = len(df_rej)

    # 取必要列
    leak = pd.to_numeric(df_rej.get("leak_ratio", np.nan), errors="coerce")
    rmse = pd.to_numeric(df_rej.get("boundary_rmse", np.nan), errors="coerce")
    inlier = pd.to_numeric(df_rej.get("inlier_ratio", np.nan), errors="coerce")

    # 改进门控需要 boundary_rmse_cnn（CNN 在同一批 boundary 点上的 rmse）
    rmse_cnn = pd.to_numeric(df_rej.get("boundary_rmse_cnn", np.nan), errors="coerce")

    # 按顺序归因（和你之前 analyze_gate_rejections 的逻辑一致）
    fail_leak = (leak.notna()) & (leak > GATE_LEAK_RATIO_MAX)
    fail_rmse = (~fail_leak) & (rmse.notna()) & (rmse > GATE_BOUNDARY_RMSE_MAX)
    fail_inlier = (~fail_leak) & (~fail_rmse) & (inlier.notna()) & (inlier < RANSAC_MIN_INLIER_RATIO)

    if GATE_REQUIRE_RMSE_IMPROVEMENT:
        # refine_rmse <= (1 - margin) * cnn_rmse 才算通过，否则 fail_improve
        # 需要 rmse 和 rmse_cnn 都有效
        cond_valid = rmse.notna() & rmse_cnn.notna()
        improved_ok = rmse <= (1.0 - RMSE_IMPROVE_MARGIN) * rmse_cnn
        fail_improve = (~fail_leak) & (~fail_rmse) & (~fail_inlier) & cond_valid & (~improved_ok)
    else:
        fail_improve = pd.Series([False] * len(df_rej))

    other = (~fail_leak) & (~fail_rmse) & (~fail_inlier) & (~fail_improve)

    return dict(
        attempted=attempted_n,
        not_attempted=not_attempted_n,
        used_ref=used_ref,
        rejected_after_attempt=rejected_after_attempt,
        fail_leak=int(fail_leak.sum()),
        fail_rmse=int(fail_rmse.sum()),
        fail_inlier=int(fail_inlier.sum()),
        fail_improve=int(fail_improve.sum()),
        other_reject=int(other.sum()),
    )


# =========================
# 3) 主流程
# =========================
def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"找不到目录：{ROOT.resolve()}（请确认 PyCharm 的 Working Directory 在工程根目录）")

    k_folders = find_k_folders(ROOT)
    if not k_folders:
        raise RuntimeError(f"在 {ROOT.resolve()} 下没有找到 k010/k020... 这种子目录")

    rows = []
    rej_rows = []

    for k, d in k_folders:
        full_csv = d / "full_eval_test.csv"
        gate_csv = d / "gate_stats.csv"

        if not full_csv.exists():
            print(f"[Skip] missing {full_csv}")
            continue

        df = pd.read_csv(full_csv)

        # 核心列（与你当前 full_eval_test.csv 一致）
        rho_cnn = _to_float_series(df, "rho_abs_px_orig_cnn")
        rho_final = _to_float_series(df, "rho_abs_px_orig_final")
        ld_cnn = _to_float_series(df, "line_dist_px_orig_cnn")
        ld_final = _to_float_series(df, "line_dist_px_orig_final")
        ey_cnn = _to_float_series(df, "edgey_px_orig_cnn")
        ey_final = _to_float_series(df, "edgey_px_orig_final")
        th_final = _to_float_series(df, "theta_abs_deg_final")

        n = len(df)
        used_ref_full = int(pd.to_numeric(df.get("used_ref", 0), errors="coerce").fillna(0).astype(int).sum())
        refine_rate = used_ref_full / n if n > 0 else np.nan

        # topk 子集统计（如果存在 in_topk）
        if "in_topk" in df.columns:
            df_topk = df[df["in_topk"] == 1].copy()
        else:
            df_topk = df.iloc[0:0].copy()

        ey_topk_cnn = _to_float_series(df_topk, "edgey_px_orig_cnn")
        ey_topk_final = _to_float_series(df_topk, "edgey_px_orig_final")
        topk_used_ref = int(pd.to_numeric(df_topk.get("used_ref", 0), errors="coerce").fillna(0).astype(int).sum()) if len(df_topk) else 0

        # gate 拒绝统计
        rej = None
        if gate_csv.exists():
            gdf = pd.read_csv(gate_csv)
            rej = compute_rejections(gdf)
        else:
            rej = dict(
                attempted=np.nan, not_attempted=np.nan,
                used_ref=np.nan, rejected_after_attempt=np.nan,
                fail_leak=np.nan, fail_rmse=np.nan, fail_inlier=np.nan, fail_improve=np.nan, other_reject=np.nan
            )

        # overall stats
        s_rho_f = stats_series(rho_final)
        s_ld_f = stats_series(ld_final)
        s_ey_f = stats_series(ey_final)

        # 构建汇总行
        row = dict(
            K=k,
            N=n,
            used_ref=used_ref_full,
            refine_rate=refine_rate,

            rho_mean_final=s_rho_f["mean"],
            rho_p95_final=s_rho_f["p95"],
            lined_mean_final=s_ld_f["mean"],
            edgey_mean_final=s_ey_f["mean"],
            edgey_p95_final=s_ey_f["p95"],

            rho_mean_cnn=float(rho_cnn.mean()),
            lined_mean_cnn=float(ld_cnn.mean()),
            edgey_mean_cnn=float(ey_cnn.mean()),

            delta_edgey_mean=float(ey_final.mean() - ey_cnn.mean()),

            # 阈值命中率（最终结果）
            rho_le_5_pct=pct_le(rho_final, 5.0),
            rho_le_10_pct=pct_le(rho_final, 10.0),
            edgey_le_5_pct=pct_le(ey_final, 5.0),
            edgey_le_10_pct=pct_le(ey_final, 10.0),
            theta_le_1_pct=pct_le(th_final, 1.0),
            theta_le_2_pct=pct_le(th_final, 2.0),

            # TopK 子集
            topk_n=len(df_topk),
            topk_used_ref=topk_used_ref,
            topk_edgey_mean_cnn=float(ey_topk_cnn.mean()) if len(df_topk) else np.nan,
            topk_edgey_mean_final=float(ey_topk_final.mean()) if len(df_topk) else np.nan,
            topk_edgey_delta=float(ey_topk_final.mean() - ey_topk_cnn.mean()) if len(df_topk) else np.nan,
        )
        row.update({f"rej_{k2}": v for k2, v in rej.items()})
        rows.append(row)

        # 另存一份 reject 明细表
        rej_rows.append(dict(K=k, **rej))

    df_sum = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    df_rej = pd.DataFrame(rej_rows).sort_values("K").reset_index(drop=True)

    # 输出文件
    out_csv = ROOT / "summary_topk_sweep_all.csv"
    out_md = ROOT / "summary_topk_sweep_all.md"
    out_rej = ROOT / "gate_rejections_detail.csv"
    df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_rej.to_csv(out_rej, index=False, encoding="utf-8-sig")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Top-K sweep 汇总（含 gate 拒绝原因）\n\n")
        f.write(markdown_table(df_sum))

    # =========================
    # 4) 画图（PNG）
    # =========================
    # 图1：overall EdgeY mean（CNN vs Final）
    fig = plt.figure()
    plt.plot(df_sum["K"], df_sum["edgey_mean_cnn"], marker="o")
    plt.plot(df_sum["K"], df_sum["edgey_mean_final"], marker="o")
    plt.xlabel("K")
    plt.ylabel("EdgeY mean (px, orig)")
    plt.title("K sweep (overall): EdgeY mean")
    plt.legend(["CNN", "Final"])
    fig.savefig(ROOT / "plot_edgey_mean_overall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 图2：refine_rate
    fig = plt.figure()
    plt.plot(df_sum["K"], df_sum["refine_rate"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Refine rate (used_ref / N)")
    plt.title("K sweep: refinement trigger rate")
    fig.savefig(ROOT / "plot_refine_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 图3：TopK 子集 EdgeY mean（CNN vs Final）
    fig = plt.figure()
    plt.plot(df_sum["K"], df_sum["topk_edgey_mean_cnn"], marker="o")
    plt.plot(df_sum["K"], df_sum["topk_edgey_mean_final"], marker="o")
    plt.xlabel("K")
    plt.ylabel("TopK EdgeY mean (px, orig)")
    plt.title("K sweep (TopK subset): EdgeY mean")
    plt.legend(["CNN", "Final"])
    fig.savefig(ROOT / "plot_topk_edgey_mean.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 图4：拒绝原因堆叠柱状图（attempted 里 rejected 的构成）
    # 只画有数字的行
    if df_rej.shape[0] > 0 and pd.to_numeric(df_rej["rejected_after_attempt"], errors="coerce").notna().any():
        fig = plt.figure()
        K = df_rej["K"].values
        a = pd.to_numeric(df_rej["fail_leak"], errors="coerce").fillna(0).values
        b = pd.to_numeric(df_rej["fail_rmse"], errors="coerce").fillna(0).values
        c = pd.to_numeric(df_rej["fail_inlier"], errors="coerce").fillna(0).values
        d = pd.to_numeric(df_rej["fail_improve"], errors="coerce").fillna(0).values
        e = pd.to_numeric(df_rej["other_reject"], errors="coerce").fillna(0).values

        bottom = np.zeros_like(K, dtype=float)
        plt.bar(K, a, bottom=bottom, label="fail_leak")
        bottom += a
        plt.bar(K, b, bottom=bottom, label="fail_rmse")
        bottom += b
        plt.bar(K, c, bottom=bottom, label="fail_inlier")
        bottom += c
        plt.bar(K, d, bottom=bottom, label="fail_improve")
        bottom += d
        plt.bar(K, e, bottom=bottom, label="other_reject")

        plt.xlabel("K")
        plt.ylabel("Count (rejected after attempt)")
        plt.title("Gate rejection breakdown by K")
        plt.legend()
        fig.savefig(ROOT / "plot_gate_rejections_stacked.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("[Saved]", out_csv)
    print("[Saved]", out_md)
    print("[Saved]", out_rej)
    print("[Saved] plot_edgey_mean_overall.png / plot_refine_rate.png / plot_topk_edgey_mean.png / plot_gate_rejections_stacked.png")


if __name__ == "__main__":
    main()
