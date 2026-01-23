import os
import re
import numpy as np
import pandas as pd

# ========= 无参配置 =========
CSV_PATH = os.path.join("eval_full_outputs", "full_eval_test.csv")

# 你实验1只要 CNN-only，所以这里默认选择 CNN-only 的列（优先含 cnn，其次无后缀）
PREFER_TAG = "cnn"          # 优先包含该词的列
AVOID_TAGS = ["final", "refine"]  # 尽量避免这些列（除非找不到）


# ========= 工具函数 =========
def _norm(s: str) -> str:
    return s.strip().lower()

def _quantile(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p))

def _stats(series: pd.Series):
    x = series.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return None
    return dict(
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        p90=_quantile(x, 0.90),
        p95=_quantile(x, 0.95),
        max=float(np.max(x)),
    )

def _pick_col(df: pd.DataFrame, include_all=None, include_any=None, exclude_any=None, prefer_contains=None):
    """
    自动选列：
    - include_all: 必须都包含的关键词（list[str]）
    - include_any: 至少包含其中一个关键词（list[str]）
    - exclude_any: 不能包含的关键词（list[str]）
    - prefer_contains: 优先包含的关键词（list[str]），用于打分
    """
    include_all = include_all or []
    include_any = include_any or []
    exclude_any = exclude_any or []
    prefer_contains = prefer_contains or []

    cols = list(df.columns)
    cols_l = [_norm(c) for c in cols]

    cand = []
    for c, cl in zip(cols, cols_l):
        ok = True
        for k in include_all:
            if k not in cl:
                ok = False
                break
        if not ok:
            continue
        if include_any:
            if not any(k in cl for k in include_any):
                continue
        if exclude_any:
            if any(k in cl for k in exclude_any):
                continue
        cand.append(c)

    if not cand:
        return None

    # 打分：优先含 prefer_contains，其次列名更短（更“基础”）
    def score(cname: str):
        cl = _norm(cname)
        s = 0
        for k in prefer_contains:
            if k in cl:
                s += 10
        # 避免 final/refine
        for bad in AVOID_TAGS:
            if bad in cl:
                s -= 5
        # 越短越基础
        s -= len(cl) * 0.01
        return s

    cand = sorted(cand, key=score, reverse=True)
    return cand[0]

def _print_table(title, rows):
    print(f"\n=== {title} ===")
    print("| 指标 | Mean | Median | P90 | P95 | Max |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, d in rows:
        if d is None:
            print(f"| {name} | NA | NA | NA | NA | NA |")
        else:
            print(f"| {name} | {d['mean']:.4f} | {d['median']:.4f} | {d['p90']:.4f} | {d['p95']:.4f} | {d['max']:.4f} |")

def _hit_rate(df, col, thr):
    x = df[col].to_numpy(dtype=float)
    return float(np.mean(x <= thr) * 100.0)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV：{CSV_PATH}（请确认工作目录是项目根目录）")

    df = pd.read_csv(CSV_PATH)
    print("Loaded:", CSV_PATH, "| N =", len(df))

    # 先打印列名，便于你核对
    print("\nColumns:")
    print(list(df.columns))

    # ========= 自动找 CNN-only 的列 =========
    # UNet 空间（1024x576）
    col_rho_unet = _pick_col(
        df,
        include_all=["rho"],
        include_any=["err", "error", "abs"],
        exclude_any=["gt", "pred", "orig", "original"],
        prefer_contains=[PREFER_TAG],
    )

    col_theta = _pick_col(
        df,
        include_all=["theta"],
        include_any=["err", "error", "deg", "angle"],
        exclude_any=["gt", "pred"],
        prefer_contains=[PREFER_TAG, "deg"],
    )

    col_linedist_unet = _pick_col(
        df,
        include_any=["line_dist", "linedist", "point", "p2l"],
        exclude_any=["orig", "original"],
        prefer_contains=[PREFER_TAG],
    )

    col_edgey_unet = _pick_col(
        df,
        include_any=["edge_y", "edgey"],
        exclude_any=["orig", "original"],
        prefer_contains=[PREFER_TAG],
    )

    # 原始尺度近似
    col_rho_orig = _pick_col(
        df,
        include_all=["rho"],
        include_any=["orig", "original"],
        exclude_any=["gt", "pred"],
        prefer_contains=[PREFER_TAG],
    )

    col_linedist_orig = _pick_col(
        df,
        include_any=["line_dist_orig", "linedist_orig", "line_dist", "linedist"],
        exclude_any=[],
        prefer_contains=[PREFER_TAG, "orig"],
    )

    col_edgey_orig = _pick_col(
        df,
        include_any=["edge_y_orig", "edgey_orig", "edge_y", "edgey"],
        exclude_any=[],
        prefer_contains=[PREFER_TAG, "orig"],
    )

    chosen = {
        "rho_unet": col_rho_unet,
        "theta_deg": col_theta,
        "line_dist_unet": col_linedist_unet,
        "edge_y_unet": col_edgey_unet,
        "rho_orig": col_rho_orig,
        "line_dist_orig": col_linedist_orig,
        "edge_y_orig": col_edgey_orig,
    }

    print("\n[Auto Column Mapping]")
    for k, v in chosen.items():
        print(f"{k:>16} -> {v}")

    # 如果找不到关键列，直接给提示
    must_have = ["rho_unet", "theta_deg", "line_dist_unet", "edge_y_unet"]
    missing = [k for k in must_have if chosen[k] is None]
    if missing:
        print("\n[ERROR] 找不到关键列：", missing)
        print("请把上面 Columns 列表发我，我会按你的真实列名再匹配一次。")
        return

    # ========= 生成实验1三张表 =========
    rows_unet = [
        ("|ρ| 绝对误差（px, UNet空间）", _stats(df[col_rho_unet])),
        ("θ 角度误差（deg, wrap-aware）", _stats(df[col_theta])),
        ("点到线平均距离 LineDist（px, UNet空间）", _stats(df[col_linedist_unet]) if col_linedist_unet else None),
        ("Edge-Y 误差（px, UNet空间）", _stats(df[col_edgey_unet]) if col_edgey_unet else None),
    ]
    _print_table("表6-1：CNN-only 基线性能（UNet空间 1024x576）", rows_unet)

    if col_rho_orig or col_linedist_orig or col_edgey_orig:
        rows_orig = []
        if col_rho_orig:
            rows_orig.append(("|ρ| 绝对误差（px, 原始尺度近似）", _stats(df[col_rho_orig])))
        if col_linedist_orig:
            rows_orig.append(("LineDist（px, 原始尺度近似）", _stats(df[col_linedist_orig])))
        if col_edgey_orig:
            rows_orig.append(("EdgeY（px, 原始尺度近似）", _stats(df[col_edgey_orig])))
        _print_table("表6-2：CNN-only 基线性能（原始尺度近似）", rows_orig)
    else:
        print("\n[WARN] 没找到 orig/original 相关列，将跳过原始尺度近似表（表6-2）。")

    # 阈值命中率（用原始尺度近似更直观；如果没有，就退回 UNet 空间）
    print("\n=== 表6-3：阈值命中率（CNN-only） ===")
    # theta
    for thr in [1, 2, 5]:
        hit = _hit_rate(df, col_theta, thr)
        print(f"Theta <= {thr}°: {hit:.2f}%")

    # position-like (prefer orig)
    def pick_or_fallback(primary, fallback):
        return primary if primary is not None else fallback

    col_rho_thr = pick_or_fallback(col_rho_orig, col_rho_unet)
    col_ld_thr = pick_or_fallback(col_linedist_orig, col_linedist_unet)
    col_ey_thr = pick_or_fallback(col_edgey_orig, col_edgey_unet)

    for col, name in [(col_rho_thr, "Rho（px）"), (col_ld_thr, "LineDist（px）"), (col_ey_thr, "EdgeY（px）")]:
        if col is None:
            continue
        for thr in [5, 10, 20]:
            hit = _hit_rate(df, col, thr)
            print(f"{name} <= {thr}: {hit:.2f}%   (using column: {col})")


if __name__ == "__main__":
    main()
