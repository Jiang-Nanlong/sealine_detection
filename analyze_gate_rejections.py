#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze refine gate rejection reasons (NO-ARGS / PyCharm-friendly).

Run this file directly in PyCharm (no command-line parameters needed).

Inputs:
  - One or more CSVs produced by evaluate_full_pipeline.py:
      eval_full_outputs/gate_stats.csv
    If you saved copies per experiment, set them in CONFIG.

Outputs (by default):
  - eval_full_outputs/gate_rejection_analysis/
      <tag>_gate_stats_with_reasons.csv
      <tag>_reason_lists/rejected_<reason>.txt
      rejection_summary.csv

Notes:
  - The script is robust to minor column-name differences.
  - If you set FOCUS="topk" but the CSV has no topk column, it falls back to all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================
# CONFIG (edit here if needed)
# ==============================

# Prefer analyzing saved copies per K (recommended).
# If they don't exist, the script will fall back to eval_full_outputs/gate_stats.csv.
CSV_PATHS: List[str] = [
    r"eval_full_outputs/gate_stats_k50.csv",
    r"eval_full_outputs/gate_stats_k100.csv",
]

# Focus on which subset:
#   - "topk" : analyze only Top-K samples (requires in_topk column)
#   - "all"  : analyze the entire split
FOCUS: str = "topk"

OUTDIR: str = r"eval_full_outputs/gate_rejection_analysis"

# Gate thresholds (match your 'best' settings)
LEAK_MAX: float = 0.05
RMSE_MAX: float = 4.0
INLIER_MIN: float = 0.83
REQUIRE_IMPROVE: bool = True
IMPROVE_MARGIN: float = 0.05

# ==============================


@dataclass
class Cols:
    idx: Optional[str]
    img: Optional[str]
    in_topk: Optional[str]
    used_ref: Optional[str]
    refine_ok: Optional[str]
    conf: Optional[str]
    leak: Optional[str]
    rmse: Optional[str]
    rmse_cnn: Optional[str]
    inlier: Optional[str]


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]
    return None


def detect_cols(df: pd.DataFrame) -> Cols:
    return Cols(
        idx=_pick_col(df, ["idx", "index", "sample_idx"]),
        img=_pick_col(df, ["img_name", "image", "filename", "name"]),
        in_topk=_pick_col(df, ["in_topk", "is_topk", "topk"]),
        used_ref=_pick_col(df, ["used_ref", "used_refine", "use_ref", "refined_used"]),
        refine_ok=_pick_col(df, ["refine_ok", "refine_success", "refine_done"]),
        conf=_pick_col(df, ["conf", "confidence", "score"]),
        leak=_pick_col(df, ["leak_ratio", "leak", "leakrate"]),
        rmse=_pick_col(df, ["boundary_rmse", "rmse", "rmse_refine", "boundaryrmse"]),
        rmse_cnn=_pick_col(df, ["boundary_rmse_cnn", "rmse_cnn", "boundaryrmse_cnn", "rmse_before"]),
        inlier=_pick_col(df, ["inlier_ratio", "inlier", "inliers", "ransac_inlier_ratio"]),
    )


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_attempted(df: pd.DataFrame, cols: Cols) -> pd.Series:
    """Infer whether refine was attempted.

    We treat 'attempted' as: any finite gate metric exists (leak/rmse/inlier).
    """
    metrics = []
    for c in [cols.leak, cols.rmse, cols.inlier]:
        if c is not None:
            metrics.append(to_num(df[c]))

    if not metrics:
        if cols.refine_ok is not None:
            return to_num(df[cols.refine_ok]).notna()
        return pd.Series(False, index=df.index)

    attempted = pd.Series(False, index=df.index)
    for s in metrics:
        attempted = attempted | np.isfinite(s)
    return attempted


def classify_rejections(
    df: pd.DataFrame,
    cols: Cols,
    leak_max: float,
    rmse_max: float,
    inlier_min: float,
    require_improve: bool,
    improve_margin: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cols.used_ref is None:
        raise RuntimeError(f"Cannot find used_ref column. Columns={list(df.columns)}")

    used_ref = to_num(df[cols.used_ref]).fillna(0).astype(int)
    refine_ok = to_num(df[cols.refine_ok]).fillna(0).astype(int) if cols.refine_ok else pd.Series(0, index=df.index)
    leak = to_num(df[cols.leak]) if cols.leak else pd.Series(np.nan, index=df.index)
    rmse = to_num(df[cols.rmse]) if cols.rmse else pd.Series(np.nan, index=df.index)
    inlier = to_num(df[cols.inlier]) if cols.inlier else pd.Series(np.nan, index=df.index)
    rmse_cnn = to_num(df[cols.rmse_cnn]) if cols.rmse_cnn else pd.Series(np.nan, index=df.index)

    attempted = compute_attempted(df, cols)

    fail_leak = attempted & np.isfinite(leak) & (leak > leak_max)
    fail_rmse = attempted & np.isfinite(rmse) & (rmse > rmse_max)
    fail_inlier = attempted & np.isfinite(inlier) & (inlier < inlier_min)

    if require_improve:
        target = (1.0 - improve_margin) * rmse_cnn
        fail_improve = attempted & (
            (~np.isfinite(rmse_cnn)) | (~np.isfinite(rmse)) | (rmse > target)
        )
    else:
        fail_improve = pd.Series(False, index=df.index)

    reason = pd.Series("", index=df.index, dtype=object)

    not_attempted = ~attempted
    rejected = attempted & (used_ref == 0)

    reason[not_attempted] = "not_attempted"

    # Assign a primary reason (ordered)
    remaining = rejected.copy()
    for mask, name in [
        (fail_leak, "fail_leak"),
        (fail_rmse, "fail_rmse"),
        (fail_inlier, "fail_inlier"),
        (fail_improve, "fail_improve"),
    ]:
        pick = remaining & mask
        reason[pick] = name
        remaining = remaining & (~pick)

    reason[remaining] = "other_reject"

    reason[used_ref == 1] = "used_ref"

    out = df.copy()
    out["attempted_refine"] = attempted.astype(int)
    out["reject_reason"] = reason
    out["fail_leak"] = fail_leak.astype(int)
    out["fail_rmse"] = fail_rmse.astype(int)
    out["fail_inlier"] = fail_inlier.astype(int)
    out["fail_improve"] = fail_improve.astype(int)

    # Summary
    summary_rows = []

    def add_summary(scope_name: str, mask_scope: pd.Series):
        sub = out[mask_scope]
        attempted_n = int(sub["attempted_refine"].sum())
        used_n = int((sub["reject_reason"] == "used_ref").sum())
        not_attempted_n = len(sub) - attempted_n
        rejected_n = int(((sub["attempted_refine"] == 1) & (to_num(sub[cols.used_ref]).fillna(0).astype(int) == 0)).sum())

        row = {
            "scope": scope_name,
            "N": len(sub),
            "attempted": attempted_n,
            "not_attempted": not_attempted_n,
            "used_ref": used_n,
            "rejected_after_attempt": rejected_n,
        }

        for r in ["fail_leak", "fail_rmse", "fail_inlier", "fail_improve", "other_reject"]:
            row[r] = int((sub["reject_reason"] == r).sum())

        summary_rows.append(row)

    add_summary("all", pd.Series(True, index=out.index))

    if cols.in_topk is not None:
        in_topk = to_num(out[cols.in_topk]).fillna(0).astype(int) == 1
        add_summary("topk", in_topk)
        add_summary("non_topk", ~in_topk)

    summary = pd.DataFrame(summary_rows)
    return out, summary


def save_reason_lists(df: pd.DataFrame, cols: Cols, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    img_col = cols.img
    if img_col is None:
        return

    for reason in sorted(df["reject_reason"].unique().tolist()):
        sub = df[df["reject_reason"] == reason]
        if len(sub) == 0:
            continue
        path = os.path.join(outdir, f"rejected_{reason}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for _, r in sub.iterrows():
                idx_val = r.get(cols.idx, "") if cols.idx else ""
                f.write(f"{idx_val}\t{r[img_col]}\n")


def _resolve_csv_paths(paths: List[str]) -> List[str]:
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        return existing

    fallback = r"eval_full_outputs/gate_stats.csv"
    if os.path.exists(fallback):
        print(f"[Info] None of CSV_PATHS exist. Falling back to: {fallback}")
        return [fallback]

    raise FileNotFoundError(
        "Cannot find any gate_stats CSV. Please set CSV_PATHS in CONFIG or generate gate_stats.csv first."
    )


def run_one(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    cols = detect_cols(df)

    out, summary = classify_rejections(
        df,
        cols,
        leak_max=LEAK_MAX,
        rmse_max=RMSE_MAX,
        inlier_min=INLIER_MIN,
        require_improve=REQUIRE_IMPROVE,
        improve_margin=IMPROVE_MARGIN,
    )

    tag = os.path.splitext(os.path.basename(csv_path))[0]

    # Focus filter
    out_focus = out
    if FOCUS.lower() == "topk":
        if cols.in_topk is None:
            print(f"[Warn] {tag}: no in_topk column found; FOCUS=topk -> using ALL samples.")
        else:
            in_topk = to_num(out[cols.in_topk]).fillna(0).astype(int) == 1
            out_focus = out[in_topk].copy()

    # Save outputs
    os.makedirs(OUTDIR, exist_ok=True)
    out_csv = os.path.join(OUTDIR, f"{tag}_gate_stats_with_reasons.csv")
    out_focus.to_csv(out_csv, index=False)

    save_reason_lists(out_focus, cols, os.path.join(OUTDIR, f"{tag}_reason_lists"))

    print(f"\n=== {tag} ({FOCUS}) ===")
    print(summary.to_string(index=False))
    print(f"[Saved] {out_csv}")

    summary2 = summary.copy()
    summary2.insert(0, "run", tag)

    return out_focus, summary2, tag


def main():
    paths = _resolve_csv_paths(CSV_PATHS)
    print("\n=== Gate rejection analysis (no-args) ===")
    print(f"FOCUS = {FOCUS}")
    print(f"CSV files:")
    for p in paths:
        print(f"  - {p}")
    print(
        f"Thresholds: leak<={LEAK_MAX}, rmse<={RMSE_MAX}, inlier>={INLIER_MIN}, "
        f"require_improve={REQUIRE_IMPROVE}, improve_margin={IMPROVE_MARGIN}"
    )

    summaries = []
    for p in paths:
        _, s, _ = run_one(p)
        summaries.append(s)

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_path = os.path.join(OUTDIR, "rejection_summary.csv")
    summary_all.to_csv(summary_path, index=False)
    print(f"\n[Saved] combined summary -> {summary_path}")


if __name__ == "__main__":
    main()
