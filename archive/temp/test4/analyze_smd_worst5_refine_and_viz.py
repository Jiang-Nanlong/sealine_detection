# test4/analyze_smd_worst5_refine_and_viz.py
# -*- coding: utf-8 -*-
"""
SMD Experiment A - Diagnostics & Refinement Script (PyCharm-friendly, no argparse)

What it does:
1) Find worst 5% samples by line_dist_px_unet from eval_smd_test_per_sample.csv
2) Identify top videos contributing to worst 5% and export visualization frames with overlays:
   - GT line (green)
   - Pred line (red)
   - Refined line (blue)
3) Perform a lightweight post-processing refinement for worst 5%:
   - Use the last channel (seg_sino) in FusionCache input (Radon sinogram 2240x180)
   - Local argmax search around predicted (rho_idx, theta_idx) to snap to strongest Radon evidence
   - Recompute errors and save a refined CSV + summary stats

Assumptions (match your pipeline):
- Cache: test4/FusionCache_SMD_1024x576/test/{idx}.npy
- Frames: test4/smd_frames/{img_name}
- CSV: test4/eval_smd_test_per_sample.csv
- UNet size: 1024x576
- Radon sinogram size: 2240x180
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import cv2

# =========================
# CONFIG (edit in PyCharm)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PER_SAMPLE_CSV = PROJECT_ROOT / "test4" / "eval_smd_test_per_sample.csv"
CACHE_TEST_DIR = PROJECT_ROOT / "test4" / "FusionCache_SMD_1024x576" / "test"
FRAMES_DIR = PROJECT_ROOT / "test4" / "smd_frames"

OUT_DIR = PROJECT_ROOT / "test4" / "analysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Worst tail definition
WORST_PERCENT = 0.05  # 5%

# Refinement search window in sinogram index space
# (rho is vertical: 0..2239, theta is horizontal: 0..179)
DELTA_RHO = 60
DELTA_THETA = 8

# How many videos to visualize (top contributors to worst tail)
TOP_VIDEOS = 3

# For each selected video, how many frames to export
FRAMES_PER_VIDEO_WORST = 8   # pick worst frames in that video
FRAMES_PER_VIDEO_RANDOM = 4  # plus a few random frames from that video

# Line metric sampling points (unet space)
N_SAMPLES = 50

# Denorm config (must match your cache label definition)
UNET_W = 1024
UNET_H = 576
RESIZE_H = 2240
RESIZE_W = 180

# =========================


def theta_wrap_dist_deg(a: float, b: float, period: float = 180.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def denorm_label_to_rho_theta(rho_norm: float, theta_norm: float) -> Tuple[float, float]:
    """
    Inverse of make_fusion_cache_smd.calculate_radon_label() output.
    Returns:
      rho: in UNet-centered coordinate system for line equation:
           (x-cx)*cos(theta) + (y-cy)*sin(theta) = rho
      theta_deg: [0, 180)
    """
    theta_deg = float(theta_norm) * 180.0

    original_diag = math.sqrt(UNET_W ** 2 + UNET_H ** 2)
    pad_top = (RESIZE_H - original_diag) / 2.0

    final_rho_idx = float(rho_norm) * (RESIZE_H - 1)
    rho_pixel_pos = final_rho_idx - pad_top
    rho = rho_pixel_pos - original_diag / 2.0
    return rho, theta_deg


def line_y_at_x(rho: float, theta_deg: float, x: float, w: int, h: int, eps: float = 1e-8) -> float:
    """
    Compute y on the line at given x in image coordinates [0,w-1] with center (cx,cy).
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    if abs(sin_t) < eps:
        return cy
    return cy + (rho - ((x - cx) * cos_t)) / sin_t


def line_distance_px(rho_gt: float, theta_gt_deg: float, rho_pr: float, theta_pr_deg: float,
                     w: int, h: int, n_samples: int = 50) -> float:
    """
    Distance metric consistent with evaluation: sample points along GT line (within x-range)
    and compute mean orthogonal distance to predicted line.
    """
    theta_p = math.radians(theta_pr_deg)
    cos_p, sin_p = math.cos(theta_p), math.sin(theta_p)

    cx, cy = w / 2.0, h / 2.0
    # Choose two points on GT line within [0,w-1]
    x0, x1 = 0.0, w - 1.0
    y0 = float(np.clip(line_y_at_x(rho_gt, theta_gt_deg, x0, w, h), 0, h - 1))
    y1 = float(np.clip(line_y_at_x(rho_gt, theta_gt_deg, x1, w, h), 0, h - 1))

    xs = np.linspace(x0, x1, n_samples, dtype=np.float64)
    ys = np.linspace(y0, y1, n_samples, dtype=np.float64)

    # signed distance to predicted line: (x-cx)*cos + (y-cy)*sin - rho
    d = (xs - cx) * cos_p + (ys - cy) * sin_p - rho_pr
    return float(np.mean(np.abs(d)))


def intersections_for_draw(rho: float, theta_deg: float, w: int, h: int, eps: float = 1e-8) -> List[Tuple[int, int]]:
    """
    Compute up to 2 boundary intersections to draw the line segment.
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    pts = []

    # x=0
    if abs(sin_t) > eps:
        y = cy + (rho - ((0 - cx) * cos_t)) / sin_t
        if -1 <= y <= h:
            pts.append((0.0, float(y)))

    # x=w-1
    if abs(sin_t) > eps:
        x = w - 1.0
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if -1 <= y <= h:
            pts.append((float(x), float(y)))

    # y=0
    if abs(cos_t) > eps:
        x = cx + (rho - ((0 - cy) * sin_t)) / cos_t
        if -1 <= x <= w:
            pts.append((float(x), 0.0))

    # y=h-1
    if abs(cos_t) > eps:
        y = h - 1.0
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if -1 <= x <= w:
            pts.append((float(x), float(y)))

    # unique
    uniq = []
    for p in pts:
        if all(abs(p[0]-q[0]) > 1e-6 or abs(p[1]-q[1]) > 1e-6 for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return []

    # farthest two
    best = (uniq[0], uniq[1])
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            dx = uniq[i][0] - uniq[j][0]
            dy = uniq[i][1] - uniq[j][1]
            d = dx*dx + dy*dy
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])

    (x1, y1), (x2, y2) = best
    x1 = int(np.clip(round(x1), 0, w-1))
    y1 = int(np.clip(round(y1), 0, h-1))
    x2 = int(np.clip(round(x2), 0, w-1))
    y2 = int(np.clip(round(y2), 0, h-1))
    return [(x1, y1), (x2, y2)]


def parse_video_id(img_name: str) -> str:
    # img_name format: Domain__VideoStem__000123.jpg
    parts = img_name.split("__")
    if len(parts) >= 3:
        return parts[1]
    return "UnknownVideo"


def refine_one_by_sinogram(input_tensor: np.ndarray,
                           rho_pred_norm: float,
                           theta_pred_norm: float,
                           delta_rho: int = 60,
                           delta_theta: int = 8) -> Tuple[float, float]:
    """
    input_tensor: (C, 2240, 180), last channel is seg_sino.
    returns refined (rho_norm, theta_norm) in [0,1].
    """
    if input_tensor.ndim != 3:
        return rho_pred_norm, theta_pred_norm
    seg_sino = input_tensor[-1]  # (2240,180)

    rho_idx = int(round(float(rho_pred_norm) * (RESIZE_H - 1)))
    th_idx = int(round(float(theta_pred_norm) * (RESIZE_W - 1)))

    r0 = max(0, rho_idx - delta_rho)
    r1 = min(RESIZE_H - 1, rho_idx + delta_rho)
    t0 = max(0, th_idx - delta_theta)
    t1 = min(RESIZE_W - 1, th_idx + delta_theta)

    window = seg_sino[r0:r1+1, t0:t1+1]
    if window.size == 0:
        return rho_pred_norm, theta_pred_norm

    # argmax in window
    flat = int(np.argmax(window))
    rr, tt = np.unravel_index(flat, window.shape)
    best_r = r0 + int(rr)
    best_t = t0 + int(tt)

    rho_ref = best_r / (RESIZE_H - 1)
    th_ref = best_t / (RESIZE_W - 1)
    return float(rho_ref), float(th_ref)


def draw_overlay(img_bgr: np.ndarray,
                 gt: Tuple[float, float],
                 pr: Tuple[float, float],
                 rf: Tuple[float, float] | None = None,
                 thickness: int = 2) -> np.ndarray:
    """
    Draw GT(red?) We'll use:
      GT: green
      Pred: red
      Refined: blue
    All in UNet space -> we draw in original image by scaling intersections.
    """
    H0, W0 = img_bgr.shape[:2]
    scale_x = W0 / UNET_W
    scale_y = H0 / UNET_H

    out = img_bgr.copy()

    def draw_one(rho, theta_deg, color):
        pts = intersections_for_draw(rho, theta_deg, UNET_W, UNET_H)
        if len(pts) != 2:
            return
        (x1, y1), (x2, y2) = pts
        p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
        p2 = (int(round(x2 * scale_x)), int(round(y2 * scale_y)))
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)

    rho_gt, th_gt = gt
    rho_pr, th_pr = pr
    draw_one(rho_gt, th_gt, (0, 255, 0))   # green
    draw_one(rho_pr, th_pr, (0, 0, 255))   # red
    if rf is not None:
        rho_rf, th_rf = rf
        draw_one(rho_rf, th_rf, (255, 0, 0))  # blue

    return out


def main():
    print("=== Analyze SMD worst tail + refine + visualize ===")
    print("PER_SAMPLE_CSV:", PER_SAMPLE_CSV)
    print("CACHE_TEST_DIR:", CACHE_TEST_DIR)
    print("FRAMES_DIR    :", FRAMES_DIR)
    print("OUT_DIR       :", OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PER_SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Per-sample CSV not found: {PER_SAMPLE_CSV}")
    if not CACHE_TEST_DIR.exists():
        raise FileNotFoundError(f"Cache test dir not found: {CACHE_TEST_DIR}")
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames dir not found: {FRAMES_DIR}")

    df = pd.read_csv(PER_SAMPLE_CSV)
    if "line_dist_px_unet" not in df.columns:
        raise KeyError("CSV missing 'line_dist_px_unet' column.")

    # Worst tail by lineDist
    thr = float(df["line_dist_px_unet"].quantile(1.0 - WORST_PERCENT))
    worst = df[df["line_dist_px_unet"] >= thr].copy()
    print(f"Worst {int(WORST_PERCENT*100)}% threshold: {thr:.4f} px, count={len(worst)} / {len(df)}")

    # Top videos in worst tail
    worst["video"] = worst["img_name"].apply(parse_video_id)
    top_videos = (worst["video"].value_counts().head(TOP_VIDEOS)).to_dict()
    print("Top videos in worst tail:", top_videos)

    # ---------------------------
    # Refinement on worst tail
    # ---------------------------
    refined_rows = []
    improved = 0

    for _, row in worst.iterrows():
        idx = int(row["idx"])
        cache_path = CACHE_TEST_DIR / f"{idx}.npy"
        if not cache_path.exists():
            continue
        data = np.load(str(cache_path), allow_pickle=True).item()
        inp = data["input"]  # (C,2240,180)

        rho_gt_norm = float(row["rho_gt_norm"])
        th_gt_norm = float(row["theta_gt_norm"])
        rho_pr_norm = float(row["rho_pred_norm"])
        th_pr_norm = float(row["theta_pred_norm"])

        rho_rf_norm, th_rf_norm = refine_one_by_sinogram(inp, rho_pr_norm, th_pr_norm,
                                                         delta_rho=DELTA_RHO, delta_theta=DELTA_THETA)

        rho_gt, th_gt = denorm_label_to_rho_theta(rho_gt_norm, th_gt_norm)
        rho_pr, th_pr = denorm_label_to_rho_theta(rho_pr_norm, th_pr_norm)
        rho_rf, th_rf = denorm_label_to_rho_theta(rho_rf_norm, th_rf_norm)

        dist_before = float(row["line_dist_px_unet"])
        dist_after = line_distance_px(rho_gt, th_gt, rho_rf, th_rf, UNET_W, UNET_H, n_samples=N_SAMPLES)

        if dist_after < dist_before:
            improved += 1

        refined_rows.append({
            "idx": idx,
            "img_name": row["img_name"],
            "domain": row["domain"],
            "video": row["video"],
            "rho_gt_norm": rho_gt_norm,
            "theta_gt_norm": th_gt_norm,
            "rho_pred_norm": rho_pr_norm,
            "theta_pred_norm": th_pr_norm,
            "line_dist_before": dist_before,
            "rho_refined_norm": rho_rf_norm,
            "theta_refined_norm": th_rf_norm,
            "line_dist_after": dist_after,
            "delta": dist_after - dist_before,
        })

    refined_df = pd.DataFrame(refined_rows)
    refined_csv = OUT_DIR / "worst5_refined.csv"
    refined_df.to_csv(refined_csv, index=False, encoding="utf-8")
    print(f"Saved refined worst-tail CSV: {refined_csv}")
    print(f"Refinement improved {improved}/{len(refined_df)} samples ({(improved/max(1,len(refined_df)))*100:.2f}%).")

    # Summary stats
    def summary(arr: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(arr, dtype=np.float64)
        return {
            "N": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
            "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "max": float(np.max(arr)) if arr.size else 0.0,
        }

    before_stats = summary(refined_df["line_dist_before"].values)
    after_stats = summary(refined_df["line_dist_after"].values)

    # Domain stats
    domain_stats = {}
    for d in refined_df["domain"].unique():
        sub = refined_df[refined_df["domain"] == d]
        domain_stats[d] = {
            "before": summary(sub["line_dist_before"].values),
            "after": summary(sub["line_dist_after"].values),
        }

    summary_path = OUT_DIR / "refine_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"WorstPercent={WORST_PERCENT}\n")
        f.write(f"Threshold(lineDist)={thr}\n")
        f.write(f"Window(delta_rho,delta_theta)=({DELTA_RHO},{DELTA_THETA})\n\n")
        f.write("Before:\n")
        f.write(str(before_stats) + "\n\n")
        f.write("After:\n")
        f.write(str(after_stats) + "\n\n")
        f.write("By domain:\n")
        for d, st in domain_stats.items():
            f.write(f"{d}\n  before={st['before']}\n  after ={st['after']}\n")
    print(f"Saved refinement summary: {summary_path}")

    # ---------------------------
    # Visualization (top videos)
    # ---------------------------
    viz_dir = OUT_DIR / "viz_top_videos"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Build a mapping from img_name -> row for quick lookup
    df_map = df.set_index("img_name")

    for video, cnt in top_videos.items():
        vdir = viz_dir / video
        vdir.mkdir(parents=True, exist_ok=True)

        # pick worst frames within this video
        v_worst = worst[worst["video"] == video].sort_values("line_dist_px_unet", ascending=False)
        pick_worst = v_worst.head(FRAMES_PER_VIDEO_WORST)["img_name"].tolist()

        # pick some random frames from this video (from full df)
        v_all = df[df["img_name"].str.contains(f"__{video}__", regex=False)]
        pick_rand = []
        if len(v_all) > 0:
            pick_rand = v_all.sample(min(FRAMES_PER_VIDEO_RANDOM, len(v_all)), random_state=42)["img_name"].tolist()

        picks = []
        for name in pick_worst + pick_rand:
            if name not in picks:
                picks.append(name)

        print(f"[VIZ] {video}: exporting {len(picks)} frames (worst={len(pick_worst)}, random={len(pick_rand)})")

        for name in picks:
            img_path = FRAMES_DIR / name
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            r = df_map.loc[name]
            rho_gt, th_gt = denorm_label_to_rho_theta(float(r["rho_gt_norm"]), float(r["theta_gt_norm"]))
            rho_pr, th_pr = denorm_label_to_rho_theta(float(r["rho_pred_norm"]), float(r["theta_pred_norm"]))

            # If this is in refined_df, also draw refined
            rf = None
            sub = refined_df[refined_df["img_name"] == name]
            if len(sub) == 1:
                rho_rf, th_rf = denorm_label_to_rho_theta(float(sub.iloc[0]["rho_refined_norm"]),
                                                         float(sub.iloc[0]["theta_refined_norm"]))
                rf = (rho_rf, th_rf)

            vis = draw_overlay(img, (rho_gt, th_gt), (rho_pr, th_pr), rf=rf, thickness=2)

            # annotate text
            dist = float(r["line_dist_px_unet"])
            txt = f"{r['domain']}  dist={dist:.1f}px"
            cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)

            out_path = vdir / name.replace(".jpg", "_overlay.jpg")
            cv2.imwrite(str(out_path), vis)

    print("\nDONE. Outputs:")
    print(" -", refined_csv)
    print(" -", summary_path)
    print(" -", viz_dir)


if __name__ == "__main__":
    main()
