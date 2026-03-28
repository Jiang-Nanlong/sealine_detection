# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
import numpy as np

import cv2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

CSV_STRONG = os.path.join(PROJ_DIR, "eval_full_outputs", "full_eval_test_strong.csv")
CSV_WEAK   = os.path.join(PROJ_DIR, "eval_full_outputs", "full_eval_test_weak.csv")

IMG_ROOT = os.path.join(PROJ_DIR, "Hashmani's Dataset", "MU-SID")

OUT_DIR = os.path.join(PROJ_DIR, "test_2", "figs_refine_gain", "overlays")

UNET_W, UNET_H = 1024, 576

N_WORST_STRONG = 6   # 强权重：误修正最大
N_BEST_WEAK    = 6   # 弱权重：纠偏最好

EPS = 1e-12

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_img(img_name: str):
    p = os.path.join(IMG_ROOT, img_name)
    if os.path.exists(p):
        return cv2.imread(p)
    # 少量样本找不到时，递归搜索（只对少数case，不会太慢）
    for root, _, files in os.walk(IMG_ROOT):
        if img_name in files:
            return cv2.imread(os.path.join(root, img_name))
    return None

import math
from typing import List, Tuple

def line_intersections_in_image(rho: float, theta_deg: float, w: int, h: int) -> List[Tuple[float, float]]:
    """
    Same convention as evaluate_full_pipeline.py:
        (x-cx)*cos + (y-cy)*sin = rho
    Return intersections with image rectangle (top-left coords).
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = w / 2.0, h / 2.0

    pts = []

    # x = 0
    x = 0.0
    if abs(sin_t) > 1e-8:
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if 0.0 <= y <= (h - 1.0):
            pts.append((x, y))

    # x = w-1
    x = float(w - 1)
    if abs(sin_t) > 1e-8:
        y = cy + (rho - ((x - cx) * cos_t)) / sin_t
        if 0.0 <= y <= (h - 1.0):
            pts.append((x, y))

    # y = 0
    y = 0.0
    if abs(cos_t) > 1e-8:
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if 0.0 <= x <= (w - 1.0):
            pts.append((x, y))

    # y = h-1
    y = float(h - 1)
    if abs(cos_t) > 1e-8:
        x = cx + (rho - ((y - cy) * sin_t)) / cos_t
        if 0.0 <= x <= (w - 1.0):
            pts.append((x, y))

    return pts

def farthest_pair(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    best = (pts[0], pts[1])
    best_d2 = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                best = (pts[i], pts[j])
    return best

def polar_to_line_pts(theta_deg: float, rho: float, w: int, h: int):
    pts = line_intersections_in_image(rho, theta_deg, w, h)
    if len(pts) >= 2:
        p0, p1 = farthest_pair(pts)
    elif len(pts) == 1:
        p0 = p1 = pts[0]
    else:
        # fallback mid horizontal
        p0, p1 = (0.0, h / 2.0), (w - 1.0, h / 2.0)
    return (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))

def make_one_overlay(row, tag: str, out_path: str):
    img_name = str(row["img_name"])
    im = load_img(img_name)
    if im is None:
        print("[WARN] image not found:", img_name)
        return

    im = cv2.resize(im, (UNET_W, UNET_H))

    # 颜色约定：GT绿 / CNN红 / Final蓝
    # GT: green
    p1, p2 = polar_to_line_pts(float(row["theta_gt"]), float(row["rho_gt"]), UNET_W, UNET_H)
    cv2.line(im, p1, p2, (0, 255, 0), 2)

    # CNN: red
    p1, p2 = polar_to_line_pts(float(row["theta_cnn"]), float(row["rho_cnn"]), UNET_W, UNET_H)
    cv2.line(im, p1, p2, (0, 0, 255), 2)

    # Final: blue
    p1, p2 = polar_to_line_pts(float(row["theta_final"]), float(row["rho_final"]), UNET_W, UNET_H)
    cv2.line(im, p1, p2, (255, 0, 0), 2)

    delta = float(row["edgey_px_orig_final"]) - float(row["edgey_px_orig_cnn"])
    txt1 = f"{tag}  used_ref={int(row['used_ref'])}  in_topk={int(row['in_topk'])}  conf={float(row['conf']):.3f}"
    txt2 = f"EdgeY cnn={float(row['edgey_px_orig_cnn']):.2f}  final={float(row['edgey_px_orig_final']):.2f}  delta={delta:+.2f}"

    cv2.putText(im, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(im, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(im, txt2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(im, txt2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, im)

def main():
    ensure_dir(OUT_DIR)

    df_s = pd.read_csv(CSV_STRONG)
    df_w = pd.read_csv(CSV_WEAK)

    # ΔEdgeY
    df_s["delta"] = df_s["edgey_px_orig_final"] - df_s["edgey_px_orig_cnn"]
    df_w["delta"] = df_w["edgey_px_orig_final"] - df_w["edgey_px_orig_cnn"]

    # 只挑“发生替换”的样本（否则 delta=0 没意义）
    df_s_used = df_s[df_s["used_ref"] == 1].copy()
    df_w_used = df_w[df_w["used_ref"] == 1].copy()

    # 强权重：误修正最大（delta 最大）
    worst_s = df_s_used.sort_values("delta", ascending=False).head(N_WORST_STRONG)
    # 弱权重：纠偏最好（delta 最小）
    best_w  = df_w_used.sort_values("delta", ascending=True).head(N_BEST_WEAK)

    # 输出 overlays
    for i, r in worst_s.iterrows():
        stem = os.path.splitext(os.path.basename(str(r["img_name"])))[0]
        out = os.path.join(OUT_DIR, f"strong_worsen_{int(r['idx']):04d}_{stem}.png")
        make_one_overlay(r, "strong", out)

    for i, r in best_w.iterrows():
        stem = os.path.splitext(os.path.basename(str(r["img_name"])))[0]
        out = os.path.join(OUT_DIR, f"weak_improve_{int(r['idx']):04d}_{stem}.png")
        make_one_overlay(r, "weak", out)

    print("[DONE] overlays saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
