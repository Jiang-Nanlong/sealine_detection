#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stage1.py — Stage-1/2 测试入口（真实 ScaleLSD + entropy injection）

在原有 line score/角度/长度筛选基础上，加入：
1. 局部熵差（线下侧 - 线上侧）
2. 法向梯度一致性（法向强、切向弱）

用于更稳地从候选线段中挑选海天线。
"""

import os
import sys
import json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCALELSD_REPO = os.path.join(_PROJECT_ROOT, "scalelsd")
if os.path.isdir(_SCALELSD_REPO) and _SCALELSD_REPO not in sys.path:
    sys.path.insert(0, _SCALELSD_REPO)
os.chdir(_PROJECT_ROOT)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1_scalelsd_entropy.data.musid_entropy_dataset import MUSIDEntropyDataset

# ============================================================
# 全局配置（PyCharm 顶部改参数后直接运行）
# ============================================================
MODE = "entropy"

IMG_DIR     = "Hashmani's Dataset/MU-SID"
ENTROPY_DIR = "Hashmani's Dataset/MU-SID_entropy_blue"
CSV_TEST    = "splits_musid/GroundTruth_test.csv"

IMG_H = 576
IMG_W = 1024

WEIGHTS_PATH = "stage1_scalelsd_entropy/weights/best_stage1_entropy.pth"

BATCH_SIZE  = 1
NUM_WORKERS = 0
USE_LSD     = True
USE_NMS     = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_JSON = ""
VIS_DIR   = ""
VIS_TOP_K = 50
SAVE_ROOT = "stage1_scalelsd_entropy/test_outputs"

# ------- 第二阶段：候选线重排序参数 -------
MAX_HORIZON_DEV_DEG = 15.0
MIN_LENGTH_RATIO    = 0.20
LINE_SAMPLE_POINTS  = 41
STRIP_HALF_WIDTHS   = (3.0, 6.0, 9.0)

# 总分权重
W_NET   = 0.35
W_ANGLE = 0.20
W_LEN   = 0.15
W_ENT   = 0.15
W_GRAD  = 0.15


def _safe_slug(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace(" ", "_")


def _build_output_dirs(mode: str, weights_path: str):
    tag = f"{mode}__{_safe_slug(weights_path)}"
    out_dir = os.path.join(SAVE_ROOT, tag)
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    if VIS_DIR != "":
        os.makedirs(vis_dir, exist_ok=True)
    return out_dir, vis_dir


# ============================================================
# Collate
# ============================================================
def collate_test(batch):
    images = torch.stack([s["image"] for s in batch], dim=0)
    entropy_maps = torch.stack([s["entropy_map"] for s in batch], dim=0)
    gt_lines = []
    stems = []
    for s in batch:
        ep = s["annotation"]["resized_endpoints"]
        gt_line = np.array([ep[0, 0], ep[0, 1], ep[1, 0], ep[1, 1]], dtype=np.float32)
        gt_lines.append(torch.from_numpy(gt_line))
        stems.append(s["annotation"]["stem"])
    gt_lines = torch.stack(gt_lines, dim=0)
    return images, entropy_maps, gt_lines, stems


# ============================================================
# 模型构建
# ============================================================
def build_model(mode, weights_path, device):
    if mode == "baseline":
        from scalelsd.ssl.models.detector import ScaleLSD
        model = ScaleLSD(gray_scale=True)
    elif mode == "entropy":
        from stage1_scalelsd_entropy.models.scalelsd_with_entropy import ScaleLSDWithEntropy
        model = ScaleLSDWithEntropy(gray_scale=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 官方 configure() 流程外手动补推理参数
    model.num_junctions_inference = 512
    model.junction_threshold_hm = 0.008

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        if "model_state" in state:
            state = state["model_state"]
        elif "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    return model.to(device)


# ============================================================
# 第二阶段：候选线重排序辅助函数
# ============================================================
def _normalize_scores(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < 1e-8:
        return np.ones_like(x, dtype=np.float32)
    return ((x - xmin) / (xmax - xmin)).astype(np.float32)


def _bilinear_sample(map2d: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = map2d.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)

    Ia = map2d[y0, x0]
    Ib = map2d[y0, x1]
    Ic = map2d[y1, x0]
    Id = map2d[y1, x1]
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def _prepare_gradients(gray_img_01: np.ndarray):
    gray32 = gray_img_01.astype(np.float32)
    gx = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _line_geom(line: np.ndarray):
    x1, y1, x2, y2 = line.astype(np.float32)
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.sqrt(dx * dx + dy * dy) + 1e-8)
    tx, ty = dx / length, dy / length
    # 约定 normal 指向“线的下侧”，便于 deltaE = lower - upper
    nx, ny = -ty, tx
    return (x1, y1, x2, y2), length, (tx, ty), (nx, ny)


def _compute_entropy_delta(ent_map_01: np.ndarray, line: np.ndarray,
                           n_samples: int = LINE_SAMPLE_POINTS,
                           offsets: tuple = STRIP_HALF_WIDTHS) -> float:
    (_, _, _, _), _, (tx, ty), (nx, ny) = _line_geom(line)
    x1, y1, x2, y2 = line.astype(np.float32)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    xs = x1 + ts * (x2 - x1)
    ys = y1 + ts * (y2 - y1)

    lower_vals = []
    upper_vals = []
    for d in offsets:
        xl = xs + nx * d
        yl = ys + ny * d
        xu = xs - nx * d
        yu = ys - ny * d
        lower_vals.append(_bilinear_sample(ent_map_01, xl, yl))
        upper_vals.append(_bilinear_sample(ent_map_01, xu, yu))

    lower_mean = float(np.mean(np.concatenate(lower_vals)))
    upper_mean = float(np.mean(np.concatenate(upper_vals)))
    return lower_mean - upper_mean


def _compute_grad_consistency(gray_img_01: np.ndarray, line: np.ndarray,
                              n_samples: int = LINE_SAMPLE_POINTS) -> float:
    gx, gy = _prepare_gradients(gray_img_01)
    (_, _, _, _), _, (tx, ty), (nx, ny) = _line_geom(line)
    x1, y1, x2, y2 = line.astype(np.float32)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    xs = x1 + ts * (x2 - x1)
    ys = y1 + ts * (y2 - y1)

    gxs = _bilinear_sample(gx, xs, ys)
    gys = _bilinear_sample(gy, xs, ys)

    g_normal = np.abs(gxs * nx + gys * ny)
    g_tangent = np.abs(gxs * tx + gys * ty)

    ratio = float(np.mean(g_normal) / (np.mean(g_tangent) + 1e-6))
    # 映射到 [0,1]，ratio>1 越大越好
    return float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))


def rerank_horizon_candidates(lines, scores, gray_img_01, ent_map_01, img_h, img_w,
                              max_horizon_dev_deg=MAX_HORIZON_DEV_DEG,
                              min_length_ratio=MIN_LENGTH_RATIO):
    """
    在原本的 line score / 角度 / 长度基础上，
    增加：
      - 局部熵差：deltaE = E_lower - E_upper
      - 法向梯度一致性：normal strong / tangent weak
    """
    if len(lines) == 0:
        return None, None, None, None

    lines_np = np.asarray(lines, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)

    dx = lines_np[:, 2] - lines_np[:, 0]
    dy = lines_np[:, 3] - lines_np[:, 1]
    raw_angles = np.abs(np.degrees(np.arctan2(dy, dx)))
    horizon_dev = np.minimum(raw_angles, 180.0 - raw_angles)
    lengths = np.sqrt(dx ** 2 + dy ** 2)

    net_norm = _normalize_scores(scores_np)
    angle_score = np.clip(1.0 - horizon_dev / max_horizon_dev_deg, 0.0, 1.0).astype(np.float32)
    len_score = np.clip(lengths / (img_w * 0.8), 0.0, 1.0).astype(np.float32)

    entropy_delta = np.zeros(len(lines_np), dtype=np.float32)
    grad_consistency = np.zeros(len(lines_np), dtype=np.float32)

    for i, line in enumerate(lines_np):
        entropy_delta[i] = _compute_entropy_delta(ent_map_01, line)
        grad_consistency[i] = _compute_grad_consistency(gray_img_01, line)

    # 熵差偏正值更好；经验上把 [-0.05, 0.15] 大致映射到 [0,1]
    ent_score = np.clip((entropy_delta + 0.05) / 0.20, 0.0, 1.0).astype(np.float32)

    total = (
        W_NET   * net_norm +
        W_ANGLE * angle_score +
        W_LEN   * len_score +
        W_ENT   * ent_score +
        W_GRAD  * grad_consistency
    )

    valid_mask = (horizon_dev <= max_horizon_dev_deg) & (lengths >= min_length_ratio * img_w)

    if valid_mask.any():
        masked_total = total.copy()
        masked_total[~valid_mask] = -1.0
        best_idx = int(np.argmax(masked_total))
    else:
        best_idx = int(np.argmax(total))

    debug = {
        "total_score": total.tolist(),
        "net_score_norm": net_norm.tolist(),
        "angle_score": angle_score.tolist(),
        "len_score": len_score.tolist(),
        "entropy_delta": entropy_delta.tolist(),
        "entropy_score": ent_score.tolist(),
        "grad_consistency": grad_consistency.tolist(),
        "horizon_dev_deg": horizon_dev.tolist(),
        "length_px": lengths.tolist(),
        "valid_mask": valid_mask.tolist(),
    }

    return lines_np[best_idx], float(total[best_idx]), best_idx, debug


# ============================================================
# 评估指标
# ============================================================
def compute_endpoint_error(pred_line, gt_line, img_w, img_h):
    def line_y_at_x(line, x_query):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        if abs(dx) < 1e-6:
            return (y1 + y2) / 2.0
        t = (x_query - x1) / dx
        return y1 + t * (y2 - y1)

    pred_y_left = line_y_at_x(pred_line, 0)
    pred_y_right = line_y_at_x(pred_line, img_w - 1)
    gt_y_left = line_y_at_x(gt_line, 0)
    gt_y_right = line_y_at_x(gt_line, img_w - 1)

    err_left = abs(pred_y_left - gt_y_left)
    err_right = abs(pred_y_right - gt_y_right)
    return (err_left + err_right) / 2.0


# ============================================================
# 可视化
# ============================================================
def visualize_predictions(gray_img, pred_lines, pred_scores, gt_line,
                          horizon_line, save_path, top_k=50):
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    if len(pred_lines) > 0:
        sorted_idx = np.argsort(pred_scores)[::-1][:top_k]
        for i in sorted_idx:
            x1, y1, x2, y2 = pred_lines[i].astype(int)
            cv2.line(vis, (x1, y1), (x2, y2), (255, 128, 0), 1)

    gx1, gy1, gx2, gy2 = gt_line.astype(int)
    cv2.line(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

    if horizon_line is not None:
        hx1, hy1, hx2, hy2 = horizon_line.astype(int)
        cv2.line(vis, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)

    cv2.imwrite(save_path, vis)


@torch.no_grad()
def main():
    device = DEVICE
    mode = MODE

    print("=" * 60)
    print(f"Stage-2 Rerank Test — mode={mode}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Device: {device}")
    print("=" * 60)

    ds = MUSIDEntropyDataset(
        csv_file=CSV_TEST, img_dir=IMG_DIR, entropy_dir=ENTROPY_DIR,
        img_size=(IMG_H, IMG_W), gray_scale=True,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_test)
    print(f"Test samples: {len(ds)}")

    model = build_model(mode, WEIGHTS_PATH, device)
    model.eval()
    print(f"Model stride: {model.stride}")
    if mode == "entropy" and hasattr(model, "backbone") and hasattr(model.backbone, "alpha"):
        print(f"alpha = {model.backbone.alpha.item():.6f}")

    out_dir, auto_vis_dir = _build_output_dirs(mode, WEIGHTS_PATH)
    do_vis = VIS_DIR != ""
    vis_dir = auto_vis_dir if do_vis else ""

    results = []
    endpoint_errors = []
    failure_samples = []

    for img_tensor, ent_tensor, gt_line_np, stem in tqdm(loader, desc="test", ncols=90):
        img_tensor = img_tensor.to(device)
        ent_map_t = ent_tensor.to(device) if mode == "entropy" else None

        ann = {"width": IMG_W, "height": IMG_H, "use_lsd": USE_LSD, "use_nms": USE_NMS}

        if mode == "entropy":
            output_list, _ = model.forward_test(img_tensor, annotations=ann, entropy_map=ent_map_t)
        else:
            output_list, _ = model.forward_test(img_tensor, annotations=ann)

        for bi in range(len(output_list)):
            out = output_list[bi]
            cur_stem = stem[bi] if isinstance(stem, (list, tuple)) else stem

            lines_pred = out["lines_pred"].cpu().numpy()
            lines_score = out["lines_score"].cpu().numpy()
            juncs_pred = out["juncs_pred"].cpu().numpy()
            gt_line = gt_line_np[bi].cpu().numpy() if torch.is_tensor(gt_line_np) else gt_line_np[bi]

            gray_np = img_tensor[bi, 0].detach().cpu().numpy().astype(np.float32)
            ent_np = ent_tensor[bi, 0].detach().cpu().numpy().astype(np.float32)

            horizon_line, horizon_score, horizon_idx, rank_debug = rerank_horizon_candidates(
                lines_pred, lines_score, gray_np, ent_np, IMG_H, IMG_W
            )

            ep_err = None
            if horizon_line is not None:
                ep_err = compute_endpoint_error(horizon_line, gt_line, IMG_W, IMG_H)
                endpoint_errors.append(ep_err)
            else:
                failure_samples.append({
                    "stem": cur_stem,
                    "num_lines": int(len(lines_pred)),
                    "num_junctions": int(len(juncs_pred)),
                    "has_candidate": False,
                })

            rec = {
                "stem": cur_stem,
                "num_lines": int(len(lines_pred)),
                "num_junctions": int(len(juncs_pred)),
                "horizon_line": horizon_line.tolist() if horizon_line is not None else None,
                "horizon_score": float(horizon_score) if horizon_score is not None else None,
                "horizon_idx": int(horizon_idx) if horizon_idx is not None else None,
                "endpoint_error_px": float(ep_err) if ep_err is not None else None,
                "gt_line": gt_line.tolist(),
                "lines_pred": lines_pred.tolist(),
                "lines_score": lines_score.tolist(),
                "rank_debug": rank_debug,
            }
            results.append(rec)

            if do_vis:
                gray_for_vis = (gray_np * 255).astype(np.uint8)
                vis_path = os.path.join(vis_dir, f"{cur_stem}.jpg")
                visualize_predictions(
                    gray_for_vis, lines_pred, lines_score, gt_line,
                    horizon_line, vis_path, top_k=VIS_TOP_K,
                )

    # 汇总
    endpoint_errors_np = np.array(endpoint_errors, dtype=np.float32) if len(endpoint_errors) else np.array([], dtype=np.float32)
    summary = {
        "mode": mode,
        "weights_path": WEIGHTS_PATH,
        "n_samples": len(results),
        "n_valid_candidate": int(len(endpoint_errors_np)),
        "n_no_candidate": int(sum(1 for r in results if r["horizon_line"] is None)),
        "avg_num_lines": float(np.mean([r["num_lines"] for r in results])) if results else 0.0,
        "avg_num_junctions": float(np.mean([r["num_junctions"] for r in results])) if results else 0.0,
    }

    if len(endpoint_errors_np):
        summary.update({
            "mean_endpoint_error_px": float(endpoint_errors_np.mean()),
            "median_endpoint_error_px": float(np.median(endpoint_errors_np)),
            "max_endpoint_error_px": float(endpoint_errors_np.max()),
            "pct_le_5px": float((endpoint_errors_np <= 5).mean() * 100.0),
            "pct_le_10px": float((endpoint_errors_np <= 10).mean() * 100.0),
            "pct_le_20px": float((endpoint_errors_np <= 20).mean() * 100.0),
            "pct_le_50px": float((endpoint_errors_np <= 50).mean() * 100.0),
        })

    worst_samples = sorted(
        [r for r in results if r["endpoint_error_px"] is not None],
        key=lambda x: x["endpoint_error_px"],
        reverse=True
    )[:20]

    with open(os.path.join(out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "failure_samples.json"), "w", encoding="utf-8") as f:
        json.dump(failure_samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "worst_samples_top20.json"), "w", encoding="utf-8") as f:
        json.dump(worst_samples, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
