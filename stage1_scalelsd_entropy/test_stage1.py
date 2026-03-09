#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stage1.py — Stage-1 测试入口（真实 ScaleLSD + entropy injection）

功能:
  1. 加载训练好的 checkpoint
  2. 在 test split 上逐样本推理（调用 ScaleLSD.forward_test）
  3. 保存每张图的线段检测结果（lines_pred, lines_score, juncs_pred 等）
  4. 自动按 mode / weights 名称分别保存结果，避免 baseline / entropy 相互覆盖
  5. 额外导出：
       - failure_samples.json      : 无有效海天线候选的样本
       - worst_samples_topk.json   : 端点误差最大的样本
       - summary.json              : 汇总指标

用法:
  在 PyCharm 中直接修改顶部全局变量后运行。
"""

import os
import sys
import json

# 将 scalelsd 仓库目录加入 sys.path，使 from scalelsd.ssl.* 可用
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCALELSD_REPO = os.path.join(_PROJECT_ROOT, "scalelsd")
if os.path.isdir(_SCALELSD_REPO) and _SCALELSD_REPO not in sys.path:
    sys.path.insert(0, _SCALELSD_REPO)
# 确保工作目录为项目根目录，使相对路径（splits_musid/ 等）可用
os.chdir(_PROJECT_ROOT)

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage1_scalelsd_entropy.data.musid_entropy_dataset import MUSIDEntropyDataset

# ============================================================
# 全局配置（在 PyCharm 中直接修改后运行）
# ============================================================

# 模式："baseline"（原始 ScaleLSD）or "entropy"（ScaleLSDWithEntropy）
MODE = "entropy"

# 数据路径
IMG_DIR     = "Hashmani's Dataset/MU-SID"
ENTROPY_DIR = "Hashmani's Dataset/MU-SID_entropy_blue"
CSV_TEST    = "splits_musid/GroundTruth_test.csv"

# 图像尺寸
IMG_H = 576
IMG_W = 1024

# 模型权重路径
WEIGHTS_PATH = "stage1_scalelsd_entropy/weights/best_stage1_entropy.pth"

# 推理参数
BATCH_SIZE  = 1       # forward_test 内部使用 LSD，batch>1 可能导致问题
NUM_WORKERS = 0
USE_LSD     = True    # 是否使用 LSD 辅助方向预测
USE_NMS     = True    # 是否对 junction heatmap 做 NMS

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出（若留空，将自动按 mode + 权重名生成互不覆盖的目录）
SAVE_ROOT = "stage1_scalelsd_entropy/test_outputs"
SAVE_JSON = ""
VIS_DIR   = ""
VIS_TOP_K = 50        # 每张图最多画 VIS_TOP_K 条得分最高的线段
SAVE_VIS = True
WORST_TOP_K = 20      # 导出端点误差最大的前 K 个样本

# 官方 ScaleLSD 默认推理参数
NUM_JUNCTIONS_INFERENCE = 512
JUNCTION_THRESHOLD_HM = 0.008


# ============================================================
# 工具函数
# ============================================================
def _safe_name_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = base.replace(" ", "_")
    return base if base else "unnamed"

def build_output_paths(mode: str, weights_path: str):
    """
    为 baseline / entropy 自动创建分开的输出目录，防止相互覆盖。
    """
    weight_tag = _safe_name_from_path(weights_path)
    run_tag = f"{mode}__{weight_tag}"
    out_dir = os.path.join(SAVE_ROOT, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    save_json = SAVE_JSON if SAVE_JSON else os.path.join(out_dir, "test_results.json")
    vis_dir = VIS_DIR if VIS_DIR else os.path.join(out_dir, "vis")
    summary_json = os.path.join(out_dir, "summary.json")
    failure_json = os.path.join(out_dir, "failure_samples.json")
    worst_json = os.path.join(out_dir, f"worst_samples_top{WORST_TOP_K}.json")
    return out_dir, save_json, vis_dir, summary_json, failure_json, worst_json

def maybe_warn_mode_weight_mismatch(mode: str, weights_path: str):
    lower = weights_path.lower()
    if mode == "baseline" and "entropy" in lower:
        print("[WARN] MODE=baseline，但权重文件名里包含 'entropy'。请确认是否为你想要的组合。")
    if mode == "entropy" and "baseline" in lower:
        print("[WARN] MODE=entropy，但权重文件名里包含 'baseline'。这通常只用于近似对比，不是严格的 entropy 测试。")


# ============================================================
# Collate: MUSIDEntropyDataset dict → test 格式
# ============================================================
def collate_test(batch):
    """
    测试用 collate：从 MUSIDEntropyDataset dict 中提取
    images, entropy_maps, gt_lines [B,4], stems。
    """
    images = torch.stack([s["image"] for s in batch], dim=0)
    entropy_maps = torch.stack([s["entropy_map"] for s in batch], dim=0)
    gt_lines = []
    stems = []
    for s in batch:
        ep = s["annotation"]["resized_endpoints"]  # np [2, 2]
        gt_line = np.array([ep[0, 0], ep[0, 1], ep[1, 0], ep[1, 1]], dtype=np.float32)
        gt_lines.append(torch.from_numpy(gt_line))
        stems.append(s["annotation"]["stem"])
    gt_lines = torch.stack(gt_lines, dim=0)  # [B, 4]
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

    # 官方默认推理参数；不走 configure(opts) 流程时手动补上
    model.num_junctions_inference = NUM_JUNCTIONS_INFERENCE
    model.junction_threshold_hm = JUNCTION_THRESHOLD_HM

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        if "model_state" in state:
            state = state["model_state"]
        elif "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print(f"[WARN] weights not found: {weights_path}")

    return model.to(device)


# ============================================================
# 海天线候选选择
# ============================================================
def select_horizon_candidate(lines, scores, img_h, img_w,
                             max_horizon_dev_deg=15.0, min_length_ratio=0.2):
    """
    从预测线段中选出最可能的海天线候选。

    启发式规则：
      1. 优先选择接近水平的线段（相对水平线的偏差角较小）
      2. 线段长度 >= min_length_ratio * img_w
      3. 在满足条件的线段中按得分排序，取最高分
      4. 如果没有满足条件的候选，则退化为取最长线段

    Returns:
        best_line: [4] or None
        best_score: float or None
        best_idx: int or None
    """
    if len(lines) == 0:
        return None, None, None

    lines_np = np.array(lines)
    scores_np = np.array(scores)

    dx = lines_np[:, 2] - lines_np[:, 0]
    dy = lines_np[:, 3] - lines_np[:, 1]
    raw_angles = np.degrees(np.arctan2(dy, dx))
    raw_angles = np.abs(raw_angles)
    horizon_dev = np.minimum(raw_angles, 180.0 - raw_angles)

    lengths = np.sqrt(dx ** 2 + dy ** 2)
    min_len = min_length_ratio * img_w

    mask = (horizon_dev <= max_horizon_dev_deg) & (lengths >= min_len)

    if not mask.any():
        best_idx = int(np.argmax(lengths))
        return lines_np[best_idx], float(scores_np[best_idx]), best_idx

    valid_scores = scores_np.copy()
    valid_scores[~mask] = -1
    best_idx = int(np.argmax(valid_scores))
    return lines_np[best_idx], float(scores_np[best_idx]), best_idx


# ============================================================
# 评估指标：端点距离
# ============================================================
def compute_endpoint_error(pred_line, gt_line, img_w, img_h):
    """
    计算预测线与 GT 线在图像左右边界处的 y 坐标差异均值。
    pred_line, gt_line: [x1, y1, x2, y2]
    """
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
    """
    画预测线段和 GT 线到灰度图上并保存。
    """
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


def summarize_errors(endpoint_errors):
    if len(endpoint_errors) == 0:
        return {
            "n_valid_candidates": 0,
            "mean_endpoint_error_px": None,
            "median_endpoint_error_px": None,
            "max_endpoint_error_px": None,
            "pct_le_5px": None,
            "pct_le_10px": None,
            "pct_le_20px": None,
            "pct_le_50px": None,
        }
    ep_arr = np.array(endpoint_errors, dtype=np.float32)
    return {
        "n_valid_candidates": int(len(ep_arr)),
        "mean_endpoint_error_px": float(ep_arr.mean()),
        "median_endpoint_error_px": float(np.median(ep_arr)),
        "max_endpoint_error_px": float(ep_arr.max()),
        "pct_le_5px": float((ep_arr <= 5).sum() / len(ep_arr) * 100.0),
        "pct_le_10px": float((ep_arr <= 10).sum() / len(ep_arr) * 100.0),
        "pct_le_20px": float((ep_arr <= 20).sum() / len(ep_arr) * 100.0),
        "pct_le_50px": float((ep_arr <= 50).sum() / len(ep_arr) * 100.0),
    }


# ============================================================
# Main
# ============================================================
@torch.no_grad()
def main():
    device = DEVICE
    mode = MODE

    out_dir, save_json, vis_dir, summary_json, failure_json, worst_json = build_output_paths(mode, WEIGHTS_PATH)
    maybe_warn_mode_weight_mismatch(mode, WEIGHTS_PATH)

    print("=" * 60)
    print(f"Stage-1 Test — mode={mode}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    # ---- Dataset ----
    ds = MUSIDEntropyDataset(
        csv_file=CSV_TEST, img_dir=IMG_DIR, entropy_dir=ENTROPY_DIR,
        img_size=(IMG_H, IMG_W), gray_scale=True,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_test)
    print(f"Test samples: {len(ds)}")

    # ---- Model ----
    model = build_model(mode, WEIGHTS_PATH, device)
    model.eval()
    stride = model.stride
    print(f"Model stride: {stride}")

    if mode == "entropy":
        print(f"alpha = {model.backbone.alpha.item():.6f}")

    # ---- 可视化目录 ----
    do_vis = bool(SAVE_VIS and vis_dir)
    if do_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # ---- Inference ----
    results = []
    endpoint_errors = []

    for img_tensor, ent_tensor, gt_line_np, stem in tqdm(loader, desc="test", ncols=90):
        img_tensor = img_tensor.to(device)
        ent_map = ent_tensor.to(device) if mode == "entropy" else None

        ann = {
            "width": IMG_W,
            "height": IMG_H,
            "use_lsd": USE_LSD,
            "use_nms": USE_NMS,
        }

        if mode == "entropy":
            output_list, _ = model.forward_test(
                img_tensor, annotations=ann, entropy_map=ent_map)
        else:
            output_list, _ = model.forward_test(img_tensor, annotations=ann)

        for bi in range(len(output_list)):
            out = output_list[bi]
            cur_stem = stem[bi] if isinstance(stem, (list, tuple)) else stem

            lines_pred = out["lines_pred"].cpu().numpy()
            lines_score = out["lines_score"].cpu().numpy()
            juncs_pred = out["juncs_pred"].cpu().numpy()
            juncs_score = out["juncs_score"].cpu().numpy()

            gt_line = gt_line_np[bi].numpy() if torch.is_tensor(gt_line_np) else gt_line_np[bi]

            horizon_line, horizon_score, horizon_idx = select_horizon_candidate(
                lines_pred, lines_score, IMG_H, IMG_W,
            )

            ep_err = None
            if horizon_line is not None:
                ep_err = compute_endpoint_error(horizon_line, gt_line, IMG_W, IMG_H)
                endpoint_errors.append(ep_err)

            rec = {
                "stem": cur_stem,
                "num_lines": int(len(lines_pred)),
                "num_junctions": int(len(juncs_pred)),
                "has_candidate": bool(horizon_line is not None),
                "horizon_line": horizon_line.tolist() if horizon_line is not None else None,
                "horizon_score": float(horizon_score) if horizon_score is not None else None,
                "horizon_index": int(horizon_idx) if horizon_idx is not None else None,
                "endpoint_error_px": float(ep_err) if ep_err is not None else None,
                "gt_line": gt_line.tolist(),
                "lines_pred": lines_pred.tolist(),
                "lines_score": lines_score.tolist(),
                "juncs_pred": juncs_pred.tolist(),
                "juncs_score": juncs_score.tolist(),
            }
            results.append(rec)

            if do_vis:
                gray_for_vis = (img_tensor[bi, 0].cpu().numpy() * 255).astype(np.uint8)
                vis_path = os.path.join(vis_dir, f"{cur_stem}.jpg")
                visualize_predictions(
                    gray_for_vis, lines_pred, lines_score, gt_line,
                    horizon_line, vis_path, top_k=VIS_TOP_K,
                )

    # ---- 汇总指标 ----
    print("\n" + "=" * 60)
    print(f"Results ({len(results)} samples, mode={mode}):")
    avg_lines = float(np.mean([r['num_lines'] for r in results])) if results else 0.0
    avg_juncs = float(np.mean([r['num_junctions'] for r in results])) if results else 0.0
    print(f"  Avg lines per image: {avg_lines:.1f}")
    print(f"  Avg junctions per image: {avg_juncs:.1f}")

    summary = summarize_errors(endpoint_errors)
    if summary["n_valid_candidates"] > 0:
        print(f"\n  Horizon endpoint error (px) on {summary['n_valid_candidates']} samples with valid candidate:")
        print(f"    mean={summary['mean_endpoint_error_px']:.2f}  median={summary['median_endpoint_error_px']:.2f}  "
              f"max={summary['max_endpoint_error_px']:.2f}")
        for th_key, th_label in [("pct_le_5px", 5), ("pct_le_10px", 10), ("pct_le_20px", 20), ("pct_le_50px", 50)]:
            print(f"    <= {th_label}px : {summary[th_key]:.1f}%")
    else:
        print("\n  No valid horizon candidates found.")

    failure_samples = [r for r in results if not r["has_candidate"]]
    worst_samples = sorted(
        [r for r in results if r["endpoint_error_px"] is not None],
        key=lambda x: x["endpoint_error_px"],
        reverse=True,
    )[:WORST_TOP_K]

    n_no_candidate = len(failure_samples)
    print(f"\n  Samples with no horizon candidate: {n_no_candidate}/{len(results)}")

    summary_payload = {
        "mode": mode,
        "weights": WEIGHTS_PATH,
        "n_samples": len(results),
        "img_size": [IMG_H, IMG_W],
        "avg_lines_per_image": avg_lines,
        "avg_junctions_per_image": avg_juncs,
        "n_no_candidate": n_no_candidate,
        "failure_rate_pct": float(n_no_candidate / len(results) * 100.0) if results else 0.0,
        **summary,
    }

    # ---- 保存 JSON ----
    payload = {
        "mode": mode,
        "weights": WEIGHTS_PATH,
        "n_samples": len(results),
        "img_size": [IMG_H, IMG_W],
        "summary": summary_payload,
        "per_sample": results,
    }

    def _json_default(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2, default=_json_default)
    with open(failure_json, "w", encoding="utf-8") as f:
        json.dump(failure_samples, f, ensure_ascii=False, indent=2, default=_json_default)
    with open(worst_json, "w", encoding="utf-8") as f:
        json.dump(worst_samples, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\nResults saved to: {save_json}")
    print(f"Summary saved to: {summary_json}")
    print(f"Failure samples saved to: {failure_json}")
    print(f"Worst samples saved to: {worst_json}")
    if do_vis:
        print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
