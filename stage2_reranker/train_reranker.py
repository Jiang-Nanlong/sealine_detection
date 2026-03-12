"""
train_reranker_downloadable.py — 第二阶段 reranker 训练脚本（hard + fixable 两层筛选版）

基于 export_reranker_data.py 导出的候选线特征 CSV，训练轻量 MLP reranker，
联合使用：
  1) pointwise soft-target 回归（辅助）
  2) pairwise ranking（主项）

当前版本重点：
  - 先按 det_top1_err 的全局统计定义 hard image
  - 再在 hard image 内识别 fixable image
  - fixable image 高权重构造 pair
  - hard but non-fixable image 也可低权重构造少量 pair
  - 同时保存 best_reranker / best_fused 两套结果
"""

import csv
import json
import math
import os
import random
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# 项目路径设置
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ============================================================
# 顶部全局变量配置 — 在 PyCharm 中直接修改
# ============================================================
EXPORT_DIR = "stage2_reranker/exports/entropy__best_checkpoint"
TRAIN_CSV = ""
VAL_CSV = ""
OUTPUT_DIR = "stage2_reranker/reranker_output/entropy__best_checkpoint_hardfixable"

FEATURE_COLUMNS = [
    "det_score",
    "length",
    "length_norm",
    "horizon_dev_deg",
    "x_span",
    "x_span_norm",
    "y_mid",
    "y_mid_norm",
    "ent_upper",
    "ent_lower",
    "delta_ent",
    "ent_consistency",
    "grad_normal",
    "grad_tangent",
    "grad_ratio",
    "grad_consistency",
]

HIDDEN_DIMS = [64, 32]
DROPOUT = 0.3

BATCH_SIZE = 256
PAIR_BATCH_SIZE = 256
NUM_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
DEVICE = "cuda"

TAU = 10.0
T_GOOD = 10.0
T_DROP = 20.0
REG_LOSS_TYPE = "mse"  # "mse" 或 "smoothl1"

# ---- Pairwise / hard-case 训练 ----
USE_PAIRWISE = True
LOSS_W_POINT = 0.2
LOSS_W_PAIR = 1.0

HARD_IMAGE_RULE = "mean"   # "mean" / "median" / "p60" / "p70"
HARD_IMAGE_EXTRA_PAIR_WEIGHT = 1.5

FIXABLE_MIN_IMPROVEMENT = 1.0
FIXABLE_MAX_ORACLE_ERR = 30.0
USE_FIXABLE_MAX_ORACLE_ERR = True
PAIR_MIN_ERR_GAP = 0.5

ENABLE_PAIRS_FOR_HARD_NONFIXABLE = True
HARD_NONFIXABLE_PAIR_WEIGHT = 0.5
MAX_NEG_PER_IMAGE = 6
NEGATIVE_SELECT_MODE = "hard_det"
HARD_NEG_TOPK_BY_DET = 10
FIXABLE_EXTRA_PAIR_WEIGHT = 2.0

PAIRWISE_LOSS_TYPE = "logistic"  # "logistic" / "margin"
PAIRWISE_MARGIN = 0.0

# ---- 融合打分（验证/评估阶段） ----
USE_FUSED_SCORING = True
FUSION_LAMBDAS = [round(i * 0.0001, 4) for i in range(10001)]


# ============================================================
# 路径工具
# ============================================================
def _resolve(p):
    if not p:
        return p
    pp = Path(p)
    if not pp.is_absolute():
        pp = _PROJECT_ROOT / pp
    return str(pp)


def _get_csv_path(csv_arg, export_dir, split_name):
    if csv_arg:
        return _resolve(csv_arg)
    return os.path.join(_resolve(export_dir), f"{split_name}_reranker_candidates.csv")


# ============================================================
# 基础工具
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"  已保存模型: {filepath}")


# ============================================================
# 1. 数据加载与清洗
# ============================================================
def load_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    fv = float(v)
                    if not math.isfinite(fv):
                        fv = float("nan")
                    parsed[k] = fv
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    print(f"  已加载 CSV: {filepath}  ({len(rows)} 行)")
    return rows


def extract_features_and_meta(rows, feature_cols):
    n = len(rows)
    d = len(feature_cols)
    X = np.zeros((n, d), dtype=np.float32)
    meta = []

    for i, row in enumerate(rows):
        for j, col in enumerate(feature_cols):
            val = row.get(col, float("nan"))
            if isinstance(val, str):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = float("nan")
            X[i, j] = val

        def _num(name, default=float("nan")):
            vv = row.get(name, default)
            if isinstance(vv, str):
                try:
                    vv = float(vv)
                except (ValueError, TypeError):
                    vv = default
            return vv

        meta.append({
            "row_idx": i,
            "image_stem": str(row.get("image_stem", "")),
            "image_id": row.get("image_id", i),
            "candidate_rank": _num("candidate_rank", i),
            "endpoint_err": _num("endpoint_err", float("nan")),
            "is_best_match": int(_num("is_best_match", 0)),
            "det_score": _num("det_score", 0.0),
            "heuristic_score": _num("heuristic_score", float("nan")),
            "label": float(_num("label", 0.0)),
        })
    return X, meta


def clean_features(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            col[:] = 0.0
        elif nan_mask.any():
            col[nan_mask] = float(np.nanmedian(col))
        X[:, j] = col
    return np.clip(X, -1e6, 1e6)


def compute_soft_targets(meta, tau):
    targets = np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        err = float(m["endpoint_err"])
        if math.isfinite(err) and err >= 0:
            targets[i] = math.exp(-err / tau)
        else:
            targets[i] = 0.0
    return targets


def compute_image_weights(meta, t_good, t_drop):
    image_best_err = defaultdict(lambda: float("inf"))
    image_indices = defaultdict(list)
    for i, m in enumerate(meta):
        stem = m["image_stem"]
        err = float(m["endpoint_err"])
        image_indices[stem].append(i)
        if math.isfinite(err) and err < image_best_err[stem]:
            image_best_err[stem] = err

    image_weight_map = {}
    for stem, best_err in image_best_err.items():
        if not math.isfinite(best_err):
            w = 0.0
        elif best_err <= t_good:
            w = 1.0
        elif best_err >= t_drop:
            w = 0.0
        else:
            w = 1.0 - (best_err - t_good) / (t_drop - t_good)
        image_weight_map[stem] = float(w)

    weights = np.zeros(len(meta), dtype=np.float32)
    for stem, indices in image_indices.items():
        w = image_weight_map[stem]
        for idx in indices:
            weights[idx] = w

    all_w = list(image_weight_map.values())
    img_stats = {
        "num_images_total": len(all_w),
        "num_images_used_for_training": int(sum(1 for w in all_w if w > 0)),
        "num_images_dropped_by_weight": int(sum(1 for w in all_w if w <= 0)),
        "mean_image_weight": float(np.mean(all_w)) if all_w else 0.0,
        "median_image_weight": float(np.median(all_w)) if all_w else 0.0,
    }
    return weights, img_stats


# ============================================================
# 2. 标准化
# ============================================================
class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# 3. 图像级分析（hard + fixable 两层筛选）
# ============================================================
def _percentile_safe(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _choose_hard_threshold(global_stats, rule):
    rule = (rule or "mean").lower()
    if rule == "median":
        return float(global_stats["global_median_det_err"])
    if rule == "p60":
        return float(global_stats["global_p60_det_err"])
    if rule == "p70":
        return float(global_stats["global_p70_det_err"])
    return float(global_stats["global_mean_det_err"])


def compute_image_analysis(meta, hard_rule, fixable_min_improvement, fixable_max_oracle_err,
                           hard_threshold_override=None, use_fixable_max_oracle_err=True):
    image_groups = defaultdict(list)
    for m in meta:
        image_groups[m["image_stem"]].append(m)

    train_like_det_errs = []
    basic = {}
    for stem, candidates in image_groups.items():
        det_top1 = max(candidates, key=lambda x: float(x["det_score"]))
        finite_candidates = [c for c in candidates if math.isfinite(float(c["endpoint_err"]))]
        oracle_best = min(finite_candidates, key=lambda x: float(x["endpoint_err"])) if finite_candidates else None
        det_top1_err = float(det_top1["endpoint_err"]) if math.isfinite(float(det_top1["endpoint_err"])) else float("inf")
        oracle_best_err = float(oracle_best["endpoint_err"]) if oracle_best is not None else float("inf")
        best_improvement = det_top1_err - oracle_best_err if math.isfinite(det_top1_err) and math.isfinite(oracle_best_err) else float("-inf")
        train_like_det_errs.append(det_top1_err)
        basic[stem] = {
            "image_stem": stem,
            "num_candidates": len(candidates),
            "candidate_indices": [int(c["row_idx"]) for c in candidates],
            "det_top1_idx": int(det_top1["row_idx"]),
            "oracle_best_idx": int(oracle_best["row_idx"]) if oracle_best is not None else None,
            "det_top1_err": float(det_top1_err),
            "oracle_best_err": float(oracle_best_err),
            "best_improvement": float(best_improvement),
        }

    global_stats = {
        "global_mean_det_err": float(np.mean(train_like_det_errs)) if train_like_det_errs else float("nan"),
        "global_median_det_err": float(np.median(train_like_det_errs)) if train_like_det_errs else float("nan"),
        "global_p60_det_err": _percentile_safe(train_like_det_errs, 60),
        "global_p70_det_err": _percentile_safe(train_like_det_errs, 70),
        "hard_image_rule": hard_rule,
    }
    hard_threshold = float(hard_threshold_override) if hard_threshold_override is not None else _choose_hard_threshold(global_stats, hard_rule)
    global_stats["hard_image_threshold"] = hard_threshold

    analysis = {}
    num_hard = 0
    num_fixable = 0
    for stem, info in basic.items():
        det_top1_err = info["det_top1_err"]
        oracle_best_err = info["oracle_best_err"]
        best_improvement = info["best_improvement"]
        is_hard = bool(math.isfinite(det_top1_err) and math.isfinite(hard_threshold) and det_top1_err > hard_threshold)
        fixable_cond = (
            is_hard and
            math.isfinite(best_improvement) and best_improvement >= fixable_min_improvement
        )
        if fixable_cond and use_fixable_max_oracle_err:
            fixable_cond = fixable_cond and math.isfinite(oracle_best_err) and oracle_best_err <= fixable_max_oracle_err
        is_fixable = bool(fixable_cond)
        if is_hard:
            num_hard += 1
        if is_fixable:
            num_fixable += 1
        merged = dict(info)
        merged.update({
            "is_hard_image": is_hard,
            "is_fixable_image": is_fixable,
        })
        analysis[stem] = merged

    summary = dict(global_stats)
    summary.update({
        "num_images": len(analysis),
        "num_hard_images": num_hard,
        "num_fixable_images": num_fixable,
    })
    return analysis, summary


# ============================================================
# 4. Dataset
# ============================================================
class CandidateDataset(Dataset):
    def __init__(self, X, target_scores, image_weights, meta):
        self.X = torch.from_numpy(X)
        self.target_scores = torch.from_numpy(target_scores.astype(np.float32))
        self.image_weights = torch.from_numpy(image_weights.astype(np.float32))
        self.meta = meta

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.target_scores[idx], self.image_weights[idx], idx


class PairDataset(Dataset):
    def __init__(self, X, pairs):
        self.X = torch.from_numpy(X)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return (
            self.X[p["pos_idx"]],
            self.X[p["neg_idx"]],
            torch.tensor(float(p["pair_weight"]), dtype=torch.float32),
        )


# ============================================================
# 5. MLP 模型
# ============================================================
class RerankerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 6. Pair 构造
# ============================================================
def _candidate_lookup_by_stem(meta):
    groups = defaultdict(list)
    for m in meta:
        groups[m["image_stem"]].append(m)
    for stem in groups:
        groups[stem] = list(groups[stem])
    return groups


def build_pairs(meta, image_analysis):
    groups = _candidate_lookup_by_stem(meta)
    pairs = []
    by_image_counts = {}

    for stem, info in image_analysis.items():
        candidates = groups[stem]
        if not candidates or info["oracle_best_idx"] is None:
            by_image_counts[stem] = 0
            continue

        is_fixable = bool(info["is_fixable_image"])
        is_hard = bool(info["is_hard_image"])
        if not is_fixable and not (ENABLE_PAIRS_FOR_HARD_NONFIXABLE and is_hard):
            by_image_counts[stem] = 0
            continue

        pos_idx = int(info["oracle_best_idx"])
        oracle_err = float(info["oracle_best_err"])
        det_top1_idx = int(info["det_top1_idx"])

        neg_candidates = []
        seen = set()

        def try_add_neg(cand, source_tag):
            neg_idx = int(cand["row_idx"])
            if neg_idx == pos_idx or neg_idx in seen:
                return
            neg_err = float(cand["endpoint_err"])
            if not math.isfinite(neg_err):
                return
            if (neg_err - oracle_err) < PAIR_MIN_ERR_GAP:
                return
            seen.add(neg_idx)
            neg_candidates.append((source_tag, cand))

        # 优先 det top1
        det_top1_cand = next((c for c in candidates if int(c["row_idx"]) == det_top1_idx), None)
        if det_top1_cand is not None:
            try_add_neg(det_top1_cand, "det_top1")

        # 再从 det_score 高分候选里补 hard negative
        if NEGATIVE_SELECT_MODE == "hard_det":
            sorted_by_det = sorted(candidates, key=lambda x: float(x["det_score"]), reverse=True)
            for cand in sorted_by_det[:HARD_NEG_TOPK_BY_DET]:
                try_add_neg(cand, "topk_det")

        # 再从其余明显更差候选里补
        sorted_by_err_desc = sorted(
            [c for c in candidates if math.isfinite(float(c["endpoint_err"]))],
            key=lambda x: float(x["endpoint_err"]), reverse=True,
        )
        for cand in sorted_by_err_desc:
            try_add_neg(cand, "worse_err")
            if len(neg_candidates) >= MAX_NEG_PER_IMAGE:
                break

        neg_candidates = neg_candidates[:MAX_NEG_PER_IMAGE]

        img_pair_weight = 1.0
        if is_fixable:
            img_pair_weight *= HARD_IMAGE_EXTRA_PAIR_WEIGHT * FIXABLE_EXTRA_PAIR_WEIGHT
        elif is_hard:
            img_pair_weight *= HARD_NONFIXABLE_PAIR_WEIGHT

        cur_count = 0
        for source_tag, cand in neg_candidates:
            pairs.append({
                "image_stem": stem,
                "pos_idx": pos_idx,
                "neg_idx": int(cand["row_idx"]),
                "pair_weight": float(img_pair_weight),
                "source": source_tag,
                "is_fixable_image": is_fixable,
                "is_hard_image": is_hard,
            })
            cur_count += 1
        by_image_counts[stem] = cur_count

    return pairs, by_image_counts


# ============================================================
# 7. 损失函数
# ============================================================
def build_regression_criterion(loss_type):
    if loss_type == "smoothl1":
        return nn.SmoothL1Loss(reduction="none")
    return nn.MSELoss(reduction="none")


def compute_pointwise_loss(model, batch, criterion, device):
    X_batch, target_batch, weight_batch, _ = batch
    X_batch = X_batch.to(device)
    target_batch = target_batch.to(device).unsqueeze(1)
    weight_batch = weight_batch.to(device).unsqueeze(1)

    preds = torch.sigmoid(model(X_batch))
    per_sample = criterion(preds, target_batch)
    weighted = per_sample * weight_batch
    denom = weight_batch.sum()
    if denom.item() > 0:
        return weighted.sum() / denom, float(denom.item())
    return weighted.sum() * 0.0, 0.0


def compute_pairwise_loss(model, batch, device):
    X_pos, X_neg, pair_weight = batch
    X_pos = X_pos.to(device)
    X_neg = X_neg.to(device)
    pair_weight = pair_weight.to(device).unsqueeze(1)

    pos_score = model(X_pos)
    neg_score = model(X_neg)
    diff = pos_score - neg_score

    if PAIRWISE_LOSS_TYPE == "margin":
        per_pair = torch.relu(PAIRWISE_MARGIN - diff)
    else:
        per_pair = torch.nn.functional.softplus(-diff)

    weighted = per_pair * pair_weight
    denom = pair_weight.sum()
    if denom.item() > 0:
        return weighted.sum() / denom, float(denom.item())
    return weighted.sum() * 0.0, 0.0


# ============================================================
# 8. 训练 / 验证
# ============================================================
def train_one_epoch(model, point_loader, pair_loader, criterion, optimizer, device):
    model.train()

    point_iter = iter(point_loader) if point_loader is not None else None
    pair_iter = iter(pair_loader) if pair_loader is not None else None
    num_point_steps = len(point_loader) if point_loader is not None else 0
    num_pair_steps = len(pair_loader) if pair_loader is not None else 0
    num_steps = max(num_point_steps, num_pair_steps, 1)

    total_loss_sum = 0.0
    total_steps = 0
    total_point_loss_sum = 0.0
    total_pair_loss_sum = 0.0
    num_point_used = 0
    num_pair_used = 0

    for step in range(num_steps):
        point_batch = None
        pair_batch = None
        if point_loader is not None and num_point_steps > 0:
            try:
                point_batch = next(point_iter)
            except StopIteration:
                point_iter = iter(point_loader)
                point_batch = next(point_iter)
        if pair_loader is not None and num_pair_steps > 0:
            try:
                pair_batch = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                pair_batch = next(pair_iter)

        point_loss = None
        pair_loss = None
        total = None

        if point_batch is not None:
            point_loss, point_weight = compute_pointwise_loss(model, point_batch, criterion, device)
            total = LOSS_W_POINT * point_loss if total is None else total + LOSS_W_POINT * point_loss
            total_point_loss_sum += float(point_loss.detach().cpu().item())
            num_point_used += 1

        if USE_PAIRWISE and pair_batch is not None:
            pair_loss, pair_weight = compute_pairwise_loss(model, pair_batch, device)
            total = LOSS_W_PAIR * pair_loss if total is None else total + LOSS_W_PAIR * pair_loss
            total_pair_loss_sum += float(pair_loss.detach().cpu().item())
            num_pair_used += 1

        if total is None:
            continue

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss_sum += float(total.detach().cpu().item())
        total_steps += 1

    return {
        "train_total_loss": total_loss_sum / max(total_steps, 1),
        "train_point_loss": total_point_loss_sum / max(num_point_used, 1),
        "train_pair_loss": total_pair_loss_sum / max(num_pair_used, 1) if num_pair_used > 0 else 0.0,
    }


def _compute_auc(labels, scores):
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    n = len(sorted_scores)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    pos_rank_sum = float(ranks[sorted_labels == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _compute_err_stats(errs, method):
    errs = np.array(errs, dtype=np.float64)
    valid = errs[np.isfinite(errs)]
    if len(valid) == 0:
        return {
            "method": method,
            "num_images": len(errs),
            "num_images_with_candidate": 0,
            "num_images_without_candidate": len(errs),
            "mean_endpoint_err": float("nan"),
            "median_endpoint_err": float("nan"),
        }
    return {
        "method": method,
        "num_images": len(errs),
        "num_images_with_candidate": len(valid),
        "num_images_without_candidate": len(errs) - len(valid),
        "mean_endpoint_err": float(np.mean(valid)),
        "median_endpoint_err": float(np.median(valid)),
        "std_endpoint_err": float(np.std(valid)),
        "pct_le_5": float(np.mean(valid <= 5) * 100),
        "pct_le_10": float(np.mean(valid <= 10) * 100),
        "pct_le_20": float(np.mean(valid <= 20) * 100),
        "pct_le_50": float(np.mean(valid <= 50) * 100),
    }


def load_split_summary(export_dir, split_name):
    summary_path = os.path.join(_resolve(export_dir), f"{split_name}_summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def enrich_image_level_stats_with_summary(stats_dict, split_summary):
    if split_summary is None:
        return stats_dict
    total = split_summary.get("num_images", None)
    if total is None:
        return stats_dict
    n_with = stats_dict.get("num_images_with_candidate", 0)
    stats_dict["num_images"] = total
    stats_dict["num_images_with_candidate"] = n_with
    stats_dict["num_images_without_candidate"] = total - n_with
    return stats_dict


def _compute_fused_image_metrics(image_groups, lambdas, split_summary=None):
    image_det_norm = {}
    for stem, candidates in image_groups.items():
        det_vals = [c["det_score"] for c in candidates]
        min_det = min(det_vals)
        max_det = max(det_vals)
        if max_det > min_det:
            norms = [(d - min_det) / (max_det - min_det) for d in det_vals]
        else:
            norms = [0.5] * len(det_vals)
        image_det_norm[stem] = norms

    best_lambda = None
    best_mean_err = float("inf")
    best_metrics = None
    best_per_image = None
    all_lambda_metrics = []

    for lam in lambdas:
        fused_errs = []
        per_image = []
        for stem, candidates in image_groups.items():
            norms = image_det_norm[stem]
            scored = []
            for i, c in enumerate(candidates):
                fused_score = lam * norms[i] + (1.0 - lam) * c["reranker_score"]
                scored.append((fused_score, norms[i], c))
            best_fs, best_norm, best_cand = max(scored, key=lambda x: x[0])
            fused_errs.append(best_cand["endpoint_err"])
            per_image.append({
                "image_stem": stem,
                "selected_candidate_rank": best_cand["candidate_rank"],
                "det_score": best_cand["det_score"],
                "det_score_norm": best_norm,
                "reranker_score": best_cand["reranker_score"],
                "fused_score": best_fs,
                "endpoint_err": best_cand["endpoint_err"],
            })
        stats = _compute_err_stats(fused_errs, f"fused_lambda_{lam:.4f}")
        stats = enrich_image_level_stats_with_summary(stats, split_summary)
        stats["lambda"] = lam
        all_lambda_metrics.append(stats)
        mean_err = stats.get("mean_endpoint_err", float("nan"))
        if math.isfinite(mean_err) and mean_err < best_mean_err:
            best_mean_err = mean_err
            best_lambda = lam
            best_metrics = stats
            best_per_image = per_image

    return all_lambda_metrics, best_lambda, best_metrics, best_per_image


def evaluate(model, dataloader, dataset, criterion, device, split_summary=None):
    model.eval()
    all_logits = []
    all_indices = []
    total_weighted_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for X_batch, target_batch, weight_batch, idx_batch in dataloader:
            X_batch = X_batch.to(device)
            target_batch = target_batch.to(device).unsqueeze(1)
            weight_batch = weight_batch.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(X_batch))
            per_sample = criterion(preds, target_batch)
            weighted = per_sample * weight_batch
            total_weighted_loss += weighted.sum().item()
            total_weight += weight_batch.sum().item()
            all_logits.append(torch.logit(torch.clamp(preds.cpu(), min=1e-6, max=1 - 1e-6)).squeeze(1))
            all_indices.append(idx_batch)

    if not all_logits:
        raise RuntimeError("Validation produced no predictions.")

    all_logits = torch.cat(all_logits).numpy()
    all_indices = torch.cat(all_indices).numpy()
    all_scores = 1.0 / (1.0 + np.exp(-all_logits))
    avg_loss = total_weighted_loss / max(total_weight, 1e-8)

    meta = dataset.meta
    all_labels = np.array([meta[int(idx)]["label"] for idx in all_indices], dtype=np.float32)
    all_target_scores = dataset.target_scores.numpy()[all_indices.astype(int)]

    preds_binary = (all_scores >= 0.5).astype(np.float32)
    tp = float(((preds_binary == 1) & (all_labels == 1)).sum())
    fp = float(((preds_binary == 1) & (all_labels == 0)).sum())
    fn = float(((preds_binary == 0) & (all_labels == 1)).sum())
    tn = float(((preds_binary == 0) & (all_labels == 0)).sum())
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    auc = _compute_auc(all_labels, all_scores)

    candidate_metrics = {
        "val_loss": avg_loss,
        "val_soft_target_mean": float(np.mean(all_target_scores)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "num_candidates": int(len(all_labels)),
        "num_positive": int(all_labels.sum()),
        "num_negative": int((all_labels == 0).sum()),
    }

    image_groups = defaultdict(list)
    predictions = []
    for i, global_idx in enumerate(all_indices):
        m = meta[int(global_idx)]
        item = {
            "idx": int(global_idx),
            "image_id": m["image_id"],
            "image_stem": m["image_stem"],
            "candidate_rank": m["candidate_rank"],
            "label": int(m["label"]),
            "det_score": float(m["det_score"]),
            "reranker_score": float(all_scores[i]),
            "endpoint_err": float(m["endpoint_err"]),
            "is_best_match": int(m["is_best_match"]),
            "heuristic_score": float(m["heuristic_score"]) if math.isfinite(float(m["heuristic_score"])) else None,
        }
        image_groups[m["image_stem"]].append(item)
        predictions.append({
            "split": "val",
            "image_id": item["image_id"],
            "image_stem": item["image_stem"],
            "candidate_rank": item["candidate_rank"],
            "label": item["label"],
            "det_score": item["det_score"],
            "reranker_score": item["reranker_score"],
            "endpoint_err": item["endpoint_err"],
            "is_best_match": item["is_best_match"],
        })

    reranker_errs, det_errs, heuristic_errs = [], [], []
    any_valid_heur = False
    per_image_results = []
    for stem, candidates in image_groups.items():
        best_reranker = max(candidates, key=lambda c: c["reranker_score"])
        best_det = max(candidates, key=lambda c: c["det_score"])
        reranker_errs.append(best_reranker["endpoint_err"])
        det_errs.append(best_det["endpoint_err"])
        valid_heur = [c for c in candidates if c["heuristic_score"] is not None and math.isfinite(c["heuristic_score"])]
        if valid_heur:
            best_heur = max(valid_heur, key=lambda c: c["heuristic_score"])
            heuristic_errs.append(best_heur["endpoint_err"])
            any_valid_heur = True
        else:
            heuristic_errs.append(best_det["endpoint_err"])
        per_image_results.append({
            "image_stem": stem,
            "num_candidates": len(candidates),
            "reranker_selected_err": best_reranker["endpoint_err"],
            "reranker_selected_rank": best_reranker["candidate_rank"],
            "det_baseline_selected_err": best_det["endpoint_err"],
            "det_baseline_selected_rank": best_det["candidate_rank"],
        })

    image_reranker = enrich_image_level_stats_with_summary(_compute_err_stats(reranker_errs, "reranker"), split_summary)
    image_det_baseline = enrich_image_level_stats_with_summary(_compute_err_stats(det_errs, "det_score_baseline"), split_summary)

    results = {
        "candidate_level_metrics": candidate_metrics,
        "image_level_metrics_reranker": image_reranker,
        "image_level_metrics_det_score_baseline": image_det_baseline,
        "per_image_results": per_image_results,
        "predictions": predictions,
    }
    if any_valid_heur:
        results["image_level_metrics_heuristic_baseline"] = enrich_image_level_stats_with_summary(
            _compute_err_stats(heuristic_errs, "heuristic_baseline"), split_summary
        )

    if USE_FUSED_SCORING and FUSION_LAMBDAS:
        fused_all, fused_best_lambda, fused_best_metrics, fused_best_per_image = _compute_fused_image_metrics(
            image_groups, FUSION_LAMBDAS, split_summary
        )
        results["fused_all_lambdas"] = fused_all
        results["fused_best_lambda"] = fused_best_lambda
        results["fused_best_metrics"] = fused_best_metrics
        results["fused_best_per_image"] = fused_best_per_image

    return results


# ============================================================
# 9. 输出工具
# ============================================================
def save_feature_config(scaler, filepath):
    config = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "tau": TAU,
        "t_good": T_GOOD,
        "t_drop": T_DROP,
        "reg_loss_type": REG_LOSS_TYPE,
        "hard_image_rule": HARD_IMAGE_RULE,
        "fixable_min_improvement": FIXABLE_MIN_IMPROVEMENT,
        "fixable_max_oracle_err": FIXABLE_MAX_ORACLE_ERR,
        "pair_min_err_gap": PAIR_MIN_ERR_GAP,
        "use_fixable_max_oracle_err": USE_FIXABLE_MAX_ORACLE_ERR,
        "fixable_extra_pair_weight": FIXABLE_EXTRA_PAIR_WEIGHT,
    }
    save_json(config, filepath)


def save_predictions_csv(predictions, filepath):
    if not predictions:
        return
    columns = [
        "split", "image_id", "image_stem", "candidate_rank",
        "label", "det_score", "reranker_score", "endpoint_err", "is_best_match",
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in predictions:
            writer.writerow({k: row.get(k, "") for k in columns})
    print(f"  已保存 predictions: {filepath}  ({len(predictions)} 行)")


def save_fused_predictions_csv(per_image_results, best_lambda, filepath):
    if not per_image_results:
        return
    columns = [
        "image_stem", "selected_candidate_rank",
        "det_score", "det_score_norm", "reranker_score", "fused_score",
        "endpoint_err", "lambda",
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in per_image_results:
            out = {k: row.get(k, "") for k in columns}
            out["lambda"] = best_lambda
            writer.writerow(out)
    print(f"  已保存 fused predictions: {filepath}  ({len(per_image_results)} 行)")


# ============================================================
# 10. main
# ============================================================
def main():
    print("=" * 72)
    print("  Stage-2 Reranker Training (hard + fixable + pairwise)")
    print(f"  EXPORT_DIR   = {EXPORT_DIR}")
    print(f"  OUTPUT_DIR   = {OUTPUT_DIR}")
    print(f"  FEATURES     = {len(FEATURE_COLUMNS)} columns")
    print(f"  HIDDEN_DIMS  = {HIDDEN_DIMS}")
    print(f"  DROPOUT      = {DROPOUT}")
    print(f"  BATCH_SIZE   = {BATCH_SIZE}")
    print(f"  PAIR_BATCH   = {PAIR_BATCH_SIZE}")
    print(f"  NUM_EPOCHS   = {NUM_EPOCHS}")
    print(f"  LR           = {LR}")
    print(f"  TAU          = {TAU}")
    print(f"  T_GOOD/T_DROP= {T_GOOD}/{T_DROP}")
    print(f"  REG_LOSS     = {REG_LOSS_TYPE}")
    print(f"  HARD_RULE    = {HARD_IMAGE_RULE}")
    print(f"  FIXABLE_MIN  = {FIXABLE_MIN_IMPROVEMENT}")
    print(f"  FIXABLE_ORCL = {FIXABLE_MAX_ORACLE_ERR}  (use={USE_FIXABLE_MAX_ORACLE_ERR})")
    print(f"  FIXABLE_EXTRA= {FIXABLE_EXTRA_PAIR_WEIGHT}")
    print(f"  PAIR_GAP     = {PAIR_MIN_ERR_GAP}")
    print(f"  MAX_NEG/DET  = {MAX_NEG_PER_IMAGE}/{HARD_NEG_TOPK_BY_DET}")
    print(f"  FUSED        = {USE_FUSED_SCORING}  lambdas={len(FUSION_LAMBDAS)}")
    print("=" * 72)

    set_seed(RANDOM_SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    output_dir = _resolve(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/7] 加载 CSV...")
    train_csv_path = _get_csv_path(TRAIN_CSV, EXPORT_DIR, "train")
    val_csv_path = _get_csv_path(VAL_CSV, EXPORT_DIR, "val")
    train_rows = load_csv(train_csv_path)
    val_rows = load_csv(val_csv_path)
    val_split_summary = load_split_summary(EXPORT_DIR, "val")

    print("\n[2/7] 特征提取、标准化、图像分析...")
    X_train_raw, meta_train = extract_features_and_meta(train_rows, FEATURE_COLUMNS)
    X_val_raw, meta_val = extract_features_and_meta(val_rows, FEATURE_COLUMNS)

    X_train = clean_features(X_train_raw)
    X_val = clean_features(X_val_raw)
    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    target_train = compute_soft_targets(meta_train, TAU)
    target_val = compute_soft_targets(meta_val, TAU)
    weight_train, train_img_weight_stats = compute_image_weights(meta_train, T_GOOD, T_DROP)
    weight_val, val_img_weight_stats = compute_image_weights(meta_val, T_GOOD, T_DROP)

    train_analysis, train_analysis_summary = compute_image_analysis(
        meta_train,
        hard_rule=HARD_IMAGE_RULE,
        fixable_min_improvement=FIXABLE_MIN_IMPROVEMENT,
        fixable_max_oracle_err=FIXABLE_MAX_ORACLE_ERR,
        use_fixable_max_oracle_err=USE_FIXABLE_MAX_ORACLE_ERR,
    )
    hard_threshold = train_analysis_summary["hard_image_threshold"]
    val_analysis, val_analysis_summary = compute_image_analysis(
        meta_val,
        hard_rule=HARD_IMAGE_RULE,
        fixable_min_improvement=FIXABLE_MIN_IMPROVEMENT,
        fixable_max_oracle_err=FIXABLE_MAX_ORACLE_ERR,
        hard_threshold_override=hard_threshold,
        use_fixable_max_oracle_err=USE_FIXABLE_MAX_ORACLE_ERR,
    )

    train_pairs, train_pair_count_by_image = build_pairs(meta_train, train_analysis)

    print(f"  Train images: {train_analysis_summary['num_images']} total, "
          f"{train_analysis_summary['num_hard_images']} hard, "
          f"{train_analysis_summary['num_fixable_images']} fixable")
    print(f"  Val images  : {val_analysis_summary['num_images']} total, "
          f"{val_analysis_summary['num_hard_images']} hard, "
          f"{val_analysis_summary['num_fixable_images']} fixable")
    print(f"  Hard threshold ({HARD_IMAGE_RULE}) = {hard_threshold:.4f}")
    print(f"  Train pairs = {len(train_pairs)}")

    feature_config_path = os.path.join(output_dir, "feature_config.json")
    save_feature_config(scaler, feature_config_path)

    print("\n[3/7] 构建 DataLoader...")
    train_dataset = CandidateDataset(X_train, target_train, weight_train, meta_train)
    val_dataset = CandidateDataset(X_val, target_val, weight_val, meta_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    pair_loader = None
    if USE_PAIRWISE and len(train_pairs) > 0:
        pair_dataset = PairDataset(X_train, train_pairs)
        pair_loader = DataLoader(pair_dataset, batch_size=PAIR_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    print("\n[4/7] 构建模型...")
    model = RerankerMLP(input_dim=len(FEATURE_COLUMNS), hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    criterion = build_regression_criterion(REG_LOSS_TYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[5/7] 开始训练...")
    metrics_history = []

    best_reranker_metric = float("inf")
    best_reranker_epoch = -1
    best_fused_metric = float("inf")
    best_fused_epoch = -1
    best_fused_lambda = None

    best_reranker_results = None
    best_fused_results = None
    best_reranker_predictions = []
    best_fused_predictions = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = train_one_epoch(model, train_loader, pair_loader, criterion, optimizer, device)
        val_results = evaluate(model, val_loader, val_dataset, criterion, device, split_summary=val_split_summary)

        cand_m = val_results["candidate_level_metrics"]
        reranker_m = val_results["image_level_metrics_reranker"]
        det_m = val_results["image_level_metrics_det_score_baseline"]
        fused_best_m = val_results.get("fused_best_metrics", None)
        fused_best_lambda_cur = val_results.get("fused_best_lambda", None)

        reranker_err = float(reranker_m.get("mean_endpoint_err", float("nan")))
        det_err = float(det_m.get("mean_endpoint_err", float("nan")))
        fused_err = float(fused_best_m.get("mean_endpoint_err", float("nan"))) if fused_best_m else float("nan")

        if math.isfinite(reranker_err) and reranker_err < best_reranker_metric:
            best_reranker_metric = reranker_err
            best_reranker_epoch = epoch
            best_reranker_results = {
                "best_epoch": epoch,
                "best_metric": "reranker_image_mean_err",
                "best_metric_value": reranker_err,
                "candidate_level_metrics": cand_m,
                "image_level_metrics_reranker": reranker_m,
                "image_level_metrics_det_score_baseline": det_m,
                "image_weight_stats": val_img_weight_stats,
                "training_config": {
                    "tau": TAU,
                    "t_good": T_GOOD,
                    "t_drop": T_DROP,
                    "reg_loss_type": REG_LOSS_TYPE,
                    "lr": LR,
                    "hidden_dims": HIDDEN_DIMS,
                    "dropout": DROPOUT,
                    "use_pairwise": USE_PAIRWISE,
                    "loss_w_point": LOSS_W_POINT,
                    "loss_w_pair": LOSS_W_PAIR,
                    "hard_image_rule": HARD_IMAGE_RULE,
                    "fixable_min_improvement": FIXABLE_MIN_IMPROVEMENT,
                    "fixable_max_oracle_err": FIXABLE_MAX_ORACLE_ERR,
                    "use_fixable_max_oracle_err": USE_FIXABLE_MAX_ORACLE_ERR,
                    "fixable_extra_pair_weight": FIXABLE_EXTRA_PAIR_WEIGHT,
                    "pair_min_err_gap": PAIR_MIN_ERR_GAP,
                    "max_neg_per_image": MAX_NEG_PER_IMAGE,
                    "hard_neg_topk_by_det": HARD_NEG_TOPK_BY_DET,
                    "enable_pairs_for_hard_nonfixable": ENABLE_PAIRS_FOR_HARD_NONFIXABLE,
                    "hard_nonfixable_pair_weight": HARD_NONFIXABLE_PAIR_WEIGHT,
                    "fusion_lambdas": FUSION_LAMBDAS,
                },
            }
            if "image_level_metrics_heuristic_baseline" in val_results:
                best_reranker_results["image_level_metrics_heuristic_baseline"] = val_results["image_level_metrics_heuristic_baseline"]
            if fused_best_m is not None:
                best_reranker_results["best_fusion_lambda"] = fused_best_lambda_cur
                best_reranker_results["image_level_metrics_fused_best"] = fused_best_m
                best_reranker_results["image_level_metrics_fused_all_lambdas"] = val_results.get("fused_all_lambdas", [])
            best_reranker_predictions = val_results.get("predictions", [])
            save_checkpoint(model, os.path.join(output_dir, "best_reranker_model.pth"))

        if fused_best_m is not None and math.isfinite(fused_err) and fused_err < best_fused_metric:
            best_fused_metric = fused_err
            best_fused_epoch = epoch
            best_fused_lambda = fused_best_lambda_cur
            best_fused_results = {
                "best_epoch": epoch,
                "best_metric": "fused_image_mean_err",
                "best_metric_value": fused_err,
                "best_fusion_lambda": fused_best_lambda_cur,
                "candidate_level_metrics": cand_m,
                "image_level_metrics_reranker": reranker_m,
                "image_level_metrics_det_score_baseline": det_m,
                "image_level_metrics_fused_best": fused_best_m,
                "image_level_metrics_fused_all_lambdas": val_results.get("fused_all_lambdas", []),
                "image_weight_stats": val_img_weight_stats,
                "training_config": {
                    "tau": TAU,
                    "t_good": T_GOOD,
                    "t_drop": T_DROP,
                    "reg_loss_type": REG_LOSS_TYPE,
                    "lr": LR,
                    "hidden_dims": HIDDEN_DIMS,
                    "dropout": DROPOUT,
                    "use_pairwise": USE_PAIRWISE,
                    "loss_w_point": LOSS_W_POINT,
                    "loss_w_pair": LOSS_W_PAIR,
                    "hard_image_rule": HARD_IMAGE_RULE,
                    "fixable_min_improvement": FIXABLE_MIN_IMPROVEMENT,
                    "fixable_max_oracle_err": FIXABLE_MAX_ORACLE_ERR,
                    "use_fixable_max_oracle_err": USE_FIXABLE_MAX_ORACLE_ERR,
                    "fixable_extra_pair_weight": FIXABLE_EXTRA_PAIR_WEIGHT,
                    "pair_min_err_gap": PAIR_MIN_ERR_GAP,
                    "max_neg_per_image": MAX_NEG_PER_IMAGE,
                    "hard_neg_topk_by_det": HARD_NEG_TOPK_BY_DET,
                    "enable_pairs_for_hard_nonfixable": ENABLE_PAIRS_FOR_HARD_NONFIXABLE,
                    "hard_nonfixable_pair_weight": HARD_NONFIXABLE_PAIR_WEIGHT,
                    "fusion_lambdas": FUSION_LAMBDAS,
                },
            }
            if "image_level_metrics_heuristic_baseline" in val_results:
                best_fused_results["image_level_metrics_heuristic_baseline"] = val_results["image_level_metrics_heuristic_baseline"]
            best_fused_predictions = val_results.get("fused_best_per_image", [])
            save_checkpoint(model, os.path.join(output_dir, "best_fused_model.pth"))

        metrics_history.append({
            "epoch": epoch,
            **train_stats,
            "val_loss": cand_m["val_loss"],
            "val_auc": cand_m["auc"],
            "val_f1": cand_m["f1"],
            "reranker_mean_err": reranker_err,
            "det_baseline_mean_err": det_err,
            "fused_best_lambda": fused_best_lambda_cur if fused_best_lambda_cur is not None else float("nan"),
            "fused_best_image_mean_err": fused_err,
            "num_train_images": train_analysis_summary["num_images"],
            "num_hard_train_images": train_analysis_summary["num_hard_images"],
            "num_fixable_train_images": train_analysis_summary["num_fixable_images"],
            "num_train_pairs": len(train_pairs),
            "num_val_images": val_analysis_summary["num_images"],
            "num_hard_val_images": val_analysis_summary["num_hard_images"],
            "num_fixable_val_images": val_analysis_summary["num_fixable_images"],
            "best_reranker_epoch": best_reranker_epoch,
            "best_reranker_metric": best_reranker_metric,
            "best_fused_epoch": best_fused_epoch,
            "best_fused_metric": best_fused_metric,
            "best_fused_lambda": best_fused_lambda if best_fused_lambda is not None else float("nan"),
        })

        print(
            f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"train_total={train_stats['train_total_loss']:.4f}  "
            f"train_point={train_stats['train_point_loss']:.4f}  "
            f"train_pair={train_stats['train_pair_loss']:.4f}  "
            f"val={cand_m['val_loss']:.4f}  auc={cand_m['auc']:.4f}  "
            f"Train images: {train_analysis_summary['num_images']} total, {train_analysis_summary['num_hard_images']} hard, {train_analysis_summary['num_fixable_images']} fixable, pairs={len(train_pairs)}  "
            f"Val images: {val_analysis_summary['num_images']} total, {val_analysis_summary['num_hard_images']} hard, {val_analysis_summary['num_fixable_images']} fixable  "
            f"reranker_err={reranker_err:.4f}  fused={fused_err:.4f}(lambda={fused_best_lambda_cur})  det_err={det_err:.4f}  "
            f"best_reranker={best_reranker_metric:.4f}@{best_reranker_epoch}  "
            f"best_fused={best_fused_metric:.4f}@{best_fused_epoch}"
        )

    save_checkpoint(model, os.path.join(output_dir, "last_model.pth"))

    print("\n[6/7] 保存结果...")
    save_json(metrics_history, os.path.join(output_dir, "metrics_history.json"))

    train_summary = {
        "global_mean_det_err": train_analysis_summary["global_mean_det_err"],
        "global_median_det_err": train_analysis_summary["global_median_det_err"],
        "global_p60_det_err": train_analysis_summary["global_p60_det_err"],
        "global_p70_det_err": train_analysis_summary["global_p70_det_err"],
        "HARD_IMAGE_RULE": HARD_IMAGE_RULE,
        "hard_image_threshold": train_analysis_summary["hard_image_threshold"],
        "FIXABLE_MIN_IMPROVEMENT": FIXABLE_MIN_IMPROVEMENT,
        "FIXABLE_MAX_ORACLE_ERR": FIXABLE_MAX_ORACLE_ERR,
        "USE_FIXABLE_MAX_ORACLE_ERR": USE_FIXABLE_MAX_ORACLE_ERR,
        "FIXABLE_EXTRA_PAIR_WEIGHT": FIXABLE_EXTRA_PAIR_WEIGHT,
        "num_hard_train_images": train_analysis_summary["num_hard_images"],
        "num_fixable_train_images": train_analysis_summary["num_fixable_images"],
        "num_hard_val_images": val_analysis_summary["num_hard_images"],
        "num_fixable_val_images": val_analysis_summary["num_fixable_images"],
        "num_train_images": train_analysis_summary["num_images"],
        "num_val_images": val_analysis_summary["num_images"],
        "num_train_pairs": len(train_pairs),
        "best_reranker_epoch": best_reranker_epoch,
        "best_reranker_metric": best_reranker_metric,
        "best_fused_epoch": best_fused_epoch,
        "best_fused_metric": best_fused_metric,
        "best_fused_lambda": best_fused_lambda,
        "train_image_weight_stats": train_img_weight_stats,
        "val_image_weight_stats": val_img_weight_stats,
    }
    save_json(train_summary, os.path.join(output_dir, "train_summary.json"))

    if best_reranker_results is not None:
        save_json(best_reranker_results, os.path.join(output_dir, "val_results_best_reranker.json"))
        save_predictions_csv(best_reranker_predictions, os.path.join(output_dir, "val_predictions_best_reranker.csv"))

    if best_fused_results is not None:
        save_json(best_fused_results, os.path.join(output_dir, "val_results_best_fused.json"))
        save_fused_predictions_csv(best_fused_predictions, best_fused_lambda, os.path.join(output_dir, "val_predictions_fused_best_fused.csv"))

    print("\n[7/7] 训练完成")
    print(f"  输出目录: {output_dir}")
    print(f"  best_reranker: epoch={best_reranker_epoch}, mean_err={best_reranker_metric:.6f}")
    print(f"  best_fused   : epoch={best_fused_epoch}, mean_err={best_fused_metric:.6f}, lambda={best_fused_lambda}")


if __name__ == "__main__":
    main()
