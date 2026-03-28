"""
test_linea_stage1.py — 第一阶段测试/推理/评估脚本

比较 baseline（LINEA）与 entropy（LINEA_ENTROPY）在 MU-SID test split 上的表现。
逐张推理 → 候选线选择 → endpoint error 计算 → 结果保存 + 可选可视化。

用法：
  在 PyCharm 中修改顶部全局变量后直接运行本文件。
"""

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# 项目路径设置（确保 import 正常）
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent          # method1_linea_entropy/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # sealine_detection/
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "LINEA"))

# 显式导入以触发模型注册（LINEA_ENTROPY / LINEA_ENTROPY_A）
import method1_linea_entropy.models

# ============================================================
# 顶部全局变量配置 — 在 PyCharm 中直接修改
# ============================================================
MODE = "entropy_b_enhanced"       # "baseline" / "entropy" / "entropy_a" / "entropy_b" / "entropy_b_enhanced"
CONFIG_FILE = "method1_linea_entropy/configs/ablation_mslep_sai_musid.py"
WEIGHTS_PATH = "output/ablation_mslep_sai_musid_e130/best_checkpoint.pth"       # 权重文件路径（.pth）
DEVICE = "cuda"

CSV_FILE = ""                    # 留空则由 config / build_musid_dataset 自动决定
IMG_DIR = "Hashmani's Dataset/MU-SID"
ENTROPY_DIR = "Hashmani's Dataset/MU-SID_entropy_blue"
SPLIT_DIR = "splits_musid"

IMG_SIZE = 640
BATCH_SIZE = 1
NUM_WORKERS = 0

SAVE_ROOT = "method1_linea_entropy/test_outputs_v3"
SAVE_VIS = True
SAVE_VIS_MAX = 100

# 候选线选择参数
MAX_HORIZON_DEV_DEG = 15.0       # 水平线最大倾斜角（度）
MIN_LENGTH_RATIO = 0.2           # 候选线最短长度占图像宽度的比例
TOPK_LINES = 50                  # 每张图最多保留的候选线数量


# ============================================================
# 1. 配置加载
# ============================================================
def load_config(config_file):
    """用 SLConfig 加载 .py 配置文件，返回 args-like 对象。"""
    from util.slconfig import SLConfig

    # 将相对路径统一解析为相对于项目根目录的绝对路径
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    cfg = SLConfig.fromfile(str(config_path))
    args = cfg

    # 保证 eval_spatial_size 格式
    sz = getattr(args, 'eval_spatial_size', IMG_SIZE)
    if isinstance(sz, int):
        sz = [sz, sz]
    args.eval_spatial_size = sz

    # 覆盖数据目录（使用顶部全局变量，相对路径基于项目根目录解析）
    args.musid_img_dir = str(_PROJECT_ROOT / IMG_DIR) if not Path(IMG_DIR).is_absolute() else IMG_DIR
    args.musid_entropy_dir = str(_PROJECT_ROOT / ENTROPY_DIR) if not Path(ENTROPY_DIR).is_absolute() else ENTROPY_DIR
    args.musid_split_dir = str(_PROJECT_ROOT / SPLIT_DIR) if not Path(SPLIT_DIR).is_absolute() else SPLIT_DIR

    # 根据 MODE 覆盖 entropy_mode
    # entropy_a / entropy_b 也需要 entropy_map, 所以映射到 'entropy'
    # entropy_b_enhanced 使用 entropy_ms (3 通道多尺度熵图), 保留 config 中原始值
    if MODE == 'entropy_b_enhanced':
        args.entropy_mode = getattr(args, 'entropy_mode', 'entropy_ms')
    elif MODE in ('entropy', 'entropy_a', 'entropy_b'):
        args.entropy_mode = 'entropy'
    else:
        args.entropy_mode = MODE

    # 确保 pretrained 为 False（推理不需要下载预训练）
    args.pretrained = False

    return args


# ============================================================
# 2. 模型构建
# ============================================================
def build_model(args):
    """
    根据 args.modelname 用 LINEA 官方 registry 构建模型 + postprocessor。
    baseline → LINEA; entropy → LINEA_ENTROPY; entropy_a → LINEA_ENTROPY_A。
    """
    from models.registry import MODULE_BUILD_FUNCS

    _mode_to_model = {
        "baseline": "LINEA",
        "entropy": "LINEA_ENTROPY",
        "entropy_a": "LINEA_ENTROPY_A",
        "entropy_b": "LINEA_ENTROPY_B",
        "entropy_b_enhanced": "LINEA_ENTROPY_B_ENHANCED",
    }
    model_name = _mode_to_model.get(MODE, "LINEA")

    # 确保 config 中 modelname 与 MODE 匹配
    args.modelname = model_name

    assert model_name in MODULE_BUILD_FUNCS._module_dict, (
        f"模型 '{model_name}' 未注册。请检查 LINEA/models/__init__.py"
    )

    build_fn = MODULE_BUILD_FUNCS.get(model_name)
    model, postprocessor = build_fn(args)

    return model, postprocessor


def _resolve_path(p):
    """将相对路径基于项目根目录解析为绝对路径。"""
    if not p:
        return p
    pp = Path(p)
    if not pp.is_absolute():
        pp = _PROJECT_ROOT / pp
    return str(pp)


def load_weights(model, weights_path, device):
    """加载权重文件，允许 strict=False，打印 missing/unexpected keys 信息。"""
    weights_path = _resolve_path(weights_path)
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[WARN] 权重文件不存在或未指定: {weights_path}")
        print("[WARN] 使用随机初始化权重（仅供调试）")
        return

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # 兼容不同保存格式
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))

    info = model.load_state_dict(state_dict, strict=False)

    if info.missing_keys:
        print(f"[INFO] Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:20]:
            print(f"  - {k}")
        if len(info.missing_keys) > 20:
            print(f"  ... 共 {len(info.missing_keys)} 个")

    if info.unexpected_keys:
        print(f"[INFO] Unexpected keys ({len(info.unexpected_keys)}):")
        for k in info.unexpected_keys[:20]:
            print(f"  - {k}")
        if len(info.unexpected_keys) > 20:
            print(f"  ... 共 {len(info.unexpected_keys)} 个")

    print(f"[OK] 权重已加载: {weights_path}")


# ============================================================
# 3. 数据集构建
# ============================================================
def build_test_dataset(args):
    """构建 test split 数据集。"""
    from method1_linea_entropy.datasets import build_musid_dataset
    return build_musid_dataset('test', args)


def build_test_collate_fn():
    """
    测试用 collate，不做 multi-scale resize。
    直接复用 BatchImageCollateFunction（不传 base_size_repeat）。
    """
    from datasets.collate import BatchImageCollateFunction
    return BatchImageCollateFunction(base_size=IMG_SIZE)


# ============================================================
# 4. 候选线选择（单独封装，便于后续升级）
# ============================================================
def select_horizon_candidate(lines, scores, img_w, img_h,
                             topk=TOPK_LINES,
                             max_dev_deg=MAX_HORIZON_DEV_DEG,
                             min_len_ratio=MIN_LENGTH_RATIO):
    """
    从模型候选线中选出最终海天线。

    第一阶段简单规则：
      1. 按分数排序，保留 top-K
      2. 过滤掉倾斜角超过阈值的
      3. 过滤掉太短的
      4. 在剩余候选中选分数最高的

    Args:
        lines   : Tensor [N, 4] — (x1, y1, x2, y2) 像素坐标
        scores  : Tensor [N] — 对应分数 (sigmoid 后)
        img_w   : 图像宽度（像素）
        img_h   : 图像高度（像素）
        topk    : 保留 top-K 条线
        max_dev_deg : 水平偏差最大角度
        min_len_ratio : 最短长度比例（相对于图像宽度）

    Returns:
        dict 包含：
            'has_candidate' : bool
            'best_line'     : ndarray [4] 或 None — (x1, y1, x2, y2) 像素坐标
            'best_score'    : float 或 None
            'all_lines'     : ndarray [M, 4] — topK 候选线
            'all_scores'    : ndarray [M] — topK 候选分数
            'num_raw_lines' : int — 模型原始输出线段数
    """
    result = {
        'has_candidate': False,
        'best_line': None,
        'best_score': None,
        'all_lines': np.zeros((0, 4)),
        'all_scores': np.zeros(0),
        'num_raw_lines': 0,
    }

    if lines is None or len(lines) == 0:
        return result

    lines_np = lines.cpu().numpy() if torch.is_tensor(lines) else np.array(lines)
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.array(scores)

    result['num_raw_lines'] = len(lines_np)

    # ---- top-K ----
    if len(lines_np) > topk:
        top_idx = np.argsort(scores_np)[::-1][:topk]
        lines_np = lines_np[top_idx]
        scores_np = scores_np[top_idx]

    result['all_lines'] = lines_np
    result['all_scores'] = scores_np

    # ---- 过滤：倾斜角 ----
    dx = lines_np[:, 2] - lines_np[:, 0]
    dy = lines_np[:, 3] - lines_np[:, 1]
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))
    # 接近水平 → angle ≈ 0° 或 ≈ 180°
    dev_from_horizontal = np.minimum(angles, 180.0 - angles)
    angle_mask = dev_from_horizontal <= max_dev_deg

    # ---- 过滤：长度 ----
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    min_length = min_len_ratio * img_w
    length_mask = lengths >= min_length

    # ---- 组合过滤 ----
    valid_mask = angle_mask & length_mask

    if not np.any(valid_mask):
        return result

    valid_lines = lines_np[valid_mask]
    valid_scores = scores_np[valid_mask]

    # ---- 选分数最高的 ----
    best_idx = np.argmax(valid_scores)
    result['has_candidate'] = True
    result['best_line'] = valid_lines[best_idx]
    result['best_score'] = float(valid_scores[best_idx])

    return result


# ============================================================
# 5. 评估指标
# ============================================================
def compute_endpoint_error(pred_line, gt_line):
    """
    计算预测线段与 GT 海天线的 endpoint error（像素）。

    端点顺序无关：尝试两种匹配，取误差更小的。

    Args:
        pred_line : ndarray [4] — (x1, y1, x2, y2) letterbox 坐标
        gt_line   : ndarray [4] — (x1, y1, x2, y2) letterbox 坐标

    Returns:
        float — mean endpoint error (两端点误差的平均值，letterbox 像素)
    """
    p1, p2 = pred_line[:2], pred_line[2:]
    g1, g2 = gt_line[:2], gt_line[2:]

    # 匹配方式 A: p1↔g1, p2↔g2
    err_a = (np.linalg.norm(p1 - g1) + np.linalg.norm(p2 - g2)) / 2.0

    # 匹配方式 B: p1↔g2, p2↔g1
    err_b = (np.linalg.norm(p1 - g2) + np.linalg.norm(p2 - g1)) / 2.0

    return min(err_a, err_b)


def _line_to_rho_theta(line):
    """
    将端点表示 (x1, y1, x2, y2) 转换为 (ρ, θ) 极坐标参数。

    直线方程: x*cos(θ) + y*sin(θ) = ρ
    θ 为法线与 x 轴的夹角，ρ 为原点到直线的有符号距离。

    Returns:
        (rho, theta_rad) — ρ (像素), θ (弧度, 范围 [0, π))
    """
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    # 直线方向角 (与 x 轴夹角)
    alpha = math.atan2(dy, dx)
    # 法线方向 θ = alpha + π/2, 归一化到 [0, π)
    theta = alpha + math.pi / 2.0
    theta = theta % math.pi
    # ρ = x1*cos(θ) + y1*sin(θ)
    rho = x1 * math.cos(theta) + y1 * math.sin(theta)
    # 保证 ρ >= 0
    if rho < 0:
        rho = -rho
        theta = (theta + math.pi) % math.pi
    return rho, theta


def compute_unified_metrics(pred_line, gt_line, img_w):
    """
    计算与第5章对齐的统一评价指标。

    在 letterbox 640×640 坐标系下计算，输出包括：
      - rho_err:    |ρ| 误差 (px) — 法向距离位置偏移
      - theta_err:  θ 误差 (°) — 方向误差 (wrap 到 [0°, 90°])
      - line_dist:  LineDist (px) — 预测线上均匀采样点到 GT 直线的平均距离
      - edge_y:     Edge-Y (px) — 两条线在图像左右边界处 y 坐标差的平均值
      - ve:         VE (px) — 图像中心列处两条线的垂直坐标差 (带符号)
      - ae:         AE (°) — 两条直线倾角差 (wrap 到 [-90°, 90°])

    Args:
        pred_line : ndarray [4] — (x1, y1, x2, y2)
        gt_line   : ndarray [4] — (x1, y1, x2, y2)
        img_w     : int — 图像宽度 (用于 Edge-Y 和 VE 计算)

    Returns:
        dict — 各指标值
    """
    # ---- ρ, θ 误差 ----
    rho_p, theta_p = _line_to_rho_theta(pred_line)
    rho_g, theta_g = _line_to_rho_theta(gt_line)
    rho_err = abs(rho_p - rho_g)
    # θ 误差: wrap 到 [0°, 90°]（方向等价性: θ 和 θ+180° 描述同一条线）
    theta_diff_rad = abs(theta_p - theta_g)
    if theta_diff_rad > math.pi / 2:
        theta_diff_rad = math.pi - theta_diff_rad
    theta_err_deg = math.degrees(theta_diff_rad)

    # ---- LineDist: 预测线上均匀采样 N 点到 GT 直线的平均距离 ----
    n_samples = 100
    px1, py1, px2, py2 = pred_line
    gx1, gy1, gx2, gy2 = gt_line
    t = np.linspace(0, 1, n_samples)
    sample_x = px1 + t * (px2 - px1)
    sample_y = py1 + t * (py2 - py1)
    # GT 直线 ax + by + c = 0
    a = gy2 - gy1
    b = gx1 - gx2
    c = gx2 * gy1 - gx1 * gy2
    denom = math.sqrt(a * a + b * b)
    if denom < 1e-12:
        line_dist = 0.0
    else:
        dists = np.abs(a * sample_x + b * sample_y + c) / denom
        line_dist = float(np.mean(dists))

    # ---- Edge-Y: 两条线在 x=0 和 x=img_w 处 y 值之差的平均值 ----
    def _y_at_x(line, x):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        if abs(dx) < 1e-12:
            return (y1 + y2) / 2.0
        return y1 + (x - x1) * (y2 - y1) / dx

    pred_y_left = _y_at_x(pred_line, 0)
    pred_y_right = _y_at_x(pred_line, img_w)
    gt_y_left = _y_at_x(gt_line, 0)
    gt_y_right = _y_at_x(gt_line, img_w)
    edge_y = (abs(pred_y_left - gt_y_left) + abs(pred_y_right - gt_y_right)) / 2.0

    # ---- VE: 图像中心列处的垂直坐标差 (带符号, pred - gt) ----
    center_x = img_w / 2.0
    ve = _y_at_x(pred_line, center_x) - _y_at_x(gt_line, center_x)

    # ---- AE: 倾角差 (带符号, wrap 到 [-90°, 90°]) ----
    angle_pred = math.degrees(math.atan2(py2 - py1, px2 - px1))
    angle_gt = math.degrees(math.atan2(gy2 - gy1, gx2 - gx1))
    ae = angle_pred - angle_gt
    # wrap 到 [-90, 90]
    while ae > 90:
        ae -= 180
    while ae < -90:
        ae += 180

    return {
        'rho_err': rho_err,
        'theta_err': theta_err_deg,
        'line_dist': line_dist,
        'edge_y': edge_y,
        've': ve,
        'ae': ae,
    }


def gt_line_from_target(target, img_size):
    """
    从 LINEA target 中提取 GT 海天线的像素坐标（letterbox 坐标系）。

    target['lines'] 归一化在 [0, 1]，乘以 img_size 得到 letterbox 后的像素坐标，
    而非原始图像像素坐标。

    Args:
        target : dict
        img_size : int — 正方形尺寸 (640)

    Returns:
        ndarray [4] — (x1, y1, x2, y2) letterbox 像素坐标
    """
    lines = target['lines']
    if torch.is_tensor(lines):
        lines = lines.cpu().numpy()
    gt = lines[0] * img_size  # [4] 像素坐标
    return gt.astype(np.float64)


# ============================================================
# 6. 可视化
# ============================================================
def draw_result(img_tensor, gt_line, pred_result, save_path, img_size,
                mean=(0.538, 0.494, 0.453), std=(0.257, 0.263, 0.273)):
    """
    在图像上绘制 GT 和预测结果，保存为 PNG。

    Args:
        img_tensor : Tensor [3, H, W] — 归一化后的图像
        gt_line    : ndarray [4] — GT (x1, y1, x2, y2) 像素坐标
        pred_result: 由 select_horizon_candidate 返回的 dict
        save_path  : 保存路径
        img_size   : 正方形尺寸
    """
    # 反归一化
    img = img_tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 画 GT（绿色）
    gx1, gy1, gx2, gy2 = gt_line.astype(int)
    cv2.line(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(img, "GT", (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 画候选线（灰色，最多画 10 条）
    all_lines = pred_result['all_lines']
    all_scores = pred_result['all_scores']
    num_draw = min(10, len(all_lines))
    if num_draw > 0:
        sorted_idx = np.argsort(all_scores)[::-1][:num_draw]
        for idx in sorted_idx:
            lx1, ly1, lx2, ly2 = all_lines[idx].astype(int)
            cv2.line(img, (lx1, ly1), (lx2, ly2), (180, 180, 180), 1)

    # 画最终预测（红色）
    if pred_result['has_candidate']:
        px1, py1, px2, py2 = pred_result['best_line'].astype(int)
        cv2.line(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        score_text = f"Pred (s={pred_result['best_score']:.3f})"
        cv2.putText(img, score_text, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(img, "NO CANDIDATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


# ============================================================
# 7. 输出目录生成
# ============================================================
def make_output_dir(save_root, mode, weights_path, config_file=None):
    """
    自动生成输出目录名：
      {save_root} / {mode}__{parent_dir_name}
    若提供 config_file 且 config 名与 weights 父目录名不同，则追加 config 名以避免冲突：
      {save_root} / {mode}__{parent_dir_name}__{config_stem}

    例如：method1_linea_entropy/test_outputs_v5/entropy_b_enhanced__linea_entropy_b_enhanced_musid_v2_e130__ablation_mslep_sai_egab_musid
    """
    if weights_path:
        # 用权重文件的父目录名（每个实验唯一）
        parent_name = Path(weights_path).parent.name
        # 如果 config 名与 weights 父目录名不同，追加 config stem 避免冲突
        if config_file:
            config_stem = Path(config_file).stem
            if config_stem not in parent_name:
                dir_name = f"{mode}__{parent_name}__{config_stem}"
            else:
                dir_name = f"{mode}__{parent_name}"
        else:
            dir_name = f"{mode}__{parent_name}"
    else:
        dir_name = f"{mode}__no_weights"
    out_dir = os.path.join(save_root, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    if SAVE_VIS:
        os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
    return out_dir


# ============================================================
# 8. 主推理流程
# ============================================================
@torch.no_grad()
def run_inference(model, postprocessor, dataloader, device, include_entropy):
    """
    在 test split 上逐 batch 推理。

    Args:
        model          : LINEA 或 LINEAWithEntropy
        postprocessor  : PostProcess
        dataloader     : test DataLoader
        device         : torch.device
        include_entropy: bool — 数据集是否返回 entropy_map

    Returns:
        all_results : list of dict, 每个 dict 包含：
            'stem'         : 样本名称
            'gt_line'      : ndarray [4] 像素坐标
            'pred_result'  : select_horizon_candidate 返回的 dict
            'img_tensor'   : Tensor [3, H, W]（用于可视化）
            'num_lines'    : 模型输出的线段数
    """
    model.eval()
    all_results = []

    for batch_idx, batch in enumerate(dataloader):
        # ---- 解包 batch ----
        if include_entropy:
            if len(batch) == 4:
                images, targets, entropy_maps, metas = batch
            else:
                images, targets, entropy_maps = batch
                metas = None
        else:
            if len(batch) == 3:
                images, targets, metas = batch
            else:
                images, targets = batch[:2]
                metas = None
            entropy_maps = None

        images = images.to(device)
        if entropy_maps is not None:
            entropy_maps = entropy_maps.to(device)

        # ---- 前向推理 ----
        if entropy_maps is not None:
            outputs = model(images, entropy_map=entropy_maps)
        else:
            outputs = model(images)

        # 检测 horizon head 额外输出字段
        has_horizon_head = 'pred_logits_raw_det' in outputs

        # ---- PostProcess: 将归一化坐标转为像素坐标 ----
        # target_sizes: [B, 2] — (h, w) 用正方形尺寸（letterbox 后的）
        B = images.shape[0]
        target_sizes = torch.tensor([[IMG_SIZE, IMG_SIZE]], device=device).repeat(B, 1)
        results_batch = postprocessor(outputs, target_sizes)

        # ---- 逐张处理 ----
        for i in range(B):
            # 样本名称
            if metas is not None:
                stem = metas[i].get('stem', f"sample_{batch_idx * BATCH_SIZE + i}")
            else:
                stem = f"sample_{batch_idx * BATCH_SIZE + i}"

            # GT 线段（像素坐标）
            gt_line = gt_line_from_target(targets[i], IMG_SIZE)

            # 模型预测线段与分数
            pred_lines = results_batch[i]['lines']    # Tensor [num_queries, 4]
            pred_scores = results_batch[i]['scores']   # Tensor [num_queries]

            # 候选线选择
            pred_result = select_horizon_candidate(
                pred_lines, pred_scores,
                img_w=IMG_SIZE, img_h=IMG_SIZE,
            )

            result_entry = {
                'stem': stem,
                'gt_line': gt_line,
                'pred_result': pred_result,
                'img_tensor': images[i].cpu(),
                'num_lines': pred_result['num_raw_lines'],
                'has_horizon_head': has_horizon_head,
            }

            # 保存 horizon head 调试信息
            if has_horizon_head:
                raw_scores_i = outputs['pred_logits_raw_det'][i, :, 0].sigmoid().cpu()
                horizon_delta_i = outputs['pred_logits_horizon'][i, :, 0].cpu()
                combined_scores_i = outputs['pred_logits_combined'][i, :, 0].sigmoid().cpu()
                result_entry['debug_raw_det_score_mean'] = float(raw_scores_i.mean())
                result_entry['debug_horizon_delta_mean'] = float(horizon_delta_i.mean())
                result_entry['debug_horizon_delta_std'] = float(horizon_delta_i.std())
                result_entry['debug_combined_score_mean'] = float(combined_scores_i.mean())

                if 'pred_logits_raw_calibrated' in outputs:
                    raw_cal_i = outputs['pred_logits_raw_calibrated'][i, :, 0].sigmoid().cpu()
                    result_entry['debug_raw_calibrated_score_mean'] = float(raw_cal_i.mean())

                if 'pred_fusion_gate' in outputs:
                    gate_i = outputs['pred_fusion_gate'][i, :, 0].cpu()
                    result_entry['debug_fusion_gate_mean'] = float(gate_i.mean())
                    result_entry['debug_fusion_gate_std'] = float(gate_i.std())

            all_results.append(result_entry)

        if (batch_idx + 1) % 50 == 0:
            print(f"  推理进度: {batch_idx + 1}/{len(dataloader)}")

    return all_results


# ============================================================
# 9. 评估 + 保存结果
# ============================================================
def evaluate_and_save(all_results, output_dir):
    """
    计算指标、保存 JSON 结果文件。

    输出文件：
      - test_results.json   : 所有样本的逐条结果
      - summary.json        : 汇总统计
      - failure_samples.json: 无有效候选的样本
      - worst_samples_top20.json : endpoint error 最大的前 20 个样本
    """

    # ---- 逐样本计算 ----
    records = []
    errors = []
    failure_samples = []
    total_lines = 0
    num_no_lines = 0
    num_no_candidate = 0

    for r in all_results:
        stem = r['stem']
        gt_line = r['gt_line']
        pred = r['pred_result']
        num_lines = r['num_lines']
        total_lines += num_lines

        rec = {
            'stem': stem,
            'gt_line': gt_line.tolist(),
            'has_candidate': pred['has_candidate'],
            'num_raw_lines': pred['num_raw_lines'],
            'best_score': pred['best_score'],
        }

        # 保存 horizon head 调试信息到 per-sample record
        if r.get('has_horizon_head'):
            rec['has_horizon_head'] = True
            rec['debug_raw_det_score_mean'] = r.get('debug_raw_det_score_mean')
            rec['debug_raw_calibrated_score_mean'] = r.get('debug_raw_calibrated_score_mean')
            rec['debug_horizon_delta_mean'] = r.get('debug_horizon_delta_mean')
            rec['debug_horizon_delta_std'] = r.get('debug_horizon_delta_std')
            rec['debug_combined_score_mean'] = r.get('debug_combined_score_mean')
            rec['debug_fusion_gate_mean'] = r.get('debug_fusion_gate_mean')
            rec['debug_fusion_gate_std'] = r.get('debug_fusion_gate_std')

        if num_lines == 0:
            num_no_lines += 1

        if pred['has_candidate']:
            err = compute_endpoint_error(pred['best_line'], gt_line)
            metrics = compute_unified_metrics(pred['best_line'], gt_line, IMG_SIZE)
            rec['best_line'] = pred['best_line'].tolist()
            rec['endpoint_error'] = err
            rec['rho_err'] = metrics['rho_err']
            rec['theta_err'] = metrics['theta_err']
            rec['line_dist'] = metrics['line_dist']
            rec['edge_y'] = metrics['edge_y']
            rec['ve'] = metrics['ve']
            rec['ae'] = metrics['ae']
            errors.append(err)
        else:
            rec['best_line'] = None
            rec['endpoint_error'] = None
            num_no_candidate += 1
            failure_samples.append({'stem': stem, 'num_raw_lines': pred['num_raw_lines']})

        records.append(rec)

    # ---- 统计指标 ----
    errors_np = np.array(errors) if errors else np.array([])
    num_valid = len(errors_np)

    summary = {
        'mode': MODE,
        'weights_path': WEIGHTS_PATH,
        'has_horizon_head': any(r.get('has_horizon_head', False) for r in all_results),
        'num_samples': len(all_results),
        'num_images_with_no_lines': num_no_lines,
        'num_images_with_no_candidate': num_no_candidate,
        'num_valid_predictions': num_valid,
        'avg_num_lines': total_lines / max(len(all_results), 1),
    }

    if num_valid > 0:
        summary['mean_endpoint_error'] = float(np.mean(errors_np))
        summary['median_endpoint_error'] = float(np.median(errors_np))
        summary['max_endpoint_error'] = float(np.max(errors_np))
        summary['min_endpoint_error'] = float(np.min(errors_np))
        summary['std_endpoint_error'] = float(np.std(errors_np))
        summary['pct_le_5'] = float(np.mean(errors_np <= 5) * 100)
        summary['pct_le_10'] = float(np.mean(errors_np <= 10) * 100)
        summary['pct_le_20'] = float(np.mean(errors_np <= 20) * 100)
        summary['pct_le_50'] = float(np.mean(errors_np <= 50) * 100)

        # best / worst sample
        best_idx = np.argmin(errors_np)
        worst_idx = np.argmax(errors_np)
        # 找到 records 中有效预测的对应索引
        valid_records = [r for r in records if r['endpoint_error'] is not None]
        summary['best_sample'] = {
            'stem': valid_records[best_idx]['stem'],
            'endpoint_error': valid_records[best_idx]['endpoint_error'],
        }
        summary['worst_sample'] = {
            'stem': valid_records[worst_idx]['stem'],
            'endpoint_error': valid_records[worst_idx]['endpoint_error'],
        }
    else:
        summary['mean_endpoint_error'] = None
        summary['median_endpoint_error'] = None
        summary['max_endpoint_error'] = None
        summary['pct_le_5'] = None
        summary['pct_le_10'] = None
        summary['pct_le_20'] = None
        summary['pct_le_50'] = None

    # ---- 统一评价指标统计 ----
    valid_records = [r for r in records if r['endpoint_error'] is not None]
    if valid_records:
        rho_errs = np.array([r['rho_err'] for r in valid_records])
        theta_errs = np.array([r['theta_err'] for r in valid_records])
        line_dists = np.array([r['line_dist'] for r in valid_records])
        edge_ys = np.array([r['edge_y'] for r in valid_records])
        ves = np.array([r['ve'] for r in valid_records])  # 带符号
        aes = np.array([r['ae'] for r in valid_records])  # 带符号

        summary['unified_metrics'] = {
            # |ρ| 误差
            'rho_mean': float(np.mean(rho_errs)),
            'rho_median': float(np.median(rho_errs)),
            'rho_p90': float(np.percentile(rho_errs, 90)),
            'rho_p95': float(np.percentile(rho_errs, 95)),
            'rho_max': float(np.max(rho_errs)),
            'pct_rho_le_5': float(np.mean(rho_errs <= 5) * 100),
            'pct_rho_le_10': float(np.mean(rho_errs <= 10) * 100),
            'pct_rho_le_20': float(np.mean(rho_errs <= 20) * 100),
            # θ 误差
            'theta_mean': float(np.mean(theta_errs)),
            'theta_median': float(np.median(theta_errs)),
            'theta_p90': float(np.percentile(theta_errs, 90)),
            'theta_p95': float(np.percentile(theta_errs, 95)),
            'theta_max': float(np.max(theta_errs)),
            'pct_theta_le_1': float(np.mean(theta_errs <= 1) * 100),
            'pct_theta_le_2': float(np.mean(theta_errs <= 2) * 100),
            'pct_theta_le_5': float(np.mean(theta_errs <= 5) * 100),
            # LineDist
            'linedist_mean': float(np.mean(line_dists)),
            'linedist_median': float(np.median(line_dists)),
            'linedist_p95': float(np.percentile(line_dists, 95)),
            'linedist_max': float(np.max(line_dists)),
            # Edge-Y
            'edgey_mean': float(np.mean(edge_ys)),
            'edgey_median': float(np.median(edge_ys)),
            'edgey_p95': float(np.percentile(edge_ys, 95)),
            'edgey_max': float(np.max(edge_ys)),
            # VE (带符号)
            've_mean': float(np.mean(np.abs(ves))),
            've_signed_mean': float(np.mean(ves)),
            've_std': float(np.std(ves)),
            've_p95': float(np.percentile(np.abs(ves), 95)),
            'pct_ve_le_10': float(np.mean(np.abs(ves) <= 10) * 100),
            # AE (带符号)
            'ae_mean': float(np.mean(np.abs(aes))),
            'ae_signed_mean': float(np.mean(aes)),
            'ae_std': float(np.std(aes)),
            'ae_p95': float(np.percentile(np.abs(aes), 95)),
            'pct_ae_le_1': float(np.mean(np.abs(aes) <= 1) * 100),
        }

    # ---- worst top-20 ----
    valid_records = [r for r in records if r['endpoint_error'] is not None]
    valid_records_sorted = sorted(valid_records, key=lambda x: x['endpoint_error'], reverse=True)
    worst_top20 = valid_records_sorted[:20]

    # ---- 保存 ----
    def _save_json(data, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  已保存: {path}")

    _save_json(records, "test_results.json")
    _save_json(summary, "summary.json")
    _save_json(failure_samples, "failure_samples.json")
    _save_json(worst_top20, "worst_samples_top20.json")

    # ---- 打印摘要 ----
    print("\n" + "=" * 60)
    print(f"  模式: {MODE}")
    print(f"  权重: {WEIGHTS_PATH}")
    print(f"  样本数: {summary['num_samples']}")
    print(f"  无线段图像: {summary['num_images_with_no_lines']}")
    print(f"  无候选图像: {summary['num_images_with_no_candidate']}")
    print(f"  平均线段数: {summary['avg_num_lines']:.1f}")
    if num_valid > 0:
        print(f"  Mean Endpoint Error: {summary['mean_endpoint_error']:.2f} px")
        print(f"  Median Endpoint Error: {summary['median_endpoint_error']:.2f} px")
        print(f"  Max Endpoint Error: {summary['max_endpoint_error']:.2f} px")
        print(f"  ≤5px:  {summary['pct_le_5']:.1f}%")
        print(f"  ≤10px: {summary['pct_le_10']:.1f}%")
        print(f"  ≤20px: {summary['pct_le_20']:.1f}%")
        print(f"  ≤50px: {summary['pct_le_50']:.1f}%")

    # ---- 打印统一评价指标 ----
    um = summary.get('unified_metrics')
    if um:
        print(f"  --- 统一评价指标 (letterbox {IMG_SIZE}×{IMG_SIZE}) ---")
        print(f"  |ρ| mean={um['rho_mean']:.2f}  median={um['rho_median']:.2f}  P95={um['rho_p95']:.2f}  max={um['rho_max']:.2f}")
        print(f"  θ   mean={um['theta_mean']:.4f}°  median={um['theta_median']:.4f}°  P95={um['theta_p95']:.4f}°")
        print(f"  LineDist  mean={um['linedist_mean']:.2f}  P95={um['linedist_p95']:.2f}")
        print(f"  Edge-Y    mean={um['edgey_mean']:.2f}  P95={um['edgey_p95']:.2f}")
        print(f"  VE  mean={um['ve_mean']:.2f}±{um['ve_std']:.2f}  P95={um['ve_p95']:.2f}  ≤10px={um['pct_ve_le_10']:.1f}%")
        print(f"  AE  mean={um['ae_mean']:.4f}°±{um['ae_std']:.4f}°  P95={um['ae_p95']:.4f}°  ≤1°={um['pct_ae_le_1']:.1f}%")

    # ---- horizon head 汇总统计 ----
    hh_results = [r for r in all_results if r.get('has_horizon_head')]
    if hh_results:
        avg_raw = np.mean([r['debug_raw_det_score_mean'] for r in hh_results])
        avg_delta = np.mean([r['debug_horizon_delta_mean'] for r in hh_results])
        avg_combined = np.mean([r['debug_combined_score_mean'] for r in hh_results])
        summary['horizon_head_stats'] = {
            'avg_raw_det_score': float(avg_raw),
            'avg_horizon_delta': float(avg_delta),
            'avg_combined_score': float(avg_combined),
        }
        print(f"  --- Horizon Head Stats ---")
        print(f"  Avg raw_det score:  {avg_raw:.4f}")
        print(f"  Avg horizon_delta:  {avg_delta:.4f}")
        print(f"  Avg combined score: {avg_combined:.4f}")

        gate_means = [r['debug_fusion_gate_mean'] for r in hh_results if r.get('debug_fusion_gate_mean') is not None]
        if gate_means:
            gate_mean_all = np.mean(gate_means)
            gate_std_all = np.std(gate_means)
            summary['horizon_head_stats']['avg_fusion_gate'] = float(gate_mean_all)
            summary['horizon_head_stats']['std_fusion_gate'] = float(gate_std_all)
            print(f"  Avg fusion_gate:    {gate_mean_all:.4f} (std across images: {gate_std_all:.4f})")

        raw_cal = [r.get('debug_raw_calibrated_score_mean') for r in hh_results if r.get('debug_raw_calibrated_score_mean') is not None]
        if raw_cal:
            avg_raw_cal = np.mean(raw_cal)
            summary['horizon_head_stats']['avg_raw_calibrated_score'] = float(avg_raw_cal)
            print(f"  Avg raw_calibrated: {avg_raw_cal:.4f}")

    print("=" * 60)

    return summary


def save_visualizations(all_results, output_dir):
    """保存可视化图片到 {output_dir}/vis/ 目录。"""
    vis_dir = os.path.join(output_dir, "vis")
    count = 0

    for r in all_results:
        if count >= SAVE_VIS_MAX:
            break

        stem = r['stem']
        save_path = os.path.join(vis_dir, f"{stem}.png")

        draw_result(
            img_tensor=r['img_tensor'],
            gt_line=r['gt_line'],
            pred_result=r['pred_result'],
            save_path=save_path,
            img_size=IMG_SIZE,
        )
        count += 1

    print(f"  已保存 {count} 张可视化图片到 {vis_dir}")


# ============================================================
# main
# ============================================================
def main():
    print("=" * 60)
    print(f"  Stage-1 测试: MODE={MODE}")
    print(f"  CONFIG_FILE={CONFIG_FILE}")
    print(f"  WEIGHTS_PATH={WEIGHTS_PATH}")
    print("=" * 60)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # ---- 加载配置 ----
    print("\n[1/6] 加载配置...")
    args = load_config(CONFIG_FILE)

    # ---- 构建模型 ----
    print("[2/6] 构建模型...")
    model, postprocessor = build_model(args)
    model.to(device)

    # ---- 加载权重 ----
    print("[3/6] 加载权重...")
    load_weights(model, WEIGHTS_PATH, device)

    # ---- 构建数据集 ----
    print("[4/6] 构建 test 数据集...")
    include_entropy = (MODE in ("entropy", "entropy_a", "entropy_b", "entropy_b_enhanced"))
    test_dataset = build_test_dataset(args)
    print(f"  test 样本数: {len(test_dataset)}")

    collate_fn = build_test_collate_fn()
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- 推理 ----
    print("[5/6] 开始推理...")
    all_results = run_inference(model, postprocessor, test_loader, device, include_entropy)
    print(f"  推理完成，共 {len(all_results)} 个样本")

    # ---- 保存结果 ----
    print("[6/6] 保存结果...")
    output_dir = make_output_dir(SAVE_ROOT, MODE, WEIGHTS_PATH, CONFIG_FILE)
    print(f"  输出目录: {output_dir}")

    evaluate_and_save(all_results, output_dir)

    if SAVE_VIS:
        save_visualizations(all_results, output_dir)

    print("\n[DONE] 测试完成。")


if __name__ == '__main__':
    main()
