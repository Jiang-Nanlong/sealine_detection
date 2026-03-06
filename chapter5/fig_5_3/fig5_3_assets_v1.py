# -*- coding: utf-8 -*-
"""export_fig5_3_assets.py

把 Fig5-3（双流正弦图融合回归框架）里用到的“中间过程图”从你的代码真实跑出来并导出。

导出内容（每个样本一个文件夹）
- 00_input_unet.png          : 输入图 I（缩放到 1024x576）
- 01_restored.png            : Stage-1 复原图 I_rest
- 02_mask_argmax.png         : Stage-1 分割 argmax 掩码 M
- 03_mask_pp.png             : 后处理掩码 M_pp（取顶部连通域等）
- 04_prob_sky.png            : 分割概率图 P(sky)
- 05_sem_edges.png           : 语义边缘（Canny + 可选膨胀）= M_sem（与你 make_fusion_cache.py 一致）
- 10_grad_map_s1.png         : 梯度流多尺度 weight map（scale=1）
- 11_grad_map_s2.png         : 梯度流多尺度 weight map（scale=2）
- 12_grad_map_s3.png         : 梯度流多尺度 weight map（scale=3）
- 20_sino_grad_s1.png        : 梯度流正弦图（scale=1，原始尺寸）
- 21_sino_grad_s2.png        : 梯度流正弦图（scale=2，原始尺寸）
- 22_sino_grad_s3.png        : 梯度流正弦图（scale=3，原始尺寸）
- 23_sino_sem.png            : 语义边缘正弦图（原始尺寸）
- 30_sino_grad_s1_proc.png   : padding+归一化后的正弦图（CNN 输入尺寸 2240x180）
- 31_sino_grad_s2_proc.png
- 32_sino_grad_s3_proc.png
- 33_sino_sem_proc.png
- 40_stage2_heatmap.png      : (可选) Stage-2 2D heatmap A(ρ,θ)
- 41_stage2_heatmap_mark.png : (可选) heatmap 标注最大点/GT 点
- 90_preview_montage.png     : 汇总预览（方便你快速检查）
- 99_summary.txt             : 记录 img_name / GT label / (可选) CNN 预测等

使用方式
1) 把本脚本放到你的项目根目录（与 unet_model.py / gradient_radon.py / cnn_model.py 同级）
2) 只修改下面 Global Config（路径、权重、输出目录、seed 等）
3) PyCharm 直接 Run

说明
- 全程只保存 png，不 plt.show()，避免你之前 PyCharm 的 matplotlib backend 报错。
- Stage-2 heatmap：脚本会尽量按 cnn_model.py 的“soft-argmax 分布”导出；
  若你的 cnn_model.py 属性名不同，只需在 forward_with_heatmap() 里对齐几行即可。

"""

import os
import sys
import glob
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =========================
# Global Config（只改这里）
# =========================
# 数据根目录：包含 GroundTruth.csv / MU-SID / FusionCache_xxx
DATASET_ROOT = PROJECT_ROOT / "Hashmani's Dataset"

# GroundTruth.csv（make_fusion_cache.py 默认：Hashmani's Dataset/GroundTruth.csv）
CSV_PATH = DATASET_ROOT / "GroundTruth.csv"

# MU-SID 图像目录（make_fusion_cache.py 默认：Hashmani's Dataset/MU-SID）
IMG_DIR = DATASET_ROOT / "MU-SID"

# (推荐) 离线缓存目录：make_fusion_cache.py 生成的 FusionCache_1024x576 或 FusionCache_new_1024x576
# 目录结构通常是：CACHE_ROOT/{train,val,test}/*.npy
CACHE_ROOT = DATASET_ROOT / "FusionCache_new_1024x576"
CACHE_SPLIT = "test"  # "train" / "val" / "test"

# Stage-1 权重路径（make_fusion_cache.py 默认）
RGHNET_CKPT = PROJECT_ROOT / "weights" / "rghnet_best_c2.pth"
DCE_WEIGHTS = PROJECT_ROOT / "weights" / "Epoch99.pth"

# Stage-2 权重路径（可选；不需要 heatmap 就关掉 ENABLE_STAGE2）
ENABLE_STAGE2 = True
FUSION_CKPT = PROJECT_ROOT / "weights_new" / "best_fusion_cnn_1024x576.pth"

# UNet 输入尺寸（make_fusion_cache.py 默认）
UNET_IN_W = 1024
UNET_IN_H = 576

# 正弦图（CNN 输入）尺寸（make_fusion_cache.py 默认）
RESIZE_H = 2240
RESIZE_W = 180

# 语义边缘参数（make_fusion_cache.py 默认）
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

# mask 后处理参数（make_fusion_cache.py 默认）
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0

# 导出多少个样本
RANDOM_SEED = 42
N_EXAMPLES = 1

# 输出目录
OUT_DIR = Path(__file__).parent / "fig5_3_assets"
FIG_DPI = 220

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils
# =========================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_torch_load(path: str, device: str):
    """Torch 2.1+ supports weights_only=True; older versions do not."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mi, ma = float(x.min()), float(x.max())
    if ma - mi < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mi) / (ma - mi)


def to_u8_gray(x: np.ndarray) -> np.ndarray:
    y = np.clip(norm01(x) * 255.0, 0.0, 255.0).astype(np.uint8)
    return y


def save_gray(path: str, x: np.ndarray) -> None:
    cv2.imwrite(path, to_u8_gray(x))


def save_rgb(path: str, rgb_u8: np.ndarray) -> None:
    assert rgb_u8.dtype == np.uint8 and rgb_u8.ndim == 3 and rgb_u8.shape[2] == 3
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def _read_image_any_ext(img_dir, img_stem_or_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """同 make_fusion_cache.py 的逻辑：GroundTruth.csv 可能只有 stem（无扩展名）。"""
    name = str(img_stem_or_name)
    lower = name.lower()
    candidates = [name] if (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")) else []
    candidates += [name + ".JPG", name + ".jpg", name + ".jpeg", name + ".png"]

    img_dir = Path(img_dir)
    for fn in candidates:
        p = img_dir / fn
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                return im, p.name
    return None, None


def post_process_mask_top_connected(mask: np.ndarray) -> np.ndarray:
    """近似复现 make_fusion_cache.py 的 M_pp：
    - mask: 0/1（0=sea,1=sky）
    - 取与图像顶部相连（或 TOP_TOUCH_TOL 范围内）的 sky 连通域中面积最大的一个
    - 其他全部置 0
    - 可选形态学闭运算平滑
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape[:2]

    if MORPH_CLOSE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=int(MORPH_CLOSE))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m

    best_lab = None
    best_area = -1
    for lab in range(1, num):
        x, y, ww, hh, area = stats[lab]
        # 是否触顶（允许 TOP_TOUCH_TOL 像素缓冲）
        if y <= int(TOP_TOUCH_TOL):
            if area > best_area:
                best_area = area
                best_lab = lab

    if best_lab is None:
        # 若没有触顶连通域，兜底：直接取最大 sky 连通域
        for lab in range(1, num):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_lab = lab

    out = (labels == best_lab).astype(np.uint8)
    return out


def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """同 make_fusion_cache.py：归一化 + 居中 padding 到 (target_h, target_w)。"""
    sino = np.asarray(sino, dtype=np.float32)
    mi, ma = float(sino.min()), float(sino.max())
    if ma - mi > 1e-6:
        sino_norm = (sino - mi) / (ma - mi)
    else:
        sino_norm = np.zeros_like(sino, dtype=np.float32)

    h_curr = sino_norm.shape[0]
    container = np.zeros((target_h, target_w), dtype=np.float32)
    start_h = (target_h - h_curr) // 2
    if h_curr <= target_h:
        container[start_h:start_h + h_curr, :] = sino_norm
    else:
        crop_start = (h_curr - target_h) // 2
        container[:, :] = sino_norm[crop_start:crop_start + target_h, :]

    return container


# =========================
# Stage-1: Restoration+Seg
# =========================

def load_stage1_model() -> torch.nn.Module:
    from unet_model import RestorationGuidedHorizonNet

    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=str(DCE_WEIGHTS)).to(DEVICE)
    sd = safe_torch_load(str(RGHNET_CKPT), DEVICE)
    # 你在 make_fusion_cache.py 里是 strict=False
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_stage1(model: torch.nn.Module, bgr_orig: np.ndarray) -> Dict[str, Any]:
    """输入原图 bgr（任意尺寸），按 make_fusion_cache.py resize 到 1024x576 后跑 Stage-1。"""
    bgr = cv2.resize(bgr_orig, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    inp = torch.from_numpy(rgb).float() / 255.0
    inp = inp.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # make_fusion_cache.py：restored_t, seg_logits, _ = seg_model(inp, None, True, True)
    restored_t, seg_logits, _, _ = model(inp, None, True, True)

    restored_np = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0)
    restored_np = np.clip(restored_np, 0, 255).astype(np.uint8)

    mask_argmax = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # 概率图：取 sky 概率（num_classes=2 默认 [sea, sky]）
    prob = F.softmax(seg_logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)

    return {
        "input_rgb": rgb,
        "restored_rgb": restored_np,
        "mask_argmax": mask_argmax,
        "prob_sky": prob,
    }


# =========================
# Feature + Radon
# =========================

def extract_grad_and_sinos(restored_bgr: np.ndarray) -> Tuple[Dict[int, Dict[str, Any]], List[np.ndarray]]:
    """调用 TextureSuppressedMuSCoWERT.detect() 获取 debug_info 和 sinograms。"""
    from gradient_radon import TextureSuppressedMuSCoWERT

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

    try:
        _, _, debug_info, sinos = detector.detect(restored_bgr)
    except Exception:
        debug_info, sinos = {}, []

    # debug_info: {scale: {'map': weighted_map, 'blurred': blurred}}
    return debug_info, sinos


def radon_sem_edges(edges_u8: np.ndarray) -> np.ndarray:
    from gradient_radon import TextureSuppressedMuSCoWERT

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)
    return detector._radon_gpu(edges_u8, theta_scan)


# =========================
# Stage-2: Fusion CNN (可选)
# =========================

def load_fusion_model() -> Optional[torch.nn.Module]:
    if not ENABLE_STAGE2:
        return None

    # 你仓库里 evaluate_fusion_cnn.py / train_fusion_cnn.py 通常是 from cnn_model import HorizonResNet
    try:
        from cnn_model import HorizonResNet
    except Exception as e:
        print(f"[Stage-2] import cnn_model failed: {e}")
        return None

    # 尽量用最少参数初始化；如果你的 HorizonResNet 需要更多参数，就在这里补。
    try:
        model = HorizonResNet(in_channels=4, img_h=RESIZE_H, img_w=RESIZE_W).to(DEVICE)
    except TypeError:
        # 兜底：不带参数
        model = HorizonResNet().to(DEVICE)

    ckpt = safe_torch_load(str(FUSION_CKPT), DEVICE)
    # 兼容不同保存格式
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    try:
        model.load_state_dict(ckpt, strict=False)
    except Exception as e:
        print(f"[Stage-2] load_state_dict failed: {e}")
        return None

    model.eval()
    return model


@torch.no_grad()
def forward_with_heatmap(model: torch.nn.Module, x: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """尽量导出 soft-argmax 概率 heatmap。

    返回：
      pred (2,)   : [rho_norm, theta_norm]（如果能算到）
      prob (Hf,Wf): softmax 概率分布（2D heatmap）
      logits (Hf,Wf): 未 softmax 的 logits（可选）

    说明：
    - 如果你的 cnn_model.py forward 本身就返回 heatmap，你可以在这里直接用。
    - 否则，脚本尝试复刻典型结构：conv_end -> softmax(flat/temperature) -> reshape。
    """

    # 1) 如果 forward 支持直接返回更多信息，你可以自己改这里
    out = None
    try:
        out = model(x)
    except Exception:
        pass

    # 2) 优先走“显式模块”路径（更稳定地拿到 conv_end 输出）
    if hasattr(model, "conv_end") and hasattr(model, "layer4") and hasattr(model, "conv1"):
        h = x
        h = model.conv1(h)
        if hasattr(model, "bn1"):
            h = model.bn1(h)
        if hasattr(model, "relu"):
            h = model.relu(h)
        if hasattr(model, "maxpool"):
            h = model.maxpool(h)
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            if hasattr(model, lname):
                h = getattr(model, lname)(h)
        if hasattr(model, "cbam"):
            h = model.cbam(h)
        logits = model.conv_end(h)  # (B,1,Hf,Wf)

        b, c, hf, wf = logits.shape
        x_flat = logits.view(b, -1)
        temperature = float(getattr(model, "temperature", 1.0))
        prob_flat = torch.softmax(x_flat / max(temperature, 1e-6), dim=1)
        prob = prob_flat.view(b, hf, wf)

        # 软期望坐标
        # 优先使用模型里 register_buffer 的坐标网格
        if hasattr(model, "coord_grid_rho") and hasattr(model, "coord_grid_theta"):
            rho_grid = getattr(model, "coord_grid_rho")  # (hf,wf)
            theta_grid = getattr(model, "coord_grid_theta")
        else:
            rho_grid = torch.linspace(0, 1, steps=hf, device=prob.device).view(hf, 1).repeat(1, wf)
            theta_grid = torch.linspace(0, 1, steps=wf, device=prob.device).view(1, wf).repeat(hf, 1)

        rho = torch.sum(prob * rho_grid, dim=(1, 2))
        theta = torch.sum(prob * theta_grid, dim=(1, 2))
        pred = torch.stack([rho, theta], dim=1)

        prob_np = prob[0].cpu().numpy().astype(np.float32)
        logits_np = logits[0, 0].cpu().numpy().astype(np.float32)
        pred_np = pred[0].cpu().numpy().astype(np.float32)

        return pred_np, prob_np, logits_np

    # 3) 如果拿不到 heatmap，就只返回 pred（如果 model forward 有输出）
    if isinstance(out, (list, tuple)) and len(out) >= 1:
        out0 = out[0]
    else:
        out0 = out

    if torch.is_tensor(out0) and out0.numel() >= 2:
        pred_np = out0.view(-1)[:2].detach().cpu().numpy().astype(np.float32)
        return pred_np, None, None

    return None, None, None


# =========================
# Main
# =========================

def pick_cache_files() -> List[str]:
    cache_dir = Path(CACHE_ROOT) / CACHE_SPLIT
    files = sorted(cache_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No cache npy found: {cache_dir}/*.npy")

    rs = np.random.RandomState(RANDOM_SEED)
    idxs = rs.choice(len(files), size=min(N_EXAMPLES, len(files)), replace=False)
    return [str(files[i]) for i in idxs]


def load_cache_item(npy_path: str) -> Dict[str, Any]:
    d = np.load(npy_path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == ():
        d = d.item()
    if not isinstance(d, dict):
        raise ValueError(f"Cache file is not a dict: {npy_path}")
    return d


def save_heatmap_png(path: str, heat: np.ndarray, title: str = "") -> None:
    plt.figure(figsize=(4.0, 2.4))
    plt.imshow(heat, aspect="auto")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=FIG_DPI)
    plt.close()


def main():
    seed_everything(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    print("[Load] Stage-1 model...")
    stage1 = load_stage1_model()

    stage2 = None
    if ENABLE_STAGE2:
        print("[Load] Stage-2 model...")
        stage2 = load_fusion_model()

    cache_files = pick_cache_files()
    print(f"[Pick] {len(cache_files)} cache files from {CACHE_SPLIT} (seed={RANDOM_SEED})")

    for npy_path in cache_files:
        idx_name = os.path.splitext(os.path.basename(npy_path))[0]
        cache = load_cache_item(npy_path)

        img_name = cache.get("img_name", None)
        if img_name is None:
            # 兼容旧缓存：可能只有 stem
            img_name = cache.get("img", cache.get("name", idx_name))

        bgr_orig, resolved = _read_image_any_ext(IMG_DIR, img_name)
        if bgr_orig is None:
            print(f"[WARN] cannot read image for cache={npy_path}, img_name={img_name}")
            continue

        # --- Stage-1 ---
        s1 = run_stage1(stage1, bgr_orig)
        input_rgb = s1["input_rgb"]
        restored_rgb = s1["restored_rgb"]
        mask_argmax = s1["mask_argmax"]
        prob_sky = s1["prob_sky"]

        mask_pp = post_process_mask_top_connected(mask_argmax)

        # 语义边缘（与 make_fusion_cache.py 一致：Canny(mask_pp*255) + 可选膨胀）
        edges = cv2.Canny((mask_pp * 255).astype(np.uint8), int(CANNY_LOW), int(CANNY_HIGH))
        if int(EDGE_DILATE) > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=int(EDGE_DILATE))

        # --- 梯度流 + 正弦图 ---
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        debug_info, grad_sinos = extract_grad_and_sinos(restored_bgr)

        # grad_sinos 可能为空（detect 失败时），做兜底
        while len(grad_sinos) < 3:
            grad_sinos.append(np.zeros((int(np.ceil(np.sqrt(UNET_IN_W**2 + UNET_IN_H**2))), RESIZE_W), dtype=np.float32))

        # 语义边缘正弦图（GPU radon）
        sem_sino = radon_sem_edges(edges)

        # processed（CNN 输入）
        grad_sinos_proc = [process_sinogram(s, RESIZE_H, RESIZE_W) for s in grad_sinos[:3]]
        sem_sino_proc = process_sinogram(sem_sino, RESIZE_H, RESIZE_W)

        # --- 输出目录 ---
        safe_img = str(resolved or img_name).replace("/", "_").replace("\\", "_")
        out_sub = Path(OUT_DIR) / f"{idx_name}_{safe_img}"
        ensure_dir(out_sub)

        # --- Save basic images ---
        save_rgb(str(out_sub / "00_input_unet.png"), input_rgb)
        save_rgb(str(out_sub / "01_restored.png"), restored_rgb)

        save_gray(str(out_sub / "02_mask_argmax.png"), mask_argmax)
        save_gray(str(out_sub / "03_mask_pp.png"), mask_pp)
        save_gray(str(out_sub / "04_prob_sky.png"), prob_sky)
        cv2.imwrite(str(out_sub / "05_sem_edges.png"), edges)

        # --- Save grad maps ---
        for s in [1, 2, 3]:
            if s in debug_info and isinstance(debug_info[s], dict) and ("map" in debug_info[s]):
                w_map = debug_info[s]["map"]
                save_gray(str(out_sub / f"1{(s-1)}_grad_map_s{s}.png"), w_map)
            else:
                save_gray(str(out_sub / f"1{(s-1)}_grad_map_s{s}.png"), np.zeros((UNET_IN_H, UNET_IN_W), np.float32))

        # --- Save sinograms (raw + processed) ---
        for i, s in enumerate([1, 2, 3]):
            save_gray(str(out_sub / f"2{i}_sino_grad_s{s}.png"), grad_sinos[i])
            save_gray(str(out_sub / f"3{i}_sino_grad_s{s}_proc.png"), grad_sinos_proc[i])

        save_gray(str(out_sub / "23_sino_sem.png"), sem_sino)
        save_gray(str(out_sub / "33_sino_sem_proc.png"), sem_sino_proc)

        # --- Stage-2 heatmap (optional) ---
        pred2 = None
        prob2 = None
        if stage2 is not None:
            # 优先用我们刚算出来的 4 通道输入（与 make_fusion_cache.py 一致）
            combined = np.stack(grad_sinos_proc + [sem_sino_proc], axis=0).astype(np.float32)
            x = torch.from_numpy(combined).unsqueeze(0).to(DEVICE)

            pred2, prob2, _ = forward_with_heatmap(stage2, x)

            if prob2 is not None:
                save_heatmap_png(str(out_sub / "40_stage2_heatmap.png"), prob2, title="A(rho,theta)")

                # 标注最大点 + GT（若 cache 有 label）
                heat = prob2
                hm = heat.copy()
                # max point
                r_max, c_max = np.unravel_index(np.argmax(hm), hm.shape)

                # GT label（cache['label'] 是在正弦图坐标系下的 (rho_idx, theta_idx) ）
                gt = cache.get("label", None)
                gt_r = gt_c = None
                if isinstance(gt, (list, tuple, np.ndarray)) and len(gt) >= 2:
                    gt_r = int(round(float(gt[0]) * 1.0))
                    gt_c = int(round(float(gt[1]) * 1.0))

                plt.figure(figsize=(4.0, 2.4))
                plt.imshow(heat, aspect="auto")
                plt.scatter([c_max], [r_max], s=40, marker="x")
                if gt_r is not None and gt_c is not None:
                    plt.scatter([gt_c], [gt_r], s=30, marker="o")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(str(out_sub / "41_stage2_heatmap_mark.png"), dpi=FIG_DPI)
                plt.close()

        # --- Summary ---
        with open(str(out_sub / "99_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"cache: {npy_path}\n")
            f.write(f"img_name: {img_name}\n")
            if resolved:
                f.write(f"resolved_img: {resolved}\n")
            f.write(f"UNet size: {UNET_IN_W}x{UNET_IN_H}\n")
            f.write(f"Sino size(proc): {RESIZE_H}x{RESIZE_W}\n")
            if "label" in cache:
                f.write(f"GT label (rho_idx, theta_idx): {cache['label']}\n")
            if pred2 is not None:
                f.write(f"Stage-2 pred (rho_norm, theta_norm or raw): {pred2}\n")

        # --- Montage preview ---
        # 为了让你快速检查，拼一个 2x4 的小预览
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 4)

        ax = fig.add_subplot(gs[0, 0]); ax.imshow(input_rgb); ax.set_title("Input (UNet)"); ax.axis('off')
        ax = fig.add_subplot(gs[0, 1]); ax.imshow(restored_rgb); ax.set_title("Restored"); ax.axis('off')
        ax = fig.add_subplot(gs[0, 2]); ax.imshow(prob_sky, cmap='gray'); ax.set_title("P(sky)"); ax.axis('off')
        ax = fig.add_subplot(gs[0, 3]); ax.imshow(edges, cmap='gray'); ax.set_title("Semantic edges"); ax.axis('off')

        # sinos（processed，CNN 输入）
        ax = fig.add_subplot(gs[1, 0]); ax.imshow(grad_sinos_proc[0], aspect='auto'); ax.set_title("S_grad^1 (proc)"); ax.axis('off')
        ax = fig.add_subplot(gs[1, 1]); ax.imshow(grad_sinos_proc[1], aspect='auto'); ax.set_title("S_grad^2 (proc)"); ax.axis('off')
        ax = fig.add_subplot(gs[1, 2]); ax.imshow(grad_sinos_proc[2], aspect='auto'); ax.set_title("S_grad^3 (proc)"); ax.axis('off')
        ax = fig.add_subplot(gs[1, 3]); ax.imshow(sem_sino_proc, aspect='auto'); ax.set_title("S_sem (proc)"); ax.axis('off')

        plt.tight_layout()
        plt.savefig(str(out_sub / "90_preview_montage.png"), dpi=FIG_DPI)
        plt.close()

        print(f"[OK] Exported -> {out_sub}")


if __name__ == "__main__":
    main()
