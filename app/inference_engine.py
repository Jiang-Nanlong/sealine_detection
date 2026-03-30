"""
inference_engine.py — 海天线检测推理引擎

封装两种方法的推理流程：
  方法一：LINEA Entropy-Enhanced Transformer
  方法二：双分支 UNet + Gradient-Radon + ResNet-34
"""

import os
import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np

# 在导入 matplotlib 之前设为非交互后端，避免与 PyQt5 冲突
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F

# ---- 项目路径 ----
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "method1_linea_entropy"))
sys.path.insert(0, str(_PROJECT_ROOT / "method1_linea_entropy" / "LINEA"))
sys.path.insert(0, str(_PROJECT_ROOT / "method2_unet_radon"))


# ======================================================================
#  辅助函数
# ======================================================================

def _safe_load_state_dict(ckpt_path, device="cpu"):
    """兼容多种 checkpoint 格式。"""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _compute_entropy_map(img_bgr, window_sizes=(5, 11, 21)):
    """
    计算多尺度局部熵图 (MSLEP)，用于 LINEA 方法。
    返回 (3, H, W) float32 numpy array。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    channels = []
    for ws in window_sizes:
        # 局部直方图熵
        pad = ws // 2
        h, w = gray.shape
        entropy = np.zeros((h, w), dtype=np.float32)
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        for i in range(h):
            for j in range(w):
                patch = padded[i:i + ws, j:j + ws].flatten()
                hist, _ = np.histogram(patch, bins=256, range=(0, 256))
                hist = hist[hist > 0].astype(np.float32)
                hist /= hist.sum()
                entropy[i, j] = -np.sum(hist * np.log2(hist + 1e-10))
        # normalize to [0, 1]
        emin, emax = entropy.min(), entropy.max()
        if emax > emin:
            entropy = (entropy - emin) / (emax - emin)
        channels.append(entropy)
    return np.stack(channels, axis=0)


def _compute_entropy_map_fast(img_bgr, window_sizes=(5, 11, 21)):
    """
    快速版本的多尺度局部熵图（使用 blue 通道 + 滤波近似）。
    """
    blue = img_bgr[:, :, 0].astype(np.float32)  # blue channel
    channels = []
    for ws in window_sizes:
        # 使用局部方差作为熵的近似
        mean = cv2.blur(blue, (ws, ws))
        sqmean = cv2.blur(blue ** 2, (ws, ws))
        variance = np.maximum(sqmean - mean ** 2, 0)
        # log(variance) 近似局部熵
        log_var = np.log(variance + 1.0)
        vmin, vmax = log_var.min(), log_var.max()
        if vmax > vmin:
            log_var = (log_var - vmin) / (vmax - vmin)
        channels.append(log_var)
    return np.stack(channels, axis=0).astype(np.float32)


# ======================================================================
#  方法一：LINEA Entropy-Enhanced Transformer
# ======================================================================

class LINEADetector:
    """LINEA Entropy-Enhanced 海天线检测器。"""

    def __init__(self, config_path, weights_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = 640
        self.model = None
        self.postprocessor = None
        self._config_path = config_path
        self._weights_path = weights_path

    def load(self):
        """加载模型（延迟加载，避免占用不必要内存）。"""
        from util.slconfig import SLConfig
        import method1_linea_entropy.models  # 注册模型
        from models.registry import MODULE_BUILD_FUNCS

        cfg = SLConfig.fromfile(str(self._config_path))
        cfg.pretrained = False
        sz = getattr(cfg, 'eval_spatial_size', self.img_size)
        if isinstance(sz, int):
            sz = [sz, sz]
        cfg.eval_spatial_size = sz

        model_name = getattr(cfg, 'modelname', 'LINEA_ENTROPY_B_ENHANCED')
        cfg.modelname = model_name
        build_fn = MODULE_BUILD_FUNCS.get(model_name)
        self.model, self.postprocessor = build_fn(cfg)
        self.model = self.model.to(self.device)

        # Load weights
        if os.path.isfile(self._weights_path):
            ckpt = _safe_load_state_dict(self._weights_path, str(self.device))
            info = self.model.load_state_dict(ckpt, strict=False)
            print(f"[LINEA] Weights loaded: {self._weights_path}")
            if info.missing_keys:
                print(f"  Missing: {len(info.missing_keys)} keys")
        else:
            print(f"[LINEA] WARNING: No weights at {self._weights_path}, using random init")

        self.model.eval()
        # Warm-up
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)
            try:
                self.model(dummy)
            except Exception:
                pass
        print("[LINEA] Model ready.")

    def unload(self):
        """释放模型内存。"""
        del self.model
        del self.postprocessor
        self.model = None
        self.postprocessor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(self, img_bgr):
        """
        对一张 BGR 图像进行海天线检测。

        Returns:
            result_img: BGR 图像（带标注）
            info: dict 包含推理时间、检测线段等
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h_orig, w_orig = img_bgr.shape[:2]

        # --- 预处理：letterbox 到 640x640 ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # 归一化
        mean = np.array([0.538, 0.494, 0.453], dtype=np.float32)
        std = np.array([0.257, 0.263, 0.273], dtype=np.float32)
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # --- 计算熵图 ---
        img_640 = cv2.resize(img_bgr, (self.img_size, self.img_size))
        entropy_np = _compute_entropy_map_fast(img_640)  # (3, 640, 640)
        entropy_tensor = torch.from_numpy(entropy_np).unsqueeze(0).to(self.device)

        # --- 推理 ---
        t0 = time.time()
        outputs = self.model(img_tensor, entropy_map=entropy_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000

        # --- 后处理 ---
        target_sizes = torch.tensor([[self.img_size, self.img_size]], device=self.device)
        results = self.postprocessor(outputs, target_sizes)

        pred_lines = results[0]['lines'].cpu().numpy()   # (N, 4)
        pred_scores = results[0]['scores'].cpu().numpy()  # (N,)

        # --- 选择海天线候选 ---
        best_line, best_score = self._select_horizon(pred_lines, pred_scores)

        # --- 可视化 ---
        result_img = img_bgr.copy()
        if best_line is not None:
            # 从 640x640 坐标映射回原图
            sx, sy = w_orig / self.img_size, h_orig / self.img_size
            x1, y1, x2, y2 = best_line
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_img, f"Score: {best_score:.3f}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        info = {
            "method": "LINEA Entropy-Enhanced",
            "infer_ms": infer_ms,
            "fps": 1000.0 / infer_ms if infer_ms > 0 else 0,
            "num_candidates": len(pred_lines),
            "best_score": float(best_score) if best_score is not None else 0,
            "detected": best_line is not None,
        }
        return result_img, info

    def _select_horizon(self, lines, scores, topk=50, max_dev_deg=15.0, min_len_ratio=0.2):
        """选择最佳海天线候选。"""
        if len(lines) == 0:
            return None, None

        # TopK
        if len(lines) > topk:
            idx = np.argsort(scores)[::-1][:topk]
            lines, scores = lines[idx], scores[idx]

        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.abs(np.degrees(np.arctan2(dy, dx)))
        dev = np.minimum(angles, 180.0 - angles)
        lengths = np.sqrt(dx**2 + dy**2)

        valid = (dev <= max_dev_deg) & (lengths >= min_len_ratio * self.img_size)
        if not np.any(valid):
            # 放宽条件
            valid = dev <= 30.0
        if not np.any(valid):
            return None, None

        best_idx = np.argmax(scores[valid])
        valid_indices = np.where(valid)[0]
        return lines[valid_indices[best_idx]], scores[valid_indices[best_idx]]


# ======================================================================
#  方法二：双分支 UNet + Gradient-Radon + ResNet-34
# ======================================================================

class UNetRadonDetector:
    """双分支 UNet + Gradient-Radon + ResNet-34 海天线检测器。"""

    def __init__(self, unet_weights, dce_weights, cnn_weights=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._unet_weights = unet_weights
        self._dce_weights = dce_weights
        self._cnn_weights = cnn_weights
        self.unet = None
        self.radon = None
        self.cnn = None
        self.unet_w, self.unet_h = 1024, 576

    def load(self):
        """加载所有模型。"""
        from unet_model import RestorationGuidedHorizonNet
        from gradient_radon import TextureSuppressedMuSCoWERT

        # UNet
        self.unet = RestorationGuidedHorizonNet(
            num_classes=2,
            dce_weights_path=self._dce_weights
        ).to(self.device)

        if os.path.isfile(self._unet_weights):
            state = _safe_load_state_dict(self._unet_weights, str(self.device))
            self.unet.load_state_dict(state, strict=False)
            print(f"[UNet] Weights loaded: {self._unet_weights}")
        else:
            print(f"[UNet] WARNING: No weights at {self._unet_weights}")

        self.unet.eval()

        # Gradient-Radon
        self.radon = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

        # CNN (ResNet-34)
        if self._cnn_weights and os.path.isfile(self._cnn_weights):
            from cnn_model import HorizonResNet
            self.cnn = HorizonResNet(in_channels=4).to(self.device)
            state = _safe_load_state_dict(self._cnn_weights, str(self.device))
            self.cnn.load_state_dict(state, strict=False)
            self.cnn.eval()
            print(f"[CNN] Weights loaded: {self._cnn_weights}")
        else:
            print("[CNN] No weights, using UNet segmentation-only pipeline")

        print("[UNet+Radon] Model ready.")

    def unload(self):
        """释放模型内存。"""
        del self.unet, self.radon, self.cnn
        self.unet = self.radon = self.cnn = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(self, img_bgr):
        """
        对一张 BGR 图像进行海天线检测。

        Returns:
            result_img: BGR 图像（带标注）
            info: dict
        """
        if self.unet is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h_orig, w_orig = img_bgr.shape[:2]

        # 预处理
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.unet_w, self.unet_h))
        tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        # --- Stage 1: UNet ---
        t0 = time.time()
        restored, seg_logits, _ = self.unet(
            tensor, target=None, enable_restoration=True, enable_segmentation=True
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_unet = time.time()

        # 从分割结果提取海天线
        seg_mask = torch.softmax(seg_logits, dim=1).argmax(dim=1)[0].cpu().numpy()  # (576, 1024)

        # 尝试完整的 Radon + CNN 管线（4 通道：3 radon sinograms + 1 seg edge sinogram）
        horizon_line = None
        t_radon = t_unet
        t_cnn = t_unet

        if self.cnn is not None:
            try:
                # --- Stage 2: Gradient-Radon（在 UNet 复原图上做）---
                restored_np = (restored[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
                restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
                _, _, _, sinograms = self.radon.detect(restored_bgr)
                t_radon = time.time()

                # Stack 3 radon sinograms（零填充居中，和训练一致）
                radon_h, radon_w = 2240, 180
                radon_features = np.zeros((3, radon_h, radon_w), dtype=np.float32)
                for i, sino in enumerate(sinograms):
                    if sino is not None and sino.size > 0:
                        radon_features[i] = self._pad_sinogram(sino, radon_h, radon_w)

                # 第 4 通道：seg mask 后处理 + Canny 边缘 + Radon 变换
                seg_mask_pp = self._postprocess_mask(seg_mask)
                seg_edge = cv2.Canny((seg_mask_pp * 255).astype(np.uint8), 50, 150)
                k = np.ones((3, 3), np.uint8)
                seg_edge = cv2.dilate(seg_edge, k, iterations=1)
                seg_sino = self.radon._radon_gpu(seg_edge.astype(np.float32), self.radon.theta)
                radon_features_seg = self._pad_sinogram(seg_sino, radon_h, radon_w)

                cnn_input = np.concatenate([radon_features, radon_features_seg[np.newaxis]], axis=0)  # (4, 2240, 180)
                cnn_tensor = torch.from_numpy(cnn_input).unsqueeze(0).float().to(self.device)

                # --- Stage 3: ResNet-34（4 通道旧模型，不使用 film_params）---
                pred, conf = self.cnn(cnn_tensor, return_conf=True)
                t_cnn = time.time()

                rho_norm, theta_norm = pred[0].cpu().numpy()
                confidence = conf[0].item()

                # 转换为图像坐标
                horizon_line = self._rho_theta_to_line(
                    rho_norm, theta_norm, self.unet_w, self.unet_h
                )
            except Exception as e:
                print(f"[UNet+Radon] CNN pipeline error: {e}")
                horizon_line = None

        # 如果 CNN 没有结果，用分割 mask 的边界作为海天线
        if horizon_line is None:
            horizon_line = self._horizon_from_segmask(seg_mask)

        total_ms = (time.time() - t0) * 1000

        # --- 可视化 ---
        result_img = img_bgr.copy()
        if horizon_line is not None:
            sx, sy = w_orig / self.unet_w, h_orig / self.unet_h
            x1, y1, x2, y2 = horizon_line
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        info = {
            "method": "UNet + Radon + ResNet-34",
            "infer_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 0 else 0,
            "unet_ms": (t_unet - t0) * 1000,
            "radon_ms": (t_radon - t_unet) * 1000,
            "cnn_ms": (t_cnn - t_radon) * 1000,
            "detected": horizon_line is not None,
        }
        return result_img, info

    def _rho_theta_to_line(self, rho_norm, theta_norm, img_w, img_h):
        """将归一化的 (rho, theta) 转为图像坐标线段。"""
        diagonal = math.sqrt(img_w**2 + img_h**2)
        radon_h = 2240
        pad = (radon_h - diagonal) / 2
        rho_px = rho_norm * (radon_h - 1) - pad - diagonal / 2
        theta_deg = theta_norm * 180.0
        theta_rad = math.radians(theta_deg)

        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        cx, cy = img_w / 2, img_h / 2
        if abs(sin_t) > 1e-6:
            y_at_0 = (rho_px - (0 - cx) * cos_t) / sin_t + cy
            y_at_w = (rho_px - (img_w - cx) * cos_t) / sin_t + cy
            return (0, y_at_0, img_w, y_at_w)
        return None

    @staticmethod
    def _pad_sinogram(sino, target_h, target_w):
        """零填充居中（和 make_fusion_cache.py 的 process_sinogram 一致）。"""
        sino = sino.astype(np.float32)
        mi, ma = float(sino.min()), float(sino.max())
        if ma - mi > 1e-6:
            sino = (sino - mi) / (ma - mi)
        else:
            sino = np.zeros_like(sino, dtype=np.float32)
        h_curr = sino.shape[0]
        container = np.zeros((target_h, target_w), dtype=np.float32)
        start_h = (target_h - h_curr) // 2
        if h_curr <= target_h:
            container[start_h:start_h + h_curr, :] = sino
        else:
            crop_start = (h_curr - target_h) // 2
            container[:, :] = sino[crop_start:crop_start + target_h, :]
        return container

    @staticmethod
    def _postprocess_mask(mask_np):
        """连通域后处理：只保留触顶的 sky 区域（和训练一致）。"""
        sky = (mask_np == 1).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
        keep = np.zeros_like(sky, dtype=np.uint8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_TOP] <= 0:
                keep[labels == i] = 1
        out = mask_np.copy()
        out[(mask_np == 1) & (keep == 0)] = 0
        return out

    def _horizon_from_segmask(self, seg_mask):
        """从分割 mask 中提取海天线（sky=1 的下边界）。"""
        h, w = seg_mask.shape
        # 找每列中 sky 和 sea 的边界
        boundary_y = []
        step = max(1, w // 20)
        sample_cols = list(range(0, w, step))
        for c in sample_cols:
            col = seg_mask[:, c]
            transitions = np.where(np.diff(col) != 0)[0]
            if len(transitions) > 0:
                boundary_y.append((c, transitions[0]))

        if len(boundary_y) < 2:
            return None

        # 用最小二乘拟合直线
        pts = np.array(boundary_y, dtype=np.float64)
        x_pts, y_pts = pts[:, 0], pts[:, 1]
        if len(x_pts) >= 2:
            coeffs = np.polyfit(x_pts, y_pts, 1)
            y1 = coeffs[0] * 0 + coeffs[1]
            y2 = coeffs[0] * (w - 1) + coeffs[1]
            return (0, y1, w - 1, y2)
        return None
