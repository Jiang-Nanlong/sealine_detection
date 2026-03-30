"""
inference_engine_fast.py — Jetson Xavier NX 优化版海天线检测推理引擎

针对嵌入式设备的关键优化：
  1. TensorRT FP16 引擎推理 (Method2: 5~6ms/帧)
  2. PyTorch FP16 autocast (Method1: ~430ms/帧)
  3. 跳过 Zero-DCE / Radon / CNN，使用分割边界提取海天线 (Method2)
  4. 异步流水线：推理与显示解耦，视频/摄像头显示可达 30fps
  5. cudnn.benchmark + 预分配 GPU tensor
  6. 快速熵近似 (Method1)
"""

import os
import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

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
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _compute_entropy_map_fast(img_bgr, window_sizes=(5, 11, 21)):
    """快速版多尺度局部熵图（局部方差近似）。"""
    blue = img_bgr[:, :, 0].astype(np.float32)
    channels = []
    for ws in window_sizes:
        mean = cv2.blur(blue, (ws, ws))
        sqmean = cv2.blur(blue ** 2, (ws, ws))
        variance = np.maximum(sqmean - mean ** 2, 0)
        log_var = np.log(variance + 1.0)
        vmin, vmax = log_var.min(), log_var.max()
        if vmax > vmin:
            log_var = (log_var - vmin) / (vmax - vmin)
        channels.append(log_var)
    return np.stack(channels, axis=0).astype(np.float32)


def _compute_entropy_map_single(img_bgr, window_size=11):
    """单尺度局部熵图 (1通道)，用于 LINEA_ENTROPY 模型。"""
    blue = img_bgr[:, :, 0].astype(np.float32)
    mean = cv2.blur(blue, (window_size, window_size))
    sqmean = cv2.blur(blue ** 2, (window_size, window_size))
    variance = np.maximum(sqmean - mean ** 2, 0)
    log_var = np.log(variance + 1.0)
    vmin, vmax = log_var.min(), log_var.max()
    if vmax > vmin:
        log_var = (log_var - vmin) / (vmax - vmin)
    return log_var[np.newaxis, :, :].astype(np.float32)  # (1, H, W)


# ======================================================================
#  方法一：LINEA (Jetson 优化版)
# ======================================================================

class LINEADetectorFast:
    """LINEA Entropy-Enhanced 海天线检测器 — Jetson 优化版。

    优化策略：
      - FP16 推理
      - 输入降至 320×320 (原 640×640)
      - 快速熵近似
      - 预分配 tensor
    """

    def __init__(self, config_path, weights_path, device="cuda", img_size=640):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.model = None
        self.postprocessor = None
        self._config_path = config_path
        self._weights_path = weights_path
        # 预计算归一化系数
        self._mean = np.array([0.538, 0.494, 0.453], dtype=np.float32)
        self._std = np.array([0.257, 0.263, 0.273], dtype=np.float32)
        self._stream = None

    def load(self):
        from util.slconfig import SLConfig
        import method1_linea_entropy.models
        from models.registry import MODULE_BUILD_FUNCS

        cfg = SLConfig.fromfile(str(self._config_path))
        cfg.pretrained = False
        cfg.eval_spatial_size = [self.img_size, self.img_size]

        model_name = getattr(cfg, 'modelname', 'LINEA_ENTROPY_B_ENHANCED')
        cfg.modelname = model_name
        build_fn = MODULE_BUILD_FUNCS.get(model_name)
        self.model, self.postprocessor = build_fn(cfg)
        self.model = self.model.to(self.device)

        # 根据 config 的 entropy_mode 自动决定熵通道数
        self._entropy_mode = getattr(cfg, 'entropy_mode', 'entropy')
        self._entropy_channels = 1 if self._entropy_mode == 'entropy' else 3

        if os.path.isfile(self._weights_path):
            ckpt = _safe_load_state_dict(self._weights_path, str(self.device))
            self.model.load_state_dict(ckpt, strict=False)
            print(f"[LINEA-Fast] Weights loaded: {self._weights_path}")

        self.model.eval()
        # LINEA Transformer 不能完全转换为 FP16（attention 层有 dtype 限制）
        # 使用 autocast 代替硬转换
        self._use_autocast = True

        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Warmup
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._use_autocast):
            dummy = torch.randn(1, 3, self.img_size, self.img_size,
                                device=self.device)
            dummy_ent = torch.randn(1, self._entropy_channels, self.img_size,
                                    self.img_size, device=self.device)
            for _ in range(3):
                try:
                    self.model(dummy, entropy_map=dummy_ent)
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[LINEA-Fast] Ready (FP16, {self.img_size}x{self.img_size})")

    def unload(self):
        del self.model, self.postprocessor
        self.model = self.postprocessor = None
        self._stream = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(self, img_bgr):
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        h_orig, w_orig = img_bgr.shape[:2]
        sz = self.img_size

        # --- 预处理 (在 CPU 上做 resize + normalize) ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (sz, sz))
        img_norm = (img_resized.astype(np.float32) / 255.0 - self._mean) / self._std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(
            device=self.device)

        # --- 快速熵图 (根据模型自动选择单/多尺度) ---
        img_small = cv2.resize(img_bgr, (sz, sz))
        if self._entropy_channels == 1:
            entropy_np = _compute_entropy_map_single(img_small)
        else:
            entropy_np = _compute_entropy_map_fast(img_small)
        entropy_tensor = torch.from_numpy(entropy_np).unsqueeze(0).to(
            device=self.device)

        # --- 推理 (autocast FP16) ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.cuda.amp.autocast(enabled=self._use_autocast):
            outputs = self.model(img_tensor, entropy_map=entropy_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000

        # --- 后处理 ---
        target_sizes = torch.tensor([[sz, sz]], device=self.device)
        results = self.postprocessor(outputs, target_sizes)

        pred_lines = results[0]['lines'].cpu().float().numpy()
        pred_scores = results[0]['scores'].cpu().float().numpy()

        best_line, best_score = self._select_horizon(pred_lines, pred_scores)

        # --- 可视化 ---
        result_img = img_bgr.copy()
        if best_line is not None:
            sx, sy = w_orig / sz, h_orig / sz
            x1, y1, x2, y2 = best_line
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_img, f"Score: {best_score:.3f}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        info = {
            "method": "LINEA Fast (FP16)",
            "infer_ms": infer_ms,
            "fps": 1000.0 / infer_ms if infer_ms > 0 else 0,
            "num_candidates": len(pred_lines),
            "best_score": float(best_score) if best_score is not None else 0,
            "detected": best_line is not None,
        }
        return result_img, info

    def _select_horizon(self, lines, scores, topk=50, max_dev_deg=15.0, min_len_ratio=0.2):
        if len(lines) == 0:
            return None, None
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
            valid = dev <= 30.0
        if not np.any(valid):
            return None, None

        best_idx = np.argmax(scores[valid])
        valid_indices = np.where(valid)[0]
        return lines[valid_indices[best_idx]], scores[valid_indices[best_idx]]


# ======================================================================
#  方法二：UNet 分割 (Jetson 优化版 — 无 Radon / 无 CNN)
# ======================================================================

class UNetSegDetectorFast:
    """UNet 分割海天线检测器 — Jetson 优化版。

    优化策略 (vs 原版 8.2s → 目标 <33ms):
      - 跳过 Zero-DCE 增强 (节省 ~12ms)
      - 跳过 Radon 变换 (节省 ~6334ms!)
      - 跳过 CNN 阶段 (节省 ~1310ms)
      - 降低分辨率: 1024×576 → 512×288
      - FP16 推理
      - cudnn.benchmark = True
      - 海天线从分割 mask 直接提取 (快速 RANSAC 边界拟合)
    """

    def __init__(self, unet_weights, dce_weights, device="cuda",
                 res_w=512, res_h=288):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._unet_weights = unet_weights
        self._dce_weights = dce_weights
        self.unet = None
        self.res_w = res_w
        self.res_h = res_h

    def load(self):
        from unet_model import RestorationGuidedHorizonNet

        self.unet = RestorationGuidedHorizonNet(
            num_classes=2,
            dce_weights_path=self._dce_weights
        )
        # 禁用 DCE 以节省推理时间
        self.unet.dce_net = None

        self.unet = self.unet.to(self.device)

        if os.path.isfile(self._unet_weights):
            state = _safe_load_state_dict(self._unet_weights, str(self.device))
            self.unet.load_state_dict(state, strict=False)
            print(f"[UNet-Fast] Weights loaded: {self._unet_weights}")

        self.unet.eval()
        self.unet.half()  # FP16

        # Warmup — 让 cudnn autotuner 找到最优 kernel
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.res_h, self.res_w,
                                device=self.device, dtype=torch.float16) * 0.5 + 0.5
            for _ in range(5):
                self.unet(dummy, target=None,
                          enable_restoration=False, enable_segmentation=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[UNet-Fast] Ready (FP16, no-DCE, seg-only, {self.res_w}x{self.res_h})")

    def unload(self):
        del self.unet
        self.unet = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(self, img_bgr):
        if self.unet is None:
            raise RuntimeError("Model not loaded.")

        h_orig, w_orig = img_bgr.shape[:2]
        t0 = time.time()

        # --- 预处理（优化：直接 resize to BGR，跳过 cvtColor，用 GPU 转换）---
        img_resized = cv2.resize(img_bgr, (self.res_w, self.res_h))
        # BGR→RGB + HWC→CHW + normalize 一步完成
        tensor = torch.from_numpy(img_resized[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(
            device=self.device, dtype=torch.float16, non_blocking=True) * (1.0 / 255.0)

        # --- UNet 分割推理 ---
        _, seg_logits, _ = self.unet(
            tensor, target=None,
            enable_restoration=False,
            enable_segmentation=True
        )

        # argmax 在 GPU 上完成，然后整列采样也在 GPU 上
        seg_mask_gpu = seg_logits.argmax(dim=1)[0]  # (H, W) on GPU

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000

        # --- 海天线提取 (GPU 加速列采样 + CPU 拟合) ---
        t_post0 = time.time()
        horizon_line = self._horizon_from_segmask_gpu(seg_mask_gpu)
        t_post1 = time.time()
        post_ms = (t_post1 - t_post0) * 1000

        total_ms = infer_ms + post_ms

        # --- 可视化 ---
        result_img = img_bgr.copy()
        if horizon_line is not None:
            sx, sy = w_orig / self.res_w, h_orig / self.res_h
            x1, y1, x2, y2 = horizon_line
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        info = {
            "method": "UNet-Seg Fast (FP16)",
            "infer_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 0 else 0,
            "unet_ms": infer_ms,
            "post_ms": post_ms,
            "detected": horizon_line is not None,
        }
        return result_img, info

    def _horizon_from_segmask_gpu(self, seg_mask_gpu):
        """快速从 GPU 分割 mask 中提取海天线 — GPU 列采样 + CPU 拟合。"""
        h, w = seg_mask_gpu.shape

        # GPU 上均匀采样 32 列
        num_samples = 32
        cols = torch.linspace(0, w - 1, num_samples, device=seg_mask_gpu.device).long()
        sampled = seg_mask_gpu[:, cols]  # (H, 32)

        # GPU 上找每列的第一个跳变点
        diff = sampled[1:, :] - sampled[:-1, :]  # (H-1, 32)
        # 对每列找第一个非零位置
        has_transition = (diff != 0).float()  # (H-1, 32)

        # argmax 找第一个 True（如果全为 False 则返回 0）
        first_trans = has_transition.argmax(dim=0)  # (32,)
        any_trans = has_transition.any(dim=0)  # (32,) bool

        # 转到 CPU
        cols_np = cols.cpu().numpy().astype(np.float64)
        first_trans_np = first_trans.cpu().numpy().astype(np.float64)
        any_trans_np = any_trans.cpu().numpy()

        # 只保留有跳变的列
        valid = any_trans_np
        if valid.sum() < 4:
            return None

        x_pts = cols_np[valid]
        y_pts = first_trans_np[valid]

        # 最小二乘拟合 (足够快，<0.1ms)
        coeffs = np.polyfit(x_pts, y_pts, 1)
        slope, intercept = coeffs[0], coeffs[1]

        # 简单去离群点：移除残差 > 5px 的点并重新拟合
        residuals = np.abs(y_pts - (slope * x_pts + intercept))
        inlier = residuals < 5.0
        if inlier.sum() >= 2:
            coeffs = np.polyfit(x_pts[inlier], y_pts[inlier], 1)
            slope, intercept = coeffs[0], coeffs[1]

        y1 = slope * 0 + intercept
        y2 = slope * (w - 1) + intercept
        return (0, y1, w - 1, y2)


# ======================================================================
#  方法二 (TensorRT): UNet 分割 — TensorRT FP16 引擎
# ======================================================================

class UNetSegDetectorTRT:
    """UNet 分割海天线检测器 — TensorRT FP16 引擎版。

    使用预编译的 TensorRT .engine 文件:
      weights/unet_seg_288x512.engine  (由 export_trt.py 生成)

    性能: ~6ms/帧 (>160 FPS) on Jetson Xavier NX
    """

    def __init__(self, engine_path, device="cuda", res_w=512, res_h=288):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._engine_path = engine_path
        self.res_w = res_w
        self.res_h = res_h
        self._context = None
        self._input_gpu = None
        self._output_gpu = None

    def load(self):
        if not _TRT_AVAILABLE:
            raise RuntimeError("tensorrt not installed")
        if not os.path.isfile(self._engine_path):
            raise FileNotFoundError(f"TRT engine not found: {self._engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self._engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self._context = engine.create_execution_context()
        self._engine = engine

        # 预分配 GPU tensor（避免每帧分配）
        self._input_gpu = torch.zeros(
            1, 3, self.res_h, self.res_w,
            dtype=torch.float32, device=self.device
        )
        self._output_gpu = torch.zeros(
            1, 2, self.res_h, self.res_w,
            dtype=torch.float32, device=self.device
        )

        # Warmup
        for _ in range(10):
            self._context.execute_async_v2(
                [self._input_gpu.data_ptr(), self._output_gpu.data_ptr()],
                torch.cuda.current_stream().cuda_stream,
            )
        torch.cuda.synchronize()
        print(f"[UNet-TRT] Ready ({self.res_w}x{self.res_h} FP16)")

    def unload(self):
        del self._context, self._engine, self._input_gpu, self._output_gpu
        self._context = self._engine = None
        self._input_gpu = self._output_gpu = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def detect(self, img_bgr):
        if self._context is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h_orig, w_orig = img_bgr.shape[:2]
        t0 = time.time()

        # --- 预处理: resize + BGR→RGB + normalize → 预分配 GPU tensor ---
        img_resized = cv2.resize(img_bgr, (self.res_w, self.res_h))
        img_rgb = img_resized[:, :, ::-1].copy()  # BGR→RGB
        # HWC float32 [0,1] → CHW tensor
        img_f32 = img_rgb.astype(np.float32) * (1.0 / 255.0)
        self._input_gpu.copy_(
            torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0),
            non_blocking=True,
        )

        # --- TensorRT 推理 ---
        self._context.execute_async_v2(
            [self._input_gpu.data_ptr(), self._output_gpu.data_ptr()],
            torch.cuda.current_stream().cuda_stream,
        )

        # argmax on GPU
        seg_mask_gpu = self._output_gpu.argmax(dim=1)[0]  # (H, W)
        torch.cuda.synchronize()
        t1 = time.time()
        infer_ms = (t1 - t0) * 1000

        # --- 海天线提取 ---
        t_post0 = time.time()
        horizon_line = self._horizon_from_segmask_gpu(seg_mask_gpu)
        t_post1 = time.time()
        post_ms = (t_post1 - t_post0) * 1000
        total_ms = infer_ms + post_ms

        # --- 可视化 ---
        result_img = img_bgr.copy()
        if horizon_line is not None:
            sx, sy = w_orig / self.res_w, h_orig / self.res_h
            x1, y1, x2, y2 = horizon_line
            x1_px, x2_px = int(x1 * sx), int(x2 * sx)
            y1_px, y2_px = int(y1 * sy), int(y2 * sy)
            cv2.line(result_img, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 255), 2)

        info = {
            "method": "UNet-Seg TRT (FP16)",
            "infer_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 0 else 0,
            "unet_ms": infer_ms,
            "post_ms": post_ms,
            "detected": horizon_line is not None,
        }
        return result_img, info

    def _horizon_from_segmask_gpu(self, seg_mask_gpu):
        """快速从 GPU 分割 mask 中提取海天线 — GPU 列采样 + CPU 拟合。"""
        h, w = seg_mask_gpu.shape
        num_samples = 32
        cols = torch.linspace(0, w - 1, num_samples, device=seg_mask_gpu.device).long()
        sampled = seg_mask_gpu[:, cols]  # (H, 32)

        diff = sampled[1:, :] - sampled[:-1, :]
        has_transition = (diff != 0).float()
        first_trans = has_transition.argmax(dim=0)
        any_trans = has_transition.any(dim=0)

        cols_np = cols.cpu().numpy().astype(np.float64)
        first_trans_np = first_trans.cpu().numpy().astype(np.float64)
        any_trans_np = any_trans.cpu().numpy()

        valid = any_trans_np
        if valid.sum() < 4:
            return None

        x_pts = cols_np[valid]
        y_pts = first_trans_np[valid]

        coeffs = np.polyfit(x_pts, y_pts, 1)
        slope, intercept = coeffs[0], coeffs[1]

        residuals = np.abs(y_pts - (slope * x_pts + intercept))
        inlier = residuals < 5.0
        if inlier.sum() >= 2:
            coeffs = np.polyfit(x_pts[inlier], y_pts[inlier], 1)
            slope, intercept = coeffs[0], coeffs[1]

        y1 = slope * 0 + intercept
        y2 = slope * (w - 1) + intercept
        return (0, y1, w - 1, y2)


# ======================================================================
#  统一接口（兼容 app/main.py 的调用方式）
# ======================================================================

def create_detector(method, **kwargs):
    """工厂函数，创建对应方法的快速检测器。"""
    if method in ("linea", "linea_n"):
        return LINEADetectorFast(**kwargs)
    elif method == "unet_trt":
        return UNetSegDetectorTRT(**kwargs)
    elif method == "unet_seg":
        return UNetSegDetectorFast(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# ======================================================================
#  异步流水线包装器（用于达到 30fps 吞吐量）
# ======================================================================
import threading
from collections import deque


class AsyncDetectorPipeline:
    """异步推理流水线：将视频捕获与推理解耦。

    工作原理：
      - 推理线程持续从最新帧中取数据进行检测
      - 主线程负责视频捕获和显示
      - 即使单帧推理 >33ms，显示帧率仍可达 30fps
        （显示上一帧的结果叠加到当前帧）
    """

    def __init__(self, detector):
        self.detector = detector
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = None
        self._latest_info = None
        self._running = False
        self._thread = None

    def start(self):
        """启动推理线程。"""
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止推理线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit_frame(self, img_bgr):
        """提交一帧图像供推理（非阻塞，总是用最新帧）。"""
        with self._lock:
            self._latest_frame = img_bgr

    def get_result(self):
        """获取最近一次推理结果。返回 (result_img, info) 或 (None, None)。"""
        with self._lock:
            return self._latest_result, self._latest_info

    def _inference_loop(self):
        """推理线程主循环。"""
        while self._running:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None  # 避免重复处理

            if frame is None:
                time.sleep(0.001)
                continue

            try:
                result_img, info = self.detector.detect(frame)
                with self._lock:
                    self._latest_result = result_img
                    self._latest_info = info
            except Exception as e:
                print(f"[AsyncPipeline] Error: {e}")

