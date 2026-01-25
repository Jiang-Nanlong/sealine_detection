import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


# =========================================================
# 0) Utils: parse size / resize (no padding) / letterbox
# =========================================================
def _parse_hw(img_size):
    """
    img_size:
      - int -> (S, S)  (legacy)
      - (H, W) / [H, W] -> (H, W)  (recommended for 16:9: (576,1024))
    """
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        h, w = int(img_size[0]), int(img_size[1])
        return max(1, h), max(1, w)
    s = int(img_size)
    return max(1, s), max(1, s)


def resize_rgb_u8(image_rgb_u8: np.ndarray, out_h: int, out_w: int):
    """Direct resize (no padding). Return resized + meta for coordinate scaling."""
    h, w = image_rgb_u8.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        meta = dict(
            mode="resize",
            scale_x=1.0,
            scale_y=1.0,
            orig_w=w,
            orig_h=h,
            out_w=out_w,
            out_h=out_h,
        )
        return canvas, meta

    sx = out_w / float(w)
    sy = out_h / float(h)
    interp = cv2.INTER_AREA if (out_w < w or out_h < h) else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (out_w, out_h), interpolation=interp)
    meta = dict(
        mode="resize",
        scale_x=sx,
        scale_y=sy,
        orig_w=w,
        orig_h=h,
        out_w=out_w,
        out_h=out_h,
    )
    return resized, meta


def letterbox_rgb_u8(image_rgb_u8: np.ndarray, dst_size: int, pad_value: int = 0):
    """Legacy: keep aspect ratio resize to fit in dst_size x dst_size and pad."""
    h, w = image_rgb_u8.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.zeros((dst_size, dst_size, 3), dtype=np.uint8)
        meta = dict(
            mode="letterbox",
            scale=1.0,
            pad_left=0,
            pad_top=0,
            new_w=dst_size,
            new_h=dst_size,
            orig_w=w,
            orig_h=h,
        )
        return canvas, meta

    scale = min(dst_size / float(w), dst_size / float(h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = max(1, min(dst_size, new_w))
    new_h = max(1, min(dst_size, new_h))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb_u8, (new_w, new_h), interpolation=interp)

    canvas = np.full((dst_size, dst_size, 3), pad_value, dtype=np.uint8)
    pad_left = (dst_size - new_w) // 2
    pad_top = (dst_size - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = dict(
        mode="letterbox",
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        new_w=new_w,
        new_h=new_h,
        orig_w=w,
        orig_h=h,
    )
    return canvas, meta


# =========================================================
# 1) Physical-ish degradation synthesis (with clean pass-through)
#    与 test5/generate_degraded_images.py 对齐的退化类型
# =========================================================

def _add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise. img: float32 [0,1]"""
    noise = np.random.randn(*img.shape).astype(np.float32) * (sigma / 255.0)
    return np.clip(img + noise, 0, 1)


def _add_motion_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Add motion blur with random angle. img: float32 [0,1]"""
    # 随机角度：0-180度（模拟船体多方向晃动）
    angle = random.uniform(0, 180)
    
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    angle_rad = np.radians(angle)
    dx = np.cos(angle_rad) * center
    dy = np.sin(angle_rad) * center
    
    x1, y1 = int(center - dx), int(center - dy)
    x2, y2 = int(center + dx), int(center + dy)
    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)
    
    kernel = kernel / (kernel.sum() + 1e-8)
    
    # Convert to uint8 for cv2.filter2D, then back
    img_u8 = (img * 255).astype(np.uint8)
    blurred = cv2.filter2D(img_u8, -1, kernel)
    return blurred.astype(np.float32) / 255.0


def _add_rain(img: np.ndarray, severity: float) -> np.ndarray:
    """
    Add rain effect with streaks and haze.
    severity: 0.3 (light) ~ 0.9 (heavy)
    img: float32 [0,1]
    """
    h, w = img.shape[:2]
    result = img.copy()
    
    # Rain parameters based on severity
    n_lines = int(1500 + 4500 * severity)  # 1500~6000
    line_len = int(15 + 20 * severity)      # 15~35
    thickness = 1 if severity < 0.6 else 2
    haze_intensity = 0.1 + 0.25 * severity  # 0.1~0.35
    
    # Create rain streak layer
    streak = np.zeros((h, w, 3), dtype=np.float32)
    angle = random.uniform(-15, 15)  # Wind effect
    tan_a = np.tan(np.deg2rad(angle))
    
    for _ in range(n_lines):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        x1 = int(x0 + line_len * tan_a)
        y1 = int(y0 + line_len)
        cv2.line(streak, (x0, y0), (x1, y1), (0.8, 0.8, 0.8), thickness)
    
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=0.5)
    result = np.clip(result + 0.4 * streak, 0, 1)
    
    # Add atmospheric haze
    haze = np.ones_like(result) * 0.78  # ~200/255
    result = result * (1 - haze_intensity) + haze * haze_intensity
    
    return np.clip(result, 0, 1).astype(np.float32)


def _add_fog(img: np.ndarray, severity: float) -> np.ndarray:
    """
    Add fog/haze with depth-aware gradient.
    severity: 0.2 ~ 0.9
    img: float32 [0,1]
    """
    h, w = img.shape[:2]
    result = img.copy()
    
    airlight = random.uniform(0.6, 0.95)
    
    # Vertical gradient (fog thicker at distance = top)
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if random.random() > 0.5:
        base_map = np.tile(ys, (1, w))
    else:
        base_map = np.tile(1 - ys, (1, w))
    
    # Add Perlin-like noise for natural variation
    noise = np.random.randn(h, w).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=10)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    
    fog_map = 0.6 * base_map + 0.4 * noise
    fog_map = fog_map * severity
    fog_map = fog_map[..., None]
    
    result = result * (1 - fog_map) + airlight * fog_map
    return np.clip(result, 0, 1).astype(np.float32)


def _add_low_light(img: np.ndarray, gamma: float) -> np.ndarray:
    """Simulate low light / dusk / overcast. img: float32 [0,1]"""
    return np.clip(np.power(img, gamma), 0, 1).astype(np.float32)


def _add_sun_glare(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Add sun glare / strong reflection on water.
    intensity: 0.3 (light) ~ 0.6 (heavy)
    img: float32 [0,1]
    """
    h, w = img.shape[:2]
    result = img.copy()
    
    # Glare gradient (stronger at top, where horizon is)
    y_coords = np.linspace(0, 1, h).reshape(-1, 1)
    glare_mask = np.clip(1.0 - y_coords * 2.5, 0, 1) ** 2
    glare_mask = np.tile(glare_mask, (1, w))
    
    # Wave-like horizontal variation
    x_variation = np.sin(np.linspace(0, 8 * np.pi, w)) * 0.3 + 0.7
    glare_mask = glare_mask * x_variation.reshape(1, -1)
    
    # Random bright spots (sun sparkles)
    n_spots = int(50 * intensity)
    for _ in range(n_spots):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, int(h * 0.6))
        radius = random.randint(20, 80)
        y, x = np.ogrid[:h, :w]
        spot_mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        glare_mask = np.maximum(glare_mask, spot_mask * random.uniform(0.5, 1.0))
    
    glare_mask = glare_mask[:, :, np.newaxis].astype(np.float32) * intensity
    result = result * (1 - glare_mask) + 1.0 * glare_mask
    
    # Reduce contrast (overexposure)
    result = result * (1 - intensity * 0.3) + 0.5 * intensity * 0.3
    
    return np.clip(result, 0, 1).astype(np.float32)


def _add_jpeg_artifacts(img: np.ndarray, quality: int) -> np.ndarray:
    """Add JPEG compression artifacts. img: float32 [0,1]"""
    img_u8 = (img * 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img_u8, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    # Note: cv2 uses BGR, but our img is RGB. For artifacts this doesn't matter much
    return decoded.astype(np.float32) / 255.0


def _add_resolution_downscale(img: np.ndarray, scale: float) -> np.ndarray:
    """Simulate low resolution by down-then-up scaling. img: float32 [0,1]"""
    h, w = img.shape[:2]
    img_u8 = (img * 255).astype(np.uint8)
    
    small_h, small_w = int(h * scale), int(w * scale)
    small = cv2.resize(img_u8, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return restored.astype(np.float32) / 255.0


def synthesize_degradation(image_rgb_u8: np.ndarray, p_clean: float = 0.30):
    """
    海洋场景退化合成函数（扩展版）
    
    与 test5/generate_degraded_images.py 对齐的退化类型：
      - gaussian_noise: 传感器噪声
      - motion_blur: 船体晃动（随机角度）
      - low_light: 黄昏/阴天
      - fog: 海雾
      - rain: 海上降雨
      - sun_glare: 阳光海面反射
      - jpeg: 压缩伪影
      - lowres: 低分辨率
    
    输入:  uint8 RGB
    输出:  float32 RGB in [0,1]
    
    p_clean: 直接返回原图的概率（教会模型"干净图不要过度处理"）
    """
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # ---- clean pass-through ----
    if random.random() < float(p_clean):
        noise = np.random.randn(h, w, 3).astype(np.float32) * 0.005
        img = np.clip(img + noise, 0, 1)
        return img.astype(np.float32)

    # ---- 随机选择 1~2 种退化类型 ----
    # 定义退化及其参数范围
    # 2026-01-25 更新: 提高 JPEG 概率 (15%→30%)，改善压缩鲁棒性
    degradation_pool = [
        ("gaussian_noise", 0.25),  # 25% 概率
        ("motion_blur", 0.20),     # 20%
        ("low_light", 0.25),       # 25%
        ("fog", 0.30),             # 30%
        ("rain", 0.25),            # 25%
        ("sun_glare", 0.15),       # 15%
        ("jpeg", 0.30),            # 30% (原15%，提高以改善压缩鲁棒性)
        ("lowres", 0.10),          # 10%
    ]
    
    # 独立概率采样
    selected = [name for name, prob in degradation_pool if random.random() < prob]
    
    # 至少应用一种退化
    if not selected:
        selected = [random.choice([name for name, _ in degradation_pool])]
    
    # 限制最多 2 种（避免过度退化）
    if len(selected) > 2:
        selected = random.sample(selected, 2)
    
    # ---- 应用选中的退化 ----
    for deg_type in selected:
        if deg_type == "gaussian_noise":
            sigma = random.uniform(10, 35)  # σ = 10~35
            img = _add_gaussian_noise(img, sigma)
        
        elif deg_type == "motion_blur":
            kernel_size = random.choice([9, 13, 17, 21, 25])
            img = _add_motion_blur(img, kernel_size)
        
        elif deg_type == "low_light":
            gamma = random.uniform(1.5, 2.5)
            img = _add_low_light(img, gamma)
        
        elif deg_type == "fog":
            severity = random.uniform(0.2, 0.6)
            img = _add_fog(img, severity)
        
        elif deg_type == "rain":
            severity = random.uniform(0.3, 0.9)
            img = _add_rain(img, severity)
        
        elif deg_type == "sun_glare":
            intensity = random.uniform(0.2, 0.6)  # 扩展到0.6，覆盖测试时的heavy
            img = _add_sun_glare(img, intensity)
        
        elif deg_type == "jpeg":
            # Q=5~50: 覆盖极端压缩(Q=5~10)到中等压缩(Q=40~50)
            quality = random.randint(5, 50)
            img = _add_jpeg_artifacts(img, quality)
        
        elif deg_type == "lowres":
            scale = random.uniform(0.25, 0.6)  # 0.25x ~ 0.6x
            img = _add_resolution_downscale(img, scale)

    # 总是加一点传感器噪声
    noise = np.random.randn(h, w, 3).astype(np.float32) * 0.008
    img = np.clip(img + noise, 0, 1)

    return img.astype(np.float32)


# 保留旧函数名作为别名，兼容现有代码
def synthesize_rain_fog(image_rgb_u8: np.ndarray, p_clean: float = 0.30):
    """Alias for synthesize_degradation (backward compatibility)."""
    return synthesize_degradation(image_rgb_u8, p_clean=p_clean)


def _scaled_ignore_band(ignore_band_at_384: int, img_h: int) -> int:
    """ignore_band 原来按 384 设计，这里按高度线性放大。"""
    if ignore_band_at_384 <= 0:
        return 0
    return max(1, int(round(ignore_band_at_384 * (img_h / 384.0))))


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _y_on_line_at_x(p1, p2, x: float) -> float:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return y1
    t = (x - x1) / dx
    return y1 + t * (y2 - y1)


# =========================================================
# 2) Stage B/C: CSV label loader (restoration + segmentation)
# =========================================================
class HorizonImageDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        img_size=384,            # int or (H,W)
        mode="joint",            # "joint" or "segmentation"
        ignore_band=10,
        augment=True,
        rotate_prob=0.5,
        max_rotate_deg=45.0,
        val_seed_offset=100000,
        p_clean=0.35,            # NEW: clean pass-through prob
        pad_value=114,           # only used for legacy square letterbox
    ):
        """
        mask 标注:
          - sea=0, sky=1, ignore=255
          - 旋转空洞全部标为 255（ignore），配合 ignore_index=255
        """
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir

        self.out_h, self.out_w = _parse_hw(img_size)
        self.mode = mode
        self.ignore_band = int(ignore_band)

        self.augment = bool(augment)
        self.rotate_prob = float(rotate_prob)
        self.max_rotate_deg = float(max_rotate_deg)
        self.val_seed_offset = int(val_seed_offset)

        self.p_clean = float(p_clean)
        self.pad_value = int(pad_value)

        # legacy flag: square -> keep old letterbox behavior
        self.use_letterbox = (self.out_h == self.out_w)

    def __len__(self):
        return len(self.data)

    def _read_rgb(self, img_name):
        path = os.path.join(self.img_dir, img_name)
        bgr = cv2.imread(path)
        if bgr is None:
            rgb = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
            meta = dict(
                mode="resize",
                scale_x=1.0,
                scale_y=1.0,
                orig_w=self.out_w,
                orig_h=self.out_h,
                out_w=self.out_w,
                out_h=self.out_h,
            )
            return rgb, meta

        rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.use_letterbox:
            rgb, meta = letterbox_rgb_u8(rgb0, self.out_h, pad_value=self.pad_value)
            return rgb, meta

        rgb, meta = resize_rgb_u8(rgb0, self.out_h, self.out_w)
        return rgb, meta

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = str(row.iloc[0])
        try:
            x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
        except Exception:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

        rgb, meta = self._read_rgb(img_name)
        H, W = self.out_h, self.out_w

        # Map endpoints to current resolution
        if meta.get("mode") == "letterbox":
            scale = float(meta["scale"])
            pl = int(meta["pad_left"])
            pt = int(meta["pad_top"])
            p1 = (_clamp_int(x1 * scale + pl, 0, W - 1), _clamp_int(y1 * scale + pt, 0, H - 1))
            p2 = (_clamp_int(x2 * scale + pl, 0, W - 1), _clamp_int(y2 * scale + pt, 0, H - 1))

            # valid ROI where the resized image lies
            nw = int(meta["new_w"])
            nh = int(meta["new_h"])
            roi_left = pl
            roi_top = pt
            roi_right = pl + nw - 1
            roi_bottom = pt + nh - 1
        else:
            sx = float(meta["scale_x"])
            sy = float(meta["scale_y"])
            p1 = (_clamp_int(x1 * sx, 0, W - 1), _clamp_int(y1 * sy, 0, H - 1))
            p2 = (_clamp_int(x2 * sx, 0, W - 1), _clamp_int(y2 * sy, 0, H - 1))

            roi_left, roi_top = 0, 0
            roi_right, roi_bottom = W - 1, H - 1

        # Build mask: sea=0, sky=1, ignore=255
        mask = np.zeros((H, W), dtype=np.uint8)

        # If legacy letterbox, set padding to ignore to avoid learning artifacts
        if meta.get("mode") == "letterbox":
            mask[:, :] = 255  # default ignore
            # set ROI to sea first
            mask[roi_top:roi_bottom + 1, roi_left:roi_right + 1] = 0

        # Fill sky polygon above the line within ROI
        xL = float(roi_left)
        xR = float(roi_right)
        yL = _y_on_line_at_x(p1, p2, xL)
        yR = _y_on_line_at_x(p1, p2, xR)
        yL = _clamp_int(yL, roi_top, roi_bottom)
        yR = _clamp_int(yR, roi_top, roi_bottom)

        pts = np.array(
            [
                [roi_left, roi_top],
                [roi_right, roi_top],
                [roi_right, yR],
                [roi_left, yL],
            ],
            np.int32,
        )
        cv2.fillPoly(mask, [pts], 1)

        # Draw ignore band around horizon line
        thick = _scaled_ignore_band(self.ignore_band, H)
        if thick > 0:
            cv2.line(mask, p1, p2, 255, thick)

        # -------------------------
        # Rotation augmentation (train only)
        # IMPORTANT: image fill must NOT be 0 (avoid black corners)
        # -------------------------
        if self.augment and (random.random() < self.rotate_prob):
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)

            img_pil = Image.fromarray(rgb)
            mask_pil = Image.fromarray(mask)

            img_pil = TF.rotate(
                img_pil,
                angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=114,  # IMPORTANT: avoid black corners
            )
            mask_pil = TF.rotate(
                mask_pil,
                angle,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=255,  # holes are ignore
            )

            rgb = np.array(img_pil)
            mask = np.array(mask_pil)

        # tensors
        mask_tensor = torch.from_numpy(mask).long()
        clean_np = rgb.copy()
        target_clean = torch.from_numpy(clean_np.astype(np.float32) / 255.0).permute(2, 0, 1)

        # -------------------------
        # Degradation: train random / val fixed
        # 注意：分割模式也应用退化，让分割分支具备鲁棒性
        # -------------------------
        if not self.augment:
            state_random = random.getstate()
            state_numpy = np.random.get_state()

            seed = int(idx) + self.val_seed_offset
            random.seed(seed)
            np.random.seed(seed)

            degraded_np = synthesize_rain_fog(clean_np, p_clean=self.p_clean)

            random.setstate(state_random)
            np.random.set_state(state_numpy)
        else:
            degraded_np = synthesize_rain_fog(clean_np, p_clean=self.p_clean)

        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1)

        if self.mode == "segmentation":
            # 分割模式：输入是退化图，标签是基于干净图生成的 mask
            return input_tensor, mask_tensor

        # joint 模式：返回 (退化输入, 干净目标, mask)
        return input_tensor, target_clean, mask_tensor


# =========================================================
# 3) Stage A: folder loader (restoration only)
# =========================================================
class SimpleFolderDataset(Dataset):
    def __init__(
        self,
        img_dir,
        img_size=384,            # int or (H,W)
        augment=True,
        rotate_prob=0.5,
        max_rotate_deg=45.0,
        val_seed_offset=200000,
        p_clean=0.35,            # NEW
        pad_value=114,           # only used for legacy square letterbox
    ):
        self.img_dir = img_dir
        self.out_h, self.out_w = _parse_hw(img_size)
        self.augment = bool(augment)
        self.rotate_prob = float(rotate_prob)
        self.max_rotate_deg = float(max_rotate_deg)
        self.val_seed_offset = int(val_seed_offset)

        self.p_clean = float(p_clean)
        self.pad_value = int(pad_value)
        self.use_letterbox = (self.out_h == self.out_w)

        raw_paths = []
        raw_paths += glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]"))
        raw_paths += glob.glob(os.path.join(img_dir, "*.[jJ][pP][eE][gG]"))
        raw_paths += glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]"))

        # de-dup + stable sort
        self.img_paths = sorted(set(raw_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        H, W = self.out_h, self.out_w
        path = self.img_paths[idx]

        bgr = cv2.imread(path)
        if bgr is None:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.use_letterbox:
                rgb, _meta = letterbox_rgb_u8(rgb0, H, pad_value=self.pad_value)
            else:
                rgb, _meta = resize_rgb_u8(rgb0, H, W)

        # rotation augmentation (image only)
        if self.augment and (random.random() < self.rotate_prob):
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            img_pil = Image.fromarray(rgb)
            img_pil = TF.rotate(
                img_pil,
                angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=114,  # IMPORTANT: avoid black corners
            )
            rgb = np.array(img_pil)

        target_clean = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        # degradation: train random / val fixed
        if not self.augment:
            state_random = random.getstate()
            state_numpy = np.random.get_state()

            seed = int(idx) + self.val_seed_offset
            random.seed(seed)
            np.random.seed(seed)

            degraded_np = synthesize_rain_fog(rgb, p_clean=self.p_clean)

            random.setstate(state_random)
            np.random.set_state(state_numpy)
        else:
            degraded_np = synthesize_rain_fog(rgb, p_clean=self.p_clean)

        input_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1)
        return input_tensor, target_clean
