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
# =========================================================
def synthesize_rain_fog(image_rgb_u8: np.ndarray, p_clean: float = 0.35):
    """
    物理感知退化函数
    输入:  uint8 RGB
    输出: float32 RGB in [0,1]

    p_clean:
      - with probability p_clean, return the original image (plus tiny noise)
      - this teaches the model: "if already clean, do not change"
    """
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # ---- clean pass-through (critical) ----
    if random.random() < float(p_clean):
        noise = np.random.randn(h, w, 3).astype(np.float32) * 0.005
        img = np.clip(img + noise, 0, 1)
        return img.astype(np.float32)

    # ---- choose degradations (no forced fog anymore) ----
    is_rain = random.random() < 0.4
    is_fog = random.random() < 0.4
    is_dark = random.random() < 0.3

    if is_rain:
        severity = random.uniform(0.3, 0.9)
        streak = np.zeros((h, w, 3), dtype=np.float32)
        n_lines = int(800 * severity)
        angle = random.uniform(-15, 15)
        tan_a = np.tan(np.deg2rad(angle))
        for _ in range(n_lines):
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            length = random.randint(15, 45)
            x1 = int(x0 + length * tan_a)
            y1 = int(y0 + length)
            cv2.line(streak, (x0, y0), (int(x1), int(y1)), (0.8, 0.8, 0.8), 1)
        streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=0.5, sigmaY=0.5)
        img = np.clip(img + 0.4 * streak, 0, 1)

    if is_fog:
        severity = random.uniform(0.2, 0.9)
        airlight = random.uniform(0.6, 0.95)

        ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        if random.random() > 0.5:
            base_map = np.tile(ys, (1, w))
        else:
            base_map = np.tile(1 - ys, (1, w))

        noise = np.random.randn(h, w).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=10)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

        fog_map = 0.6 * base_map + 0.4 * noise
        fog_map = fog_map * severity
        fog_map = fog_map[..., None]

        img = img * (1 - fog_map) + airlight * fog_map
        img = np.clip(img, 0, 1)

    if is_dark:
        gamma = random.uniform(1.2, 2.0)
        img = np.power(img, gamma)
        img = np.clip(img, 0, 1)

    # always add a tiny sensor noise
    noise = np.random.randn(h, w, 3).astype(np.float32) * 0.01
    img = np.clip(img + noise, 0, 1)

    return img.astype(np.float32)


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
        
        # 尝试多种可能的扩展名
        possible_exts = ['', '.JPG', '.jpg', '.png', '.jpeg', '.JPEG', '.PNG']
        bgr = None
        for ext in possible_exts:
            full_path = path + ext
            if os.path.exists(full_path):
                bgr = cv2.imread(full_path)
                if bgr is not None:
                    break
        
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

        if self.mode == "segmentation":
            return target_clean, mask_tensor

        # -------------------------
        # Degradation: train random / val fixed
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
