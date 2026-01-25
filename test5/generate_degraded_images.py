# -*- coding: utf-8 -*-
"""
generate_degraded_images.py

Generate degraded versions of MU-SID test images for robustness evaluation.

Degradation types (海洋场景相关):
  === 基础退化 ===
  1. Gaussian noise (σ = 15, 30) - 传感器噪声
  2. Motion blur (kernel = 15, 25) - 船体晃动（随机角度）
  3. Low light (γ = 2.0, 2.5) - 黄昏/阴天
  4. Fog/haze (30%, 50%) - 海雾
  
  === 海洋特有退化 ===
  5. Rain (轻/中/重) - 海上降雨
  6. Sun glare / 强反光 (轻/重) - 阳光海面反射
  7. JPEG compression (Q=20, 10) - 压缩伪影/低码率
  8. Resolution downscale (0.5x, 0.25x) - 远距离/低清监控

PyCharm: 直接运行此文件，在下方配置区修改参数

注意事项:
  - Clean 基线使用 shutil.copy2 原样复制，避免重新编码
  - 随机退化使用 (img_name + deg_name) 作为种子，保证顺序无关的可复现性
  - 运动模糊采用随机角度，更符合实际船载场景
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
GLOBAL_SEED = 42  # 全局种子基数
# ============================

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
MUSID_IMG_DIR = PROJECT_ROOT / "Hashmani's Dataset" / "clear"
MUSID_GT_CSV = PROJECT_ROOT / "Hashmani's Dataset" / "GroundTruth.csv"
SPLITS_DIR = PROJECT_ROOT / "splits_musid"

OUT_DIR = TEST5_DIR / "degraded_images"

# Degradation configurations - 更贴合海洋场景
DEGRADATIONS = {
    # ========== 基础退化 ==========
    # Gaussian noise - 传感器噪声
    "gaussian_noise_15": {"type": "gaussian_noise", "sigma": 15},
    "gaussian_noise_30": {"type": "gaussian_noise", "sigma": 30},
    
    # Motion blur - 船体晃动/相机抖动
    "motion_blur_15": {"type": "motion_blur", "kernel_size": 15},
    "motion_blur_25": {"type": "motion_blur", "kernel_size": 25},
    
    # Low light - 黄昏/阴天/夜间
    "low_light_2.0": {"type": "low_light", "gamma": 2.0},
    "low_light_2.5": {"type": "low_light", "gamma": 2.5},
    
    # Fog/haze - 海雾
    "fog_0.3": {"type": "fog", "intensity": 0.3},
    "fog_0.5": {"type": "fog", "intensity": 0.5},
    
    # ========== 海洋特有退化 ==========
    # Rain - 海上降雨（雨滴+雾气）
    "rain_light": {"type": "rain", "intensity": "light"},
    "rain_medium": {"type": "rain", "intensity": "medium"},
    "rain_heavy": {"type": "rain", "intensity": "heavy"},
    
    # Sun glare / 强反光 - 阳光海面反射
    "glare_light": {"type": "glare", "intensity": 0.3},
    "glare_heavy": {"type": "glare", "intensity": 0.6},
    
    # JPEG compression artifacts - 压缩伪影/低码率视频
    "jpeg_q20": {"type": "jpeg", "quality": 20},
    "jpeg_q10": {"type": "jpeg", "quality": 10},
    
    # Resolution downscale - 远距离/低清监控
    "lowres_0.5x": {"type": "downscale", "scale": 0.5},
    "lowres_0.25x": {"type": "downscale", "scale": 0.25},
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def get_deterministic_seed(img_name: str, deg_name: str) -> int:
    """
    生成基于 (img_name + deg_name) 的确定性种子。
    保证：
      1. 同一张图 + 同一种退化 -> 总是相同的随机结果
      2. 与遍历顺序无关（顺序无关可复现性）
    """
    key = f"{img_name}_{deg_name}_{GLOBAL_SEED}"
    hash_val = hashlib.md5(key.encode()).hexdigest()
    return int(hash_val[:8], 16)  # 取前 8 位 hex，转成 int


def add_gaussian_noise(img, sigma, rng: np.random.Generator):
    """Add Gaussian noise to image."""
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_motion_blur(img, kernel_size, rng: np.random.Generator):
    """
    Add motion blur with random angle.
    
    随机角度模拟船载场景的多方向抖动（横摇、俯仰、偏航）。
    """
    # 随机角度：0-180度（覆盖所有方向，180-360与0-180对称）
    angle = rng.uniform(0, 180)
    
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    # 计算线段端点
    angle_rad = np.radians(angle)
    dx = np.cos(angle_rad) * center
    dy = np.sin(angle_rad) * center
    
    # 绘制运动模糊线
    x1, y1 = int(center - dx), int(center - dy)
    x2, y2 = int(center + dx), int(center + dy)
    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)
    
    # 归一化
    kernel = kernel / kernel.sum()
    
    return cv2.filter2D(img, -1, kernel)


def add_gaussian_blur(img, sigma):
    """Add Gaussian blur."""
    ksize = int(sigma * 6) | 1  # Ensure odd kernel size
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def add_low_light(img, gamma):
    """Simulate low light by gamma correction (gamma > 1 darkens)."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def add_fog(img, intensity):
    """Add fog/haze effect."""
    fog_layer = np.ones_like(img, dtype=np.float32) * 255
    foggy = img.astype(np.float32) * (1 - intensity) + fog_layer * intensity
    return np.clip(foggy, 0, 255).astype(np.uint8)


def add_rain(img, intensity="medium", rng: np.random.Generator = None):
    """
    Add rain effect with rain streaks and atmospheric haze.
    
    Rain simulation includes:
    - Rain streaks (diagonal lines)
    - Reduced contrast (atmospheric scattering)
    - Slight blur (rain in air)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)
    
    # Rain parameters based on intensity
    params = {
        "light": {"n_drops": 1000, "length": 15, "thickness": 1, "haze": 0.1, "blur": 1},
        "medium": {"n_drops": 3000, "length": 25, "thickness": 1, "haze": 0.2, "blur": 2},
        "heavy": {"n_drops": 6000, "length": 35, "thickness": 2, "haze": 0.35, "blur": 3},
    }
    p = params.get(intensity, params["medium"])
    
    # Create rain layer
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(p["n_drops"]):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        # Rain falls at an angle (wind effect)
        angle = rng.uniform(-0.2, 0.2)  # Near vertical
        x2 = int(x + p["length"] * angle)
        y2 = int(y + p["length"])
        cv2.line(rain_layer, (x, y), (x2, y2), 200, p["thickness"])
    
    # Apply motion blur to rain streaks
    if p["blur"] > 0:
        k = p["blur"] * 2 + 1
        rain_layer = cv2.GaussianBlur(rain_layer, (k, k), 0)
    
    # Add rain streaks
    rain_mask = rain_layer.astype(np.float32) / 255.0
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - rain_mask * 0.5) + 255 * rain_mask * 0.5
    
    # Add atmospheric haze (reduces contrast, adds white tint)
    haze = np.ones_like(result) * 200
    result = result * (1 - p["haze"]) + haze * p["haze"]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_sun_glare(img, intensity=0.3, rng: np.random.Generator = None):
    """
    Add sun glare / strong reflection effect.
    
    Simulates:
    - Bright spots (sun reflection on water)
    - Overexposure in upper portion
    - Reduced contrast in affected areas
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)
    
    # Create glare gradient (stronger at top, simulating sky reflection)
    y_coords = np.linspace(0, 1, h).reshape(-1, 1)
    # Glare strongest in upper 40% of image (where horizon typically is)
    glare_mask = np.clip(1.0 - y_coords * 2.5, 0, 1) ** 2
    glare_mask = np.tile(glare_mask, (1, w))
    
    # Add some horizontal variation (wave-like reflection)
    x_variation = np.sin(np.linspace(0, 8 * np.pi, w)) * 0.3 + 0.7
    glare_mask = glare_mask * x_variation.reshape(1, -1)
    
    # Add random bright spots (sun sparkles on water)
    n_spots = int(50 * intensity)
    for _ in range(n_spots):
        cx = rng.integers(0, w)
        cy = rng.integers(0, int(h * 0.6))  # Upper 60%
        radius = rng.integers(20, 80)
        y, x = np.ogrid[:h, :w]
        spot_mask = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        glare_mask = np.maximum(glare_mask, spot_mask * rng.uniform(0.5, 1.0))
    
    # Apply glare
    glare_mask = glare_mask[:, :, np.newaxis] * intensity
    white = np.ones_like(result) * 255
    result = result * (1 - glare_mask) + white * glare_mask
    
    # Reduce contrast slightly (overexposure effect)
    result = result * (1 - intensity * 0.3) + 128 * intensity * 0.3
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_jpeg_artifacts(img, quality):
    """
    Add JPEG compression artifacts.
    
    Simulates low bitrate video compression commonly seen in
    maritime surveillance systems.
    """
    # Encode to JPEG with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    # Decode back
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def add_resolution_downscale(img, scale):
    """
    Simulate low resolution by downscaling then upscaling.
    
    Simulates:
    - Long-distance surveillance cameras
    - Low-resolution sensors
    - Heavy video compression
    """
    h, w = img.shape[:2]
    
    # Downscale
    small_h, small_w = int(h * scale), int(w * scale)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    # Upscale back to original size (introduces blur/pixelation)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return restored


def apply_degradation(img, config, img_name: str, deg_name: str):
    """
    Apply degradation based on config.
    
    Args:
        img: Input image (BGR, uint8)
        config: Degradation configuration dict
        img_name: Image filename (for deterministic seed)
        deg_name: Degradation name (for deterministic seed)
    
    Returns:
        Degraded image (BGR, uint8)
    """
    dtype = config["type"]
    
    # Create deterministic RNG for functions that need randomness
    seed = get_deterministic_seed(img_name, deg_name)
    rng = np.random.default_rng(seed)
    
    if dtype == "gaussian_noise":
        return add_gaussian_noise(img, config["sigma"], rng=rng)
    elif dtype == "motion_blur":
        return add_motion_blur(img, config["kernel_size"], rng=rng)
    elif dtype == "gaussian_blur":
        return add_gaussian_blur(img, config["sigma"])
    elif dtype == "low_light":
        return add_low_light(img, config["gamma"])
    elif dtype == "fog":
        return add_fog(img, config["intensity"])
    elif dtype == "rain":
        return add_rain(img, config["intensity"], rng=rng)
    elif dtype == "glare":
        return add_sun_glare(img, config["intensity"], rng=rng)
    elif dtype == "jpeg":
        return add_jpeg_artifacts(img, config["quality"])
    elif dtype == "downscale":
        return add_resolution_downscale(img, config["scale"])
    else:
        raise ValueError(f"Unknown degradation type: {dtype}")


def load_test_split():
    """Load test split image names from GroundTruth_test.csv."""
    test_csv = SPLITS_DIR / "GroundTruth_test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split file not found: {test_csv}")
    
    image_names = []
    with open(test_csv, "r") as f:
        for line in f:
            if line.strip():
                # First column is image name (without extension)
                name = line.strip().split(",")[0]
                # MU-SID 原始图片使用大写 .JPG 扩展名
                image_names.append(f"{name}.JPG")
    return image_names


def main():
    # Note: We use deterministic per-image seeds (via get_deterministic_seed),
    # so GLOBAL_SEED is only for any legacy code that might use np.random directly
    np.random.seed(GLOBAL_SEED)
    
    print("=" * 60)
    print("Experiment 5: Generate Degraded Images")
    print("=" * 60)
    
    # Load test images
    test_images = load_test_split()
    print(f"\n[Load] {len(test_images)} test images from MU-SID")
    
    # Create output directories
    ensure_dir(OUT_DIR)
    
    # Also save clean images for reference
    clean_dir = OUT_DIR / "clean"
    ensure_dir(clean_dir)
    
    # Copy clean images (use shutil.copy2 to preserve original, avoid re-encoding)
    print("\n[Copy] Clean images (direct file copy, no re-encoding)...")
    for img_name in tqdm(test_images, desc="Clean"):
        src_path = MUSID_IMG_DIR / img_name
        dst_path = clean_dir / img_name
        if src_path.exists():
            shutil.copy2(str(src_path), str(dst_path))
    
    # Generate degraded versions
    for deg_name, deg_config in DEGRADATIONS.items():
        print(f"\n[Generate] {deg_name}...")
        deg_dir = OUT_DIR / deg_name
        ensure_dir(deg_dir)
        
        for img_name in tqdm(test_images, desc=deg_name):
            src_path = MUSID_IMG_DIR / img_name
            if not src_path.exists():
                continue
            
            img = cv2.imread(str(src_path))
            # Pass img_name and deg_name for deterministic per-image seed
            degraded = apply_degradation(img, deg_config, img_name, deg_name)
            cv2.imwrite(str(deg_dir / img_name), degraded)
    
    print("\n" + "=" * 60)
    print("[Done] Degraded images saved to:")
    print(f"  {OUT_DIR}")
    print(f"\nGenerated {len(DEGRADATIONS)} degradation types:")
    for name in DEGRADATIONS:
        print(f"  - {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
