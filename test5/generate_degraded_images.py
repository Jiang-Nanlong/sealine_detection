# -*- coding: utf-8 -*-
"""
generate_degraded_images.py

Generate degraded versions of MU-SID test images for robustness evaluation.

Degradation types:
  1. Gaussian noise (σ = 10, 25, 50)
  2. Motion blur (kernel size = 15, 25)
  3. Gaussian blur (σ = 2, 5)
  4. Low light simulation (gamma = 2.0, 3.0)
  5. Fog/haze simulation (intensity = 0.3, 0.5)
  6. Salt & pepper noise (ratio = 0.01, 0.05)

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
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
SEED = 42
# ============================

# ----------------------------
# Config
# ----------------------------
TEST5_DIR = PROJECT_ROOT / "test5"
MUSID_IMG_DIR = PROJECT_ROOT / "Hashmani's Dataset" / "clear"
MUSID_GT_CSV = PROJECT_ROOT / "Hashmani's Dataset" / "GroundTruth.csv"
SPLITS_DIR = PROJECT_ROOT / "splits_musid"

OUT_DIR = TEST5_DIR / "degraded_images"

# Degradation configurations
DEGRADATIONS = {
    # Gaussian noise
    "gaussian_noise_10": {"type": "gaussian_noise", "sigma": 10},
    "gaussian_noise_25": {"type": "gaussian_noise", "sigma": 25},
    "gaussian_noise_50": {"type": "gaussian_noise", "sigma": 50},
    
    # Motion blur
    "motion_blur_15": {"type": "motion_blur", "kernel_size": 15},
    "motion_blur_25": {"type": "motion_blur", "kernel_size": 25},
    
    # Gaussian blur
    "gaussian_blur_2": {"type": "gaussian_blur", "sigma": 2},
    "gaussian_blur_5": {"type": "gaussian_blur", "sigma": 5},
    
    # Low light (gamma correction)
    "low_light_2.0": {"type": "low_light", "gamma": 2.0},
    "low_light_3.0": {"type": "low_light", "gamma": 3.0},
    
    # Fog/haze
    "fog_0.3": {"type": "fog", "intensity": 0.3},
    "fog_0.5": {"type": "fog", "intensity": 0.5},
    
    # Salt & pepper noise
    "sp_noise_0.01": {"type": "salt_pepper", "ratio": 0.01},
    "sp_noise_0.05": {"type": "salt_pepper", "ratio": 0.05},
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def add_gaussian_noise(img, sigma):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_motion_blur(img, kernel_size):
    """Add horizontal motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
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


def add_salt_pepper_noise(img, ratio):
    """Add salt and pepper noise."""
    noisy = img.copy()
    h, w = img.shape[:2]
    n_salt = int(h * w * ratio / 2)
    n_pepper = int(h * w * ratio / 2)
    
    # Salt
    coords = [np.random.randint(0, i, n_salt) for i in (h, w)]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper
    coords = [np.random.randint(0, i, n_pepper) for i in (h, w)]
    noisy[coords[0], coords[1]] = 0
    
    return noisy


def apply_degradation(img, config):
    """Apply degradation based on config."""
    dtype = config["type"]
    
    if dtype == "gaussian_noise":
        return add_gaussian_noise(img, config["sigma"])
    elif dtype == "motion_blur":
        return add_motion_blur(img, config["kernel_size"])
    elif dtype == "gaussian_blur":
        return add_gaussian_blur(img, config["sigma"])
    elif dtype == "low_light":
        return add_low_light(img, config["gamma"])
    elif dtype == "fog":
        return add_fog(img, config["intensity"])
    elif dtype == "salt_pepper":
        return add_salt_pepper_noise(img, config["ratio"])
    else:
        raise ValueError(f"Unknown degradation type: {dtype}")


def load_test_split():
    """Load test split image names."""
    test_split_file = SPLITS_DIR / "test.txt"
    if not test_split_file.exists():
        raise FileNotFoundError(f"Test split file not found: {test_split_file}")
    
    with open(test_split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    np.random.seed(SEED)
    
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
    
    # Copy clean images
    print("\n[Copy] Clean images...")
    for img_name in tqdm(test_images, desc="Clean"):
        src_path = MUSID_IMG_DIR / img_name
        if src_path.exists():
            img = cv2.imread(str(src_path))
            cv2.imwrite(str(clean_dir / img_name), img)
    
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
            degraded = apply_degradation(img, deg_config)
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
