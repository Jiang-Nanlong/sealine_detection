# -*- coding: utf-8 -*-
"""
Render Fig 5-1 (Diff-Gradient texture suppression) with 5 random examples.

PyCharm-friendly:
- No argparse / CLI.
- Key paths are managed by GLOBAL variables below.
- Run this file directly.

What this script will output:
1) A grid figure (N_SAMPLES rows x 3 cols):
   [Restored] [Sobel magnitude] [DiffGrad + Trunc]
2) (Optional) A 1xN mosaic of only the restored images (useful for reviewers).

It follows your repo's pipeline logic:
- UNet input size is 1024x576 (same as make_fusion_cache.py).
- Restoration model weights default to rghnet_best_c2.pth, and DCE weights to Epoch99.pth.

Requirements:
- Put this script in your project root (same folder as unet_model.py, gradient_radon.py, etc.)
- Ensure the weights exist at the configured paths.
"""

import os
import glob
import random
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Global Config (edit here)
# -----------------------------
# Dataset paths (same defaults as make_fusion_cache.py)
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR  = r"Hashmani's Dataset/MU-SID"

# Key weights (same defaults as make_fusion_cache.py)
RGHNET_CKPT = r"rghnet_best_c2.pth"
DCE_WEIGHTS = r"Epoch99.pth"

# UNet input geometry (same as make_fusion_cache.py)
UNET_IN_W = 1024
UNET_IN_H = 576

# Random sampling
RANDOM_SEED = 42
N_SAMPLES = 5

# Diff-Gradient parameters (align with your thesis)
LAMBDA_TEXTURE = 1.5           # λ in ReLU(|Gy| - λ|Gx|)
USE_CLAHE = True
CLAHE_CLIPLIMIT = 4.0
CLAHE_TILE = (8, 8)

# Median filter (align with gradient_radon.py: ksize = 10*scale + 1)
MEDIAN_SCALE = 1               # scale=1 => ksize=11
SOBEL_KSIZE = 3

# Truncation (align with thesis: T = mean + k*std; clamp to T)
TRUNC_MODE = "fixed"           # "fixed" or "adaptive"
TRUNC_FIXED_T = 20.0           # gradient_radon.py clamps to 20
TRUNC_K = 2.0                  # used only when TRUNC_MODE="adaptive"

# Output
OUT_DIR = "fig_outputs"
OUT_FIG_PATH = os.path.join(OUT_DIR, "Fig5-1_DiffGrad_5Examples.png")
OUT_RESTORED_MOSAIC = os.path.join(OUT_DIR, "Fig5-1_RestoredOnly_5Examples.png")
FIG_DPI = 220

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _try_candidates(path_like: str, roots: List[str]) -> Optional[str]:
    """Try resolving a relative file by searching in multiple roots."""
    if os.path.isabs(path_like) and os.path.exists(path_like):
        return path_like

    # direct relative to cwd
    if os.path.exists(path_like):
        return os.path.abspath(path_like)

    # search in roots
    for r in roots:
        cand = os.path.join(r, path_like)
        if os.path.exists(cand):
            return os.path.abspath(cand)

    # glob search (filename only)
    base = os.path.basename(path_like)
    for r in roots:
        hits = glob.glob(os.path.join(r, "**", base), recursive=True)
        if hits:
            return os.path.abspath(hits[0])

    return None


def resolve_image_path(img_dir: str, stem_or_name: str) -> Optional[str]:
    """
    MU-SID GroundTruth.csv stores the *stem* without extension in many cases.
    Same idea as _read_image_any_ext() in make_fusion_cache.py.
    """
    name = str(stem_or_name)
    lower = name.lower()
    candidates = [name] if (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")) else []
    candidates += [name + ".JPG", name + ".jpg", name + ".jpeg", name + ".png"]

    for fn in candidates:
        p = os.path.join(img_dir, fn)
        if os.path.exists(p):
            return p
    return None


def safe_torch_load(path: str, device: str):
    """Torch 2.1+ may support weights_only=True; older versions won't."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def norm_to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mi = float(np.min(x))
    ma = float(np.max(x))
    if ma - mi < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mi) / (ma - mi)
    y = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
    return y


# -----------------------------
# Core: Diff-Gradient
# -----------------------------
def compute_diffgrad_maps(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      sobel_mag_u8: traditional gradient magnitude (visualized)
      diffgrad_trunc_u8: proposed map (ReLU(|Gy|-λ|Gx|)) + truncation (visualized)
    """
    assert rgb_u8.dtype == np.uint8 and rgb_u8.ndim == 3 and rgb_u8.shape[2] == 3
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIPLIMIT), tileGridSize=tuple(CLAHE_TILE))
        gray = clahe.apply(gray)

    ksize = int(10 * int(MEDIAN_SCALE) + 1)
    gray_blur = cv2.medianBlur(gray, ksize)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=int(SOBEL_KSIZE))
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=int(SOBEL_KSIZE))

    sobel_mag = np.sqrt(gx * gx + gy * gy)

    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)

    diff = abs_gy - float(LAMBDA_TEXTURE) * abs_gx
    diff = np.maximum(diff, 0.0)

    if TRUNC_MODE.lower() == "adaptive":
        t = float(np.mean(diff) + float(TRUNC_K) * np.std(diff))
    else:
        t = float(TRUNC_FIXED_T)

    # avoid weird tiny thresholds
    t = max(t, 1e-6)
    diff_trunc = np.clip(diff, 0.0, t)

    return norm_to_u8(sobel_mag), norm_to_u8(diff_trunc)


# -----------------------------
# Core: Restoration model inference
# -----------------------------
def load_restoration_model(project_root: str) -> torch.nn.Module:
    """
    Load RestorationGuidedHorizonNet exactly like make_fusion_cache.py:
      model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS)
      model.load_state_dict(torch.load(RGHNET_CKPT), strict=False)
    """
    roots = [project_root, os.getcwd()]

    ckpt = _try_candidates(RGHNET_CKPT, roots)
    dce  = _try_candidates(DCE_WEIGHTS, roots)
    if ckpt is None:
        raise FileNotFoundError(f"Cannot find RGHNET_CKPT: {RGHNET_CKPT} (searched in {roots})")
    if dce is None:
        raise FileNotFoundError(f"Cannot find DCE_WEIGHTS: {DCE_WEIGHTS} (searched in {roots})")

    # Import from your project
    from unet_model import RestorationGuidedHorizonNet

    print(f"[Load] RGHNet CKPT: {ckpt}")
    print(f"[Load] DCE weights : {dce}")
    model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=dce).to(DEVICE)

    state = safe_torch_load(ckpt, DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_restoration(model: torch.nn.Module, bgr_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: original BGR uint8 (any size).
    Output:
      inp_rgb_u8     : resized degraded image (1024x576) in RGB
      restored_rgb_u8: restored image in RGB (1024x576)
    """
    bgr_rs = cv2.resize(bgr_u8, (int(UNET_IN_W), int(UNET_IN_H)), interpolation=cv2.INTER_AREA)
    rgb_rs = cv2.cvtColor(bgr_rs, cv2.COLOR_BGR2RGB)

    inp = torch.from_numpy(rgb_rs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Forward signature in your repo's make_fusion_cache.py:
    # restored_t, seg_logits, _ = seg_model(inp, None, True, True)
    restored_t, _, _ = model(inp, None, True, False)  # restoration only

    restored_rgb = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0)
    restored_rgb = np.clip(restored_rgb, 0.0, 255.0).astype(np.uint8)
    return rgb_rs, restored_rgb


# -----------------------------
# Main: sample + render
# -----------------------------
def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    ensure_dir(OUT_DIR)
    seed_everything(RANDOM_SEED)

    # 1) read CSV and build valid list
    import pandas as pd
    df = pd.read_csv(CSV_PATH, header=None)
    valid: List[Tuple[int, str]] = []
    for i in range(len(df)):
        stem = str(df.iloc[i, 0])
        p = resolve_image_path(IMG_DIR, stem)
        if p is not None:
            valid.append((i, p))

    if len(valid) == 0:
        raise RuntimeError(f"No valid images found. Check CSV_PATH={CSV_PATH} and IMG_DIR={IMG_DIR}")

    # 2) choose N samples deterministically
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(len(valid), size=min(int(N_SAMPLES), len(valid)), replace=False).tolist()
    chosen = [valid[k] for k in chosen]

    print(f"[Sample] seed={RANDOM_SEED}  picked={len(chosen)} / valid={len(valid)}")
    for idx, p in chosen:
        print(f"  - csv_row={idx}  file={os.path.basename(p)}")

    # 3) load model
    model = load_restoration_model(project_root)

    # 4) run + collect
    rows = []
    restored_only = []
    for csv_idx, img_path in chosen:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[Warn] Failed to read: {img_path} (skip)")
            continue

        inp_rgb, restored_rgb = run_restoration(model, bgr)
        sobel_mag_u8, diff_trunc_u8 = compute_diffgrad_maps(restored_rgb)

        rows.append({
            "csv_idx": csv_idx,
            "name": os.path.basename(img_path),
            "inp_rgb": inp_rgb,
            "restored_rgb": restored_rgb,
            "sobel_mag_u8": sobel_mag_u8,
            "diff_trunc_u8": diff_trunc_u8,
        })
        restored_only.append(restored_rgb)

    if len(rows) == 0:
        raise RuntimeError("All selected samples failed to load/read. Please check your dataset paths.")

    # 5) render Fig 5-1 grid
    n = len(rows)
    col_titles = ["Restored", "Sobel |∇I|", f"DiffGrad+Trunc (λ={LAMBDA_TEXTURE})"]

    # Figure size: ~4 columns; each row about 2 inches high
    fig_w = 3 * 4.0
    fig_h = max(2.0 * n, 6.0)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(fig_w, fig_h))

    # axes could be 1D if n==1
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, item in enumerate(rows):
        # 0) restored
        axes[r, 0].imshow(item["restored_rgb"])
        axes[r, 0].axis("off")

        # 1) sobel magnitude
        axes[r, 1].imshow(item["sobel_mag_u8"], cmap="gray")
        axes[r, 1].axis("off")

        # 2) diffgrad trunc
        axes[r, 2].imshow(item["diff_trunc_u8"], cmap="gray")
        axes[r, 2].axis("off")

        # annotate left with filename
        axes[r, 0].set_ylabel(item["name"], rotation=0, labelpad=60, fontsize=9, va="center")

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12)

    plt.tight_layout()
    fig.savefig(OUT_FIG_PATH, dpi=int(FIG_DPI), bbox_inches="tight")
    print(f"[Saved] {OUT_FIG_PATH}")

    # 6) optional: restored-only mosaic (1xN)
    fig2, ax2 = plt.subplots(1, len(restored_only), figsize=(len(restored_only) * 4.0, 3.2))
    if len(restored_only) == 1:
        ax2 = [ax2]
    for i, im in enumerate(restored_only):
        ax2[i].imshow(im)
        ax2[i].axis("off")
    plt.tight_layout()
    fig2.savefig(OUT_RESTORED_MOSAIC, dpi=int(FIG_DPI), bbox_inches="tight")
    print(f"[Saved] {OUT_RESTORED_MOSAIC}")

    # show on screen
    plt.show()


if __name__ == "__main__":
    main()
