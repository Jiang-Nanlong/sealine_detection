import os
import cv2
import torch
import numpy as np
import random
import glob

from unet_model import RestorationGuidedHorizonNet
from cnn_model import HorizonResNet
from gradient_radon import TextureSuppressedMuSCoWERT
from dataset_loader import synthesize_rain_fog


# ======================== Config ========================
IMG_DIR = r"Hashmani's Dataset/clear"   # test images directory
OUT_DIR = r"demo_results"

UNET_CKPT = "rghnet_best_joint.pth"  # 你也可以换成 best_seg / best_joint
CNN_CKPT  = "splits_musid/best_fusion_cnn_1024x576.pth"  # NEW
DCE_WEIGHTS = "Epoch99.pth"

# Debug only: whether to add synthetic rain/fog on input
ENABLE_DEGRADATION = False

# ALL images are 1920x1080 (16:9) -> use 1024x576 to avoid any padding.
UNET_IN_W = 1024
UNET_IN_H = 576

# CNN sinogram size
RESIZE_H = 2240
RESIZE_W = 180

NUM_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024

# Mask post-process (MUST match make_fusion_cache.py)
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 0

# Edge for 4th channel (MUST match make_fusion_cache.py)
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE = 1

# Debug visuals
DEBUG_DRAW_TRAD = False  # True: 额外画一条传统 Radon 参考线（不影响预测）
SHOW_H = 500
# =======================================================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def process_sinogram(sino: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Normalize to [0,1] then pad/crop to (target_h, target_w) along rho-axis."""
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


def post_process_mask_top_connected(mask_np: np.ndarray,
                                    sky_id: int = 1,
                                    ignore_id: int = 255,
                                    morph_close: int = MORPH_CLOSE,
                                    top_touch_tol: int = TOP_TOUCH_TOL) -> np.ndarray:
    """
    Keep only sky connected components that touch the image top.
    MUST match make_fusion_cache.py.
    """
    valid = (mask_np != ignore_id)
    sky = ((mask_np == sky_id) & valid).astype(np.uint8)

    if morph_close and morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)

    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num_labels):
        y_top = stats[i, cv2.CC_STAT_TOP]
        if y_top <= top_touch_tol:
            keep[labels == i] = 1

    out = mask_np.copy()
    out[(mask_np == sky_id) & (keep == 0)] = 0
    return out


def clamp01(x: float) -> float:
    if not np.isfinite(x):
        return 0.5
    return float(np.clip(x, 0.0, 1.0))


def draw_line_from_rho_theta(img_bgr: np.ndarray, rho_norm: float, theta_norm: float,
                             color=(0, 0, 255), thickness: int = 3) -> np.ndarray:
    """
    Draw line from CNN normalized (rho, theta) on img (original size).
    MUST be consistent with make_fusion_cache.calculate_radon_label().
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    diag_len = np.sqrt(h ** 2 + w ** 2)

    theta_deg = float(theta_norm) * 180.0
    theta_rad = np.deg2rad(theta_deg)

    pad_top_sino = (RESIZE_H - diag_len) / 2.0
    rho_idx = float(rho_norm) * (RESIZE_H - 1)
    rho_real = rho_idx - pad_top_sino - (diag_len / 2.0)

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x0 = cos_t * rho_real
    y0 = sin_t * rho_real

    scale = max(h, w) * 2
    pt1_x = int(cx + x0 - scale * sin_t)
    pt1_y = int(cy + y0 + scale * cos_t)
    pt2_x = int(cx + x0 + scale * sin_t)
    pt2_y = int(cy + y0 - scale * cos_t)

    cv2.line(img_bgr, (pt1_x, pt1_y), (pt2_x, pt2_y), color, thickness, cv2.LINE_AA)
    return img_bgr


def resize_h(img, h):
    r = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), h))


def main():
    ensure_dir(OUT_DIR)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=== Loading Models ===")
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if not os.path.exists(UNET_CKPT):
        raise FileNotFoundError(f"UNet weights not found: {UNET_CKPT}")
    unet.load_state_dict(torch.load(UNET_CKPT, map_location=DEVICE), strict=False)
    unet.eval()
    print(f"Loaded UNet: {UNET_CKPT}")

    cnn = HorizonResNet(in_channels=4).to(DEVICE)
    if not os.path.exists(CNN_CKPT):
        raise FileNotFoundError(f"CNN weights not found: {CNN_CKPT}")
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location=DEVICE))
    cnn.eval()
    print(f"Loaded CNN: {CNN_CKPT}")

    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
    theta_scan = np.linspace(0.0, 180.0, RESIZE_W, endpoint=False)

    all_imgs = glob.glob(os.path.join(IMG_DIR, "*.[jJ][pP]*[gG]")) + glob.glob(os.path.join(IMG_DIR, "*.png"))
    all_imgs = sorted(list(set(all_imgs)))
    if len(all_imgs) == 0:
        print(f"No images found in {IMG_DIR}")
        return

    samples = random.sample(all_imgs, min(NUM_SAMPLES, len(all_imgs)))
    print(f"=== Starting Inference (synthetic_degradation={ENABLE_DEGRADATION}) ===")

    for i, img_path in enumerate(samples):
        bgr_raw = cv2.imread(img_path)
        if bgr_raw is None:
            continue
        rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb_raw.shape[:2]

        # 1) UNet input: resize (no padding)
        rgb_unet = cv2.resize(rgb_raw, (UNET_IN_W, UNET_IN_H), interpolation=cv2.INTER_AREA)

        if ENABLE_DEGRADATION:
            # 强制退化：p_clean=0.0，避免合成里“偶尔直通clean”干扰判断
            rgb_input_np = synthesize_rain_fog(rgb_unet, p_clean=0.0)
        else:
            rgb_input_np = rgb_unet.astype(np.float32) / 255.0

        inp_tensor = torch.from_numpy(rgb_input_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 2) UNet forward (统一 pipeline：restoration + segmentation 都开)
        with torch.no_grad():
            restored_t, seg_logits, _ = unet(
                inp_tensor,
                None,
                enable_restoration=True,
                enable_segmentation=True
            )

        restored_unet = (restored_t[0].permute(1, 2, 0).cpu().float().numpy() * 255.0).astype(np.uint8)
        mask_unet = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # 0/1

        # 3) Resize UNet outputs back to original resolution for radon/cnn
        restored_orig = cv2.resize(restored_unet, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        mask_orig = cv2.resize(mask_unet, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        mask_pp = post_process_mask_top_connected(mask_orig)

        restored_orig_bgr = cv2.cvtColor(restored_orig, cv2.COLOR_RGB2BGR)

        # 4) Build 4-channel CNN input (MUST match cache)
        # A) traditional radon channels
        try:
            _, _, _, trad_sinos = detector.detect(restored_orig_bgr)
        except Exception:
            trad_sinos = []

        processed_stack = []
        for s in trad_sinos[:3]:
            processed_stack.append(process_sinogram(s, RESIZE_H, RESIZE_W))
        while len(processed_stack) < 3:
            processed_stack.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

        # B) segmentation edge channel
        edges = cv2.Canny((mask_pp * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)
        if EDGE_DILATE and EDGE_DILATE > 0:
            k = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, k, iterations=int(EDGE_DILATE))

        seg_sino_raw = detector._radon_gpu(edges, theta_scan)
        processed_stack.append(process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W))

        cnn_input = np.stack(processed_stack, axis=0).astype(np.float32)
        cnn_input_t = torch.from_numpy(cnn_input).unsqueeze(0).to(DEVICE)

        # 5) CNN prediction
        with torch.no_grad():
            preds = cnn(cnn_input_t).cpu().numpy()[0]

        pred_rho = clamp01(float(preds[0]))
        pred_theta = clamp01(float(preds[1]))
        theta_deg = pred_theta * 180.0

        # console debug
        print(f"[{i+1}/{len(samples)}] {os.path.basename(img_path)}  pred_rho={pred_rho:.4f}  pred_theta={pred_theta:.4f} ({theta_deg:.2f} deg)")

        # 6) Visualization (left=input, mid=UNet restored, right=final)
        vis_final = bgr_raw.copy()
        vis_final = draw_line_from_rho_theta(vis_final, pred_rho, pred_theta, color=(0, 0, 255), thickness=3)

        # optional reference line (does not affect prediction)
        if DEBUG_DRAW_TRAD:
            try:
                final_result, _, _, _ = detector.detect(bgr_raw)
                if final_result is not None:
                    Y, alpha = final_result
                    # 简单转端点用于画线：在中心列用Y，斜率 tan(alpha)
                    cx = (w_orig - 1) / 2.0
                    k = np.tan(np.deg2rad(alpha))
                    y_left = Y + k * (0 - cx)
                    y_right = Y + k * ((w_orig - 1) - cx)
                    pt1 = (0, int(np.clip(round(y_left), 0, h_orig - 1)))
                    pt2 = (w_orig - 1, int(np.clip(round(y_right), 0, h_orig - 1)))
                    cv2.line(vis_final, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass

        vis_input_show = cv2.resize((rgb_input_np * 255).astype(np.uint8), (w_orig, h_orig))
        vis_input_show = cv2.cvtColor(vis_input_show, cv2.COLOR_RGB2BGR)
        vis_restored_show = restored_orig_bgr

        tag = "SYN_DEGRADED" if ENABLE_DEGRADATION else "ORIGINAL"
        cv2.putText(vis_final, tag, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(
            vis_final,
            f"rho={pred_rho:.3f} theta={pred_theta:.3f} ({theta_deg:.1f}deg)",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        show_1 = resize_h(vis_input_show, SHOW_H)
        show_2 = resize_h(vis_restored_show, SHOW_H)
        show_3 = resize_h(vis_final, SHOW_H)
        combined = np.hstack([show_1, show_2, show_3])

        prefix = "degraded" if ENABLE_DEGRADATION else "clean"
        save_path = os.path.join(OUT_DIR, f"{prefix}_result_{i}.png")
        cv2.imwrite(save_path, combined)
        print(f"   -> Saved: {save_path}")

    print("Done!")


if __name__ == "__main__":
    main()
