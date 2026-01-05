# make_fusion_cache.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入你的模块
from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# ================= 配置 =================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR = r"Hashmani's Dataset/MU-SID"
SAVE_DIR = r"Hashmani's Dataset/FusionCache_v2"  # 建议新目录，避免覆盖

# 权重路径
RGHNET_CKPT = "rghnet_best_joint.pth"  # 用你训练最好的那个
DCE_WEIGHTS = "Epoch99.pth"

# 尺寸配置
IMG_SIZE_UNET = 384
RESIZE_H = 2240
RESIZE_W = 180

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================================

# ================= 语义mask鲁棒后处理相关参数（你主要调这里） =================
# morph_close=0 表示禁用；常用 0 / 3 / 5（太大可能把错误区域粘连，反而变差）
MORPH_CLOSE = 3
TOP_TOUCH_TOL = 2  # “接触顶边”的容忍像素（越大越宽松）
SKY_ID = 1         # 你的分割里 sky=1, sea=0

# 提边缘 + 形态学参数
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE_ITERS = 1  # 0 表示不膨胀；1~2 通常够用
# ==========================================================================


def calculate_radon_label(x1, y1, x2, y2, img_w, img_h, resize_h, resize_w):
    """计算归一化的 rho, theta 标签 (0~1)"""
    cx, cy = img_w / 2.0, img_h / 2.0
    dx, dy = x2 - x1, y2 - y1
    line_angle = np.arctan2(dy, dx)

    theta_rad = line_angle - np.pi / 2
    while theta_rad < 0:
        theta_rad += np.pi
    while theta_rad >= np.pi:
        theta_rad -= np.pi

    mx = (x1 + x2) / 2.0 - cx
    my = (y1 + y2) / 2.0 - cy
    rho = mx * np.cos(theta_rad) + my * np.sin(theta_rad)

    # theta 归一化 (0~180 -> 0~1)
    label_theta = np.degrees(theta_rad) / 180.0

    # rho 归一化：与 padding 模式一致
    original_diag = np.sqrt(img_w ** 2 + img_h ** 2)
    rho_pixel_pos = rho + original_diag / 2.0

    pad_top = (resize_h - original_diag) / 2.0
    final_rho_idx = rho_pixel_pos + pad_top
    label_rho = final_rho_idx / (resize_h - 1)

    return np.clip(label_rho, 0, 1), np.clip(label_theta, 0, 1)


def process_sinogram(sino, target_h, target_w):
    """归一化并 Padding 到统一尺寸"""
    mi, ma = sino.min(), sino.max()
    if ma - mi > 1e-6:
        sino_norm = (sino - mi) / (ma - mi)
    else:
        sino_norm = np.zeros_like(sino, dtype=np.float32)

    h_curr = sino_norm.shape[0]
    container = np.zeros((target_h, target_w), dtype=np.float32)
    start_h = (target_h - h_curr) // 2

    if h_curr <= target_h:
        container[start_h: start_h + h_curr, :] = sino_norm
    else:
        crop_start = (h_curr - target_h) // 2
        container[:, :] = sino_norm[crop_start: crop_start + target_h, :]

    return container


def post_process_mask_top_connected_simple(mask_np: np.ndarray,
                                           sky_id: int = 1,
                                           morph_close: int = 3,
                                           top_touch_tol: int = 2) -> np.ndarray:
    """
    0/1预测mask的鲁棒清理：
      1) 只保留“接触图像顶边”的 sky 连通域（去掉海面 sky 假阳性岛）
      2) 可选：对保留下来的 sky 做轻量 close 弥合裂缝
    """
    if mask_np.ndim != 2:
        raise ValueError("mask_np must be HxW")

    sky = (mask_np == sky_id).astype(np.uint8)

    # 1) 连通域：只保留接触顶边的 sky
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=np.uint8)
    for i in range(1, num_labels):
        y_top = stats[i, cv2.CC_STAT_TOP]
        if y_top <= top_touch_tol:
            keep[labels == i] = 1

    # 2) 对保留下来的天空做轻量闭运算（注意别太大）
    if morph_close and morph_close > 0:
        k = int(morph_close)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)

    out = np.zeros_like(mask_np, dtype=np.uint8)
    out[keep == 1] = sky_id
    return out


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[Params] MORPH_CLOSE={MORPH_CLOSE}, TOP_TOUCH_TOL={TOP_TOUCH_TOL}, "
          f"CANNY=({CANNY_LOW},{CANNY_HIGH}), EDGE_DILATE_ITERS={EDGE_DILATE_ITERS}")

    # 1) 加载 RG-HNet
    print(f"Loading RG-HNet from {RGHNET_CKPT} ...")
    seg_model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS).to(DEVICE)
    if os.path.exists(RGHNET_CKPT):
        state = torch.load(RGHNET_CKPT, map_location=DEVICE)
        seg_model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {RGHNET_CKPT}")
    seg_model.eval()

    # 2) 加载传统提取器（含 GPU Radon）
    print("Loading Traditional Extractor...")
    detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)

    # 3) 读取数据
    df = pd.read_csv(CSV_PATH, header=None)
    theta_scan = np.linspace(0., 180., RESIZE_W, endpoint=False)

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_name = str(row.iloc[0])
            try:
                x1, y1, x2, y2 = float(row.iloc[1]), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4])
            except Exception:
                continue

            # 读图
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path):
                img_path += ".JPG"
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)  # BGR
            if image is None:
                continue
            h_img, w_img = image.shape[:2]

            # --- 前端处理：resize -> tensor ---
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE_UNET, IMG_SIZE_UNET))
            inp_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # --- RG-HNet 推理：复原图 + 分割 logits ---
            restored_img_tensor, seg_logits, _ = seg_model(inp_tensor, None)

            # 分割 mask（0/1）
            pred_mask = seg_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # --- A. 传统特征流：使用复原图 ---
            restored_np_rgb = (restored_img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            restored_np_full = cv2.resize(restored_np_rgb, (w_img, h_img))
            restored_cv2_bgr = cv2.cvtColor(restored_np_full, cv2.COLOR_RGB2BGR)

            _, _, _, trad_sinograms = detector.detect(restored_cv2_bgr)

            processed_trad = []
            for s in trad_sinograms[:3]:
                processed_trad.append(process_sinogram(s, RESIZE_H, RESIZE_W))
            while len(processed_trad) < 3:
                processed_trad.append(np.zeros((RESIZE_H, RESIZE_W), dtype=np.float32))

            # --- B. 语义边界流：先鲁棒清理mask，再在交界处提边缘做 Radon ---
            pred_mask_full = cv2.resize(pred_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

            # ✅ robust: 只保留顶边连通天空，去掉海面sky假阳性岛
            pred_mask_full = post_process_mask_top_connected_simple(
                pred_mask_full,
                sky_id=SKY_ID,
                morph_close=MORPH_CLOSE,
                top_touch_tol=TOP_TOUCH_TOL
            )

            # mask 边界（交界处）：Canny
            edges = cv2.Canny((pred_mask_full * 255).astype(np.uint8), CANNY_LOW, CANNY_HIGH)

            # 可选：膨胀让边界更连续
            if EDGE_DILATE_ITERS and EDGE_DILATE_ITERS > 0:
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=int(EDGE_DILATE_ITERS))

            # 对“边界图”做 Radon，得到第4通道正弦图
            seg_sino_raw = detector._radon_gpu(edges, theta_scan)
            processed_seg = process_sinogram(seg_sino_raw, RESIZE_H, RESIZE_W)

            # --- C. 融合与保存 ---
            combined_input = np.stack(processed_trad + [processed_seg], axis=0).astype(np.float32)

            # 标签
            l_rho, l_theta = calculate_radon_label(x1, y1, x2, y2, w_img, h_img, RESIZE_H, RESIZE_W)

            np.save(os.path.join(SAVE_DIR, f"{idx}.npy"), {
                "input": combined_input,
                "label": np.array([l_rho, l_theta], dtype=np.float32)
            })

    print(f"Done! Cache saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
