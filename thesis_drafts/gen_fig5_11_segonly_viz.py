"""生成图5-11: seg-only海天线提取过程可视化
4子图: 原始图像 → UNet分割mask → 列采样跳变点标注 → 拟合直线叠加在原图上
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'method2_unet_radon'))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──
IMG_PATH = '/home/jetson/Documents/Hashmani_Dataset/MU-SID/DSC_0054_3.JPG'
UNET_WEIGHTS = '../weights/rghnet_best_c2.pth'
GT_CSV = '../splits_musid/GroundTruth_test.csv'
RES_W, RES_H = 512, 288
NUM_SAMPLES = 32
OUT_PNG = 'fig5_11_segonly_process.png'
OUT_PDF = 'fig5_11_segonly_process.pdf'

# ── 加载GT ──
def load_gt(csv_path, stem):
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == stem:
                return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    return None

# ── 加载UNet ──
def load_unet(device):
    from unet_model import RestorationGuidedHorizonNet
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=None, require_dce=False)
    unet.dce_net = None
    unet = unet.to(device)
    state = torch.load(UNET_WEIGHTS, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    unet.load_state_dict(state, strict=False)
    unet.eval()
    unet.half()
    return unet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = load_unet(device)

    # 读取并预处理图像
    img_bgr = cv2.imread(IMG_PATH)
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_bgr, (RES_W, RES_H))

    tensor = torch.from_numpy(img_resized[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float16) * (1.0 / 255.0)

    # UNet推理
    with torch.no_grad():
        _, seg_logits, _ = unet(tensor, target=None,
                                 enable_restoration=False, enable_segmentation=True)
    seg_mask = seg_logits.argmax(dim=1)[0]  # (H, W) on GPU

    # 列采样
    cols = torch.linspace(0, RES_W - 1, NUM_SAMPLES, device=device).long()
    sampled = seg_mask[:, cols]
    diff = sampled[1:, :] - sampled[:-1, :]
    has_trans = (diff != 0).float()
    first_trans = has_trans.argmax(dim=0)
    any_trans = has_trans.any(dim=0)

    cols_np = cols.cpu().numpy().astype(np.float64)
    trans_np = first_trans.cpu().numpy().astype(np.float64)
    valid = any_trans.cpu().numpy()
    seg_mask_np = seg_mask.cpu().numpy()

    x_pts = cols_np[valid]
    y_pts = trans_np[valid]

    # 最小二乘拟合 + 离群点剔除
    coeffs = np.polyfit(x_pts, y_pts, 1)
    slope, intercept = coeffs
    residuals = np.abs(y_pts - (slope * x_pts + intercept))
    inlier = residuals < 5.0
    outlier = ~inlier
    if inlier.sum() >= 2:
        coeffs = np.polyfit(x_pts[inlier], y_pts[inlier], 1)
        slope, intercept = coeffs

    # 拟合线端点 (模型分辨率)
    fit_y1 = slope * 0 + intercept
    fit_y2 = slope * (RES_W - 1) + intercept

    # GT (原始分辨率 → 模型分辨率)
    stem = os.path.splitext(os.path.basename(IMG_PATH))[0]
    gt = load_gt(GT_CSV, stem)

    # ── 4子图绘制 ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))

    img_show = cv2.resize(img_rgb, (RES_W, RES_H))

    # (a) 原始图像
    axes[0].imshow(img_show)
    axes[0].set_title('(a) Input Image', fontsize=12)
    axes[0].axis('off')

    # (b) UNet分割mask
    mask_vis = np.zeros((*seg_mask_np.shape, 3), dtype=np.uint8)
    mask_vis[seg_mask_np == 1] = [135, 206, 250]  # sky = light blue
    mask_vis[seg_mask_np == 0] = [34, 85, 34]     # sea = dark green
    axes[1].imshow(mask_vis)
    axes[1].set_title('(b) Segmentation Mask', fontsize=12)
    axes[1].axis('off')

    # (c) 列采样跳变点
    axes[2].imshow(mask_vis, alpha=0.5)
    # 有效点 (inlier)
    axes[2].scatter(x_pts[inlier], y_pts[inlier], c='lime', s=40,
                    edgecolors='black', linewidths=0.5, zorder=5, label='Inlier')
    # 离群点
    if outlier.any():
        axes[2].scatter(x_pts[outlier], y_pts[outlier], c='red', s=40, marker='x',
                        linewidths=1.5, zorder=5, label='Outlier')
    # 竖线标示采样列
    for c in cols_np:
        axes[2].axvline(c, color='gray', alpha=0.2, linewidth=0.5)
    axes[2].set_xlim(0, RES_W)
    axes[2].set_ylim(RES_H, 0)
    axes[2].set_title('(c) Column Sampling', fontsize=12)
    axes[2].legend(fontsize=8, loc='lower right')
    axes[2].axis('off')

    # (d) 拟合直线叠加在原图上
    axes[3].imshow(img_show)
    axes[3].plot([0, RES_W - 1], [fit_y1, fit_y2], 'r-', linewidth=2, label='Prediction')
    if gt is not None:
        sx, sy = RES_W / w_orig, RES_H / h_orig
        gt_x1, gt_y1, gt_x2, gt_y2 = gt
        axes[3].plot([gt_x1 * sx, gt_x2 * sx], [gt_y1 * sy, gt_y2 * sy],
                     'g--', linewidth=2, label='Ground Truth')
    axes[3].set_xlim(0, RES_W)
    axes[3].set_ylim(RES_H, 0)
    axes[3].set_title('(d) Fitted Horizon Line', fontsize=12)
    axes[3].legend(fontsize=8, loc='lower right')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUT_PDF, bbox_inches='tight')
    print(f'Saved {OUT_PNG} / {OUT_PDF}')

if __name__ == '__main__':
    main()
