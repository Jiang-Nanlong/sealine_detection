import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import os
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset
from cnn_model import get_resnet34_model


def draw_pred_line(img, rho_norm, theta_norm, resize_h=2240, color=(0, 255, 0), thickness=2):
    """
    Draw the predicted horizon line on an image.

    IMPORTANT: this matches the label definition used in make_fusion_cache.py::calculate_radon_label
    (and test.py::get_line_ends):

        label_rho = (rho + diag/2 + pad_top) / (resize_h - 1)
        label_theta = theta_deg / 180

    where:
        - diag = sqrt(w^2 + h^2) computed on the *current image size*
        - pad_top = (resize_h - diag) / 2 is the vertical padding used to place the sinogram
          into a fixed container of height resize_h.

    Args:
        img: BGR image (H, W, 3)
        rho_norm: predicted rho in [0, 1]
        theta_norm: predicted theta in [0, 1]  -> degrees in [0, 180)
        resize_h: sinogram container height used during training (default 2240)
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    diag = float(np.sqrt(w ** 2 + h ** 2))
    pad_top = (float(resize_h) - diag) / 2.0

    # Invert label mapping back to centered polar line parameters:
    # (x-cx)*cos(theta) + (y-cy)*sin(theta) = rho
    rho = float(rho_norm) * (float(resize_h) - 1.0) - pad_top - (diag / 2.0)
    theta_rad = np.deg2rad(float(theta_norm) * 180.0)

    cos_t, sin_t = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    x0 = cos_t * rho
    y0 = sin_t * rho

    scale = int(max(w, h) * 2)
    pt1 = (int(cx + x0 - scale * sin_t), int(cy + y0 + scale * cos_t))
    pt2 = (int(cx + x0 + scale * sin_t), int(cy + y0 - scale * cos_t))

    cv2.line(img, pt1, pt2, color, thickness)
    return img


def visualize_results():
    # 配置
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"
    MODEL_PATH = "horizon_resnet34_last.pth"

    RESIZE_H = 2240
    RESIZE_W = 180

    device = torch.device("cuda")

    # 加载模型
    model = get_resnet34_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 加载数据集 (只用来读图和GT坐标)
    dataset = HorizonFusionDataset(
        CSV_PATH,
        IMG_DIR,
        resize_h=RESIZE_H,
        resize_w=RESIZE_W
    )

    indices = np.random.choice(range(len(dataset)), 6, replace=False)

    plt.figure(figsize=(16, 10))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 获取输入
            input_tensor, _ = dataset[idx]  # 这里的 label 暂时不用，我们直接读 CSV

            # 推理
            input_batch = input_tensor.unsqueeze(0).to(device)
            output = model(input_batch).cpu().squeeze()
            pred_rho, pred_theta = output[0].item(), output[1].item()

            # 读取原图和CSV真值
            row = dataset.data.iloc[idx]
            img_name = str(row.iloc[0])
            # 读取真值坐标 (直接用 CSV 数据，不经过任何变换，这是真理)
            gx1, gy1 = int(float(row.iloc[1])), int(float(row.iloc[2]))
            gx2, gy2 = int(float(row.iloc[3])), int(float(row.iloc[4]))

            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path): img_path += ".JPG"

            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- 画图 ---
            # 1. 红线：绝对真值 (CSV坐标)
            cv2.line(img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 5)

            # 2. 绿线：网络预测 (经过反变换)
            pred_deg = draw_pred_line(img, pred_rho, pred_theta, (0, 255, 0))

            plt.subplot(2, 3, i + 1)
            plt.imshow(img)
            plt.title(f"ID: {idx}\nPred Theta: {pred_deg:.1f}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_results()