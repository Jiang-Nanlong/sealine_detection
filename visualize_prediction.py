import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import os
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset
from cnn_model import get_resnet34_model


def draw_pred_line(img, rho_norm, theta_norm, color=(0, 255, 0)):
    """
    根据网络预测的 Radon 坐标画线。
    关键：Radon 变换是基于图像中心的，需要进行坐标转换。
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    diag = np.sqrt(h ** 2 + w ** 2)

    # 1. 还原 Theta (0~1 -> 0~180度)
    theta_deg = theta_norm * 180.0
    theta_rad = np.deg2rad(theta_deg)

    # 2. 还原 Rho (0~1 -> -Diag/2 ~ +Diag/2)
    # Radon 变换后的 sinogram 高度约为对角线长度
    # 0.5 对应中心 (Offset = 0)
    # 实际上 skimage.radon 的 rho 轴是通过 center 的
    # rho_norm = 0.5 -> offset = 0
    # rho_norm = 1.0 -> offset = +diag/2
    r_offset = (rho_norm - 0.5) * diag

    # 3. 直线方程推导
    # Radon 定义: (x-cx)cos(t) + (y-cy)sin(t) = r_offset
    # 展开: x cos + y sin = r_offset + cx cos + cy sin
    # 令 total_rho = r_offset + cx cos + cy sin
    # 标准 Hough/Polar 方程: x cos + y sin = total_rho

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    total_rho = r_offset + cx * cos_t + cy * sin_t

    # 4. 计算两个端点画线
    # x0, y0 是垂足
    x0 = cos_t * total_rho
    y0 = sin_t * total_rho

    # 沿着直线方向延伸 (-sin, cos)
    scale = 3000
    pt1 = (int(x0 - scale * sin_t), int(y0 + scale * cos_t))
    pt2 = (int(x0 + scale * sin_t), int(y0 - scale * cos_t))

    cv2.line(img, pt1, pt2, color, 3)
    return theta_deg


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