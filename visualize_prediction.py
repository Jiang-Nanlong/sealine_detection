import torch
import cv2
import numpy as np
import matplotlib

# 强制使用独立窗口，避免 PyCharm 报错
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# 导入必要的模块
# 注意：这里我们用原始的 Dataset 类，因为我们需要读取原始 JPG 图片来画图
# 离线 Dataset 里只有 .npy 数据，没有原始图
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset
from cnn_model import HorizonDetNet


def draw_line_polar(img, rho, theta, color, thickness=3):
    """
    在图片上根据 (rho, theta) 画线
    rho: 距离中心的像素距离
    theta: 法线角度 (度数)
    """
    h, w = img.shape[:2]
    # 中心坐标
    cx, cy = w / 2, h / 2

    # 将角度转为弧度
    # 注意：这里的 theta 是法线角度。
    # 直线方程: (x-cx) * cos(theta) + (y-cy) * sin(theta) = rho
    theta_rad = np.deg2rad(theta)

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # 我们需要找两个点来画线。通常找直线与图像边界的交点。
    # 也就是 x = 0, x = w, y = 0, y = h 四条边

    pts = []

    # 1. 左边界 (x=0) -> x_rel = -cx
    # (-cx) * cos + y_rel * sin = rho  =>  y_rel = (rho + cx * cos) / sin
    if abs(sin_t) > 1e-4:
        y_rel = (rho + cx * cos_t) / sin_t
        r = int(y_rel + cy)
        if 0 <= r <= h:
            pts.append((0, r))

    # 2. 右边界 (x=w) -> x_rel = cx
    if abs(sin_t) > 1e-4:
        y_rel = (rho - cx * cos_t) / sin_t
        r = int(y_rel + cy)
        if 0 <= r <= h:
            pts.append((w, r))

    # 3. 上边界 (y=0) -> y_rel = -cy
    # x_rel * cos + (-cy) * sin = rho => x_rel = (rho + cy * sin) / cos
    if abs(cos_t) > 1e-4:
        x_rel = (rho + cy * sin_t) / cos_t
        c = int(x_rel + cx)
        if 0 <= c <= w:
            pts.append((c, 0))

    # 4. 下边界 (y=h) -> y_rel = cy
    if abs(cos_t) > 1e-4:
        x_rel = (rho - cy * sin_t) / cos_t
        c = int(x_rel + cx)
        if 0 <= c <= w:
            pts.append((c, h))

    # 去重并画线
    pts = list(set(pts))
    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[1], color, thickness)


def visualize():
    # ================= 配置 =================
    # 路径要和之前一致
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"
    # 模型路径 (确保这是你最新训练出的模型)
    MODEL_PATH = "horizon_cnn_offline.pth"

    RESIZE_H = 362
    RESIZE_W = 180

    # 评估参数
    APPROX_MAX_DIAG = 2203.0
    MAX_THETA_DEG = 180.0
    # =======================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型: {MODEL_PATH} ...")

    # 1. 加载模型结构
    model = HorizonDetNet(in_channels=3, img_h=RESIZE_H, img_w=RESIZE_W).to(device)
    # 2. 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print("错误：找不到模型文件！请检查路径或先运行训练脚本。")
        return

    model.eval()

    # 3. 加载数据集 (为了读取原始图片)
    print("正在初始化数据集用于读取原图...")
    dataset = HorizonFusionDataset(CSV_PATH, IMG_DIR, resize_h=RESIZE_H, resize_w=RESIZE_W)

    # 4. 随机挑选测试集中的图片进行可视化
    # 假设测试集是从 2473 开始的
    test_start_idx = 2473
    total_samples = len(dataset)

    # 随机选 6 张
    indices = np.random.choice(range(test_start_idx, total_samples), 6, replace=False)
    # 或者手动指定看某几张
    # indices = [2475, 2500, 2600]

    plt.figure(figsize=(15, 10))

    print("开始推理并绘图...")
    with torch.no_grad():
        for plot_idx, idx in enumerate(indices):
            # 获取输入和标签
            # dataset[idx] 会返回 (input_tensor, label_tensor)
            input_tensor, label = dataset[idx]

            # 模型推理
            input_batch = input_tensor.unsqueeze(0).to(device)
            output = model(input_batch).cpu().squeeze()  # [2]

            # --- 数据准备 ---
            # 1. 读取原图 (用于画图)
            row = dataset.data.iloc[idx]
            img_name = str(row.iloc[0])
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path): img_path += ".JPG"

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue

            # BGR 转 RGB
            vis_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            h, w = vis_img.shape[:2]

            # 计算真实的对角线长度 (用于反归一化)
            real_diag = np.sqrt(h ** 2 + w ** 2)

            # --- 反归一化 ---
            # 真值 (GT)
            gt_rho_norm, gt_theta_norm = label[0].item(), label[1].item()
            # 还原公式：rho = (val - 0.5) * Diag
            # (注意：这里的反归一化公式必须和 dataset_loader 里的 calculate_radon_label 对应)
            # Dataset里是: label_rho = (rho / (diag/2) + 1) / 2
            # 逆运算: rho = (label_rho * 2 - 1) * (diag/2)

            gt_rho = (gt_rho_norm * 2 - 1) * (real_diag / 2)
            gt_theta = gt_theta_norm * 180.0

            # 预测值 (Pred)
            pred_rho_norm, pred_theta_norm = output[0].item(), output[1].item()
            pred_rho = (pred_rho_norm * 2 - 1) * (real_diag / 2)
            pred_theta = pred_theta_norm * 180.0

            # --- 画图 ---
            # 红色 = 真值 GT
            draw_line_polar(vis_img, gt_rho, gt_theta, (255, 0, 0), thickness=5)
            # 绿色 = 预测 Pred
            draw_line_polar(vis_img, pred_rho, pred_theta, (0, 255, 0), thickness=3)

            # 计算误差文本
            err_rho = abs(gt_rho - pred_rho)
            err_theta = abs(gt_theta - pred_theta)

            # 子图
            plt.subplot(2, 3, plot_idx + 1)
            plt.imshow(vis_img)
            plt.title(f"Img {idx}\nErr: Rho={err_rho:.1f}px, Theta={err_theta:.1f}°")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()