import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 导入你的模块
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset
from cnn_model import HorizonDetNet


def train_and_evaluate():
    # ================= 配置参数 =================
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"

    # 训练超参数
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 50  # 建议稍微多一点，30可能刚收敛

    # 网络输入尺寸 (必须与 Dataset 里的 resize 对应)
    RESIZE_H = 362  # Rho 轴 (对应 Dataset 的 resize_h)
    RESIZE_W = 180  # Theta 轴 (对应 Dataset 的 resize_w)

    # 评估用的反归一化参数
    # 因为 Dataset 把 rho 归一化到了 [0,1]，我们需要还原回像素看误差
    # 1080P 图片对角线约为 2203
    APPROX_MAX_DIAG = 2203.0
    MAX_THETA_DEG = 180.0

    # 数据集分割点
    SPLIT_INDEX = 2473
    # ===========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # --- 1. 加载数据 ---
    print("📂 正在加载数据集...")
    # 关键：这里 resize_h/w 必须传入，确保 Dataset 内部缩放正确
    full_dataset = HorizonFusionDataset(CSV_PATH, IMG_DIR, resize_h=RESIZE_H, resize_w=RESIZE_W)
    total_len = len(full_dataset)
    print(f"📊 数据集总数: {total_len}")

    if total_len < SPLIT_INDEX:
        raise ValueError("数据集数量不足，请检查路径是否正确！")

    # 划分训练/测试集
    train_dataset = Subset(full_dataset, range(0, SPLIT_INDEX))
    test_dataset = Subset(full_dataset, range(SPLIT_INDEX, total_len))

    # DataLoader (必须 num_workers=0，因为 Dataset 用到了 CUDA)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 2. 检查数据形状 (防止跑一半报错) ---
    first_batch, first_label = next(iter(train_loader))
    print(f"🔍 输入形状检查: {first_batch.shape}")  # 应为 [8, 3, 362, 180]
    print(f"🔍 标签形状检查: {first_label.shape}")  # 应为 [8, 2]

    if first_batch.shape[2] != RESIZE_H:
        raise ValueError(f"尺寸不匹配！Dataset输出H={first_batch.shape[2]}, 预期{RESIZE_H}")

    # --- 3. 初始化模型 ---
    # in_channels=3 对应传统方法的三个尺度
    model = HorizonDetNet(in_channels=3, img_h=RESIZE_H, img_w=RESIZE_W).to(device)

    # 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率调整：每 15 轮衰减一次
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # --- 4. 训练循环 ---
    loss_history = []
    print("\n🔥 开始训练...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # 标签已经在 Dataset 里归一化到 0-1 了

            optimizer.zero_grad()
            outputs = model(inputs)  # 输出也是预测的 0-1 值

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        scheduler.step()

        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {epoch_loss:.6f} | LR: {current_lr:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "horizon_cnn_gpu.pth")
    print("💾 模型已保存: horizon_cnn_gpu.pth")

    # --- 5. 评估 (Evaluation) ---
    print("\n🧪 正在评估测试集...")
    model.eval()

    total_mae_rho_pixel = 0.0
    total_mae_theta_degree = 0.0

    count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 预测 (0-1)
            outputs = model(inputs)

            # --- 反归一化计算真实物理误差 ---
            # 标签 Rho: 0.5是中心, 0是-Diag/2, 1是+Diag/2
            # 还原公式: real_rho = (val - 0.5) * Diag
            # 但为了算 MAE (绝对误差)，可以直接算: abs(pred - gt) * Diag

            # Rho 误差 (像素)
            diff_rho_norm = torch.abs(outputs[:, 0] - labels[:, 0])
            # Dataset里是用 original_diag / 2 做分母，这里还原回去
            # 这是一个近似值，因为每张图对角线不一样，但在评估时用平均值即可
            batch_mae_rho = torch.sum(diff_rho_norm * (APPROX_MAX_DIAG))

            # Theta 误差 (度)
            diff_theta_norm = torch.abs(outputs[:, 1] - labels[:, 1])
            batch_mae_theta = torch.sum(diff_theta_norm * MAX_THETA_DEG)

            total_mae_rho_pixel += batch_mae_rho.item()
            total_mae_theta_degree += batch_mae_theta.item()
            count += inputs.size(0)

    avg_rho_error = total_mae_rho_pixel / count
    avg_theta_error = total_mae_theta_degree / count

    print("=" * 40)
    print(f"📊 测试集评估结果 (共 {count} 张):")
    print(f"   平均 Rho 误差: {avg_rho_error:.2f} 像素 (在1080P图像中)")
    print(f"   平均 Theta 误差: {avg_theta_error:.2f} 度")
    print("=" * 40)

    # 绘图
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()