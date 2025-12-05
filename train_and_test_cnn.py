# train_fixed_split.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# 导入你之前的模块
from dataset_loader_gradient_radon_cnn import HorizonFusionDataset
from cnn_model import HorizonDetNet


def train_and_evaluate():
    # --- 1. 配置路径与参数 ---
    CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
    IMG_DIR = r"Hashmani's Dataset/MU-SID"

    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    RESIZE_H = 362
    RESIZE_W = 180

    # 分割点
    SPLIT_INDEX = 2473

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 加载全部数据 ---
    print("正在加载数据集...")
    full_dataset = HorizonFusionDataset(CSV_PATH, IMG_DIR, resize_h=RESIZE_H, resize_w=RESIZE_W)
    total_len = len(full_dataset)
    print(f"数据集总数: {total_len}")

    # --- 3. 强制按索引划分数据集 ---
    if total_len < SPLIT_INDEX:
        raise ValueError(f"数据集样本数 ({total_len}) 少于要求的训练集数量 ({SPLIT_INDEX})，请检查 CSV 文件。")

    # 创建索引列表
    train_indices = list(range(0, SPLIT_INDEX))
    test_indices = list(range(SPLIT_INDEX, total_len))

    # 使用 Subset 创建子数据集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"训练集数量: {len(train_dataset)} (前 {SPLIT_INDEX} 张)")
    print(f"测试集数量: {len(test_dataset)} (后 {len(test_dataset)} 张)")

    # 创建 DataLoader
    # 训练集打乱 (shuffle=True)，测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 4. 初始化模型 ---
    # 注意：in_channels=3 (传统方法3尺度)
    model = HorizonDetNet(in_channels=3, img_h=RESIZE_H, img_w=RESIZE_W).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 5. 训练循环 ---
    loss_history = []

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {epoch_loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "horizon_model_2473split.pth")
    print("训练完成，模型已保存。")

    # --- 6. 在测试集上评估性能 ---
    print("\n正在测试集上进行最终评估...")
    model.eval()
    test_loss = 0.0
    mae_rho = 0.0
    mae_theta = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # 计算 MSE Loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算平均绝对误差 (MAE) 用于直观理解
            # labels shape: [Batch, 2] -> (rho, theta)
            diff = torch.abs(outputs - labels)
            mae_rho += torch.sum(diff[:, 0]).item()
            mae_theta += torch.sum(diff[:, 1]).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_mae_rho = mae_rho / len(test_dataset)
    avg_mae_theta = mae_theta / len(test_dataset)

    print("=" * 30)
    print(f"测试集最终评估结果 (200张图片):")
    print(f"MSE Loss (总体误差): {avg_test_loss:.6f}")
    print(f"Rho MAE (归一化距离误差): {avg_mae_rho:.6f}")
    print(f"Theta MAE (归一化角度误差): {avg_mae_theta:.6f}")

    # 简单的解释
    # Theta 归一化是除以 180，所以 0.01 的误差大约对应 1.8 度
    print(f"估算角度平均误差: {avg_mae_theta * 180:.2f} 度")
    print("=" * 30)

    # 画 Loss 曲线
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()