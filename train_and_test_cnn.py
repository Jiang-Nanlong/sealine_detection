import matplotlib

# 强制使用 TkAgg 后端，避免在某些环境下绘图报错
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import shutil

# === 导入你的模块 ===
# 确保 dataset_loader_offline.py 在同级目录
from dataset_loader_offline import OfflineHorizonDataset
# 确保 cnn_model.py 已经更新为 ResNet34 版本
from cnn_model import get_resnet34_model


# ==========================================
# 1. 定义周期性损失函数 (解决 0度/180度 跳变问题)
# ==========================================
class HorizonPeriodicLoss(nn.Module):
    def __init__(self, theta_weight=2.0):
        super(HorizonPeriodicLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.theta_weight = theta_weight

    def forward(self, preds, targets):
        """
        preds:   [Batch, 2] -> (rho_norm, theta_norm) 范围 0~1
        targets: [Batch, 2] -> (rho_norm, theta_norm) 范围 0~1
        """
        # 1. Rho Loss (距离误差)
        # Rho 是线性的，直接用 MSE
        loss_rho = self.mse(preds[:, 0], targets[:, 0])

        # 2. Theta Loss (角度周期性误差)
        # 将归一化 [0, 1] 还原回 弧度 [0, pi]
        # dataset中 theta 范围是 0~180度
        theta_pred_rad = preds[:, 1] * np.pi
        theta_target_rad = targets[:, 1] * np.pi

        # 周期性 Loss 公式: L = 1 - cos(2 * diff)
        # 解释：直线的周期是 pi (180度)。但 cos 的周期是 2pi。
        # 所以我们需要 2 * diff。
        # 当 diff = 0 (0度) 或 diff = pi (180度) 时，cos(2*diff) = 1，Loss = 0。
        diff = theta_pred_rad - theta_target_rad
        loss_theta = torch.mean(1 - torch.cos(2 * diff))

        # 总 Loss
        return loss_rho + self.theta_weight * loss_theta


# ==========================================
# 2. 训练与评估主程序
# ==========================================
def train_and_evaluate():
    # ================= 4090 显卡 激进配置 =================
    # 缓存目录 (请确保这里面是【新代码】生成的 .npy 文件)
    CACHE_DIR = r"Hashmani's Dataset/OfflineCache"

    # 4090 显存很大，Batch Size 拉大能让 BN 层表现更好，梯度更稳
    BATCH_SIZE = 64

    # 学习率与轮数
    LEARNING_RATE = 2e-4
    EPOCHS = 100

    # 图像尺寸 (必须与 Dataset 生成时一致)
    RESIZE_H = 2240
    RESIZE_W = 180

    # 数据集分割点
    SPLIT_INDEX = 2473
    # ====================================================

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
        # 开启 cudnn 自动寻找最优算法
        torch.backends.cudnn.benchmark = True

    # --- 1. 加载数据 ---
    if not os.path.exists(CACHE_DIR) or len(os.listdir(CACHE_DIR)) == 0:
        raise FileNotFoundError(f"缓存目录为空: {CACHE_DIR}\n请先运行数据生成脚本，生成新的 .npy 数据！")

    print(f"正在从缓存加载数据: {CACHE_DIR} ...")
    full_dataset = OfflineHorizonDataset(CACHE_DIR)
    total_len = len(full_dataset)
    print(f"数据集总样本数: {total_len}")

    # 分割训练集和测试集
    # 容错处理：如果缓存数量少于分割点，自动按比例分割
    if total_len < SPLIT_INDEX:
        print(f"警告: 数据总数 ({total_len}) 少于预设分割点 ({SPLIT_INDEX})")
        split_idx = int(total_len * 0.9)
    else:
        split_idx = SPLIT_INDEX

    indices = list(range(total_len))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"训练集: {len(train_dataset)}")
    print(f"测试集: {len(test_dataset)}")

    # DataLoader
    # num_workers=8 充分利用 CPU 预取数据
    # pin_memory=True 加速内存到显存的传输
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # --- 2. 初始化模型 (ResNet-34) ---
    print("初始化 ResNet-34 + CBAM 模型...")
    try:
        model = get_resnet34_model().to(device)
    except NameError:
        raise ImportError("无法加载 get_resnet34_model，请确认 cnn_model.py 已更新且包含该函数。")

    # --- 3. 优化器与损失 ---
    criterion = HorizonPeriodicLoss(theta_weight=1.5)  # 权重可调，1.5 表示稍侧重角度

    # 使用 AdamW (带权重衰减的 Adam，泛化性更好)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 余弦退火学习率调度 (Cosine Annealing)
    # 它可以让学习率周期性地下降再重启，有助于跳出局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # --- 4. 训练循环 ---
    loss_history = []
    best_test_error = float('inf')

    print("\n开始极速训练...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # 进度条
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS}", unit="batch")

        for inputs, labels in pbar:
            # 移动数据到 GPU (non_blocking加速)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算 Loss
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 更新进度条显示的当前 Loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.1e}"})

        # 记录 Epoch 平均 Loss
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # 更新学习率
        scheduler.step()

        # --- 简单的测试集抽样评估 (每 5 轮看一次，省时间) ---
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_in, v_label in test_loader:
                    v_in, v_label = v_in.to(device), v_label.to(device)
                    v_out = model(v_in)
                    v_loss = criterion(v_out, v_label)
                    val_loss += v_loss.item()
            avg_val_loss = val_loss / len(test_loader)
            print(f"Validation Loss: {avg_val_loss:.5f}")

            # 保存最佳模型
            if avg_val_loss < best_test_error:
                best_test_error = avg_val_loss
                torch.save(model.state_dict(), "horizon_resnet34_best.pth")

    # 保存最终模型
    torch.save(model.state_dict(), "horizon_resnet34_last.pth")
    print("\n训练完成！模型已保存: horizon_resnet34_last.pth")

    # --- 5. 最终全量评估 ---
    print("\n正在进行最终评估...")
    model.load_state_dict(torch.load("horizon_resnet34_best.pth"))  # 加载最好的那个
    model.eval()

    total_rho_err = 0.0
    total_theta_err = 0.0
    count = 0

    # 评估用参数
    APPROX_DIAG = np.sqrt(1920 ** 2 + 1080 ** 2)  # 假设 1080p，约 2203 像素

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # --- 误差计算 ---
            # 1. Rho 误差 (像素级近似)
            # labels 是归一化到 [0, 1] 的 rho。
            # 1.0 代表 rho = Max_Rho, 0.0 代表 rho = 0 (我们假设是这样归一化的)
            # 实际上 dataset_loader 里的归一化是线性的，所以差值直接乘以对角线长度即可估算像素误差
            rho_diff = torch.abs(outputs[:, 0] - labels[:, 0])
            batch_rho_err = torch.sum(rho_diff * APPROX_DIAG)  # 近似还原回像素

            # 2. Theta 误差 (角度)
            # 考虑周期性: diff = min(|a-b|, 180 - |a-b|)
            # outputs[:, 1] 是 0~1 对应 0~180度
            deg_pred = outputs[:, 1] * 180.0
            deg_gt = labels[:, 1] * 180.0

            diff = torch.abs(deg_pred - deg_gt)
            # 处理跨越 0/180 度的情况 (例如 1度 和 179度，差值应该是 2度)
            diff = torch.min(diff, 180.0 - diff)

            batch_theta_err = torch.sum(diff)

            total_rho_err += batch_rho_err.item()
            total_theta_err += batch_theta_err.item()
            count += inputs.size(0)

    avg_rho = total_rho_err / count
    avg_theta = total_theta_err / count

    print("=" * 40)
    print(f"最终测试集结果 (样本数 {count}):")
    print(f"平均 Rho 误差: {avg_rho:.2f} 像素 (参考值)")
    print(f"平均 Theta 误差: {avg_theta:.2f} 度")
    print("=" * 40)

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title("ResNet-34 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Periodic Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()