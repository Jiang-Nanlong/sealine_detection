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
import random
import shutil

# === 导入你的模块 ===
# 确保 dataset_loader_offline.py 在同级目录
from dataset_loader_offline import OfflineHorizonDataset
# 确保 cnn_model.py 已经更新为 ResNet34 版本
from cnn_model import get_resnet34_model


# ==========================================
# 1. 定义周期性损失函数 (解决 0度/180度 跳变问题)
# ==========================================
# class HorizonPeriodicLoss(nn.Module):
#     def __init__(self, theta_weight=2.0):
#         super(HorizonPeriodicLoss, self).__init__()
#         self.mse = nn.MSELoss()
#         self.theta_weight = theta_weight
#
#     def forward(self, preds, targets):
#         """
#         preds:   [Batch, 2] -> (rho_norm, theta_norm) 范围 0~1
#         targets: [Batch, 2] -> (rho_norm, theta_norm) 范围 0~1
#         """
#         # 1. Rho Loss (距离误差)
#         # Rho 是线性的，直接用 MSE
#         loss_rho = self.mse(preds[:, 0], targets[:, 0])
#
#         # 2. Theta Loss (角度周期性误差)
#         # 将归一化 [0, 1] 还原回 弧度 [0, pi]
#         # dataset中 theta 范围是 0~180度
#         theta_pred_rad = preds[:, 1] * np.pi
#         theta_target_rad = targets[:, 1] * np.pi
#
#         # 周期性 Loss 公式: L = 1 - cos(2 * diff)
#         # 解释：直线的周期是 pi (180度)。但 cos 的周期是 2pi。
#         # 所以我们需要 2 * diff。
#         # 当 diff = 0 (0度) 或 diff = pi (180度) 时，cos(2*diff) = 1，Loss = 0。
#         diff = theta_pred_rad - theta_target_rad
#         loss_theta = torch.mean(1 - torch.cos(2 * diff))
#
#         # 总 Loss
#         return loss_rho + self.theta_weight * loss_theta

class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0, rho_beta=0.02, theta_beta=0.02):
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
        self.rho_loss = nn.SmoothL1Loss(beta=rho_beta)
        self.theta_loss = nn.SmoothL1Loss(beta=theta_beta)

    def forward(self, preds, targets):
        # rho: 0~1 线性
        loss_rho = self.rho_loss(preds[:, 0], targets[:, 0])

        # theta: 0~1 映射到 0~pi
        theta_p = preds[:, 1] * np.pi
        theta_t = targets[:, 1] * np.pi

        # 用 (cos 2θ, sin 2θ) 表示方向，天然满足 180° 周期
        vp = torch.stack([torch.cos(2 * theta_p), torch.sin(2 * theta_p)], dim=1)
        vt = torch.stack([torch.cos(2 * theta_t), torch.sin(2 * theta_t)], dim=1)

        loss_theta = self.theta_loss(vp, vt)

        return self.rho_weight * loss_rho + self.theta_weight * loss_theta


# ==========================================
# 2. 训练与评估主程序
# ==========================================

def theta_err_deg(pred_theta_norm, gt_theta_norm):
    # 0~1 对应 0~180°，直线周期也是 180°，所以用环形距离
    diff = (pred_theta_norm - gt_theta_norm).abs()
    diff = torch.minimum(diff, 1.0 - diff)
    return diff * 180.0

def rho_err_bins(pred_rho_norm, gt_rho_norm, H):
    # H 是 sinogram 高度(例如 2240)，bins 误差≈rho轴像素误差
    return (pred_rho_norm - gt_rho_norm).abs() * (H - 1)

def augment_sinogram_batch(x, y,
                           p_shift=0.7,
                           max_rho_shift=80,
                           max_theta_shift=8,
                           p_intensity=0.8,
                           noise_std=0.01,
                           gain_low=0.9, gain_high=1.1):
    """
    x: [B,C,H,W]  y: [B,2] (rho_norm, theta_norm)
    只对训练用：不会破坏 rho/theta 对应关系
    """
    B, C, H, W = x.shape

    # 强度增强（不改标签）
    if torch.rand(1, device=x.device).item() < p_intensity:
        gain = torch.empty((B,1,1,1), device=x.device).uniform_(gain_low, gain_high)
        x = x * gain
        x = x + torch.randn_like(x) * noise_std

    # 平移增强（要同步改标签）
    for i in range(B):
        if torch.rand(1, device=x.device).item() < p_shift:
            # rho 轴平移（不循环，卷入部分置0）
            sr = int(torch.randint(-max_rho_shift, max_rho_shift + 1, (1,), device=x.device).item())
            if sr != 0:
                x[i] = torch.roll(x[i], shifts=sr, dims=1)
                if sr > 0:
                    x[i, :, :sr, :] = 0
                else:
                    x[i, :, sr:, :] = 0
                y[i, 0] = torch.clamp(y[i, 0] + sr / (H - 1), 0.0, 1.0)

            # theta 轴平移（周期循环）
            st = int(torch.randint(-max_theta_shift, max_theta_shift + 1, (1,), device=x.device).item())
            if st != 0:
                x[i] = torch.roll(x[i], shifts=st, dims=2)
                theta_idx = y[i, 1] * (W - 1)
                theta_idx = torch.remainder(theta_idx + st, W)
                y[i, 1] = theta_idx / (W - 1)

    return x, y


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

    # indices = list(range(total_len))
    # train_indices = indices[:split_idx]
    # test_indices = indices[split_idx:]

    SEED = 42  # 你可以换成任意整数，保持不变就能复现划分结果

    # 1) 让 numpy / python / torch 都尽量可复现（可选但推荐）
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 2) 打乱索引后再切分
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(total_len).tolist()

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    os.makedirs("splits", exist_ok=True)
    np.save("splits/train_indices.npy", np.array(train_indices))
    np.save("splits/test_indices.npy", np.array(test_indices))
    print("已保存随机划分到 splits/ 目录")

    # 保存 SEED，方便下次直接加载
    # train_indices = np.load("splits/train_indices.npy").tolist()
    # test_indices = np.load("splits/test_indices.npy").tolist()

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
    # criterion = HorizonPeriodicLoss(theta_weight=1.5)  # 权重可调，1.5 表示稍侧重角度
    criterion = HorizonPeriodicLoss(rho_weight=1.0, theta_weight=2.0)

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

            inputs, labels = augment_sinogram_batch(inputs, labels)

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
    val_loss = 0.0
    val_rho_bins = 0.0
    val_theta_deg = 0.0
    n_val = 0

    with torch.no_grad():
        for v_in, v_label in test_loader:
            v_in = v_in.to(device, non_blocking=True)
            v_label = v_label.to(device, non_blocking=True).float()

            v_out = model(v_in)
            loss = criterion(v_out, v_label)
            val_loss += loss.item()

            H = v_in.shape[2]
            bsz = v_in.size(0)
            val_rho_bins += rho_err_bins(v_out[:, 0], v_label[:, 0], H).sum().item()
            val_theta_deg += theta_err_deg(v_out[:, 1], v_label[:, 1]).sum().item()
            n_val += bsz

    avg_val_loss = val_loss / len(test_loader)
    avg_rho_bins = val_rho_bins / n_val
    avg_theta_deg = val_theta_deg / n_val

    print(f"[VAL] loss={avg_val_loss:.4f} | rho_err≈{avg_rho_bins:.2f} bins | theta_err={avg_theta_deg:.2f}°")

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