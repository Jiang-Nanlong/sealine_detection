# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sympy import false
# from torch.utils.data import DataLoader, Subset
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from tqdm import tqdm
#
# # 导入你的模块
# from dataset_loader_gradient_radon_cnn import HorizonFusionDataset
# from dataset_loader_offline import OfflineHorizonDataset
# from cnn_model import HorizonDetNet
#
#
# def train_and_evaluate():
#     # ================= 配置参数 =================
#     CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
#     IMG_DIR = r"Hashmani's Dataset/MU-SID"
#     CACHE_DIR = r"Hashmani's Dataset/OfflineCache"
#
#     # 训练超参数
#     BATCH_SIZE = 8
#     LEARNING_RATE = 1e-4
#     EPOCHS = 35  # 建议稍微多一点，30可能刚收敛
#
#     # 网络输入尺寸 (必须与 Dataset 里的 resize 对应)
#     RESIZE_H = 362  # Rho 轴 (对应 Dataset 的 resize_h)
#     RESIZE_W = 180  # Theta 轴 (对应 Dataset 的 resize_w)
#
#     # 评估用的反归一化参数
#     # 因为 Dataset 把 rho 归一化到了 [0,1]，我们需要还原回像素看误差
#     # 1080P 图片对角线约为 2203
#     APPROX_MAX_DIAG = 2203.0
#     MAX_THETA_DEG = 180.0
#
#     # 数据集分割点
#     SPLIT_INDEX = 2473
#     # ===========================================
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
#
#     # --- 1. 加载数据 ---
#     print("正在加载数据集...")
#     # 关键：这里 resize_h/w 必须传入，确保 Dataset 内部缩放正确
#     full_dataset = HorizonFusionDataset(CSV_PATH, IMG_DIR, resize_h=RESIZE_H, resize_w=RESIZE_W)
#     total_len = len(full_dataset)
#     print(f"数据集总数: {total_len}")
#
#     FAST_DEBUG = False
#
#     if FAST_DEBUG:
#         debug_train_size = 32
#         debug_test_size = 8
#
#         # 确保不会越界
#         if total_len < (debug_train_size + debug_test_size):
#             raise ValueError(f"数据总数 {total_len} 太少了，不够调试用")
#
#         train_indices = list(range(0, debug_train_size))
#         test_indices = list(range(debug_train_size, debug_train_size + debug_test_size))
#     else:
#         # 全量
#         if total_len < SPLIT_INDEX:
#             raise ValueError("数据集数量不足，请检查路径是否正确！")
#
#         train_indices = list(range(0, SPLIT_INDEX))
#         test_indices = list(range(SPLIT_INDEX, total_len))
#
#     # 划分训练/测试集
#     train_dataset = Subset(full_dataset, train_indices)
#     test_dataset = Subset(full_dataset, test_indices)
#
#     # DataLoader (必须 num_workers=0，因为 Dataset 用到了 CUDA)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#
#     # --- 2. 检查数据形状 (防止跑一半报错) ---
#     first_batch, first_label = next(iter(train_loader))
#     print(f"输入形状检查: {first_batch.shape}")  # 应为 [8, 3, 362, 180]
#     print(f"标签形状检查: {first_label.shape}")  # 应为 [8, 2]
#
#     if first_batch.shape[2] != RESIZE_H:
#         raise ValueError(f"尺寸不匹配！Dataset输出H={first_batch.shape[2]}, 预期{RESIZE_H}")
#
#     # --- 3. 初始化模型 ---
#     # in_channels=3 对应传统方法的三个尺度
#     model = HorizonDetNet(in_channels=3, img_h=RESIZE_H, img_w=RESIZE_W).to(device)
#
#     # 损失函数与优化器
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     # 学习率调整：每 15 轮衰减一次
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
#
#     # --- 4. 训练循环 ---
#     loss_history = []
#     print("\n开始训练...")
#
#     for epoch in range(EPOCHS):
#         model.train()
#         running_loss = 0.0
#
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")
#
#         for i, (inputs, labels) in enumerate(progress_bar):
#             inputs = inputs.to(device)
#             labels = labels.to(device).float()  # 标签已经在 Dataset 里归一化到 0-1 了
#
#             optimizer.zero_grad()
#             outputs = model(inputs)  # 输出也是预测的 0-1 值
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             progress_bar.set_postfix({'loss': loss.item()})
#
#         epoch_loss = running_loss / len(train_loader)
#         loss_history.append(epoch_loss)
#         scheduler.step()
#
#         # 打印进度
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch + 1} Done | Loss: {epoch_loss:.6f}")
#
#     # 保存模型
#     torch.save(model.state_dict(), "horizon_cnn_gpu.pth")
#     print("模型已保存: horizon_cnn_gpu.pth")
#
#     print("\n回测训练集...")
#     model.eval()
#     train_mae_rho = 0.0
#     with torch.no_grad():
#         # 只测一个 Batch 看看就行
#         for inputs, labels in train_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#
#             diff = torch.abs(outputs[:, 0] - labels[:, 0])
#             train_mae_rho += torch.sum(diff * APPROX_MAX_DIAG).item()
#             break  # 只看第一批
#
#     print(f"训练集 Batch 1 平均 Rho 误差: {train_mae_rho / BATCH_SIZE:.2f} 像素")
#
#     # --- 5. 评估 (Evaluation) ---
#     print("\n正在评估测试集...")
#     model.eval()
#
#     total_mae_rho_pixel = 0.0
#     total_mae_theta_degree = 0.0
#
#     count = 0
#
#     with torch.no_grad():
#         test_bar = tqdm(test_loader, desc="Testing", unit="batch")
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             # 预测 (0-1)
#             outputs = model(inputs)
#
#             # --- 反归一化计算真实物理误差 ---
#             # 标签 Rho: 0.5是中心, 0是-Diag/2, 1是+Diag/2
#             # 还原公式: real_rho = (val - 0.5) * Diag
#             # 但为了算 MAE (绝对误差)，可以直接算: abs(pred - gt) * Diag
#
#             # Rho 误差 (像素)
#             diff_rho_norm = torch.abs(outputs[:, 0] - labels[:, 0])
#             # Dataset里是用 original_diag / 2 做分母，这里还原回去
#             # 这是一个近似值，因为每张图对角线不一样，但在评估时用平均值即可
#             batch_mae_rho = torch.sum(diff_rho_norm * (APPROX_MAX_DIAG))
#
#             # Theta 误差 (度)
#             diff_theta_norm = torch.abs(outputs[:, 1] - labels[:, 1])
#             batch_mae_theta = torch.sum(diff_theta_norm * MAX_THETA_DEG)
#
#             total_mae_rho_pixel += batch_mae_rho.item()
#             total_mae_theta_degree += batch_mae_theta.item()
#             count += inputs.size(0)
#
#     avg_rho_error = total_mae_rho_pixel / count
#     avg_theta_error = total_mae_theta_degree / count
#
#     print("=" * 40)
#     print(f"测试集评估结果 (共 {count} 张):")
#     print(f"平均 Rho 误差: {avg_rho_error:.2f} 像素 (在1080P图像中)")
#     print(f"平均 Theta 误差: {avg_theta_error:.2f} 度")
#     print("=" * 40)
#
#     # 绘图
#     plt.plot(loss_history)
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE Loss")
#     plt.grid(True)
#     plt.show()
#
#
# if __name__ == "__main__":
#     train_and_evaluate()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === 关键修改：导入离线数据集类 ===
# 请确保同目录下有 dataset_loader_offline.py 文件
from dataset_loader_offline import OfflineHorizonDataset
from cnn_model import HorizonDetNet


def train_and_evaluate():
    # ================= 配置参数 =================
    # 指向你生成好的缓存文件夹
    CACHE_DIR = r"Hashmani's Dataset/OfflineCache"

    # 训练超参数
    BATCH_SIZE = 16  # 离线模式速度快，可以稍微加大 Batch
    LEARNING_RATE = 1e-4
    EPOCHS = 50  # 跑 50 轮没问题，很快就能跑完

    # 网络输入尺寸
    RESIZE_H = 362
    RESIZE_W = 180

    # 评估参数 (用于反归一化看误差)
    APPROX_MAX_DIAG = 2203.0
    MAX_THETA_DEG = 180.0

    # 数据集分割点
    SPLIT_INDEX = 2473
    # ===========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 加载数据 (离线极速版) ---
    print(f"正在从缓存加载数据: {CACHE_DIR} ...")

    # 实例化离线数据集
    full_dataset = OfflineHorizonDataset(CACHE_DIR)
    total_len = len(full_dataset)
    print(f"缓存文件总数: {total_len}")

    if total_len == 0:
        raise ValueError("未找到缓存文件！请先运行 prepare_offline_data.py 生成数据。")

    # ================= 调试开关 =================
    FAST_DEBUG = False  # <--- 正式训练时设为 False，调试设为 True

    if FAST_DEBUG:
        print("\n警告：当前处于【极速调试模式】 ⚠️⚠️⚠️")
        train_indices = list(range(0, 64))
        test_indices = list(range(64, 80))
    else:
        # --- 全量模式 ---
        if total_len < SPLIT_INDEX:
            # 容错：如果缓存没生成完，就按实际数量分割
            print(f"注意：缓存数量 ({total_len}) 少于预期 ({SPLIT_INDEX})，将自动调整分割点。")
            real_split = int(total_len * 0.9)
            train_indices = list(range(0, real_split))
            test_indices = list(range(real_split, total_len))
        else:
            train_indices = list(range(0, SPLIT_INDEX))
            test_indices = list(range(SPLIT_INDEX, total_len))
    # ===========================================

    # 划分训练/测试集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"训练集样本: {len(train_dataset)}")
    print(f"测试集样本: {len(test_dataset)}")

    # DataLoader
    # 注意：现在 num_workers 可以设置为 4 或 8 了！因为不再涉及 CUDA 冲突
    # persistent_workers=True 可以加速 epoch 之间的切换
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    # --- 2. 初始化模型 ---
    model = HorizonDetNet(in_channels=3, img_h=RESIZE_H, img_w=RESIZE_W).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 学习率策略：前 15 轮快跑，后面减速精调
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # --- 3. 训练循环 ---
    loss_history = []
    print("\n开始极速训练...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 打印本轮总结
        # print(f"Epoch {epoch + 1} Done | Avg Loss: {epoch_loss:.6f} | LR: {current_lr:.1e}")

    # 保存模型
    torch.save(model.state_dict(), "horizon_cnn_offline.pth")
    print("\n模型已保存: horizon_cnn_offline.pth")

    # --- 4. 评估 ---
    print("\n正在评估测试集...")
    model.eval()

    total_mae_rho_pixel = 0.0
    total_mae_theta_degree = 0.0
    count = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for inputs, labels in test_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # 反归一化计算真实物理误差
            diff_rho_norm = torch.abs(outputs[:, 0] - labels[:, 0])
            batch_mae_rho = torch.sum(diff_rho_norm * (APPROX_MAX_DIAG))

            diff_theta_norm = torch.abs(outputs[:, 1] - labels[:, 1])
            batch_mae_theta = torch.sum(diff_theta_norm * MAX_THETA_DEG)

            total_mae_rho_pixel += batch_mae_rho.item()
            total_mae_theta_degree += batch_mae_theta.item()
            count += inputs.size(0)

    avg_rho_error = total_mae_rho_pixel / count
    avg_theta_error = total_mae_theta_degree / count

    print("=" * 40)
    print(f"最终测试结果 (共 {count} 张):")
    print(f"平均 Rho 误差: {avg_rho_error:.2f} 像素")
    print(f"平均 Theta 误差: {avg_theta_error:.2f} 度")
    print("=" * 40)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Offline Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()