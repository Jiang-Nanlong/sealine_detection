import matplotlib
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

# 导入模块
from dataset_loader_offline import OfflineHorizonDataset
from cnn_model import get_resnet34_model, HorizonResNet # 确保 cnn_model.py 正确

# ================= 配置 =================
CACHE_DIR = r"Hashmani's Dataset/FusionCache" # 刚才生成的缓存路径
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
SPLIT_INDEX = 2473
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ========================================

class HorizonPeriodicLoss(nn.Module):
    def __init__(self, rho_weight=1.0, theta_weight=2.0):
        super().__init__()
        self.rho_weight = rho_weight
        self.theta_weight = theta_weight
        self.l1 = nn.SmoothL1Loss()

    def forward(self, preds, targets):
        # preds/targets: [B, 2] (rho, theta)
        
        # 1. Rho Loss (L1)
        loss_rho = self.l1(preds[:, 0], targets[:, 0])
        
        # 2. Theta Loss (Periodic)
        # 0~1 -> 0~pi (180度)
        t_pred = preds[:, 1] * np.pi
        t_gt = targets[:, 1] * np.pi
        
        # 构造向量 (cos 2x, sin 2x) 来解决 180 度周期性问题
        # 2x 是因为直线方向周期是 180度 (pi)，放大2倍变成 360度 (2pi) 周期
        vp = torch.stack([torch.cos(2*t_pred), torch.sin(2*t_pred)], dim=1)
        vt = torch.stack([torch.cos(2*t_gt), torch.sin(2*t_gt)], dim=1)
        
        loss_theta = self.l1(vp, vt)
        
        return self.rho_weight * loss_rho + self.theta_weight * loss_theta

def train():
    # 1. 数据集
    full_ds = OfflineHorizonDataset(CACHE_DIR)
    print(f"Dataset size: {len(full_ds)}")
    
    # 检查数据形状
    sample, _ = full_ds[0]
    print(f"Input Shape: {sample.shape}") # 应该是 [4, 2240, 180]
    
    # 划分
    indices = list(range(len(full_ds)))
    # 简单的按顺序划分，或者你可以加载之前的 random split npy
    train_ds = Subset(full_ds, indices[:SPLIT_INDEX])
    test_ds = Subset(full_ds, indices[SPLIT_INDEX:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. 模型 (4通道输入)
    # 你的 cnn_model.py 中 get_resnet34_model 默认是 4 通道，或者你可以显式指定
    # model = HorizonResNet(in_channels=4).to(DEVICE)
    model = get_resnet34_model().to(DEVICE) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    criterion = HorizonPeriodicLoss()
    
    # 3. 训练循环
    best_loss = 1e9
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        run_loss = 0
        
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # 简单的 Mixup 数据增强 (可选)
            # if random.random() < 0.5: ...
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # Eval
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x)
                    val_loss += criterion(pred, y).item()
            avg_val = val_loss / len(test_loader)
            print(f"Val Loss: {avg_val:.5f}")
            
            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), "fusion_resnet_best.pth")
                
    print("Done!")

if __name__ == "__main__":
    train()