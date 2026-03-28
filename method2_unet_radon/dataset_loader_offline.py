import torch
from torch.utils.data import Dataset
import numpy as np
import os


class OfflineHorizonDataset(Dataset):
    def __init__(self, cache_dir, total_len=2672):
        self.cache_dir = cache_dir
        # 这里的 total_len 最好和实际生成的 .npy 数量一致
        # 你可以写代码自动检测，也可以手动填
        self.files = [f for f in os.listdir(cache_dir) if f.endswith('.npy')]
        self.total_len = len(self.files)
        print(f"离线数据集加载成功，共找到 {self.total_len} 个缓存文件")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 直接读取 .npy
        path = os.path.join(self.cache_dir, f"{idx}.npy")

        # 容错：如果某个索引没生成成功
        if not os.path.exists(path):
            # 返回全0，避免报错
            return torch.zeros((3, 362, 180)), torch.zeros(2)

        # 加载
        data = np.load(path, allow_pickle=True).item()

        # 转 Tensor
        input_tensor = torch.from_numpy(data['input']).float()
        label_tensor = torch.from_numpy(data['label']).float()

        return input_tensor, label_tensor