import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet_model import RestorationGuidedHorizonNet
from gradient_radon import TextureSuppressedMuSCoWERT

# 配置
ckpt_path = r"weights/rghnet_best_c2.pth"  # 你的最新权重
img_path = r"Hashmani's Dataset/MU-SID/DSC_0622_2.JPG"  # 找一张典型的图

# 1. 加载模型
device = "cuda"
model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path="weights/Epoch99.pth").to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
model.eval()

# 2. 读取并复原图像
bgr = cv2.imread(img_path)
h, w = bgr.shape[:2]
# Resize 到 U-Net 输入大小
inp_bgr = cv2.resize(bgr, (1024, 576))
inp_rgb = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)
inp_tensor = torch.from_numpy(inp_rgb / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

with torch.no_grad():
    restored, _, _ = model(inp_tensor, None, True, True)

# 转回 numpy (这是送给 Traditional Method 的输入)
restored_np = (restored.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

# 3. 运行传统特征提取 (使用修改参数后的类)
# 注意：这里你需要去修改 gradient_radon.py 里的参数
detector = TextureSuppressedMuSCoWERT(scales=[1, 2, 3], full_scan=True)
# 这里调用内部函数方便看中间结果
_, _, debug_info, _ = detector.detect(restored_bgr)

# 4. 可视化对比
plt.figure(figsize=(15, 10))

# 显示复原图
plt.subplot(3, 1, 1)
plt.imshow(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))
plt.title("Restored Image (Input to Radon)")
plt.axis('off')

# 显示三个尺度的梯度图 (如果这些图是黑的，说明传统参数太严了)
for i, scale in enumerate([1, 2, 3]):
    if scale in debug_info:
        plt.subplot(3, 3, 4 + i)
        plt.imshow(debug_info[scale]['map'], cmap='gray')
        plt.title(f"Scale {scale} Gradient Map")
        plt.axis('off')

        plt.subplot(3, 3, 7 + i)
        plt.imshow(debug_info[scale]['blurred'], cmap='gray')
        plt.title(f"Scale {scale} Blurred")
        plt.axis('off')

plt.tight_layout()
plt.show()