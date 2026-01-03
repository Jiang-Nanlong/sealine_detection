# import torch
# ckpt = torch.load("Epoch99.pth", map_location="cpu")
# print(type(ckpt), list(ckpt.keys())[:30] if isinstance(ckpt, dict) else "not dict")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet_model import RestorationGuidedHorizonNet
from dataset_loader import SimpleFolderDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def psnr(x, y):
    mse = F.mse_loss(x, y, reduction="mean")
    return 10 * torch.log10(1.0 / (mse + 1e-12))

# val folder：放干净图（跟你训练一样）
VAL_DIR = r"Hashmani's Dataset/clear_val"
ds = SimpleFolderDataset(VAL_DIR, img_size=384)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path="Epoch99.pth").to(DEVICE)
model.load_state_dict(torch.load("rghnet_stage_a.pth", map_location=DEVICE), strict=False)
model.eval()

psnr_baseline, psnr_model, n = 0.0, 0.0, 0
with torch.no_grad():
    for img, target in loader:
        img = img.to(DEVICE)
        target = target.to(DEVICE)

        # baseline：DCE(degraded) vs DCE(clean)
        x_dce = model._dce_enhance(img)
        t_dce = model._dce_enhance(target)

        # model：restored vs DCE(clean)
        restored, _, t_dce2 = model(img, target, enable_restoration=True, enable_segmentation=False)

        psnr_baseline += float(psnr(x_dce, t_dce))
        psnr_model += float(psnr(restored, t_dce2))
        n += 1

print("Baseline(DCE) PSNR:", psnr_baseline / n)
print("Model PSNR:", psnr_model / n)
