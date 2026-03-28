import torch

WEIGHTS = r"F:\code_manager\Menglong Cao\sealine_detection\horizon_resnet34_best_1.pth"

ckpt = torch.load(WEIGHTS, map_location="cpu")
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
if isinstance(ckpt, dict) and "model" in ckpt:
    ckpt = ckpt["model"]

# 去掉 DataParallel 的 module.
if any(k.startswith("module.") for k in ckpt.keys()):
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

w = ckpt["conv1.weight"]   # shape: [64, in_channels, kH, kW]
print("conv1.weight.shape =", tuple(w.shape))
print("in_channels =", w.shape[1])
