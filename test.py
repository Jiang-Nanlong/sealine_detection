import torch
ckpt = torch.load("Epoch99.pth", map_location="cpu")
print(type(ckpt), list(ckpt.keys())[:30] if isinstance(ckpt, dict) else "not dict")
