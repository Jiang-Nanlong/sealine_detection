#!/usr/bin/env python3
"""
export_trt.py — 将模型导出为 TensorRT 引擎

用法：
  cd sealine_detection
  python3 app_fast/export_trt.py

会生成：
  weights/unet_seg_288x512.engine     (Method 2 UNet 分割)
  weights/unet_seg_288x512.onnx       (中间 ONNX)
"""

import sys
import os
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "method2_unet_radon"))
sys.path.insert(0, str(_ROOT / "method1_linea_entropy"))
sys.path.insert(0, str(_ROOT / "method1_linea_entropy" / "LINEA"))

WEIGHTS_DIR = _ROOT / "weights"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


# ======================================================================
#  Step 1: Patch StripPooling for ONNX compatibility
# ======================================================================

def _strip_pooling_forward_onnx(self, x):
    """StripPooling.forward 的 ONNX 兼容版本。
    
    替换 F.adaptive_avg_pool2d(y, (h,1)) → y.mean(dim=3, keepdim=True)
    替换 F.adaptive_avg_pool2d(y, (1,w)) → y.mean(dim=2, keepdim=True)
    数学等价，但避免了 ONNX 不支持的动态 adaptive pooling。
    """
    b, c, h, w = x.shape
    y = F.relu(self.bn1(self.reduce(x)), inplace=False)  # inplace=False for trace

    # 水平条带池化：沿宽度方向平均 → (b, c, h, 1)
    y_h = y.mean(dim=3, keepdim=True)
    y_h = self.conv_h(y_h)
    y_h = F.interpolate(y_h, size=(h, w), mode="bilinear", align_corners=False)

    # 垂直条带池化：沿高度方向平均 → (b, c, 1, w)
    y_w = y.mean(dim=2, keepdim=True)
    y_w = self.conv_w(y_w)
    y_w = F.interpolate(y_w, size=(h, w), mode="bilinear", align_corners=False)

    y = y_h + y_w
    y = F.relu(self.bn2(y), inplace=False)
    y = self.bn3(self.expand(y))
    return F.relu(x + y, inplace=False)


class UNetSegWrapper(nn.Module):
    """UNet 分割模型的 ONNX 导出包装器。只输出 seg_logits。"""
    
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    
    def forward(self, x):
        _, seg_logits, _ = self.unet(
            x, target=None,
            enable_restoration=False,
            enable_segmentation=True
        )
        return seg_logits


def export_unet_onnx(res_h=288, res_w=512):
    """导出 UNet 分割模型为 ONNX。"""
    from unet_model import RestorationGuidedHorizonNet, StripPooling
    
    # Monkey-patch StripPooling for ONNX compatibility
    StripPooling.forward = _strip_pooling_forward_onnx
    
    model = RestorationGuidedHorizonNet(
        num_classes=2,
        dce_weights_path=str(WEIGHTS_DIR / "Epoch99.pth")
    )
    model.dce_net = None  # 移除 DCE 子网络
    
    unet_ckpt = WEIGHTS_DIR / "rghnet_best_c2.pth"
    if unet_ckpt.exists():
        state = torch.load(str(unet_ckpt), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
    
    model.eval()
    wrapper = UNetSegWrapper(model)
    wrapper.eval()
    
    onnx_path = WEIGHTS_DIR / f"unet_seg_{res_h}x{res_w}.onnx"
    dummy = torch.randn(1, 3, res_h, res_w)
    
    print(f"[Export] UNet seg → ONNX ({res_h}x{res_w})...")
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["seg_logits"],
        do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"[Export] ONNX saved: {onnx_path} ({size_mb:.1f} MB)")
    return str(onnx_path)


def build_trt_engine(onnx_path, engine_path=None, fp16=True):
    """使用 trtexec 将 ONNX 转换为 TensorRT 引擎。"""
    if engine_path is None:
        engine_path = onnx_path.replace(".onnx", ".engine")
    
    if not os.path.isfile(TRTEXEC):
        print(f"[TRT] trtexec not found at {TRTEXEC}")
        return None
    
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=1024",  # 1GB workspace
    ]
    if fp16:
        cmd.append("--fp16")
    
    print(f"[TRT] Building engine (this may take several minutes)...")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"[TRT] Build FAILED:")
        # Print last 20 lines of stderr for diagnostics
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"  {line}")
        return None
    
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"[TRT] Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


def main():
    print("=" * 60)
    print("  TensorRT Engine Export")
    print("=" * 60)
    
    # --- UNet Segmentation ---
    print("\n--- UNet Segmentation Model ---")
    onnx_path = export_unet_onnx(res_h=288, res_w=512)
    engine_path = build_trt_engine(onnx_path, fp16=True)
    
    if engine_path:
        print(f"\n✓ TensorRT engine ready: {engine_path}")
    else:
        print("\n✗ TensorRT conversion failed. Will fall back to PyTorch FP16.")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
