#!/usr/bin/env python3
"""Quick test: which weight files are compatible with the model architectures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "method2_unet_radon"))

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DCE_WEIGHTS = "weights/Epoch99.pth"

# All weight files to test
UNET_WEIGHTS = [
    "weights/method2_group1/rghnet_best_c2.pth",
    "weights/method2_group2/rghnet_best_c2.pth",
]

CNN_WEIGHTS = [
    "weights/method2_group1/best_fusion_cnn_1024x576.pth",
    "weights/method2_group1/best_fusion_cnn_1024x576_2.pth",
    "weights/method2_group1/best_fusion_cnn_1024x576_3.pth",
    "weights/method2_group2/best_fusion_cnn_1024x576.pth",
    "weights/method2_group2/best_fusion_cnn_1024x576_1.pth",
]


def test_unet(path):
    try:
        from unet_model import RestorationGuidedHorizonNet
        model = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=DCE_WEIGHTS, require_dce=True)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.eval().to(DEVICE)
        # Quick forward test with dummy input
        with torch.no_grad():
            dummy = torch.randn(1, 3, 576, 1024).to(DEVICE)
            out = model(dummy)
        return True, f"OK (output types: {[type(o).__name__ for o in out]})"
    except Exception as e:
        return False, str(e)[:200]


def test_cnn(path):
    try:
        from cnn_model import HorizonResNet
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        in_ch = ckpt["conv1.weight"].shape[1] if "conv1.weight" in ckpt else 7
        model = HorizonResNet(in_channels=in_ch)
        model.load_state_dict(ckpt, strict=True)
        model.eval().to(DEVICE)
        # Quick forward test with dummy input
        with torch.no_grad():
            dummy = torch.randn(1, in_ch, 2240, 180).to(DEVICE)
            out = model(dummy)
        return True, f"OK (in_ch={in_ch}, output shape: {out[0].shape if isinstance(out, tuple) else out.shape})"
    except Exception as e:
        return False, str(e)[:200]


def test_combo(unet_path, cnn_path):
    """Test a UNet + CNN combination by loading both and checking they can coexist."""
    unet_ok, unet_msg = test_unet(unet_path)
    cnn_ok, cnn_msg = test_cnn(cnn_path)
    return unet_ok and cnn_ok, f"UNet: {unet_msg} | CNN: {cnn_msg}"


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("Testing UNet (RestorationGuidedHorizonNet) weights")
    print("=" * 70)
    for p in UNET_WEIGHTS:
        ok, msg = test_unet(p)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {p}\n         {msg}")

    print()
    print("=" * 70)
    print("Testing Fusion-CNN (HorizonResNet) weights")
    print("=" * 70)
    for p in CNN_WEIGHTS:
        ok, msg = test_cnn(p)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {p}\n         {msg}")

    print()
    print("=" * 70)
    print("Testing UNet + CNN combinations")
    print("=" * 70)
    for up in UNET_WEIGHTS:
        for cp in CNN_WEIGHTS:
            ok, msg = test_combo(up, cp)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {os.path.basename(up)} + {os.path.basename(cp)}")
            if not ok:
                print(f"         {msg}")
    print()
    print("Done.")
