#!/usr/bin/env python3
"""
benchmark_fast.py — 测试优化版推理引擎在 Jetson 上的速度

用法：
  cd sealine_detection
  python3 app_fast/benchmark_fast.py
"""

import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app_fast"))
sys.path.insert(0, str(_ROOT / "method1_linea_entropy"))
sys.path.insert(0, str(_ROOT / "method1_linea_entropy" / "LINEA"))
sys.path.insert(0, str(_ROOT / "method2_unet_radon"))

WEIGHTS_DIR = _ROOT / "weights"
SAMPLE_IMG = str(_ROOT / "sample_images" / "DSC_0042_9.JPG")

LINEA_CONFIG = str(_ROOT / "method1_linea_entropy" / "configs" / "linea_entropy_b_enhanced_musid.py")
LINEA_WEIGHTS = str(WEIGHTS_DIR / "linea_entropy_b_enhanced_best.pth")
UNET_WEIGHTS = str(WEIGHTS_DIR / "rghnet_best_c2.pth")
DCE_WEIGHTS = str(WEIGHTS_DIR / "Epoch99.pth")


def benchmark(name, detector, img, warmup=5, runs=30):
    """基准测试一个检测器。"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    detector.load()

    # Warmup
    for _ in range(warmup):
        detector.detect(img)

    # Benchmark
    times = []
    for i in range(runs):
        t0 = time.time()
        result_img, info = detector.detect(img)
        t1 = time.time()
        times.append(t1 - t0)

    times = np.array(times)
    avg_ms = times.mean() * 1000
    p50_ms = np.percentile(times, 50) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    fps = 1000.0 / avg_ms

    print(f"  Average : {avg_ms:.1f} ms  ({fps:.1f} FPS)")
    print(f"  Median  : {p50_ms:.1f} ms  ({1000/p50_ms:.1f} FPS)")
    print(f"  P95     : {p95_ms:.1f} ms  ({1000/p95_ms:.1f} FPS)")
    print(f"  Detected: {info.get('detected', False)}")

    # 详细分解
    for k, v in info.items():
        if k.endswith("_ms"):
            print(f"    {k}: {v:.1f} ms")

    detector.unload()
    return avg_ms, fps


def main():
    img = cv2.imread(SAMPLE_IMG)
    if img is None:
        print(f"Cannot read {SAMPLE_IMG}")
        sys.exit(1)
    print(f"Test image: {img.shape[1]}x{img.shape[0]}")

    results = {}

    # --- Method 2: UNet TRT FP16 ---
    from inference_engine_fast import UNetSegDetectorTRT
    trt_engine = str(_ROOT / "weights" / "unet_seg_288x512.engine")
    if os.path.isfile(trt_engine):
        det = UNetSegDetectorTRT(trt_engine)
        ms, fps = benchmark("Method 2: UNet-Seg TRT (FP16, 512×288)", det, img)
        results["UNet-Seg TRT"] = (ms, fps)
    else:
        print(f"\n  [SKIP] TRT engine not found: {trt_engine}")
        print(f"         Run: python3 app_fast/export_trt.py")

    # --- Method 2: UNet PyTorch FP16 (fallback) ---
    from inference_engine_fast import UNetSegDetectorFast
    det = UNetSegDetectorFast(UNET_WEIGHTS, DCE_WEIGHTS)
    ms, fps = benchmark("Method 2: UNet-Seg PyTorch (FP16, 512×288, no-DCE)", det, img)
    results["UNet-Seg PyTorch"] = (ms, fps)

    # --- Method 1: LINEA Fast ---
    from inference_engine_fast import LINEADetectorFast
    det = LINEADetectorFast(LINEA_CONFIG, LINEA_WEIGHTS, img_size=640)
    ms, fps = benchmark("Method 1: LINEA Fast (FP16, 640×640)", det, img)
    results["LINEA Fast 640"] = (ms, fps)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    target_ms = 33.3
    for name, (ms, fps) in results.items():
        status = "✓ 30fps" if ms <= target_ms else f"✗ {fps:.0f}fps"
        print(f"  {name:30s}: {ms:6.1f} ms  {fps:5.1f} FPS  [{status}]")
    print()


if __name__ == "__main__":
    main()
