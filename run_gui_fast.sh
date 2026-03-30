#!/bin/bash
# 启动 Jetson 优化版海天线检测 GUI (TensorRT 加速)
cd "$(dirname "$0")"

# 设置 Jetson 最高性能模式 (需要 sudo 密码)
echo "Setting Jetson to max performance mode..."
echo "jsrcdj" | sudo -S nvpmodel -m 0 2>/dev/null
echo "jsrcdj" | sudo -S jetson_clocks 2>/dev/null

# 检查 TRT 引擎是否存在
if [ ! -f "weights/unet_seg_288x512.engine" ]; then
    echo "TensorRT engine not found. Building (may take ~10 minutes)..."
    python3 app_fast/export_trt.py
fi

export DISPLAY=${DISPLAY:-:0}

echo "Starting optimized sealine detection GUI (TensorRT)..."
python3 app_fast/main.py
