#!/bin/bash
# 海天线检测系统启动脚本
cd "$(dirname "$0")"
export DISPLAY=${DISPLAY:-:0}
export PATH=$HOME/.local/bin:$PATH
python3 app/main.py "$@"
