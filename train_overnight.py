# -*- coding: utf-8 -*-
"""
train_overnight.py

一站式夜间训练脚本（简化版）
  1. 训练 UNet 所有阶段 (A → B → C1 → B2 → C2)
  2. 生成 FusionCache
  3. 训练 Fusion CNN

新权重保存到: weights_new/ (不影响现有权重)
新缓存保存到: FusionCache_new/

PyCharm: 直接运行此文件
预计时间: 6-9 小时
"""

import os
import sys
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime

# ----------------------------
# 路径配置
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# 新权重/缓存目录（不影响现有的）
NEW_WEIGHTS_DIR = PROJECT_ROOT / "weights_new"
NEW_CACHE_DIR = PROJECT_ROOT / "Hashmani's Dataset" / "FusionCache_new_1024x576"

# 脚本路径
TRAIN_UNET_SCRIPT = PROJECT_ROOT / "train_unet.py"
MAKE_CACHE_SCRIPT = PROJECT_ROOT / "make_fusion_cache.py"
TRAIN_CNN_SCRIPT = PROJECT_ROOT / "train_fusion_cnn.py"

# UNet 阶段
UNET_STAGES = ["A", "B", "C1", "B2", "C2"]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def modify_script_variable(script_path: Path, var_name: str, new_value: str):
    """修改脚本中的变量值"""
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 匹配 VAR_NAME = "xxx" 或 VAR_NAME = r"xxx" 或 VAR_NAME = 'xxx'
    pattern = rf'^({var_name}\s*=\s*)["\']?r?["\']?.*["\']?'
    replacement = f'{var_name} = r"{new_value}"'
    
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def modify_unet_stage(stage: str):
    """修改 train_unet.py 中的 STAGE 变量"""
    with open(TRAIN_UNET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    
    new_content = re.sub(
        r'^STAGE\s*=\s*["\'][A-Z0-9]+["\']',
        f'STAGE = "{stage}"',
        content,
        flags=re.MULTILINE
    )
    
    with open(TRAIN_UNET_SCRIPT, "w", encoding="utf-8") as f:
        f.write(new_content)


def backup_and_modify_paths():
    """备份原始路径配置并修改为新路径"""
    log("配置新的权重和缓存路径...")
    
    ensure_dir(NEW_WEIGHTS_DIR)
    ensure_dir(NEW_CACHE_DIR)
    
    # 修改 train_unet.py 中的权重保存路径
    # 我们需要修改保存路径前缀
    with open(TRAIN_UNET_SCRIPT, "r", encoding="utf-8") as f:
        unet_content = f.read()
    
    # 替换 weights/ 为 weights_new/
    unet_content = unet_content.replace(
        '"weights/rghnet_best_',
        '"weights_new/rghnet_best_'
    ).replace(
        '"weights/rghnet_last_',
        '"weights_new/rghnet_last_'
    ).replace(
        "'weights/rghnet_best_",
        "'weights_new/rghnet_best_"
    ).replace(
        "'weights/rghnet_last_",
        "'weights_new/rghnet_last_"
    )
    
    with open(TRAIN_UNET_SCRIPT, "w", encoding="utf-8") as f:
        f.write(unet_content)
    
    # 修改 make_fusion_cache.py
    with open(MAKE_CACHE_SCRIPT, "r", encoding="utf-8") as f:
        cache_content = f.read()
    
    cache_content = re.sub(
        r'RGHNET_CKPT\s*=\s*r?["\']weights/rghnet_best_c2\.pth["\']',
        'RGHNET_CKPT = r"weights_new/rghnet_best_c2.pth"',
        cache_content
    )
    # 使用 re.escape 避免路径中的反斜杠问题
    new_cache_path_escaped = str(NEW_CACHE_DIR).replace("\\", "/")
    cache_content = re.sub(
        r'SAVE_ROOT\s*=\s*r?["\'].*FusionCache[^"\']*["\']',
        f'SAVE_ROOT = r"{new_cache_path_escaped}"',
        cache_content
    )
    
    with open(MAKE_CACHE_SCRIPT, "w", encoding="utf-8") as f:
        f.write(cache_content)
    
    # 修改 train_fusion_cnn.py
    with open(TRAIN_CNN_SCRIPT, "r", encoding="utf-8") as f:
        cnn_content = f.read()
    
    cnn_content = re.sub(
        r'CACHE_ROOT\s*=\s*r?["\'].*FusionCache[^"\']*["\']',
        f'CACHE_ROOT = r"{new_cache_path_escaped}"',
        cnn_content
    )
    cnn_content = re.sub(
        r'BEST_PATH\s*=\s*r?["\']weights/best_fusion_cnn[^"\']*["\']',
        'BEST_PATH = "weights_new/best_fusion_cnn_1024x576.pth"',
        cnn_content
    )
    
    with open(TRAIN_CNN_SCRIPT, "w", encoding="utf-8") as f:
        f.write(cnn_content)
    
    log(f"  权重保存到: {NEW_WEIGHTS_DIR}")
    log(f"  缓存保存到: {NEW_CACHE_DIR}")


def run_script(script_path: Path, description: str) -> bool:
    """运行脚本"""
    log(f"开始: {description}")
    log(f"脚本: {script_path}")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    
    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}min"
    
    if result.returncode == 0:
        log(f"完成: {description} (耗时 {elapsed_str})")
        return True
    else:
        log(f"失败: {description} (返回码 {result.returncode})")
        return False


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("       一站式夜间训练脚本")
    print("=" * 60)
    log(f"工作目录: {PROJECT_ROOT}")
    log(f"新权重目录: {NEW_WEIGHTS_DIR}")
    log(f"新缓存目录: {NEW_CACHE_DIR}")
    print("=" * 60)
    
    # 1. 修改路径配置
    backup_and_modify_paths()
    
    # 2. 训练 UNet 所有阶段
    print("\n" + "=" * 60)
    print("  阶段 1: 训练 UNet")
    print("=" * 60)
    
    for stage in UNET_STAGES:
        modify_unet_stage(stage)
        
        stage_names = {
            "A": "复原分支 (50 epochs)",
            "B": "分割分支 (20 epochs)",
            "C1": "微调复原 (10 epochs)",
            "B2": "微调分割 (5 epochs)",
            "C2": "联合微调 (40 epochs)",
        }
        
        success = run_script(TRAIN_UNET_SCRIPT, f"Stage {stage}: {stage_names[stage]}")
        
        if not success:
            log(f"Stage {stage} 失败，终止训练")
            return
    
    # 3. 生成 FusionCache
    print("\n" + "=" * 60)
    print("  阶段 2: 生成 FusionCache")
    print("=" * 60)
    
    success = run_script(MAKE_CACHE_SCRIPT, "生成 FusionCache")
    if not success:
        log("FusionCache 生成失败，终止训练")
        return
    
    # 4. 训练 Fusion CNN
    print("\n" + "=" * 60)
    print("  阶段 3: 训练 Fusion CNN")
    print("=" * 60)
    
    success = run_script(TRAIN_CNN_SCRIPT, "训练 Fusion CNN (100 epochs)")
    if not success:
        log("Fusion CNN 训练失败")
        return
    
    # 完成
    total_time = time.time() - start_time
    total_hours = total_time / 3600
    
    print("\n" + "=" * 60)
    print("       训练完成!")
    print("=" * 60)
    log(f"总耗时: {total_hours:.2f} 小时")
    print()
    print("生成的权重文件:")
    if NEW_WEIGHTS_DIR.exists():
        for f in sorted(NEW_WEIGHTS_DIR.glob("*.pth")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
    print()
    print("下一步:")
    print("  1. 运行实验5测试退化鲁棒性:")
    print("     python test5/run_experiment5.py")
    print("  2. 或运行完整评估:")
    print("     python evaluate_full_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
