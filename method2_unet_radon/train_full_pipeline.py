# -*- coding: utf-8 -*-
"""
train_full_pipeline.py

一站式训练脚本：
  1. Stage A: 训练复原分支 (50 epochs)
  2. Stage B: 训练分割分支 (20 epochs)
  3. Stage C1: 微调复原 (10 epochs)
  4. Stage B2: 微调分割 (5 epochs)
  5. Stage C2: 联合微调 (40 epochs)
  6. 生成 FusionCache
  7. 训练 Fusion CNN (100 epochs)

PyCharm: 直接运行此文件，在下方配置区修改参数
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ============================
# PyCharm 配置区 (在这里修改)
# ============================
SKIP_UNET = False       # True = 跳过 UNet 训练（如果已有权重）
SKIP_CACHE = False      # True = 跳过 FusionCache 生成
SKIP_CNN = False        # True = 跳过 CNN 训练

# UNet 阶段控制:
#   None = 训练全部阶段 (A → B → C1 → B2 → C2)
#   ["A", "B"] = 只训练指定阶段
#   从某阶段开始: 设置 FROM_STAGE = "B" (会训练 B, C1, B2, C2)
UNET_STAGES_OVERRIDE = None   # 例如: ["A"] 或 ["B", "C1"]
FROM_STAGE = None             # 例如: "B" 表示从 B 开始

# ============================

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# UNet 阶段配置
UNET_STAGES = ["A", "B", "C1", "B2", "C2"]

# 脚本路径
TRAIN_UNET_SCRIPT = PROJECT_ROOT / "train_unet.py"
MAKE_CACHE_SCRIPT = PROJECT_ROOT / "make_fusion_cache.py"
TRAIN_CNN_SCRIPT = PROJECT_ROOT / "train_fusion_cnn.py"

# 日志目录
LOG_DIR = PROJECT_ROOT / "train_logs"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_script(script_path: Path, description: str, env_override: dict = None) -> bool:
    """运行 Python 脚本并返回是否成功"""
    print("\n" + "=" * 70)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"[Script] {script_path}")
    print("=" * 70 + "\n")
    
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n[✓] {description} 完成 (耗时 {elapsed/60:.1f} 分钟)")
            return True
        else:
            print(f"\n[✗] {description} 失败 (return code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n[✗] {description} 异常: {e}")
        return False


def modify_stage_in_script(stage: str):
    """修改 train_unet.py 中的 STAGE 变量"""
    script_path = TRAIN_UNET_SCRIPT
    
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找并替换 STAGE = "X"
    import re
    new_content = re.sub(
        r'^STAGE\s*=\s*["\'][A-Z0-9]+["\']',
        f'STAGE = "{stage}"',
        content,
        flags=re.MULTILINE
    )
    
    if new_content != content:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"[Config] STAGE 已设置为 '{stage}'")
    else:
        print(f"[Config] STAGE 已经是 '{stage}'")


def train_unet_all_stages(stages: list = None):
    """训练 UNet 所有阶段"""
    if stages is None:
        stages = UNET_STAGES
    
    print("\n" + "#" * 70)
    print("#  UNet 分阶段训练")
    print("#  Stages:", " → ".join(stages))
    print("#" * 70)
    
    for stage in stages:
        modify_stage_in_script(stage)
        
        stage_desc = {
            "A": "Stage A: 训练复原分支 (Restoration)",
            "B": "Stage B: 训练分割分支 (Segmentation)",
            "C1": "Stage C1: 微调复原分支",
            "B2": "Stage B2: 微调分割分支",
            "C2": "Stage C2: 联合微调 (Joint Fine-tuning)",
        }
        
        success = run_script(
            TRAIN_UNET_SCRIPT,
            stage_desc.get(stage, f"Stage {stage}")
        )
        
        if not success:
            print(f"\n[Error] Stage {stage} 训练失败，终止流程")
            return False
    
    print("\n[✓] UNet 所有阶段训练完成！")
    return True


def generate_fusion_cache():
    """生成 FusionCache"""
    return run_script(
        MAKE_CACHE_SCRIPT,
        "生成 FusionCache (Gradient-Radon 特征提取)"
    )


def train_fusion_cnn():
    """训练 Fusion CNN"""
    return run_script(
        TRAIN_CNN_SCRIPT,
        "训练 Fusion CNN (回归 ρ, θ)"
    )


def print_summary(results: dict, total_time: float):
    """打印训练总结"""
    print("\n")
    print("=" * 70)
    print("                      训 练 完 成 总 结")
    print("=" * 70)
    
    print(f"\n总耗时: {total_time/3600:.2f} 小时 ({total_time/60:.1f} 分钟)")
    
    print("\n各阶段状态:")
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    print("\n生成的权重文件:")
    weights_dir = PROJECT_ROOT / "weights"
    if weights_dir.exists():
        for f in sorted(weights_dir.glob("*.pth")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")
    
    print("\n下一步:")
    print("  1. 运行实验5测试退化鲁棒性:")
    print("     python test5/run_experiment5.py")
    print("  2. 或运行完整评估:")
    print("     python evaluate_full_pipeline.py")
    
    print("=" * 70)


def main():
    # =============================================
    # 使用 PyCharm 配置区的变量（忽略命令行参数）
    # =============================================
    skip_unet = SKIP_UNET
    skip_cache = SKIP_CACHE
    skip_cnn = SKIP_CNN
    
    # 确定要训练的 UNet 阶段
    unet_stages = None
    if UNET_STAGES_OVERRIDE:
        unet_stages = [s.strip().upper() for s in UNET_STAGES_OVERRIDE]
        print(f"\n[Config] 只训练指定阶段: {unet_stages}")
    elif FROM_STAGE:
        from_stage = FROM_STAGE.strip().upper()
        if from_stage in UNET_STAGES:
            idx = UNET_STAGES.index(from_stage)
            unet_stages = UNET_STAGES[idx:]
            print(f"\n[Config] 从 {from_stage} 开始训练: {unet_stages}")
        else:
            print(f"[Error] 无效的阶段: {from_stage}")
            return
    
    print("=" * 70)
    print("       一站式训练脚本 (UNet + FusionCache + CNN)")
    print("=" * 70)
    print(f"\n启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {PROJECT_ROOT}")
    print(f"\n[Config]")
    print(f"  SKIP_UNET  = {skip_unet}")
    print(f"  SKIP_CACHE = {skip_cache}")
    print(f"  SKIP_CNN   = {skip_cnn}")
    if unet_stages:
        print(f"  UNet 阶段  = {unet_stages}")
    else:
        print(f"  UNet 阶段  = 全部 ({' → '.join(UNET_STAGES)})")
    
    start_time = time.time()
    results = {}
    
    # Step 1: Train UNet
    if not skip_unet:
        success = train_unet_all_stages(unet_stages)
        results["UNet 训练"] = success
        if not success and not UNET_STAGES_OVERRIDE:  # 如果是全量训练且失败，则终止
            print("\n[Error] UNet 训练失败，终止流程")
            print_summary(results, time.time() - start_time)
            return
    else:
        print("\n[Skip] 跳过 UNet 训练")
        results["UNet 训练"] = "跳过"
    
    # Step 2: Generate FusionCache
    if not skip_cache:
        success = generate_fusion_cache()
        results["FusionCache 生成"] = success
        if not success:
            print("\n[Error] FusionCache 生成失败，终止流程")
            print_summary(results, time.time() - start_time)
            return
    else:
        print("\n[Skip] 跳过 FusionCache 生成")
        results["FusionCache 生成"] = "跳过"
    
    # Step 3: Train Fusion CNN
    if not skip_cnn:
        success = train_fusion_cnn()
        results["Fusion CNN 训练"] = success
    else:
        print("\n[Skip] 跳过 Fusion CNN 训练")
        results["Fusion CNN 训练"] = "跳过"
    
    # Summary
    total_time = time.time() - start_time
    print_summary(results, total_time)


if __name__ == "__main__":
    main()
