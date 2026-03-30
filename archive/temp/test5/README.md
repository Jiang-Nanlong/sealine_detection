# Experiment 5: Degradation Robustness / 恶劣天气下的海天线检测鲁棒性

## 实验目的

测试海天线检测模型在**海洋场景常见退化条件**下的鲁棒性，包括：

### 基础退化
- **传感器噪声** (σ = 15, 30) - 低光 ISO 提升
- **运动模糊** (kernel = 15, 25) - 船体晃动
- **低光照** (γ = 2.0, 2.5) - 黄昏/阴天
- **海雾** (30%, 50%) - 能见度降低

### 海洋特有退化 ⭐
- **降雨** (轻/中/重) - 海上降雨，雨滴+雾气
- **炫光/强反光** (轻/重) - 阳光海面反射，常见于正午
- **压缩伪影** (Q=20, 10) - 低码率视频传输
- **分辨率下降** (0.5x, 0.25x) - 远距离监控/低清设备

## 文件结构

```
test5/
├── run_experiment5.py              # 主控脚本（推荐）
├── generate_degraded_images.py     # 生成退化图像
├── make_fusion_cache_degraded.py   # 生成 FusionCache
├── evaluate_degraded.py            # 评估性能
├── summarize_degraded_results.py   # 生成汇总表格
├── visualize_degraded.py           # 可视化对比
├── README.md                       # 本文件
│
├── degraded_images/                # [生成] 退化图像
│   ├── clean/
│   ├── rain_light/
│   ├── rain_medium/
│   ├── rain_heavy/
│   ├── glare_light/
│   ├── glare_heavy/
│   ├── jpeg_q20/
│   ├── jpeg_q10/
│   ├── lowres_0.5x/
│   ├── lowres_0.25x/
│   └── ...
│
├── FusionCache_Degraded/           # [生成] 缓存文件
│   └── ...
│
├── eval_results/                   # [生成] 评估结果
│   └── degradation_results.csv
│
├── experiment5_results/            # [生成] 论文表格
│   ├── summary_table.md
│   └── summary_table.tex
│
└── visualization/                  # [生成] 可视化图片
    ├── rain_medium/
    ├── glare_heavy/
    └── ...
```

## 退化类型详解（共 17 种 + 1 基准）

| 类别 | 退化名称 | 参数 | 模拟场景 |
|------|----------|------|----------|
| 基准 | clean | - | 原始清晰图像 |
| 噪声 | gaussian_noise_15/30 | σ=15,30 | 传感器噪声 |
| 模糊 | motion_blur_15/25 | k=15,25 | 船体晃动 |
| 光照 | low_light_2.0/2.5 | γ=2.0,2.5 | 黄昏阴天 |
| 天气 | fog_0.3/0.5 | 30%,50% | 海雾 |
| **降雨** | rain_light/medium/heavy | 轻/中/重 | 海上降雨 ⭐ |
| **反光** | glare_light/heavy | 30%,60% | 阳光海面反射 ⭐ |
| **压缩** | jpeg_q20/q10 | Q=20,10 | 低码率视频 ⭐ |
| **低清** | lowres_0.5x/0.25x | 0.5x,0.25x | 远距离监控 ⭐ |

## 运行方式

### 方式一：一键运行（推荐）

在 PyCharm 中直接运行 `run_experiment5.py`：

```python
# 配置区
SKIP_GENERATE = False   # 首次运行设为 False
SKIP_CACHE = False      # 首次运行设为 False
SKIP_VIS = False        # 生成可视化
```

### 方式二：分步运行

1. 生成退化图像：运行 `generate_degraded_images.py`
2. 生成缓存：运行 `make_fusion_cache_degraded.py`
3. 评估：运行 `evaluate_degraded.py`
4. 生成表格：运行 `summarize_degraded_results.py`
5. 可视化：运行 `visualize_degraded.py`

## 论文价值

本实验直接支撑论文标题"**恶劣天气下的海天线检测**"：

1. **降雨**：验证模型在雨天场景的检测能力
2. **反光**：验证模型对海面强反射的抵抗能力
3. **低质量视频**：验证模型在实际监控系统中的适用性
4. **远距离监控**：验证模型对低分辨率输入的鲁棒性

## 注意事项

1. **首次运行**需要生成所有退化图像和缓存，耗时约 15-30 分钟
2. **磁盘空间**：退化图像约 3-4 GB，缓存约 1-2 GB
3. 如果只想测试部分退化类型，可以修改 `generate_degraded_images.py` 中的 `DEGRADATIONS` 字典
