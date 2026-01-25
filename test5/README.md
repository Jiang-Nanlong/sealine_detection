# Experiment 5: Degradation Robustness / 图像退化鲁棒性测试

## 实验目的

测试海天线检测模型在各种图像退化条件下的鲁棒性，包括：
- 高斯噪声 (σ = 10, 25, 50)
- 运动模糊 (kernel = 15, 25)
- 高斯模糊 (σ = 2, 5)
- 低光照 (γ = 2.0, 3.0)
- 雾霾 (intensity = 0.3, 0.5)
- 椒盐噪声 (ratio = 0.01, 0.05)

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
│   ├── gaussian_noise_10/
│   ├── gaussian_noise_25/
│   └── ...
│
├── FusionCache_Degraded/           # [生成] 缓存文件
│   ├── clean/
│   ├── gaussian_noise_25/
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
    ├── gaussian_noise_25/
    ├── motion_blur_25/
    └── ...
```

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

## 输出结果

### 评估指标

| 指标 | 说明 |
|------|------|
| ρ Mean | 位置误差均值 (像素) |
| ρ ≤ 10px | 位置误差 ≤10 像素的比例 |
| θ Mean | 角度误差均值 (度) |
| θ ≤ 2° | 角度误差 ≤2° 的比例 |

### 预期结果

- **角度预测**：对大多数退化类型保持鲁棒（θ ≤ 2° > 90%）
- **位置预测**：
  - 轻度退化（噪声 σ=10, 模糊 σ=2）：影响较小
  - 中度退化（噪声 σ=25, 低光照）：有一定下降
  - 重度退化（噪声 σ=50, 运动模糊）：下降明显

### 论文使用

1. 将 `summary_table.tex` 中的表格复制到论文
2. 选择典型可视化图片放入论文
3. 讨论模型在不同退化条件下的表现

## 注意事项

1. **首次运行**需要生成所有退化图像和缓存，耗时较长（约 10-30 分钟）
2. **磁盘空间**：退化图像约 2-3 GB，缓存约 1-2 GB
3. 如果只想测试部分退化类型，可以修改 `generate_degraded_images.py` 中的 `DEGRADATIONS` 字典
