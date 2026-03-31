# 第五章 海天线检测系统嵌入式部署与优化（大纲）

预估总字数：7000~8500字（含表格，不含图）
预估图片数量：10~12 张
预估表格数量：4~5 张


## 5.1 引言（约 400~500字）

阐述将海天线检测算法从服务器环境迁移到嵌入式平台的实际需求，包括船舶自主导航和海上监控等应用场景对实时性和低功耗的要求。分析嵌入式环境面临的核心挑战，如算力受限、显存紧张、推理延迟敏感等，引出本章在模型轻量化、推理加速和流水线优化三方面的工作思路。

📷 **图 5-1** 本章工作总览流程图（从服务器训练模型 → 轻量化 → TensorRT 编译 → seg-only 流水线 → GUI 部署）


## 5.2 嵌入式平台与系统架构（约 900~1100字）

### 5.2.1 硬件平台

介绍 Jetson Xavier NX 开发板的硬件规格（6核 ARMv8.2 CPU、384 CUDA 核心 Volta GPU、6.7GB LPDDR4x 内存）以及与服务器 GPU（RTX 4090 D，14592 CUDA 核心 Ada Lovelace，24GB GDDR6X）在算力和带宽上的差距。Jetson Xavier NX 支持9种功耗模式，覆盖10W/15W/20W三档TDP，每档可选2核/4核/6核CPU配置（如 MODE 10W 2CORE、MODE 15W 4CORE、MODE 20W 6CORE等），本文实验统一使用最高性能的 MODE 20W 6CORE 模式。给出各功耗模式下的CPU/GPU频率与理论算力数据，说明选择该平台的原因。

📷 **图 5-2** Jetson Xavier NX 开发板实物照片（正面+接口标注）

**表 5-1** Jetson Xavier NX 与服务器 GPU（RTX 4090 D）参数对比（CUDA 核心数、显存、算力 TOPS、功耗、价格）

### 5.2.2 软件栈

描述部署环境的软件配置，包括 JetPack 4.6.1、CUDA 10.2/11.4、cuDNN 8.2、TensorRT 8.5.2.2、PyTorch 2.1.0 等关键组件的版本信息。

### 5.2.3 系统整体架构与 GUI 设计

介绍基于 PyQt5 构建的海天线检测系统界面设计，包括四种检测方式的下拉选项切换（LINEA-L、LINEA-N、UNet seg-only TRT、UNet 完整流水线），以及对图片、视频文件、摄像头实时流三种输入模式的支持。

📷 **图 5-3** 系统软件架构图（分层：GUI 层 → 推理引擎层 → 模型层 → TensorRT/PyTorch 后端）
📷 **图 5-4** GUI 界面截图（展示图片检测模式，含检测结果叠加显示、方法切换下拉框、状态栏信息）


## 5.3 模型轻量化：LINEA-N（约 800~1000字）

从 LINEA-L（HGNetv2-B4 backbone，hidden_dim=256，6层 decoder，27.3M 参数）出发，说明面向嵌入式部署的轻量化设计。具体措施包括将 backbone 替换为 HGNetv2-B0、将 Transformer decoder 层数从6层缩减至3层、隐层维度从256降至128，以及将多尺度熵注入（3通道）简化为单尺度（1通道）。通过这些修改得到 LINEA-N（约4.07M 参数，相比 LINEA-L 减少约85%）。分析每项修改对模型容量和计算量的影响，以及为何这些精简在海天线检测这一相对简单的视觉任务上是可接受的。

📷 **图 5-5** LINEA-L 与 LINEA-N 结构对比图（并排展示两个模型的 backbone → encoder → decoder 结构，标注各层维度差异，箭头标注轻量化修改点）

**表 5-2** LINEA-L 与 LINEA-N 结构参数对照表（backbone、hidden_dim、dec_layers、entropy_mode、参数量、模型文件大小）


## 5.4 TensorRT 推理加速（约 800~1000字）

### 5.4.1 ONNX 导出与 TensorRT Engine 编译

说明将 PyTorch UNet 模型导出为 ONNX 格式的流程（含 StripPooling 算子的 monkey-patch 处理），以及在 Jetson 上使用 trtexec 编译 TensorRT engine 的步骤。重点描述 FP16 量化的启用方式和固定输入尺寸 (1,3,288,512) 的设置。

📷 **图 5-6** TensorRT 部署流程图（PyTorch .pth → ONNX export → trtexec 编译 → .engine → Runtime 推理）

### 5.4.2 加速效果

以 UNet 分割模型为例，对比 PyTorch FP32、PyTorch FP16 和 TensorRT FP16 三种模式在 Jetson Xavier NX 上的单帧推理时间。结果：TensorRT FP16 约 16ms（62 FPS），PyTorch FP16 约 51ms（20 FPS），PyTorch FP32 约 156ms（6.4 FPS）。TensorRT 相比 PyTorch FP16 加速比约 3.2 倍，相比 FP32 加速比约 9.8 倍。分析 TensorRT 在算子融合、内存优化和 kernel auto-tuning 方面带来的收益。

📷 **图 5-7** UNet 推理时间柱状图（三种模式对比：FP32 / FP16 / TRT FP16，含误差棒）


## 5.5 推理流水线优化（约 800~1000字）

### 5.5.1 seg-only 快速流水线

对比完整流水线（UNet + Gradient-Radon + CNN，约7347ms/帧）与 seg-only 方案（仅 UNet 分割 + mask 列采样拟合直线，约40ms/帧）的流程差异。seg-only 跳过了耗时的 Radon 变换和 CNN 推理，直接从分割 mask 中通过 32 列均匀采样找到天海交界跳变点，再用最小二乘法拟合直线。结合离群点剔除（残差阈值5像素），在绝大多数场景下获得足够精度。

📷 **图 5-8** 完整流水线 vs seg-only 流水线流程对比图（左：UNet→Radon→CNN 三阶段全流程；右：UNet→列采样→polyfit 快速流程。用删除线或灰色标注被跳过的模块，标注各阶段耗时）

### 5.5.2 GPU 端列采样与异步解耦

说明 seg-only 方案中列采样和差分运算在 GPU 上完成以避免不必要的 CPU-GPU 数据搬运，仅在最后的 polyfit 阶段回到 CPU。描述视频流场景下推理线程与 GUI 渲染线程的异步解耦设计，避免推理阻塞界面刷新。

📷 **图 5-9** seg-only 海天线提取过程可视化（4 子图：原始图像 → UNet 分割 mask → 列采样跳变点标注 → 拟合直线叠加在原图上）


## 5.6 实验与分析（约 2200~2800字）

### 5.6.1 实验 1：LINEA-L vs LINEA-N 精度对比

在 MU-SID 测试集（268张）上对比 LINEA-L 和 LINEA-N 的 sAP 指标。

**表 5-3** LINEA-L vs LINEA-N 实验结果

| 指标 | LINEA-L | LINEA-N | 变化 |
|------|---------|---------|------|
| Backbone | HGNetv2-B4 | HGNetv2-B0 | — |
| 参数量 | 27,302,689 | 4,066,586 | -85.1% (6.7x) |
| 推理速度 (RTX 4090 D) | 31.2 ms | 16.0 ms | 1.9x 加速 |
| sAP@5 | 94.0 | 93.7 | -0.3 |
| sAP@10 | 94.4 | 94.7 | **+0.3** |
| sAP@15 | 94.8 | 95.2 | **+0.5** |

分析：LINEA-N 参数减少 85%、速度提升近 2 倍，但 sAP@10/15 反而略高于 LINEA-L，说明在海天线这种单直线检测任务上大模型存在参数冗余，轻量化不仅没损精度反而有小幅提升。论证 LINEA-N 在嵌入式场景下完全可行。

📷 **图 5-10** LINEA-L 与 LINEA-N 检测结果对比可视化（选 4 张不同难度的 MU-SID 测试图，每张图上下两行并排展示两个模型的预测线 vs GT 线）

### 5.6.2 实验 3：seg-only vs 完整流水线精度对比

在 MU-SID 测试集（268张）上，在 Jetson Xavier NX 上对比 UNet seg-only 和 UNet+Radon+CNN 完整流水线。

**表 5-4** seg-only vs 完整流水线实验结果

| 指标 | seg-only | 完整流水线 | 变化 |
|------|----------|-----------|------|
| VE_mean (px) | 40.46 | 36.69 | 完整流水线低 3.77 px |
| SVE (px) | 99.56 | 136.12 | — |
| AE_mean (°) | 1.72 | 1.27 | 完整流水线低 0.45° |
| SA (°) | 4.76 | 4.83 | — |
| 命中率 VE<10px | 64.1% | 78.0% | +13.9% |
| 命中率 VE<20px | 77.7% | 88.4% | +10.7% |
| 推理时间 | **40.3 ms** | 7346.8 ms | **seg-only 快 182 倍** |
| FPS | **24.8** | 0.1 | — |
| 检测失败率 | 6.3% (17/268) | 0% (0/268) | — |

分析：完整流水线精度略优（命中率高 ~11%），但速度差距悬殊（182 倍）。seg-only 方案在 Jetson 上已达到 24.8 FPS，接近实时。结合 TensorRT 加速（16ms/帧，62 FPS）可完全满足实时需求。完整流水线 7.3 秒/帧的延迟在实际应用中完全不可接受，Radon 变换是主要瓶颈（占 80% 以上耗时）。seg-only 有 6.3% 的检测失败率，可通过 fallback（使用上一帧结果或图像中心水平线）改善。

📷 **图 5-11** seg-only vs 完整流水线检测结果对比（选 4 张典型图：2 张两种方法都准确的简单场景 + 1 张 seg-only 偏差较大的困难场景 + 1 张 seg-only 失败的极端场景。红线=预测，绿线=GT）

### 5.6.3 部署综合对比

**表 5-5** Jetson Xavier NX 上各方法综合对比

| 方法 | 精度指标 | 推理速度 | FPS | 参数量 | 适用场景 |
|------|---------|---------|-----|--------|---------|
| LINEA-L | sAP@10=94.4 | ~438 ms | ~2.3 | 27.3M | 离线高精度分析 |
| LINEA-N | sAP@10=94.7 | ~200 ms | ~5.0 | 4.07M | 嵌入式准实时 |
| UNet seg-only (PyTorch FP16) | VE=40.5px | 40 ms | 24.8 | 15M | 实时视频流 |
| UNet seg-only (TRT FP16) | VE≈40.5px | **16 ms** | **62** | 15M | **实时首选** |
| UNet+Radon+CNN | VE=36.7px | 7347 ms | 0.1 | 97M | 仅服务器端 |

从精度和速度两个维度讨论各方法的适用场景。结论：UNet seg-only TRT 是嵌入式实时部署的最佳选择；LINEA-N 适合对线段端点定位有较高要求但帧率可适当降低的场景。

📷 **图 5-12** 精度-速度散点图（横轴 log(FPS)，纵轴精度指标，标注各方法数据点，直观展示 Pareto 前沿）


## 5.7 本章小结（约 300~400字）

总结本章在嵌入式部署方面的主要工作和结论：(1) LINEA-N 轻量化使参数减少 85% 而精度不降，验证了 Transformer 方案在边缘设备部署的可行性；(2) TensorRT FP16 加速为 UNet 带来 3.2 倍提速，达到 62 FPS；(3) seg-only 流水线跳过 Radon 和 CNN，以可接受的精度代价换取 182 倍加速至实时水平；(4) 基于 PyQt5 的 GUI 系统集成四种检测方式，支持三种输入模式。指出当前方案的局限性（seg-only 失败率 6.3%、LINEA-N 在 Jetson 上的 TensorRT 部署尚未完成）以及未来可能的改进方向（INT8 量化、模型蒸馏、多帧时序平滑）。


---

# 图片清单（共 12 张）

| 编号 | 图名 | 类型 | 制作方式 |
|------|------|------|---------|
| 图 5-1 | 本章工作总览流程图 | 流程图 | 手绘/PPT/draw.io |
| 图 5-2 | Jetson Xavier NX 实物照片 | 照片 | 拍照 |
| 图 5-3 | 系统软件架构图 | 架构图 | PPT/draw.io |
| 图 5-4 | GUI 界面截图 | 截图 | 在板子上运行 GUI 截图 |
| 图 5-5 | LINEA-L vs LINEA-N 结构对比图 | 网络结构图 | PPT/draw.io |
| 图 5-6 | TensorRT 部署流程图 | 流程图 | PPT/draw.io |
| 图 5-7 | UNet 推理时间柱状图 | 数据图 | matplotlib 脚本生成 |
| 图 5-8 | 完整 vs seg-only 流水线对比图 | 流程图 | PPT/draw.io |
| 图 5-9 | seg-only 海天线提取过程可视化 | 可视化 | matplotlib 脚本生成 |
| 图 5-10 | LINEA-L vs LINEA-N 检测结果对比 | 可视化 | matplotlib 脚本生成 |
| 图 5-11 | seg-only vs 完整流水线检测结果对比 | 可视化 | matplotlib 脚本生成 |
| 图 5-12 | 精度-速度散点图 | 数据图 | matplotlib 脚本生成 |
