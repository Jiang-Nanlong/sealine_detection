# 第五章 海天线检测系统嵌入式部署与优化（大纲）

预估总字数：6500~7500字


## 5.1 引言（约 400~500字）

阐述将海天线检测算法从服务器环境迁移到嵌入式平台的实际需求，包括船舶自主导航和海上监控等应用场景对实时性和低功耗的要求。分析嵌入式环境面临的核心挑战，如算力受限、显存紧张、推理延迟敏感等，引出本章在模型轻量化、推理加速和流水线优化三方面的工作思路。


## 5.2 嵌入式平台与系统架构（约 800~1000字）

### 5.2.1 硬件平台

介绍 Jetson Xavier NX 开发板的硬件规格（6核 ARMv8.2 CPU、384 CUDA 核心 Maxwell GPU、6.7GB LPDDR4x 内存）以及与服务器 GPU 在算力和带宽上的差距。给出其在 10W/15W 功耗模式下的理论算力数据，说明选择该平台的原因。

### 5.2.2 软件栈

描述部署环境的软件配置，包括 JetPack SDK 版本、CUDA、cuDNN、TensorRT 8.5.2.2、PyTorch 等关键组件的版本信息。

### 5.2.3 系统整体架构与 GUI 设计

介绍基于 PyQt5 构建的海天线检测系统界面设计，包括四种检测方式的下拉选项切换（LINEA-L、LINEA-N、UNet seg-only、UNet 完整流水线），以及对图片、视频文件、摄像头实时流三种输入模式的支持。给出系统架构框图。


## 5.3 模型轻量化：LINEA-N（约 800~1000字）

从 LINEA-L（HGNetv2-B4 backbone，hidden_dim=256，6层 decoder，27M 参数）出发，说明面向嵌入式部署的轻量化设计。具体措施包括将 backbone 替换为 HGNetv2-B0、将 Transformer decoder 层数从6层缩减至3层、隐层维度从256降至128，以及将多尺度熵注入简化为单尺度。通过这些修改得到 LINEA-N（约4.27M 参数，相比 LINEA-L 减少约84%）。分析每项修改对模型容量和计算量的影响，以及为何这些精简在海天线检测这一相对简单的视觉任务上是可接受的。


## 5.4 TensorRT 推理加速（约 800~1000字）

### 5.4.1 ONNX 导出与 TensorRT Engine 编译

说明将 PyTorch 模型导出为 ONNX 格式的流程，以及在 Jetson 上使用 trtexec 编译 TensorRT engine 的步骤。重点描述 FP16 量化的启用方式和 dynamic shape 的设置。

### 5.4.2 加速效果

以 UNet 分割模型为例，对比 PyTorch FP32、PyTorch FP16 和 TensorRT FP16 三种模式的单帧推理时间。TensorRT FP16 下 UNet 推理约 16ms（62 FPS），而 PyTorch FP16 约 51ms，加速比约 3.2 倍。分析 TensorRT 在算子融合、内存优化和 kernel auto-tuning 方面带来的收益。


## 5.5 推理流水线优化（约 800~1000字）

### 5.5.1 seg-only 快速流水线

对比完整流水线（UNet + Gradient-Radon + CNN，约7940ms/帧）与 seg-only 方案（仅 UNet 分割 + mask 列采样拟合直线）的流程差异。seg-only 跳过了耗时的 Radon 变换和 CNN 推理，直接从分割 mask 中通过 32 列采样找到天海交界点，再用最小二乘法拟合直线。结合离群点剔除（残差阈值5像素），在绝大多数场景下获得足够精度。

### 5.5.2 GPU 端列采样与异步解耦

说明 seg-only 方案中列采样和差分运算在 GPU 上完成以避免不必要的 CPU-GPU 数据搬运，仅在最后的 polyfit 阶段回到 CPU。描述视频流场景下推理线程与 GUI 渲染线程的异步解耦设计，避免推理阻塞界面刷新。


## 5.6 实验与分析（约 2000~2500字）

### 5.6.1 实验 1：LINEA-L vs LINEA-N（sAP 精度 + 速度 + 参数量）

在 MU-SID 测试集上对比 LINEA-L 和 LINEA-N 的 sAP5/sAP10/sAP15 指标、Jetson 上的推理时间、参数量和模型文件大小。表格呈现实验结果，分析轻量化后精度损失的幅度，论证 LINEA-N 在嵌入式场景下的可行性。

（数据来源：eval_exp1_linea_compare.py 输出，LINEA-L ~438ms/帧，LINEA-N 速度待测）

### 5.6.2 实验 3：seg-only vs Radon+CNN 完整流水线（VE/AE + 速度）

在 MU-SID 测试集（268张）上对比 seg-only 和完整流水线的 VE_mean、SVE、AE_mean、SA 四项指标，以及平均推理时间和命中率（VE<10px/20px 的比例）。分析 seg-only 在精度上能否满足实际应用需求，以及其在速度上的巨大优势（约 51ms vs 7940ms）是否足以弥补可能的精度差距。讨论 Radon 变换在嵌入式平台上的瓶颈，以及 seg-only 为何是更实用的部署选择。

（数据来源：eval_exp3_segonly_vs_radon.py 输出）

### 5.6.3 部署综合对比

汇总所有方法（LINEA-L、LINEA-N、UNet seg-only PyTorch、UNet seg-only TRT、UNet+Radon+CNN 完整流水线）在 Jetson Xavier NX 上的综合对比表，涵盖精度指标、推理速度、参数量、显存占用。从精度和速度两个维度给出各方法的适用场景建议。


## 5.7 本章小结（约 300~400字）

总结本章在嵌入式部署方面的主要工作和结论：LINEA-N 轻量化使 Transformer 方案可在边缘设备运行；TensorRT 加速为 UNet 带来数倍提速；seg-only 流水线在保持可接受精度的同时将帧率提升至实时水平。指出当前方案的局限性以及未来可能的改进方向，如 INT8 量化和模型蒸馏等。
