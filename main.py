import torch
from enhancement_module import EnhancementModule

if __name__ == '__main__':
    # --- 示例用法 ---

    # 创建一个假的输入图像张量 (Batch_size, Channels, Height, Width)
    # MobileNetV3 要求输入尺寸至少为 32x32，且最好是32的倍数
    dummy_input_image = torch.randn(2, 3, 256, 256)

    # 实例化完整的增强模块
    # pretrained_unet_encoder=True 会尝试下载预训练的MobileNetV3权重
    enhancement_model = EnhancementModule(pretrained_unet_encoder=True)

    # 将模型设置为评估模式
    enhancement_model.eval()

    # 执行前向传播
    with torch.no_grad():  # 在推理时不需要计算梯度
        final_output = enhancement_model(dummy_input_image)

    # 打印输出张量的形状，确认流程正确
    print(f"\nInput shape: {dummy_input_image.shape}")
    print(f"Final output shape: {final_output.shape}")

    # 检查输出形状是否与输入一致
    assert dummy_input_image.shape == final_output.shape
    print("\nSuccessfully executed the enhancement module!")