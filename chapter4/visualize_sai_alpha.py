"""
visualize_sai_alpha.py — 可视化 SAI 模块的空间注意力图 α

功能:
  加载训练好的增强模型 → 对指定图像运行前向推理 →
  提取 SAI 模块的注意力图 α [B,1,H,W] → 保存为热力图。

用法:
  在 PyCharm 中修改下方全局变量后直接运行。
"""
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# 项目路径
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "LINEA"))

import stage1_linea_entropy.models  # noqa: F401 — 触发模型注册

# ============================================================
# 全局变量 — 在 PyCharm 中直接修改
# ============================================================
CONFIG_FILE = "stage1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py"
WEIGHTS_PATH = "output/linea_entropy_b_enhanced_musid_v1_e150/best_checkpoint.pth"

# 测试图像 (原图 + 预计算熵图)
IMAGE_PATH = "Hashmani's Dataset/MU-SID/1.jpg"
ENTROPY_DIR = "Hashmani's Dataset/MU-SID_entropy_blue"

IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出
OUTPUT_DIR = "chapter4/fig_sai_alpha"
CMAP = "inferno"          # 热力图配色: inferno / magma / hot / jet
DPI = 200


# ============================================================
# 工具函数
# ============================================================
def load_config(config_file):
    from util.slconfig import SLConfig
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    cfg = SLConfig.fromfile(str(config_path))
    cfg.pretrained = False
    sz = getattr(cfg, 'eval_spatial_size', IMG_SIZE)
    if isinstance(sz, int):
        sz = [sz, sz]
    cfg.eval_spatial_size = sz
    return cfg


def letterbox_image(img_bgr, target_size):
    """等比缩放 + padding (居中) → [target_size, target_size]"""
    h, w = img_bgr.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, new_w, new_h


def prepare_image(image_path):
    """读取并预处理图像 → [1, 3, H, W] float tensor"""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    canvas, scale, new_w, new_h = letterbox_image(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor, img_bgr, canvas


def prepare_entropy(image_path, entropy_dir, multiscale_kernels=(1, 5, 11)):
    """加载预计算熵图并构造多尺度 3 通道 → [1, 3, H, W]"""
    stem = Path(image_path).stem
    ent_path = Path(entropy_dir) / f"{stem}.npy"
    if not ent_path.exists():
        raise FileNotFoundError(f"未找到熵图: {ent_path}")

    ent_raw = np.load(str(ent_path)).astype(np.float32)  # [H_orig, W_orig]

    # letterbox
    h, w = ent_raw.shape
    scale = min(IMG_SIZE / h, IMG_SIZE / w)
    new_w, new_h = int(w * scale), int(h * scale)
    ent_resized = cv2.resize(ent_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    top = (IMG_SIZE - new_h) // 2
    left = (IMG_SIZE - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = ent_resized

    # 归一化到 [0, 1]
    max_val = canvas.max()
    if max_val > 0:
        canvas = canvas / max_val

    ent_tensor = torch.from_numpy(canvas).unsqueeze(0)  # [1, H, W]

    # 多尺度
    channels = []
    for k in multiscale_kernels:
        if k <= 1:
            channels.append(ent_tensor)
        else:
            pad = k // 2
            smoothed = F.avg_pool2d(
                ent_tensor.unsqueeze(0), kernel_size=k,
                stride=1, padding=pad,
            ).squeeze(0)
            channels.append(smoothed)
    ms_tensor = torch.cat(channels, dim=0).unsqueeze(0)  # [1, 3, H, W]
    return ms_tensor


def extract_sai_alpha(model, image_tensor, entropy_tensor):
    """
    手动执行 backbone → encoder 前半部分, 提取 SAI 的 α 注意力图。

    Returns:
        alpha: [1, 1, H_feat, W_feat] tensor (值在 0~1)
        entropy_feat: [1, 256, H_feat, W_feat]
    """
    model.eval()
    with torch.no_grad():
        # 1. backbone
        feats = model.backbone(image_tensor)

        # 2. encoder: input_proj
        encoder = model.encoder
        proj_feats = [encoder.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 3. entropy branch
        entropy_feat = encoder.entropy_branch(entropy_tensor)

        # 对齐分辨率
        target_h, target_w = proj_feats[0].shape[2], proj_feats[0].shape[3]
        if entropy_feat.shape[2] != target_h or entropy_feat.shape[3] != target_w:
            entropy_feat = F.interpolate(
                entropy_feat, size=(target_h, target_w),
                mode='bilinear', align_corners=False,
            )

        # 4. SAI: 直接调用 attn_conv 获取 alpha
        alpha = torch.sigmoid(encoder.sai.attn_conv(entropy_feat))  # [1, 1, H, W]

    return alpha, entropy_feat


def save_alpha_figure(alpha_np, canvas_bgr, output_dir, stem):
    """保存 alpha 热力图和叠加图"""
    os.makedirs(output_dir, exist_ok=True)

    h, w = canvas_bgr.shape[:2]
    alpha_resized = cv2.resize(alpha_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. 纯 alpha 热力图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(alpha_resized, cmap=CMAP, vmin=0, vmax=1)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("SAI Attention Map (α)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_alpha.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # 2. alpha 叠加到原图
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha_color = plt.cm.get_cmap(CMAP)(alpha_resized)[:, :, :3]
    overlay = 0.5 * canvas_rgb + 0.5 * alpha_color

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    ax2.imshow(overlay)
    ax2.set_axis_off()
    ax2.set_title("α overlaid on image", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{stem}_alpha_overlay.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig2)

    print(f"[SAVED] {output_dir}/{stem}_alpha.png")
    print(f"[SAVED] {output_dir}/{stem}_alpha_overlay.png")


# ============================================================
# 主流程
# ============================================================
def main():
    # 路径解析
    image_path = Path(IMAGE_PATH)
    if not image_path.is_absolute():
        image_path = _PROJECT_ROOT / image_path
    entropy_dir = Path(ENTROPY_DIR)
    if not entropy_dir.is_absolute():
        entropy_dir = _PROJECT_ROOT / entropy_dir
    weights_path = Path(WEIGHTS_PATH)
    if not weights_path.is_absolute():
        weights_path = _PROJECT_ROOT / weights_path
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir

    # 加载配置 + 构建模型
    print(f"[CONFIG] {CONFIG_FILE}")
    args = load_config(CONFIG_FILE)
    from models.registry import MODULE_BUILD_FUNCS
    model, _ = MODULE_BUILD_FUNCS.get(args.modelname)(args)
    model.to(DEVICE)

    # 加载权重
    if weights_path.exists():
        ckpt = torch.load(str(weights_path), map_location=DEVICE)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"[WEIGHTS] 已加载: {weights_path}")
    else:
        print(f"[WARNING] 权重文件不存在: {weights_path}")
        print("          将使用随机初始化权重（α 无实际意义，仅验证流程）")

    # 准备输入
    image_tensor, img_bgr, canvas = prepare_image(str(image_path))
    image_tensor = image_tensor.to(DEVICE)

    entropy_tensor = prepare_entropy(str(image_path), str(entropy_dir))
    entropy_tensor = entropy_tensor.to(DEVICE)

    print(f"[INPUT]  image: {tuple(image_tensor.shape)}")
    print(f"[INPUT]  entropy: {tuple(entropy_tensor.shape)}")

    # 提取 alpha
    alpha, entropy_feat = extract_sai_alpha(model, image_tensor, entropy_tensor)
    alpha_np = alpha.squeeze().cpu().numpy()  # [H_feat, W_feat]

    print(f"[ALPHA]  shape={alpha_np.shape}  "
          f"min={alpha_np.min():.4f}  max={alpha_np.max():.4f}  "
          f"mean={alpha_np.mean():.4f}")

    # 保存
    stem = Path(IMAGE_PATH).stem
    save_alpha_figure(alpha_np, canvas, str(output_dir), stem)

    print("[DONE]")


if __name__ == '__main__':
    main()
