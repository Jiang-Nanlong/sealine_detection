import os
import cv2
import torch
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

from unet_model import RestorationGuidedHorizonNet
from zerodce import C_DCE_Net
from dataset_loader import synthesize_rain_fog  # 你原脚本就在用

# ================= 配置区域 =================
# 1) 数据与输出路径
IMG_DIR = r"test4/smd_frames"                 # 原始图片目录
CSV_PATH = r"test4/SMD_GroundTruth.csv"      # 仍沿用你的 CSV（只用于取文件名索引）
SPLIT_DIR = r"test4/splits"                           # test_indices.npy 所在目录
OUT_DIR = r"paper_figures_4panels"                    # 输出文件夹（建议新建，避免覆盖）

# 2) 权重路径
UNET_CKPT = r"rghnet_best_c2.pth"
DCE_WEIGHTS = r"Epoch99.pth"  # 你说的 Epoth99.pth/ Epoch99.pth 两种写法都可能，下面会自动兜底

# 3) 输入设置
ENABLE_DEGRADATION = True   # True: 清晰图合成雨雾/低照度（作为“原图/输入图”）；False: 直接用清晰图
UNET_W, UNET_H = 1024, 576  # UNet 输入尺寸

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2024

# 4) 可视化设置
# mask 可视化：二值（白=sky, 黑=nosky） or 伪彩色
MASK_VIS_MODE = "binary"  # "binary" or "color"
COLOR_MASK_SKY = (235, 206, 135)   # BGR
COLOR_MASK_SEA = (100, 149, 237)   # BGR

# 5) 输出控制
SAVE_COMBINED = True        # 保存 1x4 拼图
SAVE_INDIVIDUAL = False     # 是否把四张也分别保存（方便你排版）
MAX_IMAGES = None           # 只想导出前N张用于论文：例如 20；想全跑则 None
SHOW_PREVIEW = False        # True 会弹窗预览（跑很多图不建议）
PREVIEW_EVERY = 1           # 每隔多少张预览一次（SHOW_PREVIEW=True 时生效）
# ===========================================


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def safe_load_state_dict(ckpt_path: str):
    """兼容：state_dict / {'state_dict':...} / {'model':...} / DataParallel('module.')"""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state)}")

    # strip "module."
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def find_img_path(name_no_ext: str):
    candidates = [
        os.path.join(IMG_DIR, name_no_ext),
        os.path.join(IMG_DIR, name_no_ext + ".JPG"),
        os.path.join(IMG_DIR, name_no_ext + ".jpg"),
        os.path.join(IMG_DIR, name_no_ext + ".png"),
        os.path.join(IMG_DIR, name_no_ext + ".jpeg"),
        os.path.join(IMG_DIR, name_no_ext + ".JPEG"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def to_tensor_rgb01(rgb_uint8: np.ndarray) -> torch.Tensor:
    """rgb_uint8: HWC uint8 [0,255] -> tensor [1,3,H,W] float [0,1]"""
    x = torch.from_numpy(rgb_uint8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return x


def tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    """t: [1,3,H,W] in [0,1] -> BGR uint8"""
    arr = (t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def make_mask_vis(mask_np: np.ndarray) -> np.ndarray:
    """
    mask_np: HxW, {0,1}（假设 1=sky, 0=nosky/sea）
    返回 BGR uint8
    """
    h, w = mask_np.shape[:2]
    if MASK_VIS_MODE == "binary":
        vis = np.zeros((h, w), dtype=np.uint8)
        vis[mask_np == 1] = 255
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        return vis_bgr
    else:
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[mask_np == 0] = COLOR_MASK_SEA
        vis[mask_np == 1] = COLOR_MASK_SKY
        return vis


def main():
    ensure_dir(OUT_DIR)
    seed_everything(SEED)

    # -------- 1) 处理 DCE 权重路径的“Epoch/Epoth”兜底 --------
    dce_path = DCE_WEIGHTS
    if not os.path.exists(dce_path):
        alt = "Epoth99.pth"
        if os.path.exists(alt):
            dce_path = alt
        else:
            # 也尝试在当前脚本目录下找
            cand = os.path.join(os.path.dirname(__file__), DCE_WEIGHTS)
            if os.path.exists(cand):
                dce_path = cand

    if not os.path.exists(dce_path):
        raise FileNotFoundError(f"找不到 Zero-DCE++ 权重：{DCE_WEIGHTS}（也未找到 Epoth99.pth）")

    # -------- 2) 加载 Zero-DCE++（独立调用，用于输出第2张图）--------
    print("Loading Zero-DCE++ model...")
    dce = C_DCE_Net(scale_factor=1, n=32, enhance_iters=8).to(DEVICE)
    dce.load_state_dict(safe_load_state_dict(dce_path), strict=True)
    dce.eval()
    for p in dce.parameters():
        p.requires_grad = False

    # -------- 3) 加载 UNet（内部也会用 DCE，输出第3/4张图）--------
    print("Loading RG-HNet (UNet) model...")
    unet = RestorationGuidedHorizonNet(num_classes=2, dce_weights_path=dce_path).to(DEVICE)
    unet.load_state_dict(safe_load_state_dict(UNET_CKPT), strict=False)
    unet.eval()

    # -------- 4) 加载测试索引与CSV（沿用你的方式）--------
    print("Loading test set indices & csv...")
    test_indices = np.load(os.path.join(SPLIT_DIR, "test_indices.npy"))
    df = pd.read_csv(CSV_PATH, header=None)

    print(f"Total test images: {len(test_indices)}")
    print(f"Output dir: {OUT_DIR}")
    if MAX_IMAGES is not None:
        print(f"Will export only first {MAX_IMAGES} images.")

    count = 0
    for k, idx in enumerate(tqdm(test_indices)):
        if MAX_IMAGES is not None and count >= MAX_IMAGES:
            break

        row = df.iloc[idx]
        name = str(row[0])  # 仍按你CSV第0列是名字的方式

        img_path = find_img_path(name)
        if img_path is None:
            continue

        bgr_orig = cv2.imread(img_path)
        if bgr_orig is None:
            continue

        rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)
        rgb_unet = cv2.resize(rgb_orig, (UNET_W, UNET_H), interpolation=cv2.INTER_LINEAR)

        # -------- Panel1: 原图/输入图 --------
        if ENABLE_DEGRADATION:
            # synthesize_rain_fog 返回 float [0,1]
            rgb_degraded = synthesize_rain_fog(rgb_unet, p_clean=0.0)
            panel1_bgr = cv2.cvtColor((rgb_degraded * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            input_tensor = torch.from_numpy(rgb_degraded).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        else:
            panel1_bgr = cv2.cvtColor(rgb_unet, cv2.COLOR_RGB2BGR)
            input_tensor = to_tensor_rgb01(rgb_unet).to(DEVICE)

        # -------- Panel2: Zero-DCE++ 增强图（独立调用）--------
        with torch.no_grad():
            enh = dce(input_tensor)  # [1,3,H,W] in [0,1]
        panel2_bgr = tensor_to_bgr_uint8(enh)

        # -------- Panel3 & Panel4: UNet 输出（复原 + 分割）--------
        with torch.no_grad():
            restored_t, seg_logits, _ = unet(input_tensor, None, True, True)

        panel3_bgr = tensor_to_bgr_uint8(restored_t)

        # seg mask: argmax(softmax)
        if seg_logits is None:
            continue
        mask_score = torch.softmax(seg_logits, dim=1)
        mask_np = torch.argmax(mask_score, dim=1)[0].cpu().numpy().astype(np.uint8)  # HxW, {0,1}
        panel4_bgr = make_mask_vis(mask_np)

        # -------- 保存输出 --------
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if SAVE_INDIVIDUAL:
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_1_input.png"), panel1_bgr)
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_2_dce.png"), panel2_bgr)
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_3_unet_rest.png"), panel3_bgr)
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_4_unet_mask.png"), panel4_bgr)

        if SAVE_COMBINED:
            combined = np.hstack([panel1_bgr, panel2_bgr, panel3_bgr, panel4_bgr])
            cv2.imwrite(os.path.join(OUT_DIR, f"{base_name}_4panels.png"), combined)

        # -------- 预览（可选）--------
        if SHOW_PREVIEW and (count % PREVIEW_EVERY == 0):
            preview = np.hstack([panel1_bgr, panel2_bgr, panel3_bgr, panel4_bgr])
            cv2.imshow("4-panels preview (Input | DCE | UNet-Rest | UNet-Mask)", preview)
            key = cv2.waitKey(0)
            if key == 27:  # ESC 退出
                break

        count += 1

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("All done.")


if __name__ == "__main__":
    main()
