"""生成图5-12: LINEA-L与LINEA-N检测结果对比可视化
选4张不同难度的MU-SID测试图，每张图两行并排展示两个模型的预测线vs GT线
使用正确的letterbox预处理（与训练一致）
"""
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'method1_linea_entropy'))
sys.path.insert(0, os.path.join(_ROOT, 'method1_linea_entropy', 'LINEA'))

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──
IMG_DIR = '/home/jetson/Documents/Hashmani_Dataset/MU-SID'
GT_CSV = os.path.join(_ROOT, 'splits_musid/GroundTruth_test.csv')
IMG_SIZE = 640
OUT_PNG = 'fig5_12_linea_compare.png'
OUT_PDF = 'fig5_12_linea_compare.pdf'

# 手动挑选4张: 2张简单 + 1张中等 + 1张偏难
SELECTED_STEMS = ['DSC_0054_3', 'DSC_0049_6', 'DSC_0036_5', 'DSC_1631_6']

LINEA_L_CFG = os.path.join(_ROOT, 'method1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py')
LINEA_L_WTS = os.path.join(_ROOT, 'weights/linea_entropy_b_enhanced_best.pth')
LINEA_N_CFG = os.path.join(_ROOT, 'method1_linea_entropy/configs/linea_entropy_n_musid.py')
LINEA_N_WTS = os.path.join(_ROOT, 'weights/linea_entropy_n_best.pth')

MEAN = [0.538, 0.494, 0.453]
STD  = [0.257, 0.263, 0.273]

def load_gt_all(csv_path):
    gt = {}
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(',')
            gt[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    return gt

def letterbox_pil(pil_img, sz):
    """等比例缩放 + 右下角零填充到 (sz, sz)"""
    orig_w, orig_h = pil_img.size
    scale = min(sz / orig_w, sz / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pil_resized = TF.resize(pil_img, [new_h, new_w])
    pad_right = sz - new_w
    pad_bottom = sz - new_h
    pil_out = TF.pad(pil_resized, [0, 0, pad_right, pad_bottom], fill=0)
    return pil_out, scale, new_w, new_h

def compute_entropy_single(img_bgr, sz, new_w, new_h):
    """单尺度方差近似熵 with letterbox"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray, (new_w, new_h))
    k = 9
    mean = cv2.blur(gray, (k, k))
    sq_mean = cv2.blur(gray ** 2, (k, k))
    var = np.clip(sq_mean - mean ** 2, 0, None)
    ent = np.log(var + 1e-8)
    ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
    ent_sq = np.zeros((sz, sz), dtype=np.float32)
    ent_sq[:new_h, :new_w] = ent
    return torch.from_numpy(ent_sq[np.newaxis]).float()

def compute_entropy_multi(img_bgr, sz, new_w, new_h):
    """多尺度方差近似熵 with letterbox"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray, (new_w, new_h))
    channels = []
    for k in [5, 9, 15]:
        mean = cv2.blur(gray, (k, k))
        sq_mean = cv2.blur(gray ** 2, (k, k))
        var = np.clip(sq_mean - mean ** 2, 0, None)
        ent = np.log(var + 1e-8)
        ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
        ent_sq = np.zeros((sz, sz), dtype=np.float32)
        ent_sq[:new_h, :new_w] = ent
        channels.append(ent_sq)
    return torch.from_numpy(np.stack(channels, axis=0)).float()

def load_linea(cfg_path, wts_path, device):
    from util.slconfig import SLConfig
    import method1_linea_entropy.models
    from models.registry import MODULE_BUILD_FUNCS

    cfg = SLConfig.fromfile(cfg_path)
    cfg.pretrained = False
    cfg.eval_spatial_size = [IMG_SIZE, IMG_SIZE]
    model_name = getattr(cfg, 'modelname', 'LINEA_ENTROPY_B_ENHANCED')
    cfg.modelname = model_name
    build_fn = MODULE_BUILD_FUNCS.get(model_name)
    model, postprocessor = build_fn(cfg)
    model = model.to(device)

    entropy_mode = getattr(cfg, 'entropy_mode', 'entropy')

    if os.path.isfile(wts_path):
        ckpt = torch.load(wts_path, map_location=device)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, postprocessor, entropy_mode

@torch.no_grad()
def detect_linea(model, postprocessor, img_bgr, entropy_mode, device):
    sz = IMG_SIZE
    h_orig, w_orig = img_bgr.shape[:2]

    # ── Letterbox预处理（与训练完全一致）──
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_lb, scale, new_w, new_h = letterbox_pil(pil_img, sz)
    img_tensor = TF.to_tensor(pil_lb)
    img_tensor = TF.normalize(img_tensor, MEAN, STD)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # ── Entropy with letterbox ──
    if entropy_mode == 'entropy':
        ent = compute_entropy_single(img_bgr, sz, new_w, new_h).unsqueeze(0).to(device)
    else:
        ent = compute_entropy_multi(img_bgr, sz, new_w, new_h).unsqueeze(0).to(device)

    # ── 推理 ──
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(img_tensor, entropy_map=ent)

    # ── 后处理：在letterbox空间(sz x sz)中获取线段坐标 ──
    target_sizes = torch.tensor([[sz, sz]], device=device)
    results = postprocessor(outputs, target_sizes)

    if results and len(results[0]['lines']) > 0:
        lines = results[0]['lines'].cpu().float().numpy()
        scores = results[0]['scores'].cpu().float().numpy()

        # 过滤：只保留近水平线 + 足够长的线
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        angles = np.abs(np.degrees(np.arctan2(dy, dx)))
        dev = np.minimum(angles, 180.0 - angles)
        lengths = np.sqrt(dx**2 + dy**2)
        valid = (dev <= 15.0) & (lengths >= 0.2 * sz)
        if not np.any(valid):
            valid = dev <= 30.0
        if not np.any(valid):
            best_idx = scores.argmax()
        else:
            valid_idx = np.where(valid)[0]
            best_idx = valid_idx[scores[valid].argmax()]

        line_lb = lines[best_idx]  # [x1, y1, x2, y2] in letterbox space

        # ── 映射回原始图像坐标 ──
        # letterbox: 原图缩放scale倍，右下角填充，无偏移
        line_orig = line_lb / scale
        return line_orig, scores[best_idx]
    return None, 0.0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt_all = load_gt_all(GT_CSV)

    stems = SELECTED_STEMS
    print(f"Selected images: {stems}")

    print("Loading LINEA-L...")
    model_l, post_l, ent_mode_l = load_linea(LINEA_L_CFG, LINEA_L_WTS, device)
    print("Loading LINEA-N...")
    model_n, post_n, ent_mode_n = load_linea(LINEA_N_CFG, LINEA_N_WTS, device)

    # ── Warmup ──
    dummy_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(2):
        detect_linea(model_l, post_l, dummy_bgr, ent_mode_l, device)
        detect_linea(model_n, post_n, dummy_bgr, ent_mode_n, device)

    fig, axes = plt.subplots(2, 4, figsize=(20, 7))

    for col, stem in enumerate(stems):
        for ext in ['.JPG', '.jpg', '.png', '.PNG']:
            img_path = os.path.join(IMG_DIR, stem + ext)
            if os.path.exists(img_path):
                break
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        gt = gt_all.get(stem)

        line_l, score_l = detect_linea(model_l, post_l, img_bgr, ent_mode_l, device)
        line_n, score_n = detect_linea(model_n, post_n, img_bgr, ent_mode_n, device)

        for row, (line, label, score) in enumerate([
            (line_l, 'LINEA-L', score_l),
            (line_n, 'LINEA-N', score_n)
        ]):
            axes[row, col].imshow(img_rgb)
            if gt is not None:
                axes[row, col].plot([gt[0], gt[2]], [gt[1], gt[3]],
                                    'g-', linewidth=2.5, label='GT')
            if line is not None:
                axes[row, col].plot([line[0], line[2]], [line[1], line[3]],
                                    'r--', linewidth=2, label=label)
            axes[row, col].set_xlim(0, w)
            axes[row, col].set_ylim(h, 0)
            axes[row, col].axis('off')
            # 左侧标注模型名
            if col == 0:
                axes[row, col].text(-0.02, 0.5, label, transform=axes[row, col].transAxes,
                                     fontsize=14, fontweight='bold', va='center', ha='right',
                                     rotation=90)
            if col == 3:
                axes[row, col].legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUT_PDF, bbox_inches='tight')
    print(f'Saved {OUT_PNG} / {OUT_PDF}')

    del model_l, model_n
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
