"""
linea_entropy_a_musid.py — LINEA + 多层门控 FiLM 熵注入，MU-SID 数据集训练配置

用法：
  python LINEA/main.py -c method1_linea_entropy/configs/linea_entropy_a_musid.py

模型：LINEA_ENTROPY_A（LINEAWithEntropyA，encoder 带多层门控 FiLM 熵注入）
数据：MU-SID，letterbox 到 640×640，entropy_mode='entropy'
"""

# ---- 继承官方 LINEA 公共配置 ----
_base_ = [
    '../LINEA/configs/linea/include/optimizer.py',
    '../LINEA/configs/linea/include/linea.py',
    '../LINEA/configs/linea/include/dataset.py',
]

# ============================================================
# 输出目录（独立于旧实验）
# ============================================================
output_dir = 'output/linea_entropy_a_musid_v2_e150'

# ============================================================
# 模型
# ============================================================
modelname = 'LINEA_ENTROPY_A'

# ---- backbone ----
backbone = 'HGNetv2_B4'
param_dict_type = 'hgnetv2_b4'
use_lab = False

# ---- encoder ----
feat_strides = [8, 16, 32]
hidden_dim = 256
dim_feedforward = 1024
nheads = 8
expansion = 0.5
depth_mult = 1.0
use_lmap = False

# ---- decoder ----
feat_channels_decoder = [256, 256, 256]
dec_layers = 6
num_select = 300
reg_max = 16
reg_scale = 4

# ---- criterion ----
weight_dict = {'loss_logits': 4, 'loss_line': 5}

# ============================================================
# 主线 A 新增配置 — 多层门控 FiLM 熵注入 + 自适应门控
# ============================================================
enable_multiscale_entropy = True
entropy_inject_levels = [0, 1, 2]
entropy_fusion_type = 'gated_film'
entropy_use_spatial_gate = True
entropy_use_channel_modulation = True
entropy_use_adaptive_gate = True       # 实例自适应残差门控 (IARG)
entropy_use_quality_gate = True        # 熵先验质量评估门控

# ============================================================
# 数据集
# ============================================================
dataset_file = 'musid'
entropy_mode = 'entropy'          # 'baseline' / 'entropy'

musid_img_dir     = "Hashmani's Dataset/MU-SID"
musid_entropy_dir = "Hashmani's Dataset/MU-SID_entropy_blue"
musid_split_dir   = 'splits_musid'

# ============================================================
# 训练
# ============================================================
batch_size_train = 4
batch_size_val = 8

epochs = 150
lr = 0.00025
lr_drop_list = [110, 135]
clip_max_norm = 0.1
save_checkpoint_interval = 10
use_warmup = True
warmup_iters = 200

use_ema = False
ema_epoch = 0

# ---- optimizer param groups ----
model_parameters = [
    {
        'params': '^(?=.*backbone)(?!.*norm|bn).*$',
        'lr': 0.0000125,
    },
    {
        'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$',
        'weight_decay': 0.0,
    },
]
