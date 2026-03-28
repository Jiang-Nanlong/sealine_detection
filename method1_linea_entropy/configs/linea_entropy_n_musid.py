"""
linea_entropy_n_musid.py — LINEA-N (最小变体) + 局部熵注入，MU-SID 数据集训练配置

与 linea_entropy_musid.py 相比：
  - backbone:  HGNetv2_B4 → HGNetv2_B0  (参数大幅减少)
  - hidden_dim: 256 → 128
  - dim_feedforward: 1024 → 512
  - dec_layers: 6 → 3
  - expansion: 0.5 → 0.34
  - depth_mult: 1.0 → 0.5
  预期参数量约 5-8M（原 ~26M），推理速度提升 3-5 倍。

用法：
  python LINEA/main.py -c method1_linea_entropy/configs/linea_entropy_n_musid.py
"""

# ---- 继承官方 LINEA 公共配置 ----
_base_ = [
    '../LINEA/configs/linea/include/optimizer.py',
    '../LINEA/configs/linea/include/linea.py',
    '../LINEA/configs/linea/include/dataset.py',
]

# ============================================================
# 输出目录
# ============================================================
output_dir = 'output/linea_entropy_n_musid_e200'

# ============================================================
# 模型（LINEA-N + 单尺度熵注入）
# ============================================================
modelname = 'LINEA_ENTROPY'

# ---- backbone (LINEA-N: HGNetv2-B0) ----
backbone = 'HGNetv2_B0'
param_dict_type = 'hgnetv2_b0'
use_lab = True
freeze_norm = False
freeze_stem_only = True

# ---- encoder (LINEA-N 尺寸) ----
feat_strides = [8, 16, 32]
hidden_dim = 128
dim_feedforward = 512
nheads = 8
in_channels_encoder = [256, 512, 1024]
expansion = 0.34
depth_mult = 0.5
use_lmap = False

# ---- decoder (LINEA-N 尺寸) ----
feat_channels_decoder = [128, 128, 128]
dec_layers = 3
num_queries = 1100
num_select = 300
reg_max = 16
reg_scale = 4
eval_idx = 2

# ---- criterion ----
weight_dict = {'loss_logits': 4, 'loss_line': 5}

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
batch_size_train = 8
batch_size_val = 16

epochs = 200
lr = 0.0004
lr_drop_list = [140, 170]
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
        'lr': 0.00002,
    },
    {
        'params': '^(?=.*backbone)(?=.*norm|bn).*$',
        'lr': 0.00002,
        'weight_decay': 0.0,
    },
    {
        'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$',
        'weight_decay': 0.0,
    },
]
