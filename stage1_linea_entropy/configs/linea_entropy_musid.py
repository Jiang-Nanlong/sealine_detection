"""
linea_entropy_musid.py — LINEA + 局部熵注入，MU-SID 数据集训练配置

用法：
  python LINEA/main.py -c stage1_linea_entropy/configs/linea_entropy_musid.py

模型：LINEA_ENTROPY（LINEAWithEntropy，encoder 带 entropy_branch + alpha）
数据：MU-SID，letterbox 到 640×640，entropy_mode='entropy'
"""

# ---- 继承官方 LINEA 公共配置 ----
_base_ = [
    '../../LINEA/configs/linea/include/optimizer.py',
]

# ============================================================
# 输出目录
# ============================================================
output_dir = 'output/linea_entropy_musid'

# ============================================================
# 模型
# ============================================================
modelname = 'LINEA_ENTROPY'
criterionname = 'LINEACRITERION'

eval_spatial_size = [640, 640]
eval_idx = 5                  # 6 decoder layers → index 5
num_classes = 2

# ---- backbone ----
backbone = 'HGNetv2_B4'
param_dict_type = 'hgnetv2_b4'
pretrained = True
use_lab = False
use_checkpoint = False
return_interm_indices = [1, 2, 3]
freeze_norm = True
freeze_stem_only = True

# ---- encoder ----
hybrid_encoder = 'hybrid_encoder_asymmetric_conv'
in_channels_encoder = [512, 1024, 2048]
feat_strides = [8, 16, 32]
hidden_dim = 256
dim_feedforward = 1024
nheads = 8
pe_temperatureH = 20
pe_temperatureW = 20
expansion = 0.5
depth_mult = 1.0
use_lmap = False

transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

# ---- decoder ----
feat_channels_decoder = [256, 256, 256]
dec_layers = 6
num_queries = 1100
num_select = 300
reg_max = 16
reg_scale = 4
query_dim = 4
num_feature_levels = 3
dec_n_points = [4, 1, 1]

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

# MU-SID: collate 用到的 data_aug 参数（MU-SID 已在 dataset 内完成 letterbox，
# collate 的 multi-scale resize 仍需以下字段）
data_aug_scales = [(640, 640)]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None

# ============================================================
# 训练
# ============================================================
batch_size_train = 4
batch_size_val = 8

epochs = 24
lr = 0.00025
lr_drop_list = [18, 22]
clip_max_norm = 0.1
save_checkpoint_interval = 2
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
