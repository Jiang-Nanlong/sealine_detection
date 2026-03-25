"""
ablation_mslep_sai_egab_musid.py — 消融实验：MSLEP + SAI + EGAB

用法：
  python LINEA/main.py -c stage1_linea_entropy/configs/ablation_mslep_sai_egab_musid.py

模型：LINEA_ENTROPY_B_ENHANCED（关闭 HASH）
  - MSLEP: 多尺度局部熵先验提取 (3 通道: 5×5, 11×11, 21×21 窗口)  ✓
  - SAI:   空间自适应注入 (sigmoid attention-weighted injection)    ✓
  - EGAB:  熵引导注意力偏置 (post-attention residual correction)   ✓
  - HASH:  关闭                                                     ✗
数据：MU-SID，letterbox 到 640×640，entropy_mode='entropy_ms'

消融表第 5 行：Baseline → 单尺度熵 → MSLEP → MSLEP+SAI → **MSLEP+SAI+EGAB**
"""

# ---- 继承官方 LINEA 公共配置 ----
_base_ = [
    '../../LINEA/configs/linea/include/optimizer.py',
    '../../LINEA/configs/linea/include/linea.py',
    '../../LINEA/configs/linea/include/dataset.py',
]

# ============================================================
# 输出目录
# ============================================================
output_dir = 'output/ablation_mslep_sai_egab_musid_standalone_e130'

# ============================================================
# 模型
# ============================================================
modelname = 'LINEA_ENTROPY_B_ENHANCED'

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

# ---- criterion（标准 criterion，无 ranking）----
criterionname = 'LINEACRITERION'
weight_dict = {'loss_logits': 4, 'loss_line': 5}
losses = ['labels', 'lines']

# ============================================================
# 方向一：MSLEP — 多尺度局部熵先验提取  ✓ 开启
# ============================================================
entropy_in_channels = 3          # 3 通道: 5×5, 11×11, 21×21 窗口熵图

# ============================================================
# 方向二：SAI — 空间自适应注入  ✓ 开启
# ============================================================
use_sai = True

# ============================================================
# 方向三：EGAB — 熵引导注意力偏置  ✓ 开启
# ============================================================
enable_egab = True
egab_mode = 'post_attn'

# ============================================================
# 主线 B：HASH — 关闭
# ============================================================
enable_horizon_head = False

# ============================================================
# 数据集
# ============================================================
dataset_file = 'musid'
entropy_mode = 'entropy_ms'      # 多尺度熵图模式 (3 通道)

musid_img_dir     = "Hashmani's Dataset/MU-SID"
musid_entropy_dir = "Hashmani's Dataset/MU-SID_entropy_blue"
musid_split_dir   = 'splits_musid'

# ============================================================
# 训练
# ============================================================
batch_size_train = 4
batch_size_val = 8

epochs = 130
lr = 0.00025
lr_drop_list = [100, 115]
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
