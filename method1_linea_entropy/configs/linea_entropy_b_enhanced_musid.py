"""
linea_entropy_b_enhanced_musid.py — MSLEP + SAI + EGAB + HASH + Ranking 增强版，MU-SID 训练配置

用法：
  python LINEA/main.py -c method1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py

模型：LINEA_ENTROPY_B_ENHANCED
  - MSLEP: 多尺度局部熵先验提取 (3 通道: 5×5, 11×11, 21×21 窗口)
  - SAI:   空间自适应注入 (sigmoid attention-weighted injection)
  - EGAB:  熵引导注意力偏置 (post-attention residual correction)
  - HASH:  Horizon-Aware Scoring Head (内部重评分, tanh 限幅)
  - Ranking: 图内排序损失 (detector-side pairwise ranking)
数据：MU-SID，letterbox 到 640×640，entropy_mode='entropy_ms'
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
output_dir = 'output/linea_entropy_b_enhanced_musid_v2_e130'

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

# ---- criterion ----
criterionname = 'LINEACRITERION_RANKING'
weight_dict = {'loss_logits': 4, 'loss_line': 5, 'loss_ranking': 2}
losses = ['labels', 'lines', 'ranking']
ranking_margin = 0.3
ranking_num_pairs = 50

# ============================================================
# 方向一：MSLEP — 多尺度局部熵先验提取
# ============================================================
entropy_in_channels = 3          # 3 通道: 5×5, 11×11, 21×21 窗口熵图

# ============================================================
# 方向二：SAI — 空间自适应注入
# ============================================================
use_sai = True                   # True: SAI 注入; False: 退化为 alpha 加法注入

# ============================================================
# 方向三：EGAB — 熵引导注意力偏置
# ============================================================
enable_egab = True
egab_mode = 'post_attn'          # 'post_attn' 或 'memory_bias'

# ============================================================
# 主线 B：HASH — Horizon-Aware Scoring Head
# ============================================================
enable_horizon_head = True
horizon_num_sample_points = 16
horizon_band_widths = [2.0, 4.0, 8.0]
horizon_use_feat_context = True
horizon_use_entropy_context = True
horizon_use_gradient_context = True
horizon_use_geometry = True
horizon_use_adaptive_fusion_gate = False   # 禁用 gate（防塌缩）
horizon_use_multilayer_query = True
horizon_num_query_layers = 3
horizon_use_scale_attention = True
horizon_score_init_scale = 0.1             # 限制 delta 幅度（配合 tanh）
horizon_clamp_delta = True                 # tanh 限幅 horizon_delta ∈ [-1, 1]
horizon_hidden_dim = 256
horizon_use_raw_calibration = False        # 禁用 calibration 快捷路径

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
