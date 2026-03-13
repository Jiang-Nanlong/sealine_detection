"""
horizon_scoring_head.py — Horizon-aware scoring head for LINEA_ENTROPY_B

Components:
  - SobelGradient              : Fixed-kernel image gradient computation
  - DualSideContextSampler     : Grid-sample based dual-side feature sampling along lines
  - MultiScaleContextSampler   : Multi-bandwidth wrapper over DualSideContextSampler
  - extract_geometry_features  : Geometry feature extraction from predicted lines
  - HorizonScoringHead         : Main head combining query feat, geometry, and context
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sobel gradient (fixed kernel, no trainable params)
# ============================================================
class SobelGradient(nn.Module):
    """
    Computes image gradient magnitude using fixed Sobel kernels.
    Input:  [B, 3, H, W] RGB images
    Output: [B, 1, H, W] gradient magnitude
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('rgb_weights',
                             torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    @torch.no_grad()
    def forward(self, images):
        gray = (images * self.rgb_weights).sum(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = (gx ** 2 + gy ** 2).sqrt()
        return mag


GEOM_DIM = 15  # updated geometry feature dimension


# ============================================================
# Geometry feature extraction (enhanced)
# ============================================================
def extract_geometry_features(pred_lines):
    """
    Extract normalized geometry features from predicted line segments.

    Args:
        pred_lines: [B, N, 4] normalized [0,1] coords (x1, y1, x2, y2)
    Returns:
        geom: [B, N, GEOM_DIM]
    """
    x1, y1, x2, y2 = pred_lines.unbind(dim=-1)
    dx = x2 - x1
    dy = y2 - y1
    length = (dx ** 2 + dy ** 2).sqrt().clamp(min=1e-6)
    angle = torch.atan2(dy, dx)
    sin_a = torch.sin(angle)
    cos_a = torch.cos(angle)
    horizon_dev = torch.abs(angle)
    horizon_dev = torch.min(horizon_dev, math.pi - horizon_dev)
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    x_span = torch.abs(dx)
    y_span = torch.abs(dy)

    geom = torch.stack([
        x1, y1, x2, y2,
        dx, dy, length,
        sin_a, cos_a,
        horizon_dev / (math.pi / 2),
        x_mid, y_mid, x_span, y_span,
        length * cos_a.abs(),  # horizontal projection of length
    ], dim=-1)
    return geom


# ============================================================
# Dual-side context sampler (single bandwidth)
# ============================================================
class DualSideContextSampler(nn.Module):
    """
    Samples features along and around predicted line segments.
    For each line, samples K points along the center, upper band, and lower band.
    Returns pooled (mean over K) features per line.
    """

    def __init__(self, num_sample_points=16, band_width=3.0):
        super().__init__()
        self.num_sample_points = num_sample_points
        self.band_width = band_width
        self.register_buffer('t_vals', torch.linspace(0, 1, num_sample_points))

    def forward(self, feature_map, pred_lines, img_h, img_w):
        """
        Args:
            feature_map : [B, C, fH, fW]
            pred_lines  : [B, N, 4] normalized [0,1] (x1,y1,x2,y2), detached
            img_h, img_w: int, image spatial size (for band_width scaling)
        Returns:
            center_feat, upper_feat, lower_feat, diff_feat, abs_diff_feat
            each [B, N, C]
        """
        B, N, _ = pred_lines.shape
        K = self.num_sample_points

        x1 = pred_lines[..., 0]
        y1 = pred_lines[..., 1]
        x2 = pred_lines[..., 2]
        y2 = pred_lines[..., 3]

        t = self.t_vals.view(1, 1, K)

        cx = x1.unsqueeze(-1) * (1 - t) + x2.unsqueeze(-1) * t
        cy = y1.unsqueeze(-1) * (1 - t) + y2.unsqueeze(-1) * t

        dx = x2 - x1
        dy = y2 - y1
        length = (dx ** 2 + dy ** 2).sqrt().clamp(min=1e-6)
        nx = -dy / length
        ny = dx / length

        bw_x = self.band_width / img_w
        bw_y = self.band_width / img_h

        ux = cx + bw_x * nx.unsqueeze(-1)
        uy = cy + bw_y * ny.unsqueeze(-1)
        lx = cx - bw_x * nx.unsqueeze(-1)
        ly = cy - bw_y * ny.unsqueeze(-1)

        def _sample(px, py):
            gx = px * 2 - 1
            gy = py * 2 - 1
            grid = torch.stack([gx, gy], dim=-1)
            grid = grid.reshape(B, N * K, 1, 2)
            sampled = F.grid_sample(feature_map, grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            C = sampled.shape[1]
            sampled = sampled.squeeze(-1).reshape(B, C, N, K)
            return sampled.mean(dim=-1).permute(0, 2, 1)

        center_feat = _sample(cx, cy)
        upper_feat = _sample(ux, uy)
        lower_feat = _sample(lx, ly)
        diff_feat = upper_feat - lower_feat
        abs_diff_feat = diff_feat.abs()

        return center_feat, upper_feat, lower_feat, diff_feat, abs_diff_feat


# ============================================================
# Multi-bandwidth context sampler
# ============================================================
class MultiScaleContextSampler(nn.Module):
    """
    Runs DualSideContextSampler at multiple bandwidths and concatenates results.
    For each bandwidth, produces 5 output tensors (c, u, l, d, ad).
    Returns concatenation along last dim: [B, N, C * 5 * num_widths].
    """

    def __init__(self, num_sample_points=16, band_widths=(2.0, 4.0, 8.0)):
        super().__init__()
        self.band_widths = list(band_widths)
        self.samplers = nn.ModuleList([
            DualSideContextSampler(num_sample_points, bw)
            for bw in self.band_widths
        ])

    def forward(self, feature_map, pred_lines, img_h, img_w):
        """
        Returns:
            cat of [center, upper, lower, diff, abs_diff] from all widths
            shape: [B, N, C * 5 * len(band_widths)]
        """
        all_parts = []
        for sampler in self.samplers:
            c, u, l, d, ad = sampler(feature_map, pred_lines, img_h, img_w)
            all_parts.extend([c, u, l, d, ad])
        return torch.cat(all_parts, dim=-1)

    @property
    def num_widths(self):
        return len(self.band_widths)


# ============================================================
# Horizon-aware scoring head (enhanced)
# ============================================================
class HorizonScoringHead(nn.Module):
    """
    Combines query features, geometry features, and dual-side context
    to produce horizon-aware logits for each candidate line.

    Enhancements over v1:
      - Multi-bandwidth dual-side sampling
      - Learnable encoders for entropy/gradient context
      - Per-candidate adaptive fusion gate
    """

    def __init__(self, d_model=256, hidden_dim=256, num_classes=2,
                 num_sample_points=16, band_widths=(2.0, 4.0, 8.0),
                 use_feat_context=True, use_entropy_context=True,
                 use_gradient_context=True, use_geometry=True,
                 use_adaptive_fusion_gate=True,
                 score_init_scale=0.1):
        super().__init__()
        self.use_feat_context = use_feat_context
        self.use_entropy_context = use_entropy_context
        self.use_gradient_context = use_gradient_context
        self.use_geometry = use_geometry
        self.use_adaptive_fusion_gate = use_adaptive_fusion_gate
        self.num_classes = num_classes

        n_widths = len(band_widths)

        if use_gradient_context:
            self.sobel = SobelGradient()

        # --- Feature context: multi-bandwidth sampler + projection ---
        if use_feat_context:
            self.feat_sampler = MultiScaleContextSampler(num_sample_points, band_widths)
            # 5 parts (c,u,l,d,ad) * n_widths * d_model channels
            feat_ctx_raw = d_model * 5 * n_widths
            self.feat_ctx_proj = nn.Sequential(
                nn.Linear(feat_ctx_raw, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )

        # --- Entropy context: multi-bandwidth + learnable encoder ---
        if use_entropy_context:
            self.ent_sampler = MultiScaleContextSampler(num_sample_points, band_widths)
            ent_ctx_raw = 1 * 5 * n_widths  # 1-ch entropy map
            ent_proj_dim = max(32, ent_ctx_raw)
            self.entropy_ctx_proj = nn.Sequential(
                nn.Linear(ent_ctx_raw, ent_proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ent_proj_dim, ent_proj_dim),
                nn.ReLU(inplace=True),
            )
            self._ent_proj_dim = ent_proj_dim

        # --- Gradient context: multi-bandwidth + learnable encoder ---
        if use_gradient_context:
            self.grad_sampler = MultiScaleContextSampler(num_sample_points, band_widths)
            grad_ctx_raw = 1 * 5 * n_widths
            grad_proj_dim = max(32, grad_ctx_raw)
            self.gradient_ctx_proj = nn.Sequential(
                nn.Linear(grad_ctx_raw, grad_proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(grad_proj_dim, grad_proj_dim),
                nn.ReLU(inplace=True),
            )
            self._grad_proj_dim = grad_proj_dim

        # --- Compute total input dim for the scoring MLP ---
        in_dim = d_model  # query feature always present

        if use_feat_context:
            in_dim += hidden_dim
        if use_entropy_context:
            in_dim += self._ent_proj_dim
        if use_gradient_context:
            in_dim += self._grad_proj_dim
        if use_geometry:
            in_dim += GEOM_DIM

        # --- Scoring MLP ---
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # --- Global scale (auxiliary) ---
        self.score_scale = nn.Parameter(torch.tensor(score_init_scale))

        # --- Per-candidate adaptive fusion gate ---
        if use_adaptive_fusion_gate:
            # gate input: query feat + geometry (always available) + encoded contexts
            gate_in_dim = d_model
            if use_geometry:
                gate_in_dim += GEOM_DIM
            if use_feat_context:
                gate_in_dim += hidden_dim
            if use_entropy_context:
                gate_in_dim += self._ent_proj_dim
            if use_gradient_context:
                gate_in_dim += self._grad_proj_dim

            self.fusion_gate_head = nn.Sequential(
                nn.Linear(gate_in_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            # init to produce ~0.5 gate (sigmoid(0) = 0.5)
            nn.init.zeros_(self.fusion_gate_head[-1].weight)
            nn.init.zeros_(self.fusion_gate_head[-1].bias)

    def forward(self, hs_last, pred_lines, encoder_feat, images,
                entropy_map=None, img_h=640, img_w=640):
        """
        Args:
            hs_last      : [B, N, d_model] — last decoder layer hidden states
            pred_lines   : [B, N, 4] — normalized [0,1] line coords
            encoder_feat : [B, C, fH, fW] — high-res encoder feature (features[0])
            images       : [B, 3, H, W] — input images
            entropy_map  : [B, 1, H, W] or None
            img_h, img_w : int
        Returns:
            horizon_logit : [B, N, num_classes]
            fusion_gate   : [B, N, num_classes] or None (when not adaptive)
            score_scale   : scalar
        """
        pred_lines_d = pred_lines.detach()
        head_parts = [hs_last]
        gate_parts = [hs_last]
        B, N = hs_last.shape[:2]

        # --- Feature context ---
        feat_ctx_enc = None
        if self.use_feat_context:
            feat_raw = self.feat_sampler(encoder_feat, pred_lines_d, img_h, img_w)
            feat_ctx_enc = self.feat_ctx_proj(feat_raw)
            head_parts.append(feat_ctx_enc)
            gate_parts.append(feat_ctx_enc)

        # --- Entropy context ---
        ent_ctx_enc = None
        if self.use_entropy_context:
            if entropy_map is not None:
                ent_raw = self.ent_sampler(entropy_map, pred_lines_d, img_h, img_w)
                ent_ctx_enc = self.entropy_ctx_proj(ent_raw)
            else:
                ent_ctx_enc = torch.zeros(B, N, self._ent_proj_dim, device=hs_last.device)
            head_parts.append(ent_ctx_enc)
            gate_parts.append(ent_ctx_enc)

        # --- Gradient context ---
        grad_ctx_enc = None
        if self.use_gradient_context:
            grad_map = self.sobel(images)
            grad_raw = self.grad_sampler(grad_map, pred_lines_d, img_h, img_w)
            grad_ctx_enc = self.gradient_ctx_proj(grad_raw)
            head_parts.append(grad_ctx_enc)
            gate_parts.append(grad_ctx_enc)

        # --- Geometry ---
        geom = None
        if self.use_geometry:
            geom = extract_geometry_features(pred_lines_d)
            head_parts.append(geom)
            gate_parts.append(geom)

        # --- Scoring MLP ---
        x = torch.cat(head_parts, dim=-1)
        horizon_logit = self.head(x)

        # --- Fusion gate ---
        fusion_gate = None
        if self.use_adaptive_fusion_gate:
            gate_x = torch.cat(gate_parts, dim=-1)
            fusion_gate = torch.sigmoid(self.fusion_gate_head(gate_x))  # [B, N, num_classes]

        return horizon_logit, fusion_gate, self.score_scale
