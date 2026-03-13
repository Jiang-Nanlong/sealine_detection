"""
horizon_scoring_head.py — Horizon-aware scoring head for LINEA_ENTROPY_B

Components:
  - SobelGradient       : Fixed-kernel image gradient computation
  - DualSideContextSampler : Grid-sample based dual-side feature sampling along lines
  - extract_geometry_features : Geometry feature extraction from predicted lines
  - HorizonScoringHead  : Main head combining query feat, geometry, and context
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


# ============================================================
# Geometry feature extraction
# ============================================================
def extract_geometry_features(pred_lines):
    """
    Extract normalized geometry features from predicted line segments.

    Args:
        pred_lines: [B, N, 4] normalized [0,1] coords (x1, y1, x2, y2)
    Returns:
        geom: [B, N, 14]
    """
    x1, y1, x2, y2 = pred_lines.unbind(dim=-1)
    dx = x2 - x1
    dy = y2 - y1
    length = (dx ** 2 + dy ** 2).sqrt().clamp(min=1e-6)
    angle = torch.atan2(dy, dx)
    abs_slope = dy.abs() / dx.abs().clamp(min=1e-6)
    abs_slope = abs_slope.clamp(max=10.0)
    horizon_dev = torch.abs(angle)
    horizon_dev = torch.min(horizon_dev, math.pi - horizon_dev)
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    x_span = torch.abs(dx)
    y_span = torch.abs(dy)

    geom = torch.stack([
        x1, y1, x2, y2,
        dx, dy, length,
        angle / math.pi,
        abs_slope / 10.0,
        horizon_dev / (math.pi / 2),
        x_mid, y_mid, x_span, y_span,
    ], dim=-1)
    return geom


# ============================================================
# Dual-side context sampler
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
        device = pred_lines.device

        x1 = pred_lines[..., 0]
        y1 = pred_lines[..., 1]
        x2 = pred_lines[..., 2]
        y2 = pred_lines[..., 3]

        t = self.t_vals.view(1, 1, K)

        # center sample points [B, N, K]
        cx = x1.unsqueeze(-1) * (1 - t) + x2.unsqueeze(-1) * t
        cy = y1.unsqueeze(-1) * (1 - t) + y2.unsqueeze(-1) * t

        # normal direction
        dx = x2 - x1
        dy = y2 - y1
        length = (dx ** 2 + dy ** 2).sqrt().clamp(min=1e-6)
        nx = -dy / length  # [B, N]
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
            grid = torch.stack([gx, gy], dim=-1)  # [B, N, K, 2]
            grid = grid.reshape(B, N * K, 1, 2)
            sampled = F.grid_sample(feature_map, grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)
            # [B, C, N*K, 1] → [B, C, N, K] → mean over K → [B, N, C]
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
# Horizon-aware scoring head
# ============================================================
class HorizonScoringHead(nn.Module):
    """
    Combines query features, geometry features, and dual-side context
    to produce horizon-aware logits for each candidate line.
    """

    def __init__(self, d_model=256, hidden_dim=256, num_classes=2,
                 num_sample_points=16, band_width=3.0,
                 use_feat_context=True, use_entropy_context=True,
                 use_gradient_context=True, use_geometry=True,
                 score_init_scale=0.1):
        super().__init__()
        self.use_feat_context = use_feat_context
        self.use_entropy_context = use_entropy_context
        self.use_gradient_context = use_gradient_context
        self.use_geometry = use_geometry

        self.sampler = DualSideContextSampler(num_sample_points, band_width)

        if use_gradient_context:
            self.sobel = SobelGradient()

        # --- compute input dim ---
        in_dim = d_model  # query feature always present

        if use_feat_context:
            self.feat_ctx_proj = nn.Sequential(
                nn.Linear(d_model * 3, hidden_dim),
                nn.ReLU(inplace=True),
            )
            in_dim += hidden_dim

        if use_entropy_context:
            in_dim += 5  # center, upper, lower, diff, abs_diff (1-ch each)

        if use_gradient_context:
            in_dim += 5

        if use_geometry:
            in_dim += 14

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        # init last layer to near-zero output
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.score_scale = nn.Parameter(torch.tensor(score_init_scale))

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
            score_scale   : scalar
        """
        pred_lines_d = pred_lines.detach()
        parts = [hs_last]

        if self.use_feat_context:
            c, _u, _l, d, ad = self.sampler(encoder_feat, pred_lines_d, img_h, img_w)
            feat_ctx = torch.cat([c, d, ad], dim=-1)  # [B, N, d_model*3]
            feat_ctx = self.feat_ctx_proj(feat_ctx)     # [B, N, hidden_dim]
            parts.append(feat_ctx)

        if self.use_entropy_context:
            if entropy_map is not None:
                ec, eu, el, ed, ead = self.sampler(entropy_map, pred_lines_d, img_h, img_w)
                ent_ctx = torch.cat([ec, eu, el, ed, ead], dim=-1)  # [B, N, 5]
            else:
                B, N = hs_last.shape[:2]
                ent_ctx = torch.zeros(B, N, 5, device=hs_last.device)
            parts.append(ent_ctx)

        if self.use_gradient_context:
            grad_map = self.sobel(images)
            gc, gu, gl, gd, gad = self.sampler(grad_map, pred_lines_d, img_h, img_w)
            grad_ctx = torch.cat([gc, gu, gl, gd, gad], dim=-1)  # [B, N, 5]
            parts.append(grad_ctx)

        if self.use_geometry:
            geom = extract_geometry_features(pred_lines_d)
            parts.append(geom)

        x = torch.cat(parts, dim=-1)
        horizon_logit = self.head(x)
        return horizon_logit, self.score_scale
