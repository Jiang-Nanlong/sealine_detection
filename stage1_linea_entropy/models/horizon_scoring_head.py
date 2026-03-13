"""
horizon_scoring_head.py — Horizon-aware scoring head for LINEA_ENTROPY_B (v3)

Components:
  - SobelGradient              : Fixed-kernel image gradient computation
  - DualSideContextSampler     : Grid-sample based dual-side feature sampling along lines
  - MultiScaleContextSampler   : Multi-bandwidth wrapper over DualSideContextSampler
  - extract_geometry_features  : Geometry feature extraction from predicted lines
  - ScaleAttentionFusion       : Learnable scale selection across bandwidths
  - ContextTokenMixer          : Lightweight cross-modal token attention
  - MultiLayerQueryAggregator  : Learnable aggregation of multi-layer decoder queries
  - RawCalibrationHead         : Lightweight calibration for raw detector logits
  - HorizonScoringHead         : Main head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sobel gradient (fixed kernel, no trainable params)
# ============================================================
class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('rgb_weights',
                             torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    @torch.no_grad()
    def forward(self, images):
        gray = (images * self.rgb_weights).sum(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2).sqrt()


GEOM_DIM = 15


def extract_geometry_features(pred_lines):
    """[B, N, 4] -> [B, N, GEOM_DIM]"""
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
    return torch.stack([
        x1, y1, x2, y2, dx, dy, length,
        sin_a, cos_a, horizon_dev / (math.pi / 2),
        x_mid, y_mid, x_span, y_span,
        length * cos_a.abs(),
    ], dim=-1)


# ============================================================
# Dual-side context sampler (single bandwidth)
# ============================================================
class DualSideContextSampler(nn.Module):
    def __init__(self, num_sample_points=16, band_width=3.0):
        super().__init__()
        self.num_sample_points = num_sample_points
        self.band_width = band_width
        self.register_buffer('t_vals', torch.linspace(0, 1, num_sample_points))

    def forward(self, feature_map, pred_lines, img_h, img_w):
        """Returns center, upper, lower each [B, N, C]."""
        B, N, _ = pred_lines.shape
        K = self.num_sample_points
        x1, y1, x2, y2 = (pred_lines[..., i] for i in range(4))
        t = self.t_vals.view(1, 1, K)
        cx = x1.unsqueeze(-1) * (1 - t) + x2.unsqueeze(-1) * t
        cy = y1.unsqueeze(-1) * (1 - t) + y2.unsqueeze(-1) * t
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2).sqrt().clamp(min=1e-6)
        nx, ny = -dy / length, dx / length
        bw_x, bw_y = self.band_width / img_w, self.band_width / img_h
        ux = cx + bw_x * nx.unsqueeze(-1)
        uy = cy + bw_y * ny.unsqueeze(-1)
        lx = cx - bw_x * nx.unsqueeze(-1)
        ly = cy - bw_y * ny.unsqueeze(-1)

        def _sample(px, py):
            grid = torch.stack([px * 2 - 1, py * 2 - 1], dim=-1).reshape(B, N * K, 1, 2)
            s = F.grid_sample(feature_map, grid, mode='bilinear',
                              padding_mode='border', align_corners=False)
            C = s.shape[1]
            return s.squeeze(-1).reshape(B, C, N, K).mean(dim=-1).permute(0, 2, 1)

        return _sample(cx, cy), _sample(ux, uy), _sample(lx, ly)


# ============================================================
# Multi-bandwidth context sampler — returns per-scale results
# ============================================================
class MultiScaleContextSampler(nn.Module):
    def __init__(self, num_sample_points=16, band_widths=(2.0, 4.0, 8.0)):
        super().__init__()
        self.band_widths = list(band_widths)
        self.samplers = nn.ModuleList([
            DualSideContextSampler(num_sample_points, bw)
            for bw in self.band_widths
        ])

    def forward(self, feature_map, pred_lines, img_h, img_w):
        """Returns list of (center, upper, lower) tuples, one per bandwidth."""
        results = []
        for sampler in self.samplers:
            c, u, l = sampler(feature_map, pred_lines, img_h, img_w)
            results.append((c, u, l))
        return results

    @property
    def num_widths(self):
        return len(self.band_widths)


# ============================================================
# Scale attention fusion
# ============================================================
class ScaleAttentionFusion(nn.Module):
    """
    Given per-scale context embeddings [B, N, S, D],
    produces a fused embedding [B, N, D] via softmax-weighted pooling.
    The attention logits are computed from each embedding + a learnable query.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, x):
        """x: [B, N, S, D] -> [B, N, D]"""
        logits = self.attn_proj(x).squeeze(-1)  # [B, N, S]
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # [B, N, S, 1]
        return (x * weights).sum(dim=2)


# ============================================================
# Context token mixer (lightweight cross-modal attention)
# ============================================================
class ContextTokenMixer(nn.Module):
    """
    Takes a set of context tokens [B, N, T, D] and mixes them via
    one layer of multi-head self-attention, then pools to [B, N, D].
    Light: single layer, small nheads.
    """
    def __init__(self, d_model, nheads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(self, tokens):
        """tokens: [B*N, T, D] -> [B*N, D]"""
        out = self.attn(tokens, tokens, tokens, need_weights=False)[0]
        out = self.norm(out + tokens)
        return self.pool_proj(out.mean(dim=1))


# ============================================================
# Multi-layer query aggregator
# ============================================================
class MultiLayerQueryAggregator(nn.Module):
    """
    Aggregates the last K layers of decoder hidden states
    via learnable weighted sum.
    """
    def __init__(self, num_layers=3):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hs_stack):
        """
        hs_stack: [L, B, N, D] — all decoder layer hidden states
        Uses the last num_layers layers.
        Returns: [B, N, D]
        """
        K = self.layer_weights.shape[0]
        hs_sel = hs_stack[-K:]  # [K, B, N, D]
        w = F.softmax(self.layer_weights, dim=0)  # [K]
        return torch.einsum('k,kbnd->bnd', w, hs_sel)


# ============================================================
# Raw logit calibration head
# ============================================================
class RawCalibrationHead(nn.Module):
    """
    Lightweight per-candidate calibration of raw detector logits.
    calibrated = raw * scale + bias, where scale/bias depend on query feat.
    """
    def __init__(self, d_model, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, num_classes * 2),  # scale + bias per class
        )
        # init to identity: scale=1, bias=0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, query_feat, raw_logits):
        """
        query_feat: [B, N, D]
        raw_logits: [B, N, C]
        Returns: calibrated logits [B, N, C]
        """
        C = raw_logits.shape[-1]
        params = self.proj(query_feat)  # [B, N, C*2]
        scale, bias = params[..., :C], params[..., C:]
        return raw_logits * (1.0 + torch.tanh(scale)) + bias


# ============================================================
# Horizon-aware scoring head (v3 — ultimate)
# ============================================================
class HorizonScoringHead(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256, num_classes=2,
                 num_sample_points=16, band_widths=(2.0, 4.0, 8.0),
                 use_feat_context=True, use_entropy_context=True,
                 use_gradient_context=True, use_geometry=True,
                 use_adaptive_fusion_gate=True,
                 use_multilayer_query=True, num_query_layers=3,
                 use_scale_attention=True,
                 score_init_scale=0.1):
        super().__init__()
        self.use_feat_context = use_feat_context
        self.use_entropy_context = use_entropy_context
        self.use_gradient_context = use_gradient_context
        self.use_geometry = use_geometry
        self.use_adaptive_fusion_gate = use_adaptive_fusion_gate
        self.use_multilayer_query = use_multilayer_query
        self.use_scale_attention = use_scale_attention
        self.num_classes = num_classes

        n_widths = len(band_widths)

        # --- Multi-layer query aggregator ---
        if use_multilayer_query:
            self.query_agg = MultiLayerQueryAggregator(num_query_layers)

        # --- Gradient computation ---
        if use_gradient_context:
            self.sobel = SobelGradient()

        # --- Per-modality: sampler + per-scale encoder + scale attention ---
        # We encode each (c, u, l) triple per scale into hidden_dim,
        # then fuse across scales via scale attention.

        # Feature context
        if use_feat_context:
            self.feat_sampler = MultiScaleContextSampler(num_sample_points, band_widths)
            self.feat_scale_encs = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model * 3, hidden_dim), nn.ReLU(inplace=True))
                for _ in range(n_widths)
            ])
            if use_scale_attention:
                self.feat_scale_attn = ScaleAttentionFusion(hidden_dim)

        # Entropy context
        if use_entropy_context:
            self.ent_sampler = MultiScaleContextSampler(num_sample_points, band_widths)
            ent_triple = 3  # 1-ch * 3 (c, u, l)
            ent_enc_dim = max(32, hidden_dim // 4)
            self.ent_scale_encs = nn.ModuleList([
                nn.Sequential(nn.Linear(ent_triple, ent_enc_dim), nn.ReLU(inplace=True))
                for _ in range(n_widths)
            ])
            if use_scale_attention:
                self.ent_scale_attn = ScaleAttentionFusion(ent_enc_dim)
            self._ent_enc_dim = ent_enc_dim

        # Gradient context
        if use_gradient_context:
            self.grad_sampler = MultiScaleContextSampler(num_sample_points, band_widths)
            grad_triple = 3
            grad_enc_dim = max(32, hidden_dim // 4)
            self.grad_scale_encs = nn.ModuleList([
                nn.Sequential(nn.Linear(grad_triple, grad_enc_dim), nn.ReLU(inplace=True))
                for _ in range(n_widths)
            ])
            if use_scale_attention:
                self.grad_scale_attn = ScaleAttentionFusion(grad_enc_dim)
            self._grad_enc_dim = grad_enc_dim

        # --- Context token mixer ---
        # Collect all modality embeddings as tokens and mix them
        n_tokens = 0
        token_dim = hidden_dim  # all tokens projected to this dim
        self._token_dim = token_dim
        if use_feat_context:
            n_tokens += 1
            # feat is already hidden_dim
        if use_entropy_context:
            n_tokens += 1
            self.ent_to_token = nn.Linear(ent_enc_dim, token_dim)
        if use_gradient_context:
            n_tokens += 1
            self.grad_to_token = nn.Linear(grad_enc_dim, token_dim)
        self._n_ctx_tokens = n_tokens

        if n_tokens >= 2:
            self.ctx_mixer = ContextTokenMixer(token_dim, nheads=4)
            ctx_out_dim = token_dim
        else:
            self.ctx_mixer = None
            ctx_out_dim = hidden_dim if use_feat_context else (
                ent_enc_dim if use_entropy_context else (
                    grad_enc_dim if use_gradient_context else 0))

        # --- Compute total input dim for the scoring MLP ---
        in_dim = d_model  # query feature
        if n_tokens > 0:
            in_dim += ctx_out_dim
        if use_geometry:
            in_dim += GEOM_DIM

        # --- Scoring MLP (produces horizon_delta) ---
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

        # --- Raw logit calibration ---
        self.raw_calibration = RawCalibrationHead(d_model, num_classes)

        # --- Per-candidate fusion gate (scalar per candidate) ---
        if use_adaptive_fusion_gate:
            gate_in = d_model  # query feat
            if use_geometry:
                gate_in += GEOM_DIM
            if n_tokens > 0:
                gate_in += ctx_out_dim
            self.fusion_gate_head = nn.Sequential(
                nn.Linear(gate_in, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),  # candidate-level single gate
            )
            nn.init.zeros_(self.fusion_gate_head[-1].weight)
            nn.init.zeros_(self.fusion_gate_head[-1].bias)

    def _encode_modality_scales(self, scale_results, scale_encs, scale_attn, B, N):
        """
        Encode per-scale (c, u, l) triples and fuse across scales.
        scale_results: list of (center, upper, lower) tuples, one per bandwidth
        Returns: [B, N, enc_dim]
        """
        S = len(scale_results)
        scale_embs = []
        for s_idx, (c, u, l) in enumerate(scale_results):
            triple = torch.cat([c, u, l], dim=-1)  # [B, N, C*3]
            emb = scale_encs[s_idx](triple)         # [B, N, enc_dim]
            scale_embs.append(emb)

        if S == 1 or scale_attn is None:
            # no scale attention: just mean pool
            return torch.stack(scale_embs, dim=2).mean(dim=2)  # [B, N, enc_dim]

        stacked = torch.stack(scale_embs, dim=2)  # [B, N, S, enc_dim]
        return scale_attn(stacked)

    def forward(self, hs_stack, pred_lines, raw_logits, encoder_feat, images,
                entropy_map=None, img_h=640, img_w=640):
        """
        Args:
            hs_stack     : [L, B, N, d_model] — all decoder layer hidden states
            pred_lines   : [B, N, 4]
            raw_logits   : [B, N, num_classes] — original detector logits
            encoder_feat : [B, C, fH, fW]
            images       : [B, 3, H, W]
            entropy_map  : [B, 1, H, W] or None
        Returns:
            horizon_delta      : [B, N, num_classes]
            fusion_gate        : [B, N, 1] or None
            raw_calibrated     : [B, N, num_classes]
            score_scale        : scalar
        """
        pred_lines_d = pred_lines.detach()

        # --- Multi-layer query aggregation ---
        if self.use_multilayer_query:
            query_feat = self.query_agg(hs_stack)  # [B, N, D]
        else:
            query_feat = hs_stack[-1]  # last layer: [B, N, D]

        B, N = query_feat.shape[:2]

        # --- Per-modality context extraction with scale attention ---
        ctx_tokens = []

        if self.use_feat_context:
            feat_scales = self.feat_sampler(encoder_feat, pred_lines_d, img_h, img_w)
            feat_attn = self.feat_scale_attn if self.use_scale_attention else None
            feat_enc = self._encode_modality_scales(
                feat_scales, self.feat_scale_encs, feat_attn, B, N)
            ctx_tokens.append(feat_enc)  # [B, N, hidden_dim]

        if self.use_entropy_context:
            if entropy_map is not None:
                ent_scales = self.ent_sampler(entropy_map, pred_lines_d, img_h, img_w)
                ent_attn = self.ent_scale_attn if self.use_scale_attention else None
                ent_enc = self._encode_modality_scales(
                    ent_scales, self.ent_scale_encs, ent_attn, B, N)
            else:
                ent_enc = torch.zeros(B, N, self._ent_enc_dim, device=query_feat.device)
            ctx_tokens.append(
                self.ent_to_token(ent_enc) if hasattr(self, 'ent_to_token') else ent_enc)

        if self.use_gradient_context:
            grad_map = self.sobel(images)
            grad_scales = self.grad_sampler(grad_map, pred_lines_d, img_h, img_w)
            grad_attn = self.grad_scale_attn if self.use_scale_attention else None
            grad_enc = self._encode_modality_scales(
                grad_scales, self.grad_scale_encs, grad_attn, B, N)
            ctx_tokens.append(
                self.grad_to_token(grad_enc) if hasattr(self, 'grad_to_token') else grad_enc)

        # --- Context token mixing ---
        if len(ctx_tokens) >= 2 and self.ctx_mixer is not None:
            tokens = torch.stack(ctx_tokens, dim=2)  # [B, N, T, D]
            BN = B * N
            tokens_flat = tokens.reshape(BN, len(ctx_tokens), self._token_dim)
            ctx_mixed = self.ctx_mixer(tokens_flat).reshape(B, N, self._token_dim)
        elif len(ctx_tokens) == 1:
            ctx_mixed = ctx_tokens[0]
        else:
            ctx_mixed = None

        # --- Assemble head input ---
        head_parts = [query_feat]
        gate_parts = [query_feat]

        if ctx_mixed is not None:
            head_parts.append(ctx_mixed)
            gate_parts.append(ctx_mixed)

        geom = None
        if self.use_geometry:
            geom = extract_geometry_features(pred_lines_d)
            head_parts.append(geom)
            gate_parts.append(geom)

        # --- Horizon delta ---
        x = torch.cat(head_parts, dim=-1)
        horizon_delta = self.head(x)

        # --- Raw calibration ---
        raw_calibrated = self.raw_calibration(query_feat, raw_logits)

        # --- Fusion gate (candidate-level, scalar) ---
        fusion_gate = None
        if self.use_adaptive_fusion_gate:
            gate_x = torch.cat(gate_parts, dim=-1)
            fusion_gate = torch.sigmoid(self.fusion_gate_head(gate_x))  # [B, N, 1]

        return horizon_delta, fusion_gate, raw_calibrated, self.score_scale
