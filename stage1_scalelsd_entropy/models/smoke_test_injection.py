"""
smoke_test_injection.py — path_1 注入冒烟测试
Smoke test: verify entropy injection at DPT path_1.

测试内容：
  1. 创建 ScaleLSDWithEntropy，打印模型参数量
  2. baseline 前向：entropy_map=None → 输出 shape 与原 ScaleLSD 一致
  3. entropy 前向：entropy_map=rand → 输出 shape 不变
  4. 验证 alpha 初始为 0 → 两种模式输出完全相等
  5. 修改 alpha → 两种模式输出不再相等

直接运行：python -m stage1_scalelsd_entropy.models.smoke_test_injection
"""

import torch
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stage1_scalelsd_entropy.models.scalelsd_with_entropy import ScaleLSDWithEntropy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke_test] device = {device}")

    # ---- 构建模型 ----
    model = ScaleLSDWithEntropy(gray_scale=True, use_layer_scale=False)
    model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    entropy_params = sum(p.numel() for n, p in model.named_parameters() if "entropy_branch" in n or n.endswith("alpha"))
    print(f"[smoke_test] total params     : {total_params:,}")
    print(f"[smoke_test] entropy params   : {entropy_params:,}")
    print(f"[smoke_test] alpha            : {model.backbone.alpha.item():.4f}")

    # ---- 准备输入 ----
    B, C, H, W = 1, 1, 576, 1024
    images = torch.randn(B, C, H, W, device=device)
    entropy_map = torch.rand(B, 1, H, W, device=device)

    with torch.no_grad():
        # ---- test 1: baseline (no entropy) ----
        out_base, feat_base, aux_base = model.forward_backbone(images, entropy_map=None)
        print(f"[smoke_test] baseline output  : {tuple(out_base.shape)}")
        assert out_base.shape == (B, 9, H // 2, W // 2), f"shape mismatch: {out_base.shape}"

        # ---- test 2: with entropy ----
        out_ent, feat_ent, aux_ent = model.forward_backbone(images, entropy_map=entropy_map)
        print(f"[smoke_test] entropy output   : {tuple(out_ent.shape)}")
        assert out_ent.shape == out_base.shape, f"shape mismatch: {out_ent.shape}"

        # ---- test 3: alpha=0 → 两者完全相等 ----
        diff_zero = (out_base - out_ent).abs().max().item()
        print(f"[smoke_test] max diff (alpha=0): {diff_zero:.6e}")
        assert diff_zero < 1e-5, f"outputs should be identical when alpha=0, got max diff {diff_zero}"

    # ---- test 4: alpha≠0 → 两者不再相等 ----
    with torch.no_grad():
        model.backbone.alpha.fill_(1.0)
        out_ent2, _, _ = model.forward_backbone(images, entropy_map=entropy_map)
        diff_nonzero = (out_base - out_ent2).abs().max().item()
        print(f"[smoke_test] max diff (alpha=1): {diff_nonzero:.6e}")
        assert diff_nonzero > 1e-3, f"outputs should differ when alpha=1, got max diff {diff_nonzero}"

    print("[smoke_test] ALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()
