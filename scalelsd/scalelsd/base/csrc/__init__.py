from torch.utils.cpp_extension import load 
import glob
import os.path as osp
import torch

__this__ = osp.dirname(__file__)

try:
    _C = load(name='_C',sources=[
        osp.join(__this__,'binding.cpp'),
        osp.join(__this__,'linesegment.cu'),
    ]
    )
except Exception as _e:
    import warnings
    warnings.warn(f"[scalelsd] CUDA extension failed to compile: {_e}. "
                  "Using pure-PyTorch fallback for encodels.", RuntimeWarning)
    _C = None

# ── pure-PyTorch fallback ────────────────────────────────────────────
if _C is None:
    class _FallbackC:
        """Drop-in replacement for the _C CUDA extension (encodels only)."""

        @staticmethod
        def encodels(lines, input_height, input_width, height, width, num_lines):
            device = lines.device
            N = num_lines
            HW = height * width

            lmap  = torch.zeros((6, height, width), device=device, dtype=torch.float32)
            label = torch.full((1, height, width), -1, device=device, dtype=torch.int32)
            tmap  = torch.zeros((1, height, width), device=device, dtype=torch.float32)

            if N == 0:
                return lmap, label, tmap

            xs = float(width)  / float(input_width)
            ys = float(height) / float(input_height)

            # pixel grid  [HW]
            gy = torch.arange(height, device=device, dtype=torch.float32)
            gx = torch.arange(width,  device=device, dtype=torch.float32)
            py, px = torch.meshgrid(gy, gx, indexing='ij')
            px = px.reshape(-1)          # [HW]
            py = py.reshape(-1)

            # scaled endpoints  [N]
            x1 = lines[:, 0] * xs;  y1 = lines[:, 1] * ys
            x2 = lines[:, 2] * xs;  y2 = lines[:, 3] * ys
            dx = x2 - x1;           dy = y2 - y1
            norm2 = dx * dx + dy * dy                   # [N]

            # broadcast  lines [N,1]  pixels [1,HW]
            px_ = px[None, :];  py_ = py[None, :]
            x1_ = x1[:, None];  y1_ = y1[:, None]
            x2_ = x2[:, None];  y2_ = y2[:, None]
            dx_ = dx[:, None];  dy_ = dy[:, None]

            # parameter t  [N, HW]
            t_raw = ((px_ - x1_) * dx_ + (py_ - y1_) * dy_) / (norm2[:, None] + 1e-6)
            in_seg = (t_raw >= 0) & (t_raw <= 1)        # [N, HW]
            t_c = t_raw.clamp(0.0, 1.0)

            # vector pixel → nearest point on line  [N, HW]
            ax = x1_ + t_c * (x2_ - x1_) - px_
            ay = y1_ + t_c * (y2_ - y1_) - py_
            dis = ax * ax + ay * ay                      # [N, HW]

            # nearest line per pixel
            min_idx = dis.argmin(dim=0)                  # [HW]
            idx = torch.arange(HW, device=device)

            best_ax = ax[min_idx, idx]
            best_ay = ay[min_idx, idx]
            best_t  = t_c[min_idx, idx]
            best_in = in_seg[min_idx, idx]

            # vectors pixel → endpoints  [N, HW]
            ux_all = x1_ - px_;  uy_all = y1_ - py_
            vx_all = x2_ - px_;  vy_all = y2_ - py_

            b_ux = ux_all[min_idx, idx];  b_uy = uy_all[min_idx, idx]
            b_vx = vx_all[min_idx, idx];  b_vy = vy_all[min_idx, idx]

            # put closer endpoint into (u), farther into (v)
            swap = (b_ux * b_ux + b_uy * b_uy) >= (b_vx * b_vx + b_vy * b_vy)
            f_ux = torch.where(swap, b_vx, b_ux)
            f_uy = torch.where(swap, b_vy, b_uy)
            f_vx = torch.where(swap, b_ux, b_vx)
            f_vy = torch.where(swap, b_uy, b_vy)

            # fill output
            lmap[0] = best_ax.reshape(height, width)
            lmap[1] = best_ay.reshape(height, width)
            lmap[2] = f_ux.reshape(height, width)
            lmap[3] = f_uy.reshape(height, width)
            lmap[4] = f_vx.reshape(height, width)
            lmap[5] = f_vy.reshape(height, width)

            label[0] = torch.where(best_in, min_idx, torch.tensor(-1, device=device)) \
                              .reshape(height, width).to(torch.int32)
            tmap[0]  = best_t.reshape(height, width)

            return lmap, label, tmap

    _C = _FallbackC()
# ─────────────────────────────────────────────────────────────────────

__all__ = ["_C"]
