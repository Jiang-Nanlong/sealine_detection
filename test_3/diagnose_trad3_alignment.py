import os, random
import numpy as np

# 改成你的 test cache
CACHE_DIR = r"F:\code_manager\Menglong Cao\sealine_detection\Hashmani's Dataset\FusionCache_1024x576\test"
N_SAMPLE = 200  # 随机抽多少张
SEED = 0

def label_to_idx(label, H, W):
    # label 是 [rho_norm, theta_norm]
    rho_idx = float(label[0]) * (H - 1)
    th_idx  = float(label[1]) * (W - 1)
    return int(round(rho_idx)), int(round(th_idx))

def circ_dist(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)

def main():
    random.seed(SEED)

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
    idxs = [int(os.path.splitext(f)[0]) for f in files]
    idxs = sorted(idxs)
    pick = random.sample(idxs, min(N_SAMPLE, len(idxs)))

    # 统计：每个通道 argmax 与 GT 的距离；以及“上下翻转后”的距离
    stats = {c: {"rho": [], "rho_flip": [], "theta": []} for c in range(4)}

    for idx in pick:
        path = os.path.join(CACHE_DIR, f"{idx}.npy")
        data = np.load(path, allow_pickle=True).item()
        x = data["input"]    # [C,H,W]
        y = data["label"]    # [2]

        C, H, W = x.shape
        gt_r, gt_t = label_to_idx(y, H, W)

        for c in range(min(C, 4)):
            ch = x[c]
            pr, pt = np.unravel_index(np.argmax(ch), ch.shape)

            # rho 原始差 & rho 上下翻转后的差
            dr = abs(pr - gt_r)
            dr_flip = abs((H - 1 - pr) - gt_r)

            dt = circ_dist(pt, gt_t, W)

            stats[c]["rho"].append(dr)
            stats[c]["rho_flip"].append(dr_flip)
            stats[c]["theta"].append(dt)

    print(f"[INFO] sampled {len(pick)} from {CACHE_DIR}")
    for c in range(4):
        if len(stats[c]["rho"]) == 0:
            continue
        rho_m = float(np.mean(stats[c]["rho"]))
        rho_flip_m = float(np.mean(stats[c]["rho_flip"]))
        th_m  = float(np.mean(stats[c]["theta"]))
        print(f"ch{c}: mean|drho|={rho_m:.2f}  mean|drho_flip|={rho_flip_m:.2f}  mean|dtheta|={th_m:.2f}")

    print("\nInterpretation:")
    print("- 如果传统通道(0/1/2)的 mean|drho_flip| 远小于 mean|drho|，基本可以判定 rho 轴上下翻转了。")
    print("- seg通道(3)一般应该是 mean|drho| 更小（与 label 同坐标系）。")

if __name__ == "__main__":
    main()
