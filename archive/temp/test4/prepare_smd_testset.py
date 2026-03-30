# test4/prepare_smd_testset.py
import csv
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


# =========================
# PyCharm 里只需要改这里
# =========================
TARGET_TOTAL_IMAGES = 3000     # 目标总图片数（约 3000）
RANDOM_SEED = 2026            # 固定随机种子（可复现）
MAX_FRAMES_PER_VIDEO = None   # 可选：限制每个视频最多抽多少帧，例如 80；None 表示不限制
DOMAINS = ["NIR", "VIS_Onboard", "VIS_Onshore"]

# 若你希望每个域大致均衡，也可以开这个（把总数按域平均分配）
BALANCE_BY_DOMAIN = True
# =========================


def find_matching_mat(mat_dir: Path, video_stem: str) -> Path | None:
    """
    SMD 官方：<video_stem>_HorizonGT.mat
    """
    p = mat_dir / f"{video_stem}_HorizonGT.mat"
    if p.exists():
        return p

    # 大小写不一致时兜底
    target = f"{video_stem}_HorizonGT.mat".lower()
    for mp in mat_dir.glob("*.mat"):
        if mp.name.lower() == target:
            return mp
    return None


def load_structXML(mat_path: Path):
    mat = sio.loadmat(str(mat_path))
    if "structXML" not in mat:
        raise KeyError(f"'structXML' not found in {mat_path.name}. Keys={list(mat.keys())}")
    struct = mat["structXML"]
    # 通常是 shape (1, N) object array
    if struct.ndim == 2 and struct.shape[0] == 1:
        struct = struct[0]
    elif struct.ndim == 2 and struct.shape[1] == 1:
        struct = struct[:, 0]
    return struct


def field_empty(entry, key: str) -> bool:
    arr = entry[key]
    if arr is None:
        return True
    if not hasattr(arr, "size"):
        arr = np.array(arr)
    return arr.size == 0


def has_any_empty_gt(struct) -> tuple[bool, int, str]:
    for i in range(len(struct)):
        e = struct[i]
        for k in ("X", "Y", "Nx", "Ny"):
            if field_empty(e, k):
                return True, i, k
    return False, -1, ""


def get_scalar(entry, key: str):
    arr = entry[key]
    if arr is None:
        return None
    if not hasattr(arr, "size"):
        arr = np.array(arr)
    if arr.size == 0:
        return None
    return float(np.array(arr).reshape(-1)[0])


def line_endpoints_from_point_normal(X, Y, Nx, Ny, W, H, eps=1e-9):
    """
    Line: Nx*x + Ny*y = c , where c = Nx*X + Ny*Y
    Return 2 endpoints within image boundary.
    """
    c = Nx * X + Ny * Y
    pts = []

    # x=0, x=W-1
    if abs(Ny) > eps:
        y = c / Ny
        if 0 <= y <= H - 1:
            pts.append((0.0, float(y)))
        y = (c - Nx * (W - 1)) / Ny
        if 0 <= y <= H - 1:
            pts.append((float(W - 1), float(y)))

    # y=0, y=H-1
    if abs(Nx) > eps:
        x = c / Nx
        if 0 <= x <= W - 1:
            pts.append((float(x), 0.0))
        x = (c - Ny * (H - 1)) / Nx
        if 0 <= x <= W - 1:
            pts.append((float(x), float(H - 1)))

    # unique
    uniq = []
    for p in pts:
        if all((abs(p[0] - q[0]) > 1e-6 or abs(p[1] - q[1]) > 1e-6) for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None

    # farthest two points
    best = (uniq[0], uniq[1])
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            dx = uniq[i][0] - uniq[j][0]
            dy = uniq[i][1] - uniq[j][1]
            d = dx * dx + dy * dy
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])

    return best  # (x1,y1),(x2,y2)


def build_video_catalog(project_root: Path):
    """
    扫描三个域下所有视频，筛掉：
      - avi 帧数 != mat 标注数
      - 任意帧 X/Y/Nx/Ny 为空
    返回：[(domain, video_path, mat_path, n_frames, W, H), ...]
    """
    smd_root = project_root / "Singapore Maritime Dataset"
    catalog = []
    skipped = []

    for domain in DOMAINS:
        domain_root = smd_root / domain / domain
        videos_dir = domain_root / "Videos"
        mats_dir = domain_root / "HorizonGT"

        if not videos_dir.exists() or not mats_dir.exists():
            skipped.append((domain, "", "Missing Videos/HorizonGT folder"))
            continue

        video_list = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() == ".avi"])
        for vp in video_list:
            mp = find_matching_mat(mats_dir, vp.stem)
            if mp is None:
                skipped.append((domain, vp.name, "No matching *_HorizonGT.mat"))
                continue

            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                skipped.append((domain, vp.name, "Cannot open video"))
                continue

            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            try:
                struct = load_structXML(mp)
            except Exception as e:
                skipped.append((domain, vp.name, f"Mat read error: {e}"))
                continue

            n_gt = len(struct)
            if n_frames != n_gt:
                skipped.append((domain, vp.name, f"Mismatch avi={n_frames} mat={n_gt}"))
                continue

            bad, bad_i, bad_k = has_any_empty_gt(struct)
            if bad:
                skipped.append((domain, vp.name, f"Empty GT at frame {bad_i}, field {bad_k}"))
                continue

            catalog.append((domain, vp, mp, n_frames, W, H))

    return catalog, skipped


def choose_frame_indices(n_frames: int, k: int, rng: np.random.Generator):
    """
    从 0..n_frames-1 选择 k 个帧索引。
    为了覆盖更均匀，使用“均匀取样 + 微小随机抖动”的方式：
      - 先按均匀间隔取中心点
      - 再在邻域内随机抖动（不会聚集到同一段）
    """
    if k <= 0:
        return []

    if k >= n_frames:
        return list(range(n_frames))

    # 均匀中心点
    centers = np.linspace(0, n_frames - 1, num=k, dtype=np.float64)

    idxs = []
    for c in centers:
        # 抖动范围：半个 bin
        # bin 大约为 n_frames/k
        bin_size = n_frames / k
        jitter = rng.uniform(-0.4 * bin_size, 0.4 * bin_size)
        x = int(round(c + jitter))
        x = max(0, min(n_frames - 1, x))
        idxs.append(x)

    # 去重（有可能抖动后重复），不足再补随机
    idxs = sorted(set(idxs))
    while len(idxs) < k:
        x = int(rng.integers(0, n_frames))
        idxs.append(x)
        idxs = sorted(set(idxs))

    # 若超了（极少），截断
    return idxs[:k]


def main():
    project_root = Path(__file__).resolve().parents[1]
    out_root = project_root / "test4"
    frames_dir = out_root / "smd_frames"
    splits_dir = out_root / "splits"
    frames_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / "SMD_GroundTruth.csv"
    skip_log_path = out_root / "skipped_videos.txt"

    rng = np.random.default_rng(RANDOM_SEED)

    # 1) 扫描 & 严格过滤视频
    catalog, skipped = build_video_catalog(project_root)
    if not catalog:
        raise RuntimeError("No valid videos found after filtering. Check dataset path / naming.")

    # 2) 计算每个视频抽多少帧（均衡）
    if BALANCE_BY_DOMAIN:
        # 每个域分配 TARGET_TOTAL / 3（再按该域视频均分）
        domain_to_videos = {d: [] for d in DOMAINS}
        for item in catalog:
            domain_to_videos[item[0]].append(item)

        target_per_domain = TARGET_TOTAL_IMAGES // len(DOMAINS)
        plan = []  # list of (domain, vp, mp, n_frames, W, H, k)
        for d in DOMAINS:
            vids = domain_to_videos[d]
            if not vids:
                continue
            k_each = max(1, target_per_domain // len(vids))
            for (domain, vp, mp, n_frames, W, H) in vids:
                k = min(k_each, n_frames)
                if MAX_FRAMES_PER_VIDEO is not None:
                    k = min(k, int(MAX_FRAMES_PER_VIDEO))
                plan.append((domain, vp, mp, n_frames, W, H, k))
    else:
        k_each = max(1, TARGET_TOTAL_IMAGES // len(catalog))
        plan = []
        for (domain, vp, mp, n_frames, W, H) in catalog:
            k = min(k_each, n_frames)
            if MAX_FRAMES_PER_VIDEO is not None:
                k = min(k, int(MAX_FRAMES_PER_VIDEO))
            plan.append((domain, vp, mp, n_frames, W, H, k))

    # 让总数尽量逼近 TARGET_TOTAL：用“余数分配”补齐
    planned_total = sum(p[-1] for p in plan)
    deficit = TARGET_TOTAL_IMAGES - planned_total
    if deficit > 0:
        # 优先给帧数更长的视频每个 +1，直到补足
        # （仍然保持均衡，不会让某个视频爆炸）
        plan_sorted = sorted(plan, key=lambda x: x[3], reverse=True)  # by n_frames
        i = 0
        while deficit > 0 and i < len(plan_sorted):
            domain, vp, mp, n_frames, W, H, k = plan_sorted[i]
            if k < n_frames and (MAX_FRAMES_PER_VIDEO is None or k < MAX_FRAMES_PER_VIDEO):
                plan_sorted[i] = (domain, vp, mp, n_frames, W, H, k + 1)
                deficit -= 1
            i = (i + 1) % len(plan_sorted)
        plan = plan_sorted

    # 3) 真正抽帧 + 生成 CSV
    rows = []
    for (domain, vp, mp, n_frames, W, H, k) in plan:
        struct = load_structXML(mp)

        # 选择该视频的 k 个帧索引
        idxs = choose_frame_indices(n_frames, k, rng)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            skipped.append((domain, vp.name, "Cannot open video during extraction"))
            continue

        stem = vp.stem
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                skipped.append((domain, vp.name, f"Failed reading frame {i}"))
                continue

            entry = struct[i]
            X = get_scalar(entry, "X")
            Y = get_scalar(entry, "Y")
            Nx = get_scalar(entry, "Nx")
            Ny = get_scalar(entry, "Ny")
            if X is None or Y is None or Nx is None or Ny is None:
                # 按你要求：一旦发现空帧就该整段跳，但我们在 catalog 阶段已过滤过
                # 这里做兜底：记录并跳过该帧
                skipped.append((domain, vp.name, f"Unexpected empty GT at frame {i}"))
                continue

            ep = line_endpoints_from_point_normal(X, Y, Nx, Ny, W, H)
            if ep is None:
                skipped.append((domain, vp.name, f"Invalid endpoints at frame {i}"))
                continue

            (x1, y1), (x2, y2) = ep

            img_name = f"{domain}__{stem}__{i:06d}.jpg"
            cv2.imwrite(str(frames_dir / img_name), frame)
            rows.append([img_name, x1, y1, x2, y2])

        cap.release()

    # 4) 写 CSV & splits（实验A：全为 test）
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_name", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    N = len(rows)
    np.save(splits_dir / "train_indices.npy", np.array([], dtype=np.int64))
    np.save(splits_dir / "val_indices.npy", np.array([], dtype=np.int64))
    np.save(splits_dir / "test_indices.npy", np.arange(N, dtype=np.int64))

    # 5) 记录跳过视频原因
    with open(skip_log_path, "w", encoding="utf-8") as f:
        for d, v, reason in skipped:
            f.write(f"{d}\t{v}\t{reason}\n")

    # 6) 汇总打印
    print("\n==== DONE (SMD prepared for Experiment A) ====")
    print(f"Target total images : {TARGET_TOTAL_IMAGES}")
    print(f"Actual total images : {N}")
    print(f"Frames dir          : {frames_dir}")
    print(f"GT CSV              : {csv_path}")
    print(f"Splits dir          : {splits_dir} (test-only)")
    print(f"Skipped log         : {skip_log_path}")
    print("Tip: set TARGET_TOTAL_IMAGES to 2800~3200 if you want closer match.")


if __name__ == "__main__":
    main()
