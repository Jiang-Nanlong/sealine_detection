# make_musid_splits.py
import os
import re
import json
import numpy as np
import pandas as pd

# ===================== 你只需要改这些全局变量 =====================
CSV_PATH = r"Hashmani's Dataset/GroundTruth.csv"
IMG_DIR  = r"Hashmani's Dataset/MU-SID"
OUT_DIR  = r"splits_musid"

SEED = 42

# 比例（建议：80/10/10；val 用于选 best_ckpt，test 只做最终一次评估）
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# 是否启用“组划分”：尽量避免相邻帧/同场景同时落入 train/test
# 规则：如果文件名 stem 形如 xxx_123 或 xxx-123，则 group_id=xxx，否则 group_id=stem
USE_GROUP_SPLIT = True

# 若你发现 group 太大影响比例，可改成 False（纯随机划分）
# ====================================================================


def resolve_image_path(img_dir: str, name_in_csv: str):
    """
    按你项目里常用方式尝试补后缀。
    GroundTruth.csv 第一列有时没后缀，有时带后缀。
    """
    base = os.path.join(img_dir, str(name_in_csv))
    candidates = [
        base,
        base + ".JPG",
        base + ".jpg",
        base + ".png",
        base + ".jpeg",
        base + ".JPEG",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def derive_group_id(filename: str):
    """
    更保守的 group_id：
    - stem = 去掉扩展名
    - 如果 stem 末尾是 _digits 或 -digits，则 group_id = stem 去掉末尾这一段
    """
    stem = os.path.splitext(os.path.basename(filename))[0]

    # xxx_123
    if "_" in stem:
        left, right = stem.rsplit("_", 1)
        if right.isdigit():
            return left

    # xxx-123
    if "-" in stem:
        left, right = stem.rsplit("-", 1)
        if right.isdigit():
            return left

    return stem


def greedy_assign_groups(group_to_indices, n_total, n_train, n_val, n_test, rng):
    """
    把 group（不同大小）分配到 train/val/test，尽量满足目标数量
    """
    groups = list(group_to_indices.items())  # [(gid, [idx...]), ...]
    rng.shuffle(groups)

    # 为了更贴近目标，按 group size 从大到小放（贪心更稳）
    groups.sort(key=lambda x: len(x[1]), reverse=True)

    splits = {"train": [], "val": [], "test": []}
    remain = {"train": n_train, "val": n_val, "test": n_test}

    for gid, idxs in groups:
        gsz = len(idxs)

        # 优先放到“剩余容量最大”的 split
        # 若都不够放，则放到当前“最不超标”的 split（最小溢出）
        best_key = None
        best_score = None

        for k in ["train", "val", "test"]:
            r = remain[k]
            # score 越大越好：优先填满剩余容量
            # 若 r>=gsz，score=r-gsz（越接近0越好但仍>=0）
            # 若 r<gsz，score=-(gsz-r)（溢出越少越好）
            if r >= gsz:
                score = 100000 + (r - gsz)  # 可放，给大正数
            else:
                score = -(gsz - r)          # 不可放，溢出惩罚（越接近0越好）

            if best_score is None or score < best_score:
                # 这里我们想要“最小 score”还是“最大 score”？
                # 我们设计为：可放的 score 很大 -> 不希望取最小
                pass

        # 重新写：直接选择“最合适”的 split：
        # 1) 优先选择 r>=gsz 且 (r-gsz) 最小（最贴合容量）
        # 2) 如果都不满足，则选择 (gsz-r) 最小（溢出最少）
        candidates_fit = []
        candidates_over = []
        for k in ["train", "val", "test"]:
            r = remain[k]
            if r >= gsz:
                candidates_fit.append((r - gsz, k))
            else:
                candidates_over.append((gsz - r, k))

        if candidates_fit:
            candidates_fit.sort(key=lambda x: x[0])
            best_key = candidates_fit[0][1]
        else:
            candidates_over.sort(key=lambda x: x[0])
            best_key = candidates_over[0][1]

        splits[best_key].extend(idxs)
        remain[best_key] -= gsz

    # 排序输出
    for k in splits:
        splits[k] = sorted(splits[k])

    return splits


def main():
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "ratios must sum to 1"

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH, header=None)

    # 过滤掉不存在图片的样本（否则后面 cache 会跳过导致索引对不上）
    valid_indices = []
    valid_names = []
    for i in range(len(df)):
        name = str(df.iloc[i, 0])
        p = resolve_image_path(IMG_DIR, name)
        if p is not None:
            valid_indices.append(i)   # 注意：这里保留的是 CSV 行号（你工程里一直用这个当样本ID）
            valid_names.append(os.path.basename(p))

    n = len(valid_indices)
    if n == 0:
        raise RuntimeError("No valid samples found (check CSV_PATH / IMG_DIR).")

    rng = np.random.default_rng(SEED)

    # 计算目标数量
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_test = n - n_train - n_val  # 保证总和等于 n

    if USE_GROUP_SPLIT:
        # group -> indices
        group_to_indices = {}
        for csv_idx, fname in zip(valid_indices, valid_names):
            gid = derive_group_id(fname)
            group_to_indices.setdefault(gid, []).append(csv_idx)

        splits = greedy_assign_groups(group_to_indices, n, n_train, n_val, n_test, rng)
    else:
        # 纯随机按样本划分
        perm = rng.permutation(valid_indices).tolist()
        splits = {
            "train": sorted(perm[:n_train]),
            "val": sorted(perm[n_train:n_train + n_val]),
            "test": sorted(perm[n_train + n_val:]),
        }

    # 保存 indices
    np.save(os.path.join(OUT_DIR, "train_indices.npy"), np.array(splits["train"], dtype=np.int64))
    np.save(os.path.join(OUT_DIR, "val_indices.npy"),   np.array(splits["val"],   dtype=np.int64))
    np.save(os.path.join(OUT_DIR, "test_indices.npy"),  np.array(splits["test"],  dtype=np.int64))

    # 生成 split CSV（保持原 CSV 的无表头格式）
    def save_split_csv(name, indices):
        sub = df.iloc[indices].copy()
        out_csv = os.path.join(OUT_DIR, f"GroundTruth_{name}.csv")
        sub.to_csv(out_csv, header=False, index=False)
        return out_csv

    train_csv = save_split_csv("train", splits["train"])
    val_csv = save_split_csv("val", splits["val"])
    test_csv = save_split_csv("test", splits["test"])

    meta = {
        "seed": SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "counts": {"valid_total": n, "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"])},
        "use_group_split": USE_GROUP_SPLIT,
        "paths": {"train_csv": train_csv, "val_csv": val_csv, "test_csv": test_csv},
        "notes": [
            "indices are CSV row indices (compatible with cache naming idx.npy if you use iterrows index).",
            "test split should never be used for UNet training or checkpoint selection to avoid leakage."
        ],
    }
    with open(os.path.join(OUT_DIR, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ Split done.")
    print(json.dumps(meta["counts"], ensure_ascii=False, indent=2))
    print("Saved to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
