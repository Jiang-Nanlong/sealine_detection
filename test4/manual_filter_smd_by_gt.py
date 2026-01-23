# test4/manual_filter_smd_by_gt.py
# -*- coding: utf-8 -*-
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# =========================
# 全局配置（你只改这里）
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入：你 prepare_smd_testset.py 生成的 GT 和抽帧图片
GT_CSV = PROJECT_ROOT / "test4" / "SMD_GroundTruth.csv"
FRAMES_DIR = PROJECT_ROOT / "test4" / "smd_frames"

# （可选）如果你想同时画 Pred 线（红色），就填入你的 per-sample 结果 CSV
# 不想画就设为 None
PER_SAMPLE_CSV = PROJECT_ROOT / "test4" / "eval_smd_test_per_sample.csv"
DRAW_PRED_LINE = True  # True: 画红线；False: 只画绿线(GT)

# 输出目录
OUT_DIR = PROJECT_ROOT / "test4" / "manual_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 是否复制保留图片到新目录（建议 True，后续更方便）
COPY_KEPT_IMAGES = True
KEPT_DIR = OUT_DIR / "kept_frames"

# 显示窗口大小（只影响显示，不改图片）
WINDOW_W = 1280

# GT 线颜色：绿色；Pred：红色
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
THICKNESS = 2

# =========================


def fit_line_from_endpoints(x1, y1, x2, y2):
    """返回 ax+by+c=0 的 (a,b,c)"""
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return float(a), float(b), float(c)


def intersect_line_with_image(a, b, c, w, h, eps=1e-9):
    """求直线与图像边界的两个交点，用于绘制"""
    pts = []

    # x=0 -> b*y + c = 0
    if abs(b) > eps:
        y = (-c) / b
        if 0 <= y <= h - 1:
            pts.append((0.0, float(y)))

    # x=w-1 -> a*(w-1) + b*y + c = 0
    if abs(b) > eps:
        x = w - 1.0
        y = (-(a * x + c)) / b
        if 0 <= y <= h - 1:
            pts.append((float(x), float(y)))

    # y=0 -> a*x + c = 0
    if abs(a) > eps:
        x = (-c) / a
        if 0 <= x <= w - 1:
            pts.append((float(x), 0.0))

    # y=h-1 -> a*x + b*(h-1) + c = 0
    if abs(a) > eps:
        y = h - 1.0
        x = (-(b * y + c)) / a
        if 0 <= x <= w - 1:
            pts.append((float(x), float(y)))

    # unique
    uniq = []
    for p in pts:
        if all(abs(p[0] - q[0]) > 1e-6 or abs(p[1] - q[1]) > 1e-6 for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None

    # 取最远两点
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

    (x1, y1), (x2, y2) = best
    return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))


def draw_line_from_endpoints(img, x1, y1, x2, y2, color, thickness=2):
    h, w = img.shape[:2]
    a, b, c = fit_line_from_endpoints(x1, y1, x2, y2)
    pts = intersect_line_with_image(a, b, c, w, h)
    if pts is None:
        return img
    p1, p2 = pts
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    return img


def resize_for_display(img, target_w=1280):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / w
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def load_pred_map(per_sample_csv: Path):
    """
    从 eval_smd_test_per_sample.csv 里读取每张图的预测线端点（如果有）
    你当前的 per-sample csv 里没有端点，但有 rho/theta norm。
    为了简单起见：这里改成直接读 “rho/theta -> 线” 会更复杂。
    所以我们优先支持：如果 per-sample csv 里包含 pred_x1/pred_y1/pred_x2/pred_y2 就画。
    若没有，就自动跳过画 Pred。
    """
    if per_sample_csv is None or (not per_sample_csv.exists()):
        return {}, False

    df = pd.read_csv(per_sample_csv)
    need = {"img_name", "pred_x1", "pred_y1", "pred_x2", "pred_y2"}
    if not need.issubset(set(df.columns)):
        return {}, False

    m = {}
    for _, r in df.iterrows():
        m[r["img_name"]] = (float(r["pred_x1"]), float(r["pred_y1"]),
                            float(r["pred_x2"]), float(r["pred_y2"]))
    return m, True


def main():
    if not GT_CSV.exists():
        raise FileNotFoundError(f"GT CSV not found: {GT_CSV}")
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames dir not found: {FRAMES_DIR}")

    decisions_path = OUT_DIR / "decisions.csv"
    filtered_csv_path = OUT_DIR / "SMD_GroundTruth_filtered.csv"
    split_dir = OUT_DIR / "splits_filtered"
    split_dir.mkdir(parents=True, exist_ok=True)

    # 读取 GT
    gt = pd.read_csv(GT_CSV)
    if "img_name" not in gt.columns:
        raise ValueError("GT CSV must contain 'img_name'")

    # 加载已有 decisions（断点续看）
    decided = {}
    if decisions_path.exists():
        old = pd.read_csv(decisions_path)
        for _, r in old.iterrows():
            decided[r["img_name"]] = r["decision"]

    # （可选）Pred 映射
    pred_map, pred_ok = load_pred_map(PER_SAMPLE_CSV)
    draw_pred = DRAW_PRED_LINE and pred_ok

    print("=== Manual GT Review ===")
    print("GT_CSV      :", GT_CSV)
    print("FRAMES_DIR  :", FRAMES_DIR)
    print("OUT_DIR     :", OUT_DIR)
    print("Resume from :", decisions_path if decisions_path.exists() else "None")
    print("Draw Pred   :", draw_pred)

    if COPY_KEPT_IMAGES:
        KEPT_DIR.mkdir(parents=True, exist_ok=True)

    # 待审列表：跳过已决定的
    todo = [row for _, row in gt.iterrows() if row["img_name"] not in decided]
    print(f"Total={len(gt)}, decided={len(decided)}, remaining={len(todo)}")

    # 用于 undo
    history = []  # list of (img_name, decision)

    # OpenCV window
    win = "ManualReview (K=keep, D=drop, A=undo, Q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0
    while idx < len(todo):
        r = todo[idx]
        name = r["img_name"]
        img_path = FRAMES_DIR / name
        if not img_path.exists():
            # 找不到就直接 drop（也记录）
            decided[name] = "drop_missing"
            history.append((name, "drop_missing"))
            idx += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            decided[name] = "drop_badimg"
            history.append((name, "drop_badimg"))
            idx += 1
            continue

        # 画 GT 线（绿色）
        vis = img.copy()
        vis = draw_line_from_endpoints(
            vis,
            r["x1"], r["y1"], r["x2"], r["y2"],
            GT_COLOR, THICKNESS
        )

        # （可选）画 Pred（红色）
        if draw_pred and name in pred_map:
            px1, py1, px2, py2 = pred_map[name]
            vis = draw_line_from_endpoints(vis, px1, py1, px2, py2, PRED_COLOR, THICKNESS)

        # 左上角文字
        domain = name.split("__")[0] if "__" in name else ""
        cv2.putText(vis, f"{domain}  {name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(vis, f"[{idx+1}/{len(todo)}]  K=keep  D=drop  A=undo  Q=quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)

        show = resize_for_display(vis, WINDOW_W)
        cv2.imshow(win, show)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('k'), ord('K')):
            decided[name] = "keep"
            history.append((name, "keep"))
            idx += 1

        elif key in (ord('d'), ord('D')):
            decided[name] = "drop"
            history.append((name, "drop"))
            idx += 1

        elif key in (ord('a'), ord('A')):
            # undo: 回退一个 decision
            if history:
                last_name, _ = history.pop()
                if last_name in decided:
                    del decided[last_name]
                # 把 todo 索引退回到上一张（如果上一张就是当前张，会覆盖显示）
                idx = max(0, idx - 1)
            else:
                print("Nothing to undo.")

        elif key in (ord('q'), ord('Q')):
            print("Quit requested. Saving progress...")
            break

        else:
            print("Unknown key. Use K/D/A/Q.")

        # 每处理 30 张自动保存一次进度
        if (len(history) % 30) == 0:
            save_decisions(decisions_path, decided)

    cv2.destroyAllWindows()

    # 最终保存 decisions
    save_decisions(decisions_path, decided)

    # 生成过滤后的 CSV 和 split
    kept_names = [n for n, d in decided.items() if d == "keep"]
    kept_df = gt[gt["img_name"].isin(kept_names)].copy()
    kept_df.to_csv(filtered_csv_path, index=False, encoding="utf-8")

    # 新 test_indices：就是 kept_df 的行号 0..N-1（因为新 CSV 也会被用作“全量 test”）
    N = len(kept_df)
    np.save(split_dir / "train_indices.npy", np.array([], dtype=np.int64))
    np.save(split_dir / "val_indices.npy", np.array([], dtype=np.int64))
    np.save(split_dir / "test_indices.npy", np.arange(N, dtype=np.int64))

    print("\n=== DONE ===")
    print("Decisions saved     :", decisions_path)
    print("Filtered GT saved   :", filtered_csv_path)
    print("Filtered splits dir :", split_dir)
    print(f"Kept {N} / {len(gt)} images")

    # 可选：复制保留图片
    if COPY_KEPT_IMAGES:
        print("Copying kept images ...")
        KEPT_DIR.mkdir(parents=True, exist_ok=True)
        for n in kept_names:
            src = FRAMES_DIR / n
            if src.exists():
                shutil.copy2(src, KEPT_DIR / n)
        print("Kept images copied to:", KEPT_DIR)


def save_decisions(path: Path, decided: dict):
    rows = [{"img_name": k, "decision": v} for k, v in decided.items()]
    rows.sort(key=lambda x: x["img_name"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_name", "decision"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
