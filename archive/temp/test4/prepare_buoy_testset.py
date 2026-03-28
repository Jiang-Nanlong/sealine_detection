# -*- coding: utf-8 -*-
"""
Prepare Buoy dataset for Experiment 4: Cross-Dataset Generalization.

This script extracts frames from Buoy videos and creates a GroundTruth CSV file.
Buoy dataset contains 10 videos with ~100 frames each, totaling ~1000 images.

Inputs (relative to project root):
  - Buoy/*.avi                    (video files)
  - Buoy/*HorizonGT.mat           (ground truth annotations)

Outputs:
  - test4/buoy_frames/            (extracted frames as JPEG images)
  - test4/Buoy_GroundTruth.csv    (CSV with img_name, x1, y1, x2, y2)
  - test4/splits_buoy/            (train/val/test split indices)

Usage:
  python test4/prepare_buoy_testset.py
"""

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


# =========================
# Configuration
# =========================
RANDOM_SEED = 2026


def find_matching_mat(buoy_dir: Path, video_stem: str) -> Path | None:
    """
    Buoy dataset: <video_stem>HorizonGT.mat
    """
    p = buoy_dir / f"{video_stem}HorizonGT.mat"
    if p.exists():
        return p

    # Fallback: case-insensitive search
    target = f"{video_stem}HorizonGT.mat".lower()
    for mp in buoy_dir.glob("*.mat"):
        if mp.name.lower() == target:
            return mp
    return None


def load_structXML(mat_path: Path):
    """Load structXML from mat file."""
    mat = sio.loadmat(str(mat_path))
    if "structXML" not in mat:
        raise KeyError(f"'structXML' not found in {mat_path.name}. Keys={list(mat.keys())}")
    struct = mat["structXML"]
    # Usually shape (1, N) object array
    if struct.ndim == 2 and struct.shape[0] == 1:
        struct = struct[0]
    elif struct.ndim == 2 and struct.shape[1] == 1:
        struct = struct[:, 0]
    return struct


def field_empty(entry, key: str) -> bool:
    """Check if a field in the entry is empty."""
    arr = entry[key]
    if arr is None:
        return True
    if not hasattr(arr, "size"):
        arr = np.array(arr)
    return arr.size == 0


def get_scalar(entry, key: str):
    """Get scalar value from entry field."""
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
    Compute line endpoints from point-normal representation.
    Line: Nx*x + Ny*y = c, where c = Nx*X + Ny*Y
    Returns two endpoints within image boundary.
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

    # Remove duplicates
    uniq = []
    for p in pts:
        if all((abs(p[0] - q[0]) > 1e-6 or abs(p[1] - q[1]) > 1e-6) for q in uniq):
            uniq.append(p)

    if len(uniq) < 2:
        return None

    # Find farthest two points
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

    return best  # ((x1, y1), (x2, y2))


def build_video_catalog(project_root: Path):
    """
    Scan Buoy directory for all videos and validate against mat annotations.
    Returns: [(video_path, mat_path, n_frames, W, H), ...]
    """
    buoy_dir = project_root / "Buoy"
    catalog = []
    skipped = []

    if not buoy_dir.exists():
        print(f"[Error] Buoy directory not found: {buoy_dir}")
        return catalog, skipped

    # Find all video files
    videos = sorted(buoy_dir.glob("*.avi"))
    print(f"[Info] Found {len(videos)} video files in {buoy_dir}")

    for video_path in videos:
        video_stem = video_path.stem  # e.g., "buoyGT_2_5_3_0"
        mat_path = find_matching_mat(buoy_dir, video_stem)

        if mat_path is None:
            skipped.append((video_path.name, "No matching mat file"))
            continue

        # Open video to get frame count and resolution
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            skipped.append((video_path.name, "Cannot open video"))
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Load mat and check frame count
        try:
            struct = load_structXML(mat_path)
            n_gt = len(struct)
        except Exception as e:
            skipped.append((video_path.name, f"Error loading mat: {e}"))
            continue

        if n_frames != n_gt:
            skipped.append((video_path.name, f"Frame count mismatch: video={n_frames}, gt={n_gt}"))
            continue

        # Check for empty GT fields
        has_empty = False
        for i in range(n_gt):
            e = struct[i]
            for k in ("X", "Y", "Nx", "Ny"):
                if field_empty(e, k):
                    has_empty = True
                    break
            if has_empty:
                break

        if has_empty:
            skipped.append((video_path.name, "Contains empty GT fields"))
            continue

        catalog.append((video_path, mat_path, n_frames, W, H))

    return catalog, skipped


def extract_frames_and_build_csv(catalog, project_root: Path):
    """
    Extract frames from videos and build GroundTruth CSV.
    """
    out_dir = project_root / "test4" / "buoy_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_root / "test4" / "Buoy_GroundTruth.csv"
    rows = []

    total_frames = sum(n for _, _, n, _, _ in catalog)
    print(f"[Info] Extracting {total_frames} frames from {len(catalog)} videos...")

    for video_path, mat_path, n_frames, W, H in catalog:
        video_stem = video_path.stem
        struct = load_structXML(mat_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Warning] Cannot open {video_path}")
            continue

        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"[Warning] Failed to read frame {frame_idx} from {video_path.name}")
                continue

            # Get GT for this frame
            e = struct[frame_idx]
            X = get_scalar(e, "X")
            Y = get_scalar(e, "Y")
            Nx = get_scalar(e, "Nx")
            Ny = get_scalar(e, "Ny")

            if any(v is None for v in [X, Y, Nx, Ny]):
                continue

            endpoints = line_endpoints_from_point_normal(X, Y, Nx, Ny, W, H)
            if endpoints is None:
                continue

            (x1, y1), (x2, y2) = endpoints

            # Save frame
            img_name = f"{video_stem}__{frame_idx:06d}.jpg"
            img_path = out_dir / img_name
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            rows.append({
                "img_name": img_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "video": video_stem,
            })

        cap.release()
        print(f"  Processed {video_path.name}: {n_frames} frames")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img_name", "x1", "y1", "x2", "y2", "video"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] Saved {len(rows)} samples to {csv_path}")
    return rows


def create_splits(rows, project_root: Path):
    """
    Create train/val/test splits.
    For zero-shot evaluation, all samples go to test set.
    """
    split_dir = project_root / "test4" / "splits_buoy"
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    indices = np.arange(n, dtype=np.int64)

    # For cross-dataset evaluation, all samples are test
    train_indices = np.array([], dtype=np.int64)
    val_indices = np.array([], dtype=np.int64)
    test_indices = indices

    np.save(split_dir / "train_indices.npy", train_indices)
    np.save(split_dir / "val_indices.npy", val_indices)
    np.save(split_dir / "test_indices.npy", test_indices)

    print(f"[Splits] train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    print(f"[Done] Saved splits to {split_dir}")


def main():
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    print("=" * 60)
    print("Prepare Buoy Dataset for Experiment 4")
    print("=" * 60)

    # Build catalog
    catalog, skipped = build_video_catalog(project_root)

    if skipped:
        print(f"\n[Skipped {len(skipped)} videos]:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    if not catalog:
        print("[Error] No valid videos found!")
        return

    print(f"\n[Valid] {len(catalog)} videos:")
    for vp, mp, nf, w, h in catalog:
        print(f"  {vp.name}: {w}x{h}, {nf} frames")

    # Extract frames and build CSV
    rows = extract_frames_and_build_csv(catalog, project_root)

    # Create splits
    if rows:
        create_splits(rows, project_root)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
