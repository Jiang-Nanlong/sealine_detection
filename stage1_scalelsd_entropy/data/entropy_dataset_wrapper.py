#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic adapter that adds entropy_map to an existing dataset sample.

Why this file exists:
- your project already has a mature MU-SID dataset / split / CSV pipeline
- I cannot safely overwrite that unknown code from here
- this wrapper lets you keep your existing dataset untouched and only add
  entropy_map loading in a strict, explicit way

Expected usage:
    base_dataset = YourExistingMUSIDDataset(...)
    dataset = EntropyDatasetWrapper(
        base_dataset=base_dataset,
        entropy_dir="Hashmani's Dataset/MU-SID_entropy_blue",
        stem_getter=lambda sample: sample["meta"]["stem"],
        image_getter=lambda sample: sample["image"],
    )

Sample contract:
- base_dataset[idx] should return a dict-like sample
- sample must allow us to get:
  1) image tensor/array, to know target H/W
  2) image stem, to load <stem>.npy from entropy_dir

This wrapper is strict by design:
- if entropy .npy is missing, it raises FileNotFoundError
- it never silently fills zeros
"""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ENTROPY_SCALE = 8.0


class EntropyDatasetWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        entropy_dir: str,
        stem_getter: Callable[[Dict[str, Any]], str],
        image_getter: Callable[[Dict[str, Any]], Any],
    ) -> None:
        self.base_dataset = base_dataset
        self.entropy_dir = entropy_dir
        self.stem_getter = stem_getter
        self.image_getter = image_getter

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _infer_hw(image_obj: Any) -> tuple[int, int]:
        if isinstance(image_obj, torch.Tensor):
            if image_obj.ndim == 3:  # [C,H,W]
                return int(image_obj.shape[-2]), int(image_obj.shape[-1])
            raise ValueError(f"Unsupported tensor image shape: {tuple(image_obj.shape)}")
        if isinstance(image_obj, np.ndarray):
            if image_obj.ndim == 3:  # [H,W,C]
                return int(image_obj.shape[0]), int(image_obj.shape[1])
            if image_obj.ndim == 2:  # [H,W]
                return int(image_obj.shape[0]), int(image_obj.shape[1])
            raise ValueError(f"Unsupported ndarray image shape: {image_obj.shape}")
        raise TypeError(f"Unsupported image object type: {type(image_obj)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError("base_dataset[idx] must return a dict-like sample")

        out = copy.deepcopy(sample)
        stem = str(self.stem_getter(out))
        image_obj = self.image_getter(out)
        target_h, target_w = self._infer_hw(image_obj)

        ent_path = os.path.join(self.entropy_dir, f"{stem}.npy")
        if not os.path.isfile(ent_path):
            raise FileNotFoundError(f"Entropy map not found: {ent_path}")

        ent_map = np.load(ent_path).astype(np.float32)
        if ent_map.shape[:2] != (target_h, target_w):
            ent_map = cv2.resize(ent_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        ent_map = np.clip(ent_map / ENTROPY_SCALE, 0.0, 1.0).astype(np.float32)
        out["entropy_map"] = torch.from_numpy(ent_map).unsqueeze(0)

        meta = out.get("meta", {})
        if not isinstance(meta, dict):
            meta = {"_old_meta": meta}
        meta["entropy_path"] = ent_path
        out["meta"] = meta
        return out
