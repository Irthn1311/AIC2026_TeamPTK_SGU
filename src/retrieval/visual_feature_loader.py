from __future__ import annotations

from pathlib import Path

import numpy as np


def load_visual_feature_file(feature_path: str | Path, mmap_mode: str = "r") -> np.ndarray:
    arr = np.load(Path(feature_path), mmap_mode=mmap_mode)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got {arr.shape} from {feature_path}")
    return arr


def sanitize_feature_batch(batch: np.ndarray) -> np.ndarray:
    x = np.asarray(batch, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Feature batch contains NaN/Inf")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

