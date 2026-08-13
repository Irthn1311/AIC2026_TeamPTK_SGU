from __future__ import annotations

import cv2
import numpy as np


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def score_quality(image_bgr: np.ndarray, cfg: dict) -> dict[str, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    poor = float(cfg.get("sharpness_poor", 30.0))
    good = float(cfg.get("sharpness_good", 350.0))
    sharpness_norm = clamp01((sharpness_raw - poor) / max(1e-6, good - poor))

    mean = float(gray.mean())
    low = float(cfg.get("exposure_low", 35.0))
    high = float(cfg.get("exposure_high", 220.0))
    if mean < low:
        exposure = clamp01(mean / max(1e-6, low))
    elif mean > high:
        exposure = clamp01((255.0 - mean) / max(1e-6, 255.0 - high))
    else:
        exposure = 1.0

    quality = clamp01(0.65 * sharpness_norm + 0.35 * exposure)
    return {
        "sharpness_raw": sharpness_raw,
        "sharpness_norm": sharpness_norm,
        "exposure_score": exposure,
        "quality_score": quality,
    }
