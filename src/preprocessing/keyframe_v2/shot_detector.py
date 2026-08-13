from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .video_metadata import VideoMetadata


@dataclass
class Shot:
    video_id: str
    shot_id: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    duration_sec: float
    num_frames: int
    detector_backend: str
    confidence: float | None = None

    def asdict(self) -> dict:
        return asdict(self)


def detect_shots(video_path: Path, meta: VideoMetadata, cfg: dict) -> tuple[list[Shot], list[str]]:
    warnings: list[str] = []
    require_transnetv2 = bool(cfg.get("require_transnetv2", False))
    backend = str(cfg.get("backend", "")).lower()
    use_fast_histdiff = (
        bool(cfg.get("use_histdiff_only", False))
        or backend in ("histdiff", "histdiff_fallback", "fast")
        or os.environ.get("AIC_FAST_SHOT_DETECTION", "0") == "1"
    )
    if not use_fast_histdiff:
        try:
            from transnetv2_pytorch import TransNetV2

            model = TransNetV2(device=str(cfg.get("transnetv2_device", "auto")))
            model.eval()
            if str(cfg.get("transnetv2_frame_reader", "opencv")) == "opencv":
                shots = _detect_transnetv2_opencv(video_path, meta, model, float(cfg.get("transnetv2_threshold", 0.5)))
            else:
                scenes = model.detect_scenes(str(video_path), threshold=float(cfg.get("transnetv2_threshold", 0.5)))
                shots = []
                for i, sc in enumerate(scenes):
                    sf = int(sc["start_frame"])
                    ef = int(sc["end_frame"])
                    shots.append(_make_shot(meta, i, sf, ef, "transnetv2", sc.get("probability")))
            if shots:
                return shots, warnings
            message = "TransNetV2 returned zero shots"
            if require_transnetv2:
                raise RuntimeError(message)
            warnings.append(f"{message}; fallback histdiff used")
        except Exception as exc:
            if require_transnetv2:
                raise RuntimeError(f"TransNetV2 required but unavailable: {exc}") from exc
            warnings.append(f"TransNetV2 unavailable: {exc}; fallback histdiff used")
    return _detect_histdiff(video_path, meta, cfg), warnings


def _detect_transnetv2_opencv(video_path: Path, meta: VideoMetadata, model, threshold: float) -> list[Shot]:
    import torch

    frames = _load_transnet_frames_opencv(video_path)
    if frames.shape[0] == 0:
        raise RuntimeError(f"No frames decoded for TransNetV2: {video_path}")
    tensor = torch.from_numpy(frames).to(model.device)
    with torch.no_grad():
        single_frame_pred, _ = model.predict_frames(tensor, quiet=True)
    pred = single_frame_pred.detach().float().cpu().numpy()
    scenes = model.predictions_to_scenes(pred, threshold=threshold)
    shots: list[Shot] = []
    for i, pair in enumerate(scenes):
        sf = int(pair[0])
        ef = int(pair[1])
        conf = float(pred[sf : ef + 1].max()) if ef >= sf else None
        shots.append(_make_shot(meta, i, sf, ef, "transnetv2", conf))
    return shots


def _load_transnet_frames_opencv(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for TransNetV2: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.zeros((0, 27, 48, 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8, copy=False)


def _make_shot(meta: VideoMetadata, shot_id: int, start: int, end: int, backend: str, confidence: float | None = None) -> Shot:
    start = max(0, int(start))
    end = max(start, min(int(end), meta.total_frames - 1))
    return Shot(
        video_id=meta.video_id,
        shot_id=shot_id,
        start_frame=start,
        end_frame=end,
        start_timestamp=start / meta.reported_fps,
        end_timestamp=end / meta.reported_fps,
        duration_sec=(end - start + 1) / meta.reported_fps,
        num_frames=end - start + 1,
        detector_backend=backend,
        confidence=confidence,
    )


def _detect_histdiff(video_path: Path, meta: VideoMetadata, cfg: dict) -> list[Shot]:
    stride = max(1, int(cfg.get("fallback_sample_stride", 15)))
    threshold = float(cfg.get("fallback_hist_threshold", 0.48))
    min_len = max(1, int(round(float(cfg.get("fallback_min_shot_seconds", 1.2)) * meta.reported_fps)))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for fallback shot detection: {video_path}")

    cuts = [0]
    prev_hist = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        hist = _frame_hist(frame)
        if prev_hist is not None:
            diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if diff >= threshold and frame_idx - cuts[-1] >= min_len:
                cuts.append(frame_idx)
        prev_hist = hist
        frame_idx += stride
        for _ in range(stride - 1):
            if not cap.grab():
                break
    cap.release()

    shots = []
    for i, start in enumerate(cuts):
        end = (cuts[i + 1] - 1) if i + 1 < len(cuts) else meta.total_frames - 1
        if end >= start:
            shots.append(_make_shot(meta, i, start, end, "histdiff_fallback", None))
    return shots or [_make_shot(meta, 0, 0, meta.total_frames - 1, "histdiff_fallback", None)]


def _frame_hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist
