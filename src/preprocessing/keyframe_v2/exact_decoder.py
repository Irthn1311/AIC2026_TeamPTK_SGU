from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DecodedFrame:
    requested_frame: int
    actual_pos_after_read: int | None
    image_bgr: np.ndarray


class ExactFrameDecoder:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

    def close(self) -> None:
        self.cap.release()

    def decode(self, internal_frame_index: int) -> DecodedFrame:
        if internal_frame_index < 0:
            raise ValueError("internal_frame_index must be non-negative")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(internal_frame_index))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot decode frame {internal_frame_index}")
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return DecodedFrame(int(internal_frame_index), pos, frame)

    def save(self, internal_frame_index: int, output_path: str | Path, quality: int = 95) -> None:
        decoded = self.decode(internal_frame_index)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(output_path), decoded.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError(f"Cannot write image: {output_path}")


def compare_images(a_bgr: np.ndarray, b_bgr: np.ndarray, resize_width: int = 320) -> float:
    def prep(img: np.ndarray) -> np.ndarray:
        if img.shape[1] != resize_width:
            scale = resize_width / float(img.shape[1])
            img = cv2.resize(img, (resize_width, max(1, int(round(img.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= gray.mean()
        norm = float(np.linalg.norm(gray))
        return gray / norm if norm > 0 else gray

    x = prep(a_bgr)
    y = prep(b_bgr)
    if x.shape != y.shape:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.clip((x * y).sum(), -1.0, 1.0))
