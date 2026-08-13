from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2


@dataclass
class VideoMetadata:
    video_id: str
    video_path: str
    width: int
    height: int
    duration_sec: float
    total_frames: int
    avg_frame_rate: str
    r_frame_rate: str
    time_base: str
    reported_fps: float
    is_cfr: bool | None
    probe_backend: str

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


def _fraction_to_float(text: str) -> float:
    try:
        return float(Fraction(text))
    except Exception:
        try:
            return float(text)
        except Exception:
            return 0.0


def probe_video(video_path: Path, cfr_tolerance: float = 0.001) -> VideoMetadata:
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,time_base,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        stream = json.loads(raw)["streams"][0]
        avg = stream.get("avg_frame_rate", "0/0")
        r_rate = stream.get("r_frame_rate", "0/0")
        fps = _fraction_to_float(avg)
        duration = float(stream.get("duration") or 0.0)
        total = int(stream.get("nb_frames") or round(duration * fps))
        is_cfr = abs(_fraction_to_float(avg) - _fraction_to_float(r_rate)) <= cfr_tolerance
        return VideoMetadata(
            video_id=video_path.stem,
            video_path=str(video_path),
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            duration_sec=duration,
            total_frames=total,
            avg_frame_rate=avg,
            r_frame_rate=r_rate,
            time_base=stream.get("time_base", ""),
            reported_fps=fps,
            is_cfr=is_cfr,
            probe_backend="ffprobe",
        )
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = total / fps if fps > 0 else 0.0
        return VideoMetadata(
            video_id=video_path.stem,
            video_path=str(video_path),
            width=width,
            height=height,
            duration_sec=duration,
            total_frames=total,
            avg_frame_rate=f"{fps:.6f}",
            r_frame_rate=f"{fps:.6f}",
            time_base="",
            reported_fps=fps,
            is_cfr=True,
            probe_backend="opencv",
        )
