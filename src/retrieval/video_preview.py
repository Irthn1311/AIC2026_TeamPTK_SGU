from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def make_preview_clip(video_path: str | Path, start_seconds: float, end_seconds: float, output_path: str | Path) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found")
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-to",
        f"{max(start_seconds, end_seconds):.3f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path

