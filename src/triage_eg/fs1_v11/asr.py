"""Deterministic MP4 -> ffmpeg PCM -> Whisper transcription boundary."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import WHISPER_REVISION


def ffprobe_audio_stream(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration",
        "-of",
        "json",
        str(video),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"FFPROBE_AUDIO_FAILED: {process.stderr}")
    import json

    payload = json.loads(process.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("FFPROBE_EXPECTED_EXACTLY_ONE_PRIMARY_AUDIO_STREAM")
    return {"command": command, "stream": streams[0]}


def ffmpeg_pcm_command(video: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]


def decode_mp4_waveform(video: Path) -> tuple[np.ndarray, list[str]]:
    command = ffmpeg_pcm_command(Path(video))
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"FFMPEG_AUDIO_DECODE_FAILED: {process.stderr.decode(errors='replace')}")
    waveform = np.frombuffer(process.stdout, dtype="<f4").copy()
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("FFMPEG_AUDIO_EMPTY_OR_NONFINITE")
    return waveform, command


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def transcribe_video(
    video_id: str,
    video: Path,
    fps: float,
    transcriber: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    audio_probe = ffprobe_audio_stream(video)
    waveform, command = decode_mp4_waveform(video)
    result = transcriber(
        {"array": waveform, "sampling_rate": 16000},
        return_timestamps=True,
        generate_kwargs={"task": "transcribe"},
    )
    segments, previous = [], -1.0
    for chunk in result.get("chunks", []):
        start, end = chunk.get("timestamp", (None, None))
        if start is None or end is None or start < previous or end < start:
            raise RuntimeError("ASR_TIMESTAMPS_NOT_MONOTONIC")
        previous = float(end)
        raw = str(chunk.get("text", ""))
        segments.append(
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "start_frame": int(round(float(start) * fps)),
                "end_frame": int(round(float(end) * fps)),
                "raw_text": raw,
                "normalized_text": normalize_text(raw),
            }
        )
    return {
        "video_id": video_id,
        "status": "PASS",
        "language": result.get("language"),
        "segments": segments,
        "model_revision": WHISPER_REVISION,
        "ffmpeg_command": command,
        "provenance": {
            "video_path": str(video),
            "sampling_rate": 16000,
            "channels": 1,
            "ffprobe": audio_probe,
        },
    }


def lexical_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for segment in row.get("segments", []):
            for token in sorted(set(re.findall(r"\w+", segment["normalized_text"].casefold()))):
                output.setdefault(token, []).append(
                    {
                        "video_id": row["video_id"],
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"],
                        "text": segment["normalized_text"],
                    }
                )
    return output
