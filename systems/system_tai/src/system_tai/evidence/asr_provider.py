"""Whisper Automatic Speech Recognition (ASR) Audio Extraction and Transcript Provider."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from system_tai.retrieval.bm25_search import BM25Document

try:
    import whisper  # type: ignore[import-untyped]
    HAS_WHISPER = True
except ImportError:
    whisper = None
    HAS_WHISPER = False


@dataclass(frozen=True, slots=True)
class ASRSegment:
    """A timestamped speech recognition segment."""

    video_id: str
    start_sec: float
    end_sec: float
    text: str
    language: str = "vi"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must not be empty")
        if self.start_sec < 0 or self.end_sec < 0:
            raise ValueError("timestamps must be non-negative")
        if self.start_sec > self.end_sec:
            raise ValueError("start_sec must not exceed end_sec")

    def to_actual_frame_id(self, fps: float) -> int:
        """Convert start timestamp to representative actual_frame_id."""
        if fps <= 0:
            raise ValueError("fps must be positive")
        return int(round(self.start_sec * fps))

    def to_frame_interval(self, fps: float) -> tuple[int, int]:
        """Convert segment timestamps to start/end frame interval."""
        if fps <= 0:
            raise ValueError("fps must be positive")
        return (int(round(self.start_sec * fps)), int(round(self.end_sec * fps)))


class WhisperASRExtractor:
    """Extracts audio and transcribes video speech using OpenAI Whisper."""

    def __init__(
        self,
        model_name: str = "base",
        *,
        device: str = "auto",
        cache_dir: Path | None = None,
        ffmpeg_executable: str = "ffmpeg",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.ffmpeg_executable = ffmpeg_executable
        self._model: Any = None
        self.has_whisper = HAS_WHISPER

    def _get_model(self) -> Any:
        if not self.has_whisper:
            return None
        if self._model is None:
            dev = self.device
            if dev == "auto":
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = whisper.load_model(self.model_name, device=dev)
        return self._model

    def extract_audio(self, video_path: Path, output_wav: Path) -> bool:
        """Extract 16kHz mono 16-bit PCM WAV audio from video using FFmpeg."""
        video_p = Path(video_path)
        output_p = Path(output_wav)
        if not video_p.exists():
            return False

        output_p.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_executable,
            "-y",
            "-i", str(video_p),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_p),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            return proc.returncode == 0 and output_p.exists() and output_p.stat().st_size > 0
        except (subprocess.SubprocessError, OSError):
            return False

    def transcribe_video(
        self,
        video_id: str,
        video_path: Path,
        *,
        language: str | None = None,
        force_recompute: bool = False,
    ) -> list[ASRSegment]:
        """Transcribe video audio to a sequence of ASRSegment."""
        if not video_id.strip():
            raise ValueError("video_id must not be empty")

        # 1. Check Cache
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{video_id}_asr.json"
            if cache_file.exists() and not force_recompute:
                return self.load_cached_transcript(cache_file)

        # 2. Extract Audio
        temp_wav = (
            self.cache_dir / f"{video_id}_temp.wav"
            if self.cache_dir is not None
            else Path(f"temp_{video_id}.wav")
        )
        try:
            success = self.extract_audio(video_path, temp_wav)
            if not success:
                return []

            # 3. Transcribe via Whisper
            model = self._get_model()
            if model is None:
                return []

            options: dict[str, Any] = {}
            if language:
                options["language"] = language

            result = model.transcribe(str(temp_wav), **options)
            segments: list[ASRSegment] = []
            detected_lang = result.get("language", language or "unknown")

            for seg in result.get("segments", []):
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                start_s = float(seg.get("start", 0.0))
                end_s = float(seg.get("end", start_s))
                if end_s < start_s:
                    end_s = start_s
                segments.append(
                    ASRSegment(
                        video_id=video_id,
                        start_sec=start_s,
                        end_sec=end_s,
                        text=text,
                        language=detected_lang,
                        confidence=float(seg.get("confidence", 1.0)),
                    )
                )

            # 4. Save to Cache
            if self.cache_dir is not None and segments:
                cache_file = self.cache_dir / f"{video_id}_asr.json"
                self.save_transcript(segments, cache_file)

            return segments
        finally:
            if temp_wav.exists():
                try:
                    temp_wav.unlink()
                except OSError:
                    pass

    @staticmethod
    def load_cached_transcript(transcript_path: Path) -> list[ASRSegment]:
        """Load transcript segments from a JSON file."""
        p = Path(transcript_path)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = []
        raw_list = data if isinstance(data, list) else data.get("segments", [])
        for item in raw_list:
            segments.append(
                ASRSegment(
                    video_id=str(item["video_id"]),
                    start_sec=float(item["start_sec"]),
                    end_sec=float(item["end_sec"]),
                    text=str(item["text"]).strip(),
                    language=str(item.get("language", "vi")),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        return segments

    @staticmethod
    def save_transcript(segments: Sequence[ASRSegment], output_path: Path) -> None:
        """Save transcript segments to a JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serialized = [asdict(s) for s in segments]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    @staticmethod
    def to_bm25_documents(
        segments: Sequence[ASRSegment],
        fps: float = 25.0,
    ) -> list[BM25Document]:
        """Convert ASR speech segments into BM25 documents for text indexing."""
        docs = []
        for idx, seg in enumerate(segments):
            frame_id = seg.to_actual_frame_id(fps)
            doc_id = f"{seg.video_id}_asr_{idx:04d}_{frame_id}"
            docs.append(
                BM25Document(
                    doc_id=doc_id,
                    video_id=seg.video_id,
                    frame_id=frame_id,
                    text=seg.text,
                    metadata={
                        "source": "asr_whisper",
                        "start_sec": seg.start_sec,
                        "end_sec": seg.end_sec,
                        "language": seg.language,
                        "confidence": seg.confidence,
                    },
                )
            )
        return docs
