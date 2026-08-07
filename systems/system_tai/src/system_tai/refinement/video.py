"""Bounded raw-video resolution, probing, and absolute-frame decoding."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from system_tai.data.corpus_discovery import VIDEO_EXTENSIONS, CorpusManifest


class CoarseDecodeStrategy(StrEnum):
    SEQUENTIAL = "sequential"
    SPARSE_VERIFIED = "sparse-verified"


class RawVideoError(RuntimeError):
    """Structured raw-video resolution, probe, or decode failure."""


@dataclass(frozen=True, slots=True)
class RawVideoRecord:
    video_id: str
    raw_video_path: Path | None
    warnings: tuple[str, ...] = ()


class RawVideoRegistry:
    def __init__(self, records: Sequence[RawVideoRecord]) -> None:
        by_video: dict[str, RawVideoRecord] = {}
        for record in records:
            if not record.video_id.strip():
                raise ValueError("raw-video video_id must not be empty")
            if record.video_id in by_video:
                raise ValueError(f"duplicate raw-video record: {record.video_id}")
            by_video[record.video_id] = record
        self._records = tuple(sorted(records, key=lambda item: item.video_id.casefold()))
        self._by_video = by_video

    @classmethod
    def from_manifest(cls, manifest: CorpusManifest) -> RawVideoRegistry:
        return cls(
            tuple(
                RawVideoRecord(
                    video_id=video.video_id,
                    raw_video_path=video.raw_video_path,
                    warnings=(
                        ()
                        if video.raw_video_path is not None
                        else (f"raw video missing for {video.video_id}",)
                    ),
                )
                for video in manifest.videos
            )
        )

    @classmethod
    def from_bounded_roots(
        cls,
        video_ids: Sequence[str],
        roots: Sequence[Path],
    ) -> RawVideoRegistry:
        resolved_roots = tuple(
            sorted(
                {Path(root).resolve(strict=False) for root in roots},
                key=lambda path: str(path).casefold(),
            )
        )
        for root in resolved_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"raw-video search root is not a directory: {root}")
        records: list[RawVideoRecord] = []
        for video_id in sorted(set(video_ids), key=str.casefold):
            matches = sorted(
                {
                    path.resolve(strict=False)
                    for root in resolved_roots
                    for path in root.rglob(f"{video_id}.*")
                    if path.is_file()
                    and path.stem == video_id
                    and path.suffix.casefold() in VIDEO_EXTENSIONS
                },
                key=lambda path: str(path).casefold(),
            )
            if len(matches) > 1:
                raise RawVideoError(
                    f"ambiguous raw videos for {video_id}: "
                    + ", ".join(str(path) for path in matches)
                )
            records.append(
                RawVideoRecord(
                    video_id=video_id,
                    raw_video_path=matches[0] if matches else None,
                    warnings=() if matches else (f"raw video missing for {video_id}",),
                )
            )
        return cls(records)

    @property
    def records(self) -> tuple[RawVideoRecord, ...]:
        return self._records

    def get(self, video_id: str) -> RawVideoRecord:
        try:
            return self._by_video[video_id]
        except KeyError as exc:
            raise KeyError(f"video_id absent from raw-video registry: {video_id}") from exc


@dataclass(frozen=True, slots=True)
class VideoProbe:
    video_id: str
    raw_video_path: Path
    decoder_backend: str
    fps: float
    total_frame_count: int
    width: int
    height: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise RawVideoError(f"invalid raw-video FPS for {self.video_id}: {self.fps}")
        if self.total_frame_count <= 0:
            raise RawVideoError(
                f"invalid raw-video frame count for {self.video_id}: {self.total_frame_count}"
            )
        if self.width <= 0 or self.height <= 0:
            raise RawVideoError(f"invalid raw-video resolution for {self.video_id}")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise RawVideoError(f"invalid raw-video duration for {self.video_id}")


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    absolute_frame_id: int
    timestamp_seconds: float
    image: Any

    def __post_init__(self) -> None:
        if self.absolute_frame_id < 0:
            raise ValueError("absolute_frame_id must be non-negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("decoded timestamp must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    probe: VideoProbe
    frame_ids: tuple[int, ...]
    max_decoded_frames: int

    def __post_init__(self) -> None:
        if not self.frame_ids:
            raise ValueError("decode request requires at least one frame ID")
        if tuple(sorted(set(self.frame_ids))) != self.frame_ids:
            raise ValueError("decode frame IDs must be sorted and unique")
        if self.frame_ids[0] < 0 or self.frame_ids[-1] >= self.probe.total_frame_count:
            raise ValueError("decode frame IDs are outside raw-video bounds")
        if self.max_decoded_frames <= 0:
            raise ValueError("max_decoded_frames must be positive")
        sequential_span = self.frame_ids[-1] - self.frame_ids[0] + 1
        if sequential_span > self.max_decoded_frames:
            raise RawVideoError(
                "decode request exceeds max_decoded_frames: "
                f"span={sequential_span}, limit={self.max_decoded_frames}"
            )


@dataclass(frozen=True, slots=True)
class DecodeResult:
    frames: tuple[DecodedFrame, ...]
    decoded_frame_count: int
    video_open_seconds: float
    decode_seconds: float
    decoder_backend: str
    warnings: tuple[str, ...]
    decode_strategy: str = "sequential_bounded"


class VideoDecoder(Protocol):
    backend_identifier: str

    def probe(self, record: RawVideoRecord) -> VideoProbe: ...

    def decode(self, request: DecodeRequest) -> DecodeResult: ...


class OpenCVVideoDecoder:
    """Seek to a bounded start, then decode sequentially with absolute frame IDs."""

    backend_identifier = "opencv-sequential-after-bounded-seek"

    def __init__(
        self,
        *,
        cv2_module: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        try:
            self._cv2 = cv2_module or importlib.import_module("cv2")
        except ImportError as exc:
            raise RawVideoError(f"OpenCV dependency unavailable: {exc}") from exc
        self._clock = clock

    def probe(self, record: RawVideoRecord) -> VideoProbe:
        path = record.raw_video_path
        if path is None:
            raise RawVideoError(f"raw video missing for {record.video_id}")
        if not path.is_file():
            raise RawVideoError(f"raw video path is not a file: {path}")
        capture = self._cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RawVideoError(f"raw video is unreadable: {path}")
            fps = float(capture.get(self._cv2.CAP_PROP_FPS))
            total = int(capture.get(self._cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        duration = total / fps if math.isfinite(fps) and fps > 0 else math.nan
        return VideoProbe(
            video_id=record.video_id,
            raw_video_path=path.resolve(strict=False),
            decoder_backend=self.backend_identifier,
            fps=fps,
            total_frame_count=total,
            width=width,
            height=height,
            duration_seconds=duration,
        )

    def decode(self, request: DecodeRequest) -> DecodeResult:
        start_frame = request.frame_ids[0]
        end_frame = request.frame_ids[-1]
        requested = set(request.frame_ids)
        open_start = self._clock()
        capture = self._cv2.VideoCapture(str(request.probe.raw_video_path))
        video_open_seconds = self._clock() - open_start
        frames: list[DecodedFrame] = []
        decoded_count = 0
        decode_start = self._clock()
        try:
            if not capture.isOpened():
                raise RawVideoError(f"raw video is unreadable: {request.probe.raw_video_path}")
            if not capture.set(self._cv2.CAP_PROP_POS_FRAMES, start_frame):
                raise RawVideoError(f"raw-video seek failed at frame {start_frame}")
            for absolute_frame_id in range(start_frame, end_frame + 1):
                if decoded_count >= request.max_decoded_frames:
                    raise RawVideoError("decode exceeded max_decoded_frames guard")
                observed_position = float(capture.get(self._cv2.CAP_PROP_POS_FRAMES))
                if math.isfinite(observed_position) and not math.isclose(
                    observed_position,
                    absolute_frame_id,
                    abs_tol=0.5,
                ):
                    raise RawVideoError(
                        "decoder position disagrees with requested absolute frame: "
                        f"expected={absolute_frame_id}, observed={observed_position}"
                    )
                ok, image = capture.read()
                decoded_count += 1
                if not ok or image is None:
                    raise RawVideoError(
                        f"raw-video decode failed at absolute frame {absolute_frame_id}"
                    )
                if absolute_frame_id in requested:
                    frames.append(
                        DecodedFrame(
                            absolute_frame_id=absolute_frame_id,
                            timestamp_seconds=absolute_frame_id / request.probe.fps,
                            image=image,
                        )
                    )
        finally:
            capture.release()
        if tuple(frame.absolute_frame_id for frame in frames) != request.frame_ids:
            raise RawVideoError("decoder did not return every requested absolute frame")
        return DecodeResult(
            frames=tuple(frames),
            decoded_frame_count=decoded_count,
            video_open_seconds=video_open_seconds,
            decode_seconds=self._clock() - decode_start,
            decoder_backend=self.backend_identifier,
            warnings=(
                "bounded seek followed by sequential absolute-frame decode with "
                "backend position verification",
            ),
            decode_strategy="sequential_bounded",
        )

    def decode_sparse_verified(
        self, request: DecodeRequest, *, fallback_to_sequential: bool = True
    ) -> DecodeResult:
        open_start = self._clock()
        capture = self._cv2.VideoCapture(str(request.probe.raw_video_path))
        video_open_seconds = self._clock() - open_start
        frames: list[DecodedFrame] = []
        decoded_count = 0
        decode_start = self._clock()
        failed = False
        fail_reason = ""

        try:
            if not capture.isOpened():
                failed = True
                fail_reason = f"raw video is unreadable: {request.probe.raw_video_path}"
            else:
                for absolute_frame_id in request.frame_ids:
                    if not capture.set(self._cv2.CAP_PROP_POS_FRAMES, absolute_frame_id):
                        failed = True
                        fail_reason = f"raw-video seek failed at frame {absolute_frame_id}"
                        break

                    observed_position = float(capture.get(self._cv2.CAP_PROP_POS_FRAMES))
                    if math.isfinite(observed_position) and not math.isclose(
                        observed_position, absolute_frame_id, abs_tol=0.5
                    ):
                        failed = True
                        fail_reason = (
                            "decoder position disagrees with requested absolute frame: "
                            f"expected={absolute_frame_id}, observed={observed_position}"
                        )
                        break

                    ok, image = capture.read()
                    decoded_count += 1
                    if not ok or image is None:
                        failed = True
                        fail_reason = (
                            f"raw-video decode failed at absolute frame {absolute_frame_id}"
                        )
                        break

                    frames.append(
                        DecodedFrame(
                            absolute_frame_id=absolute_frame_id,
                            timestamp_seconds=absolute_frame_id / request.probe.fps,
                            image=image,
                        )
                    )
        finally:
            capture.release()

        if failed or tuple(frame.absolute_frame_id for frame in frames) != request.frame_ids:
            if not failed:
                fail_reason = "decoder did not return every requested absolute frame"

            if fallback_to_sequential:
                res = self.decode(request)
                # Keep the same decoded_frame_count signature but add warning
                warnings = res.warnings + (
                    f"sparse coarse decode failed; sequential fallback used: {fail_reason}",
                )
                return DecodeResult(
                    frames=res.frames,
                    decoded_frame_count=res.decoded_frame_count + decoded_count,
                    video_open_seconds=video_open_seconds + res.video_open_seconds,
                    decode_seconds=(self._clock() - decode_start) + res.decode_seconds,
                    decoder_backend=self.backend_identifier,
                    warnings=warnings,
                    decode_strategy="sparse_verified_fallback_sequential",
                )
            else:
                raise RawVideoError(fail_reason)

        return DecodeResult(
            frames=tuple(frames),
            decoded_frame_count=decoded_count,
            video_open_seconds=video_open_seconds,
            decode_seconds=self._clock() - decode_start,
            decoder_backend=self.backend_identifier,
            warnings=("verified sparse seek",),
            decode_strategy="sparse_verified",
        )
