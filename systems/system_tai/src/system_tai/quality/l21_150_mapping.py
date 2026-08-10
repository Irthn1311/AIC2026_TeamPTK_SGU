"""Mapping and raw-video validation evidence for proposed L21-150 frame GT."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .l21_150_schema import (
    FrameInterval,
    L21150Benchmark,
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEQuery,
)

VALIDATION_STATUSES = {
    "VALIDATED",
    "MISMATCH",
    "OUT_OF_RANGE",
    "MISSING_MAPPING",
    "MISSING_VIDEO_METADATA",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    fps: float
    total_frames: int
    path: Path

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video FPS must be positive and finite")
        if type(self.total_frames) is not int or self.total_frames <= 0:
            raise ValueError("video total_frames must be a positive integer")


def timestamp_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid timestamp: {value!r}")
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0
    else:
        hours, minutes, seconds = numbers
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid timestamp: {value!r}")
    return float(hours * 3600 + minutes * 60 + seconds)


def _exact_matches(root: Path, video_id: str, suffixes: set[str]) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path.resolve()
                for path in root.rglob("*")
                if path.is_file()
                and path.stem == video_id
                and path.suffix.casefold() in suffixes
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    )


def _load_mapping(path: Path) -> tuple[tuple[int, float | None], ...]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "frame_idx" not in reader.fieldnames:
            raise ValueError("mapping CSV requires frame_idx")
        rows: list[tuple[int, float | None]] = []
        for line_number, row in enumerate(reader, start=2):
            raw_frame = (row.get("frame_idx") or "").strip()
            try:
                frame_idx = int(raw_frame)
            except ValueError as exc:
                raise ValueError(f"invalid frame_idx at line {line_number}") from exc
            if frame_idx < 0:
                raise ValueError(f"negative frame_idx at line {line_number}")
            raw_pts = (row.get("pts_time") or "").strip()
            pts_time: float | None = None
            if raw_pts:
                try:
                    pts_time = float(raw_pts)
                except ValueError as exc:
                    raise ValueError(f"invalid pts_time at line {line_number}") from exc
                if not math.isfinite(pts_time) or pts_time < 0:
                    raise ValueError(f"invalid pts_time at line {line_number}")
            rows.append((frame_idx, pts_time))
    if not rows:
        raise ValueError("mapping CSV contains no data rows")
    return tuple(rows)


def probe_video_opencv(path: Path) -> VideoMetadata:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is unavailable for raw-video metadata probing") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open raw video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMetadata(fps=fps, total_frames=total_frames, path=path)
    finally:
        capture.release()


def _entry(
    *,
    query_id: str,
    task_type: str,
    video_id: str,
    timestamp: str,
    center: int,
    interval: FrameInterval,
    event_index: int | None,
    mapping_rows: tuple[tuple[int, float | None], ...] | None,
    mapping_path: Path | None,
    mapping_error: str | None,
    metadata: VideoMetadata | None,
    metadata_required: bool,
    metadata_error: str | None,
) -> dict[str, Any]:
    nearest_frame: int | None = None
    nearest_pts: float | None = None
    timestamp_delta: float | None = None
    status = "VALIDATED"
    reason = "proposed interval contains the nearest timestamp-mapped original frame"

    if mapping_rows is None:
        status = "MISSING_MAPPING" if mapping_path is None else "MISMATCH"
        reason = mapping_error or "mapping is unavailable"
    else:
        with_pts = [(frame_idx, pts) for frame_idx, pts in mapping_rows if pts is not None]
        if not with_pts:
            status = "MISMATCH"
            reason = "mapping has no pts_time values for timestamp comparison"
        else:
            reference_seconds = timestamp_seconds(timestamp)
            nearest_frame, nearest_pts = min(
                with_pts,
                key=lambda item: (abs(float(item[1]) - reference_seconds), item[0]),
            )
            timestamp_delta = abs(float(nearest_pts) - reference_seconds)
            if not interval.start_frame_id <= nearest_frame <= interval.end_frame_id:
                status = "MISMATCH"
                reason = "nearest timestamp-mapped frame is outside the proposed interval"

        max_mapping_frame = max(frame_idx for frame_idx, _ in mapping_rows)
        if interval.start_frame_id > max_mapping_frame:
            status = "OUT_OF_RANGE"
            reason = "proposed interval begins after the maximum mapped frame"

    if metadata is not None:
        if interval.end_frame_id >= metadata.total_frames:
            status = "OUT_OF_RANGE"
            reason = "proposed interval exceeds raw-video frame bounds"
    elif metadata_required and status == "VALIDATED":
        status = "MISSING_VIDEO_METADATA"
        reason = metadata_error or "raw-video metadata is unavailable"

    return {
        "query_id": query_id,
        "task_type": task_type,
        "video_id": video_id,
        "event_index": event_index,
        "reference_timestamp": timestamp,
        "source_proposed_frame_center": center,
        "source_proposed_start_frame_id": interval.start_frame_id,
        "source_proposed_end_frame_id": interval.end_frame_id,
        "nearest_mapping_frame_idx": nearest_frame,
        "nearest_mapping_pts_time": nearest_pts,
        "timestamp_delta_seconds": timestamp_delta,
        "mapping_path": str(mapping_path) if mapping_path is not None else None,
        "raw_video_path": str(metadata.path) if metadata is not None else None,
        "raw_video_fps": metadata.fps if metadata is not None else None,
        "raw_video_total_frames": metadata.total_frames if metadata is not None else None,
        "status": status,
        "reason": reason,
        "frame_shift_applied": 0,
    }


def validate_l21_150_mapping(
    benchmark: L21150Benchmark,
    mapping_root: Path,
    *,
    video_root: Path | None = None,
    video_probe: Callable[[Path], VideoMetadata] = probe_video_opencv,
) -> dict[str, Any]:
    mapping_base = Path(mapping_root)
    video_base = Path(video_root) if video_root is not None else None
    video_ids = sorted({query.video_id for query in benchmark.queries})
    resources: dict[str, tuple[Any, ...]] = {}

    for video_id in video_ids:
        mapping_matches = _exact_matches(mapping_base, video_id, {".csv"})
        mapping_path: Path | None = None
        mapping_rows: tuple[tuple[int, float | None], ...] | None = None
        mapping_error: str | None = None
        if len(mapping_matches) == 1:
            mapping_path = mapping_matches[0]
            try:
                mapping_rows = _load_mapping(mapping_path)
            except ValueError as exc:
                mapping_error = str(exc)
        elif not mapping_matches:
            mapping_error = "no exact-stem mapping CSV found"
        else:
            mapping_error = f"ambiguous exact-stem mapping CSVs: {len(mapping_matches)}"

        metadata: VideoMetadata | None = None
        metadata_error: str | None = None
        if video_base is not None:
            video_matches = _exact_matches(video_base, video_id, VIDEO_EXTENSIONS)
            if len(video_matches) == 1:
                try:
                    metadata = video_probe(video_matches[0])
                except (OSError, RuntimeError, ValueError) as exc:
                    metadata_error = str(exc)
            elif not video_matches:
                metadata_error = "no exact-stem raw video found"
            else:
                metadata_error = f"ambiguous exact-stem raw videos: {len(video_matches)}"
        resources[video_id] = (
            mapping_rows,
            mapping_path,
            mapping_error,
            metadata,
            metadata_error,
        )

    records: list[dict[str, Any]] = []
    for query in benchmark.queries:
        mapping_rows, mapping_path, mapping_error, metadata, metadata_error = resources[
            query.video_id
        ]
        if isinstance(query, (L21150KISQuery, L21150QAQuery)):
            records.append(
                _entry(
                    query_id=query.query_id,
                    task_type=query.task_type,
                    video_id=query.video_id,
                    timestamp=query.reference_timestamp,
                    center=query.proposed_frame_center,
                    interval=query.proposed_interval,
                    event_index=None,
                    mapping_rows=mapping_rows,
                    mapping_path=mapping_path,
                    mapping_error=mapping_error,
                    metadata=metadata,
                    metadata_required=video_base is not None,
                    metadata_error=metadata_error,
                )
            )
        elif isinstance(query, L21150TRAKEQuery):
            for event in query.events:
                records.append(
                    _entry(
                        query_id=query.query_id,
                        task_type=query.task_type,
                        video_id=query.video_id,
                        timestamp=event.reference_timestamp,
                        center=event.proposed_frame_center,
                        interval=event.proposed_interval,
                        event_index=event.event_index,
                        mapping_rows=mapping_rows,
                        mapping_path=mapping_path,
                        mapping_error=mapping_error,
                        metadata=metadata,
                        metadata_required=video_base is not None,
                        metadata_error=metadata_error,
                    )
                )

    status_counts = {status: 0 for status in sorted(VALIDATION_STATUSES)}
    for record in records:
        status_counts[record["status"]] += 1
    return {
        "schema_version": 1,
        "benchmark_id": benchmark.benchmark_id,
        "validation_role": "PROPOSED_GT_MAPPING_EVIDENCE",
        "source_gt_mutated": False,
        "automatic_frame_shift_applied": False,
        "video_count": len(video_ids),
        "record_count": len(records),
        "status_counts": status_counts,
        "records": records,
    }
