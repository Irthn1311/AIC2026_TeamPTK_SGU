"""BTC mapping CSV ingestion with explicit original-frame validation."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

from system_tai.common.schemas import FrameRecord
from system_tai.data.video_catalog import BenchmarkVideoCatalog


class FrameMappingLoader:
    def __init__(self, *, fps_tolerance: float = 1e-6) -> None:
        if fps_tolerance < 0:
            raise ValueError("fps_tolerance must be non-negative")
        self.fps_tolerance = fps_tolerance

    def load(
        self,
        csv_path: Path,
        catalog: BenchmarkVideoCatalog,
        *,
        mapping_version: str,
        video_id: str | None = None,
        use_physical_clip_rows: bool = False,
    ) -> Sequence[FrameRecord]:
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"mapping CSV not found: {csv_path}")
        if not mapping_version.strip():
            raise ValueError("mapping_version must not be empty")

        records: list[FrameRecord] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            required = {"n", "pts_time", "fps", "frame_idx"}
            missing = sorted(required - fieldnames)
            if missing:
                raise ValueError(f"mapping CSV missing columns: {', '.join(missing)}")
            has_video_column = "video_id" in fieldnames
            has_clip_row_column = "clip_row" in fieldnames
            if not has_video_column and video_id is None:
                raise ValueError(
                    "mapping CSV has no video_id column; an explicit video_id argument is required"
                )

            for line_number, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue
                physical_row = len(records)
                row_video_id = self._resolve_video_id(
                    row,
                    line_number=line_number,
                    has_video_column=has_video_column,
                    explicit_video_id=video_id,
                )
                keyframe_order = self._parse_int(row, "n", line_number)
                pts_time = self._parse_float(row, "pts_time", line_number)
                fps = self._parse_float(row, "fps", line_number)
                actual_frame_id = self._parse_int(row, "frame_idx", line_number)
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"fps must be positive at line {line_number}")
                if not math.isfinite(pts_time) or pts_time < 0:
                    raise ValueError(f"pts_time must be non-negative at line {line_number}")

                if has_clip_row_column and (row.get("clip_row") or "").strip():
                    clip_row = self._parse_int(row, "clip_row", line_number)
                elif use_physical_clip_rows:
                    clip_row = physical_row
                else:
                    raise ValueError(
                        f"clip_row missing at line {line_number}; physical-row fallback disabled"
                    )

                keyframe_filename = (
                    (row.get("keyframe_filename") or row.get("filename") or "").strip()
                    or None
                )
                record = FrameRecord(
                    video_id=row_video_id,
                    actual_frame_id=actual_frame_id,
                    keyframe_order=keyframe_order,
                    clip_row=clip_row,
                    pts_time=pts_time,
                    fps=fps,
                    mapping_version=mapping_version,
                    physical_row=physical_row,
                    keyframe_filename=keyframe_filename,
                )
                self._validate_against_catalog(record, catalog, line_number)
                records.append(record)

        if not records:
            raise ValueError("mapping CSV contains no records")
        self.validate(records, catalog)
        return tuple(records)

    def validate(
        self,
        records: Sequence[FrameRecord],
        catalog: BenchmarkVideoCatalog,
    ) -> None:
        seen_keyframes: set[tuple[str, int | None]] = set()
        seen_frames: set[tuple[str, int]] = set()
        seen_clip_rows: set[int] = set()
        seen_physical_rows: set[int] = set()
        for record in records:
            self._validate_against_catalog(record, catalog, None)
            keyframe_key = (record.video_id, record.keyframe_order)
            frame_key = (record.video_id, record.actual_frame_id)
            if keyframe_key in seen_keyframes:
                raise ValueError(f"duplicate keyframe mapping: {keyframe_key}")
            if frame_key in seen_frames:
                raise ValueError(f"ambiguous duplicate actual-frame mapping: {frame_key}")
            if record.clip_row in seen_clip_rows:
                raise ValueError(f"duplicate clip_row: {record.clip_row}")
            if record.physical_row in seen_physical_rows:
                raise ValueError(f"duplicate physical_row: {record.physical_row}")
            seen_keyframes.add(keyframe_key)
            seen_frames.add(frame_key)
            seen_clip_rows.add(record.clip_row)
            seen_physical_rows.add(record.physical_row)

    def _validate_against_catalog(
        self,
        record: FrameRecord,
        catalog: BenchmarkVideoCatalog,
        line_number: int | None,
    ) -> None:
        video = catalog.get(record.video_id)
        if abs(record.fps - video.fps) > self.fps_tolerance:
            location = f" at line {line_number}" if line_number is not None else ""
            raise ValueError(
                f"mapping FPS mismatch{location}: mapping={record.fps}, catalog={video.fps}"
            )
        catalog.validate_actual_frame_id(record.video_id, record.actual_frame_id)

    @staticmethod
    def _resolve_video_id(
        row: dict[str, str | None],
        *,
        line_number: int,
        has_video_column: bool,
        explicit_video_id: str | None,
    ) -> str:
        if has_video_column:
            row_video_id = (row.get("video_id") or "").strip()
            if not row_video_id:
                raise ValueError(f"empty video_id at line {line_number}")
            if explicit_video_id is not None and row_video_id != explicit_video_id:
                raise ValueError(
                    f"video_id mismatch at line {line_number}: "
                    f"CSV={row_video_id}, argument={explicit_video_id}"
                )
            return row_video_id
        if explicit_video_id is None:  # Defensive; load checks this before iteration.
            raise ValueError("explicit video_id is required")
        return explicit_video_id

    @staticmethod
    def _parse_int(row: dict[str, str | None], field: str, line_number: int) -> int:
        raw = (row.get(field) or "").strip()
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {field} at line {line_number}: {raw!r}") from exc

    @staticmethod
    def _parse_float(row: dict[str, str | None], field: str, line_number: int) -> float:
        raw = (row.get(field) or "").strip()
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {field} at line {line_number}: {raw!r}") from exc
