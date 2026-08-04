"""Authoritative benchmark video catalog and original-frame bounds."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from system_tai.common.schemas import BenchmarkVideoRecord, FrameIndexBase

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REQUIRED_COLUMNS = {
    "video_id",
    "video_path",
    "fps",
    "duration_seconds",
    "total_frames",
    "frame_index_base",
}


class BenchmarkVideoCatalog:
    """Resolve video identifiers and authoritative original-frame bounds."""

    def __init__(
        self,
        *,
        strict_paths: bool = False,
        duration_tolerance_seconds: float = 1.0,
    ) -> None:
        if duration_tolerance_seconds < 0:
            raise ValueError("duration_tolerance_seconds must be non-negative")
        self.strict_paths = strict_paths
        self.duration_tolerance_seconds = duration_tolerance_seconds
        self._records: dict[str, BenchmarkVideoRecord] = {}

    @property
    def records(self) -> tuple[BenchmarkVideoRecord, ...]:
        return tuple(self._records.values())

    def load(
        self,
        source: Path,
        *,
        strict_paths: bool | None = None,
    ) -> tuple[BenchmarkVideoRecord, ...]:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"video catalog not found: {source}")

        enforce_paths = self.strict_paths if strict_paths is None else strict_paths
        parsed: dict[str, BenchmarkVideoRecord] = {}

        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_COLUMNS - fieldnames)
            if missing:
                raise ValueError(f"video catalog missing columns: {', '.join(missing)}")

            for line_number, row in enumerate(reader, start=2):
                if not any((value or "").strip() for value in row.values()):
                    continue
                record = self._parse_row(
                    row,
                    line_number=line_number,
                    source_dir=source.parent,
                    strict_paths=enforce_paths,
                )
                if record.video_id in parsed:
                    raise ValueError(
                        f"duplicate video_id at line {line_number}: {record.video_id}"
                    )
                parsed[record.video_id] = record

        if not parsed:
            raise ValueError("video catalog contains no records")

        self._records = parsed
        return self.records

    def get(self, video_id: str) -> BenchmarkVideoRecord:
        try:
            return self._records[video_id]
        except KeyError as exc:
            raise KeyError(f"unknown video_id: {video_id}") from exc

    def validate_actual_frame_id(self, video_id: str, actual_frame_id: int) -> None:
        record = self.get(video_id)
        if record.frame_index_base is FrameIndexBase.ZERO:
            lower, upper = 0, record.total_frames - 1
        elif record.frame_index_base is FrameIndexBase.ONE:
            lower, upper = 1, record.total_frames
        else:  # Defensive: BenchmarkVideoRecord already rejects UNKNOWN.
            raise ValueError(f"unsupported frame_index_base: {record.frame_index_base}")
        if not lower <= actual_frame_id <= upper:
            raise ValueError(
                f"actual_frame_id {actual_frame_id} is out of bounds for {video_id}: "
                f"expected [{lower}, {upper}]"
            )

    def frame_bounds(self, video_id: str) -> tuple[int, int]:
        record = self.get(video_id)
        if record.frame_index_base is FrameIndexBase.ZERO:
            return 0, record.total_frames - 1
        if record.frame_index_base is FrameIndexBase.ONE:
            return 1, record.total_frames
        raise ValueError(f"unsupported frame_index_base: {record.frame_index_base}")

    def _parse_row(
        self,
        row: dict[str, str | None],
        *,
        line_number: int,
        source_dir: Path,
        strict_paths: bool,
    ) -> BenchmarkVideoRecord:
        video_id = (row.get("video_id") or "").strip()
        if not _VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError(f"malformed video_id at line {line_number}: {video_id!r}")

        raw_video_path = (row.get("video_path") or "").strip()
        if not raw_video_path:
            raise ValueError(f"empty video_path at line {line_number}")
        video_path = Path(raw_video_path).expanduser()
        if not video_path.is_absolute():
            video_path = source_dir / video_path
        video_path = video_path.resolve(strict=False)
        if strict_paths and not video_path.is_file():
            raise FileNotFoundError(
                f"video path does not exist at line {line_number}: {video_path}"
            )

        fps = self._parse_float(row, "fps", line_number)
        duration_seconds = self._parse_float(row, "duration_seconds", line_number)
        total_frames = self._parse_int(row, "total_frames", line_number)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive at line {line_number}")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be positive at line {line_number}")
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive at line {line_number}")

        frame_index_base = self._parse_frame_index_base(
            row.get("frame_index_base"), line_number
        )
        expected_duration = total_frames / fps
        duration_error = abs(duration_seconds - expected_duration)
        if duration_error > self.duration_tolerance_seconds:
            raise ValueError(
                f"duration/FPS/frame-count inconsistency at line {line_number}: "
                f"observed={duration_seconds}, expected={expected_duration:.6f}, "
                f"error={duration_error:.6f}, "
                f"tolerance={self.duration_tolerance_seconds:.6f}"
            )

        codec = (row.get("codec") or "").strip() or None
        width = self._parse_optional_int(row.get("width"), "width", line_number)
        height = self._parse_optional_int(row.get("height"), "height", line_number)
        return BenchmarkVideoRecord(
            video_id=video_id,
            video_path=video_path,
            fps=fps,
            duration_seconds=duration_seconds,
            total_frames=total_frames,
            frame_index_base=frame_index_base,
            codec=codec,
            width=width,
            height=height,
        )

    @staticmethod
    def _parse_float(row: dict[str, str | None], field: str, line_number: int) -> float:
        raw = (row.get(field) or "").strip()
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {field} at line {line_number}: {raw!r}") from exc

    @staticmethod
    def _parse_int(row: dict[str, str | None], field: str, line_number: int) -> int:
        raw = (row.get(field) or "").strip()
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {field} at line {line_number}: {raw!r}") from exc

    @staticmethod
    def _parse_optional_int(
        raw_value: str | None,
        field: str,
        line_number: int,
    ) -> int | None:
        raw = (raw_value or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {field} at line {line_number}: {raw!r}") from exc

    @staticmethod
    def _parse_frame_index_base(raw_value: str | None, line_number: int) -> FrameIndexBase:
        raw = (raw_value or "").strip().lower()
        aliases = {
            "0": FrameIndexBase.ZERO,
            "zero": FrameIndexBase.ZERO,
            "zero_based": FrameIndexBase.ZERO,
            "zero-based": FrameIndexBase.ZERO,
            "1": FrameIndexBase.ONE,
            "one": FrameIndexBase.ONE,
            "one_based": FrameIndexBase.ONE,
            "one-based": FrameIndexBase.ONE,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise ValueError(
                f"unsupported or unknown frame_index_base at line {line_number}: {raw!r}"
            ) from exc
