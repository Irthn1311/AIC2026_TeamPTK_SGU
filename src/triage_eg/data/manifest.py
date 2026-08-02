"""CSV serialization for video manifests without pandas."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from triage_eg.common.schemas import VideoRecord

_FIELDS = [
    "video_id",
    "relative_path",
    "batch_id",
    "fps",
    "total_frames",
    "duration_ms",
    "width",
    "height",
    "has_audio",
    "dataset_version",
]


def _optional_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def _optional_bool(value: str) -> bool | None:
    if not value.strip():
        return None
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"Invalid optional boolean value: {value}")
    return lowered == "true"


def read_video_manifest_csv(path: str | Path) -> list[VideoRecord]:
    """Read video records from a standard CSV manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Video manifest does not exist: {manifest_path}")
    records: list[VideoRecord] = []
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(_FIELDS) - set(reader.fieldnames):
            raise ValueError(f"Manifest must contain fields: {', '.join(_FIELDS)}")
        for row in reader:
            records.append(
                VideoRecord(
                    video_id=row["video_id"],
                    relative_path=row["relative_path"],
                    batch_id=row["batch_id"],
                    fps=float(row["fps"]),
                    total_frames=int(row["total_frames"]),
                    duration_ms=int(row["duration_ms"]),
                    width=_optional_int(row["width"]),
                    height=_optional_int(row["height"]),
                    has_audio=_optional_bool(row["has_audio"]),
                    dataset_version=row["dataset_version"],
                )
            )
    return records


def write_video_manifest_csv(records: Iterable[VideoRecord], path: str | Path) -> None:
    """Write video records to CSV with a stable column order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in _FIELDS})


def validate_unique_video_ids(records: Iterable[VideoRecord]) -> list[str]:
    """Return sorted video IDs that occur more than once."""

    counts = Counter(record.video_id for record in records)
    return sorted(video_id for video_id, count in counts.items() if count > 1)


def validate_relative_paths(records: Iterable[VideoRecord]) -> list[str]:
    """Return invalid paths that are absolute or traverse above the data root."""

    invalid: list[str] = []
    for record in records:
        normalized = record.relative_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or not normalized:
            invalid.append(record.relative_path)
    return invalid

