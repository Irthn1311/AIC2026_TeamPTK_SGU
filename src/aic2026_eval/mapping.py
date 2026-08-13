"""BTC mapping discovery, duplicate-preserving reads, and L21 coordinate audit."""

from __future__ import annotations

import csv
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import VIDEO_ID_PATTERN
from .io import read_jsonl


def discover_keyframe_partitions(dataset_root: str | Path) -> dict[str, str]:
    """Map canonical video IDs to the actual keyframe partition containing them."""
    root = Path(dataset_root) / "keyframes" / "keyframes"
    partitions: dict[str, str] = {}
    if not root.is_dir():
        return partitions
    for partition in sorted(root.iterdir(), key=lambda path: path.name):
        content = partition / "keyframes"
        if not content.is_dir() or partition.is_symlink() or content.is_symlink():
            continue
        for directory in sorted(content.iterdir(), key=lambda path: path.name):
            if directory.is_dir() and VIDEO_ID_PATTERN.fullmatch(directory.name):
                partitions[directory.name] = partition.name
    return partitions


def asset_paths(
    dataset_root: str | Path,
    video_id: str,
    source_group: str,
    *,
    keyframe_partition: str | None = None,
) -> dict[str, Path]:
    root = Path(dataset_root)
    level = video_id.split("_", 1)[0]
    keyframe_partition = keyframe_partition or f"Keyframes_{level}"
    return {
        "video": root / source_group / "video" / f"{video_id}.mp4",
        "mapping": root / "map-keyframes-aic25-b1/map-keyframes" / f"{video_id}.csv",
        "metadata": root / "media-info-aic25-b1/media-info" / f"{video_id}.json",
        "clip": root / "clip-features-32-aic25-b1/clip-features-32" / f"{video_id}.npy",
        "objects": root / "objects-aic25-b1/objects" / video_id,
        "keyframes": root / "keyframes" / "keyframes" / keyframe_partition / "keyframes" / video_id,
        "level": Path(level),
    }


def read_mapping(path: str | Path) -> list[dict[str, Any]]:
    """Read every BTC row in CSV order; duplicate frame_idx rows are never collapsed."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "pts_time", "fps", "frame_idx"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"mapping requires columns {sorted(required)}")
        for line_number, raw in enumerate(reader, 2):
            try:
                rows.append(
                    {
                        "n": int(raw["n"]),
                        "pts_time": float(raw["pts_time"]),
                        "fps": float(raw["fps"]),
                        "frame_idx": int(raw["frame_idx"]),
                    }
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid mapping row {line_number}: {error}") from error
    return rows


def _bootstrap_files(bootstrap_zip: Path, temporary_root: Path) -> dict[str, Path]:
    with zipfile.ZipFile(bootstrap_zip) as archive:
        safe = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and ".." not in Path(name).parts
            and Path(name).name
            in {
                "l21_queries.jsonl",
                "l21_gt_draft.jsonl",
                "l21_anchor_index.jsonl",
                "manifest.json",
                "README.md",
            }
        ]
        archive.extractall(temporary_root, safe)
    return {path.name: path for path in temporary_root.rglob("*") if path.is_file()}


def _anchor_interval(value: dict[str, Any]) -> tuple[int, int] | None:
    pairs = (
        ("provisional_start_frame", "provisional_end_frame"),
        ("acceptable_start_frame", "acceptable_end_frame"),
        ("start_frame", "end_frame"),
    )
    for left, right in pairs:
        if isinstance(value.get(left), int) and isinstance(value.get(right), int):
            return int(value[left]), int(value[right])
    for key in (
        "provisional_raw_frame",
        "approx_original_frame_idx",
        "original_frame_idx",
        "frame_id",
    ):
        if isinstance(value.get(key), int):
            frame = int(value[key])
            return frame, frame
    return None


def audit_l21_bootstrap(
    bootstrap_zip: str | Path | None,
    inventory: list[dict[str, Any]],
    *,
    temporary_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if bootstrap_zip is None or not Path(bootstrap_zip).is_file():
        return [], {"status": "SKIPPED_NO_INPUT", "anchor_count": 0}
    files = _bootstrap_files(Path(bootstrap_zip), Path(temporary_root))
    sources = [
        files[name] for name in ("l21_anchor_index.jsonl", "l21_gt_draft.jsonl") if name in files
    ]
    anchors: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for source in sources:
        for index, row in enumerate(read_jsonl(source), 1):
            video_id = row.get("video_id")
            interval = _anchor_interval(row)
            if not isinstance(video_id, str) or interval is None:
                continue
            anchor_id = str(row.get("anchor_id") or row.get("query_id") or f"row_{index}")
            anchors[(anchor_id, video_id, *interval)] = row
    inventory_by_id = {row["video_id"]: row for row in inventory}
    results = []
    for anchor_id, video_id, start, end in sorted(anchors):
        item = inventory_by_id.get(video_id)
        status = "NEEDS_VISUAL_REVIEW"
        nearby: list[dict[str, Any]] = []
        if (
            not VIDEO_ID_PATTERN.fullmatch(video_id)
            or item is None
            or start < 0
            or end < start
            or end >= int(item["total_frames"])
        ):
            status = "OUT_OF_BOUNDS"
        elif not item["mapping_available"]:
            status = "NEEDS_VISUAL_REVIEW"
        else:
            mapping = read_mapping(item["mapping_path"])
            fps = float(item["fps"])
            margin = max(1, round(2 * fps))
            nearby = [row for row in mapping if start - margin <= row["frame_idx"] <= end + margin]
            inside = [row for row in nearby if start <= row["frame_idx"] <= end]
            monotonic = all(
                left["frame_idx"] <= right["frame_idx"]
                for left, right in zip(mapping, mapping[1:], strict=False)
            )
            if not monotonic:
                status = "MAPPING_AMBIGUOUS"
            elif inside:
                status = "COORDINATE_SUPPORTED"
            else:
                status = "NO_BTC_FRAME_IN_PROVISIONAL_INTERVAL"
        duplicate_counts = Counter(row["frame_idx"] for row in nearby)
        results.append(
            {
                "anchor_id": anchor_id,
                "video_id": video_id,
                "provisional_start_frame": start,
                "provisional_end_frame": end,
                "status": status,
                "semantic_interval_changed": False,
                "nearby_mapping_rows": nearby,
                "duplicate_frame_idx_groups": {
                    str(frame): count for frame, count in duplicate_counts.items() if count > 1
                },
            }
        )
    counts = Counter(row["status"] for row in results)
    return results, {
        "status": "PASS" if results and not counts.get("OUT_OF_BOUNDS") else "PARTIAL",
        "anchor_count": len(results),
        "by_status": dict(sorted(counts.items())),
        "fps_derived_frames_declared_canonical": False,
        "semantic_interval_changes": 0,
    }


__all__ = [
    "asset_paths",
    "audit_l21_bootstrap",
    "discover_keyframe_partitions",
    "read_mapping",
]
