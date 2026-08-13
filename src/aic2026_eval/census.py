"""Deterministic corpus inventory and conservative repository usage census."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .contracts import VIDEO_ID_PATTERN
from .mapping import asset_paths, discover_keyframe_partitions

VIDEO_ID_SEARCH = re.compile(r"(?<![A-Za-z0-9])L\d+_V\d+(?![A-Za-z0-9])", re.ASCII)
TEXT_SUFFIXES = {".py", ".ipynb", ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt"}
SCAN_ROOTS = {"src", "configs", "notebooks", "tests", "docs"}
TIER_ORDER = {
    "T0_UNREFERENCED": 0,
    "T1_INFRA_ONLY": 1,
    "T2_EXPERIMENT_USED": 2,
    "T3_GT_OR_QC_USED": 3,
}


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower(), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        try:
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                result = float(numerator) / float(denominator)
            else:
                result = float(value)
        except (ValueError, ZeroDivisionError):
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _metadata_values(path: Path) -> tuple[float | None, int | None, float | None]:
    if not path.is_file():
        return None, None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, None
    flattened = list(_walk(value))

    def first(keys: set[str]) -> float | None:
        for key, raw in flattened:
            if key in keys:
                result = _number(raw)
                if result is not None and result > 0:
                    return result
        return None

    fps = first({"fps", "average_fps", "avg_frame_rate", "r_frame_rate", "frame_rate"})
    frames = first({"total_frames", "frame_count", "nb_frames", "number_of_frames"})
    duration = first({"duration_sec", "duration_seconds", "duration"})
    return fps, int(frames) if frames is not None else None, duration


def _opencv_probe(path: Path) -> tuple[float, int, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"VIDEO_OPEN_FAILED: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"VIDEO_METADATA_INVALID: {path}")
    return fps, total_frames, total_frames / fps


def build_corpus_inventory(
    dataset_root: str | Path,
    *,
    metadata_probe: Callable[[Path], tuple[float, int, float]] = _opencv_probe,
    video_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    root = Path(dataset_root).expanduser().resolve(strict=True)
    records, issues = [], []
    keyframe_partitions = discover_keyframe_partitions(root)
    paths = sorted(
        path
        for group in root.glob("Videos_L*")
        if group.is_dir()
        for path in (group / "video").glob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".avi", ".mkv"}
        and VIDEO_ID_PATTERN.fullmatch(path.stem)
        and (video_ids is None or path.stem in video_ids)
    )
    for video_path in paths:
        source_group = video_path.parents[1].name
        assets = asset_paths(
            root,
            video_path.stem,
            source_group,
            keyframe_partition=keyframe_partitions.get(video_path.stem),
        )
        fps, total_frames, duration = _metadata_values(assets["metadata"])
        metadata_source = "MEDIA_INFO"
        if fps is None or total_frames is None:
            fps, total_frames, probed_duration = metadata_probe(video_path)
            duration = duration or probed_duration
            metadata_source = "VIDEO_HEADER_FALLBACK"
            issues.append(
                {
                    "severity": "INFO",
                    "code": "VIDEO_HEADER_METADATA_FALLBACK",
                    "video_id": video_path.stem,
                }
            )
        if duration is None or duration <= 0:
            duration = total_frames / fps
        record = {
            "video_id": video_path.stem,
            "top_level": video_path.stem.split("_", 1)[0],
            "source_group": source_group,
            "video_path": str(video_path),
            "file_size": video_path.stat().st_size,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "duration_sec": float(duration),
            "metadata_source": metadata_source,
            "mapping_available": assets["mapping"].is_file(),
            "mapping_path": str(assets["mapping"]),
            "keyframes_available": assets["keyframes"].is_dir(),
            "keyframe_directory": str(assets["keyframes"]),
            "objects_available": assets["objects"].is_dir(),
            "clip_available": assets["clip"].is_file(),
            "metadata_available": assets["metadata"].is_file(),
        }
        record["valid_frame_metadata"] = bool(record["fps"] > 0 and record["total_frames"] > 0)
        records.append(record)
    by_group = Counter(row["source_group"] for row in records)
    summary = {
        "status": "PASS"
        if records and all(row["valid_frame_metadata"] for row in records)
        else "FAIL",
        "dataset_root": str(root),
        "video_count": len(records),
        "by_source_group": dict(sorted(by_group.items())),
        "mapping_available": sum(row["mapping_available"] for row in records),
        "keyframes_available": sum(row["keyframes_available"] for row in records),
        "objects_available": sum(row["objects_available"] for row in records),
        "clip_available": sum(row["clip_available"] for row in records),
        "metadata_fallback_count": sum(
            row["metadata_source"] == "VIDEO_HEADER_FALLBACK" for row in records
        ),
    }
    return records, summary, issues


def _tier_for_path(path: str) -> str:
    lowered = path.lower()
    if any(
        token in lowered
        for token in (
            "ground_truth",
            "pseudo_gt",
            "benchmark",
            "annotation",
            "manual_review",
            "ai_review",
            "review_key",
            "qc",
        )
    ):
        return "T3_GT_OR_QC_USED"
    if any(
        token in lowered
        for token in (
            "experiment",
            "candidate",
            "diagnostic",
            "moment_m",
            "reference_rt",
            "temporal_t",
        )
    ):
        return "T2_EXPERIMENT_USED"
    return "T1_INFRA_ONLY"


def _tracked_files(repository_root: Path) -> tuple[list[str], str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        return (
            [item.decode("utf-8") for item in result.stdout.split(b"\0") if item],
            "GIT_TRACKED_FILES",
        )
    files = [
        path.relative_to(repository_root).as_posix()
        for root_name in sorted(SCAN_ROOTS)
        if (repository_root / root_name).is_dir()
        for path in (repository_root / root_name).rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]
    return files, "TEXT_TREE_FALLBACK_NO_GIT_METADATA"


def build_usage_census(
    inventory: list[dict[str, Any]],
    repository_root: str | Path,
    *,
    manual_exclude_path: str | Path | None = None,
    tracked_files: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repository = Path(repository_root).expanduser().resolve(strict=True)
    known = {row["video_id"] for row in inventory}
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    if tracked_files is None:
        files, scan_mode = _tracked_files(repository)
    else:
        files, scan_mode = tracked_files, "CALLER_PROVIDED_FILES"
    for relative in sorted(files):
        path = Path(relative)
        if (
            not path.parts
            or path.parts[0] not in SCAN_ROOTS
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        source = repository / path
        if not source.is_file() or source.stat().st_size > 25 * 1024 * 1024:
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        tier = _tier_for_path(relative)
        for video_id in sorted(set(VIDEO_ID_SEARCH.findall(text)) & known):
            evidence[video_id].append({"tier": tier, "path": path.as_posix()})
    manual_path = Path(manual_exclude_path) if manual_exclude_path else None
    if manual_path is not None and manual_path.is_file():
        for line in manual_path.read_text(encoding="utf-8").splitlines():
            video_id = line.split("#", 1)[0].strip()
            if video_id in known:
                evidence[video_id].append({"tier": "T3_GT_OR_QC_USED", "path": str(manual_path)})
    rows = []
    for video_id in sorted(known):
        items = evidence.get(video_id, [])
        tier = max(
            (item["tier"] for item in items),
            key=TIER_ORDER.__getitem__,
            default="T0_UNREFERENCED",
        )
        rows.append(
            {
                "video_id": video_id,
                "usage_tier": tier,
                "evidence_paths": sorted({item["path"] for item in items}),
                "evidence": items,
                "ambiguous_assignments_choose_more_contaminated": True,
            }
        )
    counts = Counter(row["usage_tier"] for row in rows)
    return rows, {
        "status": "PASS",
        "video_count": len(rows),
        "by_usage_tier": {tier: counts.get(tier, 0) for tier in TIER_ORDER},
        "repository_scan_mode": scan_mode,
        "scanned_file_count": len(files),
        "manual_exclude_path": str(manual_path) if manual_path else None,
    }


__all__ = [
    "TIER_ORDER",
    "build_corpus_inventory",
    "build_usage_census",
]
