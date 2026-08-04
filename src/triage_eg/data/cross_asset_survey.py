"""Targeted, read-only cross-asset contract survey for TRIAGE-EG."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within

SURVEY_VERSION = "0.2.0"
DEFAULT_DATASET_ROOT = Path("/kaggle/input/datasets/nadkli/dataset-aic")
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/cross_asset_survey_v02")
VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CANDIDATES = ("L21_V017", "L24_V010", "L26_V215", "L28_V019", "L30_V025")
EXPECTED_MAPPING_COLUMNS = ("n", "pts_time", "fps", "frame_idx")


@dataclass(frozen=True)
class CrossAssetLimits:
    max_videos: int = 5
    max_object_json_total: int = 25
    max_object_json_bytes: int = 1_048_576
    max_mapping_rows: int = 10_000
    seed: int = 2026

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not 3 <= self.max_videos <= 5:
            raise ValueError("max_videos must be between 3 and 5")
        if self.max_object_json_total > 25:
            raise ValueError("max_object_json_total must not exceed 25")


@dataclass
class CrossAssetResult:
    summary: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    object_samples: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_dataset_root(value: str | Path | None) -> Path:
    selected = value or os.environ.get("AIC_DATA_ROOT") or DEFAULT_DATASET_ROOT
    return Path(selected).expanduser().resolve(strict=False)


def validate_root(root: str | Path, *, strict_root: bool = False) -> Path:
    resolved = Path(root).expanduser().resolve(strict=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {resolved}")
    if strict_root and not _is_within(resolved, KAGGLE_INPUT_ROOT):
        raise ValueError(f"Strict dataset root must be below {KAGGLE_INPUT_ROOT}: {resolved}")
    return resolved


def make_issue(
    severity: str,
    code: str,
    *,
    video_id: str | None = None,
    asset_type: str,
    path: str | Path,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "video_id": video_id,
        "asset_type": asset_type,
        "path": str(path),
        "message": message,
        "evidence": evidence or {},
    }


def _direct_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix),
        key=lambda path: path.name,
    )


def _direct_directories(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_dir()), key=lambda path: path.name
    )


def collect_asset_paths(
    root: Path,
) -> tuple[dict[str, dict[str, Path]], dict[str, Any], list[dict[str, Any]]]:
    """Collect complete ID sets by scanning only known direct roots."""

    issues: list[dict[str, Any]] = []
    assets: dict[str, dict[str, Path]] = {
        "video": {},
        "mapping": {},
        "clip": {},
        "metadata": {},
        "object": {},
        "keyframe": {},
    }
    video_partitions: dict[str, str] = {}
    for partition in sorted(root.glob("Videos_*"), key=lambda path: path.name):
        if not partition.is_dir():
            continue
        for path in _direct_files(partition / "video", ".mp4"):
            assets["video"][path.stem] = path
            video_partitions[path.stem] = partition.name
    fixed_roots = {
        "mapping": (root / "map-keyframes-aic25-b1" / "map-keyframes", ".csv"),
        "clip": (root / "clip-features-32-aic25-b1" / "clip-features-32", ".npy"),
        "metadata": (root / "media-info-aic25-b1" / "media-info", ".json"),
    }
    for asset_type, (directory, suffix) in fixed_roots.items():
        assets[asset_type] = {path.stem: path for path in _direct_files(directory, suffix)}
    object_root = root / "objects-aic25-b1" / "objects"
    assets["object"] = {path.name: path for path in _direct_directories(object_root)}

    keyframe_root = root / "keyframes" / "keyframes"
    layout_partitions: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for partition in _direct_directories(keyframe_root):
        content = partition / "keyframes"
        children = _direct_directories(content)
        direct_images = (
            [path for path in content.iterdir() if path.is_file()] if content.is_dir() else []
        )
        layout_partitions.append(
            {
                "partition_root": str(partition),
                "content_root": str(content),
                "per_video_directory_count": len(children),
                "direct_file_count": len(direct_images),
                "per_video_examples": [path.name for path in children[:5]],
            }
        )
        if children:
            for child in children:
                assets["keyframe"][child.name] = child
        elif direct_images:
            unsupported.append(str(content))
    if unsupported:
        issues.append(
            make_issue(
                "ERROR",
                "UNSUPPORTED_KEYFRAME_LAYOUT",
                asset_type="KEYFRAME",
                path=keyframe_root,
                message=(
                    "Some partition content roots contain files instead of per-video directories"
                ),
                evidence={"roots": unsupported},
            )
        )
    for asset_type, mapping in assets.items():
        for identifier, path in mapping.items():
            if not VIDEO_ID_PATTERN.fullmatch(identifier):
                issues.append(
                    make_issue(
                        "WARNING",
                        "INVALID_VIDEO_ID_PATTERN",
                        video_id=identifier,
                        asset_type=asset_type.upper(),
                        path=path,
                        message="Asset identifier does not match ^L\\d+_V\\d+$",
                    )
                )
    keyframe_layout = {
        "root": str(keyframe_root),
        "organization": "PARTITION_THEN_PER_VIDEO_DIRECTORY" if assets["keyframe"] else "UNKNOWN",
        "partitions": layout_partitions,
        "unsupported_roots": unsupported,
    }
    context = {"video_partitions": video_partitions, "keyframe_layout": keyframe_layout}
    return assets, context, issues


def compare_id_sets(assets: dict[str, dict[str, Path]]) -> dict[str, Any]:
    names = ("video", "mapping", "clip", "metadata")
    sets = {name: set(assets[name]) for name in names}
    union = set().union(*(sets[name] for name in names))
    intersection = set.intersection(*(sets[name] for name in names)) if sets else set()
    return {
        "counts": {name: len(sets[name]) for name in names},
        "union_count": len(union),
        "intersection_count": len(intersection),
        "all_equal": all(sets[name] == sets["video"] for name in names[1:]),
        "missing_from": {name: sorted(union - sets[name]) for name in names},
        "extra_vs_video": {name: sorted(sets[name] - sets["video"]) for name in names[1:]},
        "object_coverage": {
            "count": len(assets["object"]),
            "missing_video_ids": sorted(sets["video"] - set(assets["object"])),
            "extra_ids": sorted(set(assets["object"]) - sets["video"]),
            "covers_all_videos": set(assets["object"]) == sets["video"],
        },
        "keyframe_coverage": {
            "count": len(assets["keyframe"]),
            "missing_video_ids": sorted(sets["video"] - set(assets["keyframe"])),
            "extra_ids": sorted(set(assets["keyframe"]) - sets["video"]),
            "covers_all_videos": set(assets["keyframe"]) == sets["video"],
        },
    }


def parse_mapping(path: Path, max_rows: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != EXPECTED_MAPPING_COLUMNS:
            raise ValueError(f"Expected columns {EXPECTED_MAPPING_COLUMNS}; found {columns}")
        for index, raw in enumerate(reader):
            if index >= max_rows:
                raise ValueError(f"Mapping exceeds max_mapping_rows={max_rows}")
            if any(raw.get(name) in {None, ""} for name in EXPECTED_MAPPING_COLUMNS):
                raise ValueError(f"Missing mapping value at data row {index + 1}")
            rows.append(
                {
                    "row_index": index,
                    "n": int(raw["n"]),
                    "pts_time": float(raw["pts_time"]),
                    "fps": float(raw["fps"]),
                    "frame_idx": int(raw["frame_idx"]),
                }
            )
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["frame_idx"]].append(row)
    duplicates = [
        {
            "frame_idx": frame_idx,
            "csv_row_indices": [row["row_index"] for row in group],
            "n_values": [row["n"] for row in group],
            "pts_time_values": [row["pts_time"] for row in group],
        }
        for frame_idx, group in sorted(groups.items())
        if len(group) > 1
    ]

    def monotonic(values: list[int | float]) -> bool:
        return all(left <= right for left, right in zip(values, values[1:], strict=False))

    return {
        "columns": list(columns),
        "row_count": len(rows),
        "rows": rows,
        "n_start": rows[0]["n"] if rows else None,
        "n_end": rows[-1]["n"] if rows else None,
        "n_unique": len({row["n"] for row in rows}) == len(rows),
        "n_monotonic": monotonic([row["n"] for row in rows]),
        "pts_time_monotonic": monotonic([row["pts_time"] for row in rows]),
        "frame_idx_monotonic": monotonic([row["frame_idx"] for row in rows]),
        "duplicate_pts_time_count": len(rows) - len({row["pts_time"] for row in rows}),
        "fps_values": sorted({row["fps"] for row in rows}),
        "duplicate_frame_idx_groups": duplicates,
    }


def inspect_numeric_files(directory: Path, suffixes: set[str]) -> dict[str, Any]:
    files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=lambda path: path.name,
    )
    stems = [path.stem for path in files]
    numeric = [int(stem) for stem in stems if stem.isdigit()]
    suffix_counts = Counter(path.suffix.lower() for path in files)
    return {
        "paths": files,
        "count": len(files),
        "filename_stems": stems,
        "numeric_values": numeric,
        "numeric_filename_count": len(numeric),
        "numeric_min": min(numeric) if numeric else None,
        "numeric_max": max(numeric) if numeric else None,
        "numeric_widths": sorted({len(stem) for stem in stems if stem.isdigit()}),
        "suffix_distribution": dict(sorted(suffix_counts.items())),
        "duplicate_stem_count": len(stems) - len(set(stems)),
        "non_numeric_filenames": [path.name for path in files if not path.stem.isdigit()],
        "examples": [path.name for path in files[:5]],
    }


def filename_hypothesis(values: list[int], mapping: dict[str, Any]) -> dict[str, str]:
    observed = set(values)
    rows = mapping["rows"]
    hypotheses = {
        "CSV_N": {row["n"] for row in rows},
        "CSV_N_MINUS_ONE": {row["n"] - 1 for row in rows},
        "FRAME_IDX": {row["frame_idx"] for row in rows},
    }
    matches = [name for name, expected in hypotheses.items() if observed == expected]
    if len(matches) == 1 and len(values) == len(rows):
        return {"hypothesis": matches[0], "confidence": "HIGH"}
    if matches:
        return {"hypothesis": "AMBIGUOUS:" + ",".join(matches), "confidence": "LOW"}
    return {"hypothesis": "NO_SIMPLE_RELATION", "confidence": "UNKNOWN"}


def inspect_clip(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "shape": [int(value) for value in array.shape],
        "ndim": int(array.ndim),
        "row_count": int(array.shape[0]) if array.ndim else 0,
        "dimension": int(array.shape[1]) if array.ndim == 2 else None,
        "dtype": str(array.dtype),
    }


def _schema_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value)[:30]}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    return {"type": type(value).__name__, "example": str(value)[:120]}


def inspect_object_json(path: Path, max_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size > max_bytes:
        raise OverflowError(f"Object JSON exceeds {max_bytes} bytes")
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise OverflowError(f"Object JSON exceeds {max_bytes} bytes")
    payload = json.loads(raw.decode("utf-8-sig"))
    detections: list[Any]
    if isinstance(payload, list):
        detections = payload
    elif isinstance(payload, dict):
        list_values = [value for value in payload.values() if isinstance(value, list)]
        detections = list_values[0] if len(list_values) == 1 else []
    else:
        detections = []
    dict_detections = [item for item in detections if isinstance(item, dict)]
    fields = sorted({str(key) for item in dict_detections[:20] for key in item})
    label_fields = [
        name for name in fields if name.lower() in {"label", "class", "class_name", "name"}
    ]
    score_fields = [name for name in fields if name.lower() in {"score", "confidence", "conf"}]
    bbox_fields = [name for name in fields if name.lower() in {"bbox", "box", "bounding_box"}]
    bbox_lengths: set[int] = set()
    coordinates: list[float] = []
    for item in dict_detections[:20]:
        for name in bbox_fields:
            value = item.get(name)
            if isinstance(value, list):
                bbox_lengths.add(len(value))
                coordinates.extend(float(x) for x in value if isinstance(x, int | float))
    return {
        "path": str(path),
        "top_level": _schema_value(payload),
        "is_empty": payload in ({}, []),
        "detection_count": len(detections),
        "detection_fields": fields,
        "candidate_label_fields": label_fields,
        "candidate_score_fields": score_fields,
        "candidate_bbox_fields": bbox_fields,
        "bbox_lengths": sorted(bbox_lengths),
        "coordinate_min": min(coordinates) if coordinates else None,
        "coordinate_max": max(coordinates) if coordinates else None,
        "bbox_order": "UNKNOWN",
    }


def choose_samples(
    complete_ids: set[str],
    *,
    requested: list[str] | None,
    maximum: int,
    seed: int,
) -> tuple[list[str], dict[str, str]]:
    reasons: dict[str, str] = {}
    preferred = requested or list(DEFAULT_CANDIDATES)
    selected: list[str] = []
    used_partitions: set[str] = set()
    for identifier in preferred:
        if identifier in complete_ids and identifier not in selected:
            selected.append(identifier)
            used_partitions.add(identifier.split("_", 1)[0])
            reasons[identifier] = "requested/preferred ID with all required counterparts"
            if len(selected) == maximum:
                return selected, reasons
    remaining = sorted(complete_ids - set(selected))
    random.Random(seed).shuffle(remaining)
    for identifier in remaining:
        partition = identifier.split("_", 1)[0]
        if partition not in used_partitions:
            selected.append(identifier)
            used_partitions.add(partition)
            reasons[identifier] = "deterministic fallback from a different partition"
        if len(selected) == maximum:
            return selected, reasons
    for identifier in remaining:
        if identifier not in selected:
            selected.append(identifier)
            reasons[identifier] = "deterministic fallback from complete intersection"
        if len(selected) == maximum:
            break
    return selected, reasons


def _selected_object_paths(
    paths: list[Path], duplicate_n: set[int], seed: int, remaining_budget: int
) -> list[Path]:
    if not paths or remaining_budget <= 0:
        return []
    ordered = sorted(paths, key=lambda path: path.name)
    candidates = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    candidates.extend(
        path for path in ordered if path.stem.isdigit() and int(path.stem) in duplicate_n
    )
    candidates.append(random.Random(seed).choice(ordered))
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique[: min(5, remaining_budget)]


def survey_cross_assets(
    dataset_root: str | Path,
    *,
    limits: CrossAssetLimits | None = None,
    video_ids: list[str] | None = None,
    strict_root: bool = False,
) -> CrossAssetResult:
    started_at = _now()
    selected_limits = limits or CrossAssetLimits()
    root = validate_root(dataset_root, strict_root=strict_root)
    assets, context, issues = collect_asset_paths(root)
    comparison = compare_id_sets(assets)
    if not comparison["all_equal"]:
        issues.append(
            make_issue(
                "ERROR",
                "ID_SET_MISMATCH",
                asset_type="ALL",
                path=root,
                message="Video, mapping, CLIP and metadata ID sets are not equal",
                evidence=comparison["counts"],
            )
        )
    required = ("video", "mapping", "clip", "metadata", "keyframe", "object")
    complete_ids = set.intersection(*(set(assets[name]) for name in required))
    selected, reasons = choose_samples(
        complete_ids,
        requested=video_ids,
        maximum=selected_limits.max_videos,
        seed=selected_limits.seed,
    )
    if len(selected) < 3:
        issues.append(
            make_issue(
                "ERROR",
                "INSPECTION_LIMIT_REACHED",
                asset_type="ALL",
                path=root,
                message="Fewer than three IDs have all required counterparts",
                evidence={"selected": selected, "complete_count": len(complete_ids)},
            )
        )
    records: list[dict[str, Any]] = []
    object_samples: list[dict[str, Any]] = []
    duplicate_cases: list[dict[str, Any]] = []
    for offset, identifier in enumerate(selected):
        record_issues: list[dict[str, Any]] = []
        try:
            mapping = parse_mapping(
                assets["mapping"][identifier], selected_limits.max_mapping_rows
            )
        except (OSError, TypeError, ValueError) as error:
            issues.append(
                make_issue(
                    "ERROR",
                    "MALFORMED_MAPPING",
                    video_id=identifier,
                    asset_type="MAPPING",
                    path=assets["mapping"][identifier],
                    message=str(error),
                )
            )
            continue
        try:
            clip = inspect_clip(assets["clip"][identifier])
        except (OSError, TypeError, ValueError) as error:
            issues.append(
                make_issue(
                    "ERROR",
                    "MAPPING_CLIP_COUNT_MISMATCH",
                    video_id=identifier,
                    asset_type="CLIP",
                    path=assets["clip"][identifier],
                    message=f"CLIP metadata could not be inspected: {error}",
                )
            )
            continue
        keyframes = inspect_numeric_files(assets["keyframe"][identifier], IMAGE_SUFFIXES)
        objects = inspect_numeric_files(assets["object"][identifier], {".json"})
        keyframe_contract = filename_hypothesis(keyframes["numeric_values"], mapping)
        object_contract = filename_hypothesis(objects["numeric_values"], mapping)
        if keyframe_contract["confidence"] in {"LOW", "UNKNOWN"}:
            record_issues.append(
                make_issue(
                    "WARNING",
                    "AMBIGUOUS_KEYFRAME_FILENAME",
                    video_id=identifier,
                    asset_type="KEYFRAME",
                    path=assets["keyframe"][identifier],
                    message="No unique keyframe filename contract was established",
                    evidence=keyframe_contract,
                )
            )
        if object_contract["confidence"] in {"LOW", "UNKNOWN"}:
            record_issues.append(
                make_issue(
                    "WARNING",
                    "AMBIGUOUS_OBJECT_FILENAME",
                    video_id=identifier,
                    asset_type="OBJECT",
                    path=assets["object"][identifier],
                    message="No unique Object filename contract was established",
                    evidence=object_contract,
                )
            )
        if set(objects["filename_stems"]) != set(keyframes["filename_stems"]):
            record_issues.append(
                make_issue(
                    "WARNING",
                    "OBJECT_KEYFRAME_SET_MISMATCH",
                    video_id=identifier,
                    asset_type="ALIGNMENT",
                    path=assets["object"][identifier],
                    message="Object and keyframe filename stem sets differ",
                )
            )
        for field_name, code in (
            ("n_monotonic", "NON_MONOTONIC_N"),
            ("pts_time_monotonic", "NON_MONOTONIC_PTS"),
            ("frame_idx_monotonic", "NON_MONOTONIC_FRAME_IDX"),
        ):
            if not mapping[field_name]:
                record_issues.append(
                    make_issue(
                        "WARNING",
                        code,
                        video_id=identifier,
                        asset_type="MAPPING",
                        path=assets["mapping"][identifier],
                        message=f"Mapping field failed monotonic validation: {field_name}",
                    )
                )
        mapping_count = mapping["row_count"]
        count_checks = {
            "mapping_equals_clip": mapping_count == clip["row_count"],
            "mapping_equals_keyframes": mapping_count == keyframes["count"],
            "clip_equals_keyframes": clip["row_count"] == keyframes["count"],
        }
        for valid, code, message in (
            (
                count_checks["mapping_equals_clip"],
                "MAPPING_CLIP_COUNT_MISMATCH",
                "Mapping and CLIP row counts differ",
            ),
            (
                count_checks["mapping_equals_keyframes"],
                "MAPPING_KEYFRAME_COUNT_MISMATCH",
                "Mapping and keyframe counts differ",
            ),
            (
                count_checks["clip_equals_keyframes"],
                "CLIP_KEYFRAME_COUNT_MISMATCH",
                "CLIP and keyframe counts differ",
            ),
        ):
            if not valid:
                record_issues.append(
                    make_issue(
                        "WARNING",
                        code,
                        video_id=identifier,
                        asset_type="ALIGNMENT",
                        path=root,
                        message=message,
                    )
                )
        if mapping["duplicate_frame_idx_groups"]:
            record_issues.append(
                make_issue(
                    "WARNING",
                    "DUPLICATE_FRAME_IDX",
                    video_id=identifier,
                    asset_type="MAPPING",
                    path=assets["mapping"][identifier],
                    message="Duplicate original frame indices are preserved",
                    evidence={"group_count": len(mapping["duplicate_frame_idx_groups"])},
                )
            )
        duplicate_n = {
            value for group in mapping["duplicate_frame_idx_groups"] for value in group["n_values"]
        }
        chosen_objects = _selected_object_paths(
            objects["paths"],
            duplicate_n,
            selected_limits.seed + offset,
            selected_limits.max_object_json_total - len(object_samples),
        )
        for path in chosen_objects:
            try:
                sample = {
                    "video_id": identifier,
                    **inspect_object_json(path, selected_limits.max_object_json_bytes),
                }
                object_samples.append(sample)
                if sample["is_empty"]:
                    record_issues.append(
                        make_issue(
                            "INFO",
                            "EMPTY_OBJECT_JSON",
                            video_id=identifier,
                            asset_type="OBJECT",
                            path=path,
                            message="Object JSON is an empty list or object",
                        )
                    )
            except OverflowError as error:
                record_issues.append(
                    make_issue(
                        "WARNING",
                        "OBJECT_JSON_TOO_LARGE",
                        video_id=identifier,
                        asset_type="OBJECT",
                        path=path,
                        message=str(error),
                    )
                )
            except json.JSONDecodeError as error:
                record_issues.append(
                    make_issue(
                        "WARNING",
                        "MALFORMED_OBJECT_JSON",
                        video_id=identifier,
                        asset_type="OBJECT",
                        path=path,
                        message=str(error),
                    )
                )
        keyframe_by_stem = {path.stem: path for path in keyframes["paths"]}
        object_stems = set(objects["filename_stems"])
        for group in mapping["duplicate_frame_idx_groups"]:
            related = []
            for n_value in group["n_values"]:
                candidates = [str(n_value), str(n_value - 1), str(group["frame_idx"])]
                related.append(
                    {
                        "n": n_value,
                        "keyframe_files": [
                            {
                                "name": keyframe_by_stem[value].name,
                                "size_bytes": keyframe_by_stem[value].stat().st_size,
                            }
                            for value in candidates
                            if value in keyframe_by_stem
                        ],
                        "npy_row_exists": 0 <= n_value - 1 < clip["row_count"],
                        "object_files": [
                            value + ".json" for value in candidates if value in object_stems
                        ],
                    }
                )
            case = {
                "video_id": identifier,
                **group,
                "related_assets": related,
                "classification": "DUPLICATE_MAPPING_PRESERVED",
            }
            duplicate_cases.append(case)
        clip_contract = {
            "hypothesis": "NPY row i maps to CSV row i / n=i+1"
            if count_checks["mapping_equals_clip"] and mapping["n_start"] == 1
            else "COUNT_ONLY_ALIGNMENT",
            "confidence": "MEDIUM" if count_checks["mapping_equals_clip"] else "UNKNOWN",
            "limitation": (
                "Semantic vector identity cannot be verified from count and filename "
                "evidence alone."
            ),
        }
        record = {
            "video_id": identifier,
            "selection_reason": reasons[identifier],
            "partition": context["video_partitions"].get(identifier),
            "video_path": str(assets["video"][identifier]),
            "mapping_path": str(assets["mapping"][identifier]),
            "clip_path": str(assets["clip"][identifier]),
            "metadata_path": str(assets["metadata"][identifier]),
            "keyframe_directory": str(assets["keyframe"][identifier]),
            "object_directory": str(assets["object"][identifier]),
            "mapping_row_count": mapping_count,
            "clip_row_count": clip["row_count"],
            "keyframe_image_count": keyframes["count"],
            "object_json_count": objects["count"],
            **count_checks,
            "mapping_observations": {key: value for key, value in mapping.items() if key != "rows"},
            "clip_observations": clip,
            "keyframe_observations": {
                key: value
                for key, value in keyframes.items()
                if key not in {"paths", "filename_stems", "numeric_values"}
            },
            "object_observations": {
                **{
                    key: value
                    for key, value in objects.items()
                    if key not in {"paths", "filename_stems", "numeric_values"}
                },
                "object_matches_keyframe_filename_count": len(
                    set(objects["filename_stems"]) & set(keyframes["filename_stems"])
                ),
                "object_matches_csv_n_count": len(
                    set(objects["numeric_values"]) & {row["n"] for row in mapping["rows"]}
                ),
                "object_matches_csv_n_minus_one_count": len(
                    set(objects["numeric_values"]) & {row["n"] - 1 for row in mapping["rows"]}
                ),
                "object_matches_frame_idx_count": len(
                    set(objects["numeric_values"]) & {row["frame_idx"] for row in mapping["rows"]}
                ),
            },
            "keyframe_filename_contract": keyframe_contract,
            "clip_row_contract": clip_contract,
            "object_filename_contract": object_contract,
            "duplicate_frame_idx_groups": mapping["duplicate_frame_idx_groups"],
            "issues": record_issues,
        }
        records.append(record)
        issues.extend(record_issues)
    schema_fields = sorted(
        {field for sample in object_samples for field in sample.get("detection_fields", [])}
    )
    object_summary = {
        "files_inspected": len(object_samples),
        "top_level_types": dict(Counter(sample["top_level"]["type"] for sample in object_samples)),
        "detection_fields": schema_fields,
        "candidate_label_fields": sorted(
            {
                field
                for sample in object_samples
                for field in sample.get("candidate_label_fields", [])
            }
        ),
        "candidate_score_fields": sorted(
            {
                field
                for sample in object_samples
                for field in sample.get("candidate_score_fields", [])
            }
        ),
        "candidate_bbox_fields": sorted(
            {
                field
                for sample in object_samples
                for field in sample.get("candidate_bbox_fields", [])
            }
        ),
        "bbox_order": "UNKNOWN",
    }
    verified = [
        f"[VERIFIED] ID set equality for video/mapping/CLIP/metadata = {comparison['all_equal']}",
        "[VERIFIED] Object directory coverage equals video IDs = "
        f"{comparison['object_coverage']['covers_all_videos']}",
        "[VERIFIED] Keyframe directory coverage equals video IDs = "
        f"{comparison['keyframe_coverage']['covers_all_videos']}",
    ]
    inferred = [
        "[INFERRED] NPY row-to-CSV order is supported only by count/order evidence where reported."
    ]
    unknown = [
        "[UNKNOWN] CLIP model, preprocessing and semantic row identity.",
        "[UNKNOWN] Object bbox coordinate order unless an explicit schema proves it.",
        "[UNKNOWN] Video codec, FPS/timebase and decode health.",
    ]
    critical = any(item["severity"] == "ERROR" for item in issues)
    summary = {
        "survey_version": SURVEY_VERSION,
        "dataset_root": str(root),
        "started_at": started_at,
        "completed_at": _now(),
        "limits": asdict(selected_limits),
        "selected_video_ids": selected,
        "selection_reasons": reasons,
        "id_set_comparison": comparison,
        "keyframe_layout": context["keyframe_layout"],
        "cross_asset_records": records,
        "object_schema_summary": object_summary,
        "duplicate_frame_idx_case_studies": duplicate_cases,
        "verified_contracts": verified,
        "inferred_contracts": inferred,
        "unknown_contracts": unknown,
        "issues_summary": {
            "total": len(issues),
            "by_code": dict(Counter(item["code"] for item in issues)),
            "by_severity": dict(Counter(item["severity"] for item in issues)),
        },
        "next_stage_readiness": {
            "status": "NOT_READY_FOR_STAGE_0_DESIGN"
            if critical or len(records) < 3
            else "READY_FOR_STAGE_0_DESIGN",
            "reason": "Blocking errors remain"
            if critical
            else "At least three complete samples were inspected",
        },
        "disclaimer": "This is a targeted cross-asset contract survey, not a complete Data Audit.",
    }
    return CrossAssetResult(summary, records, object_samples, issues)


def _markdown(summary: dict[str, Any]) -> str:
    sections = [
        (
            "# 1. Executive Summary",
            [
                summary["disclaimer"],
                f"[VERIFIED] Selected IDs: {', '.join(summary['selected_video_ids'])}",
                f"[VERIFIED] Readiness: {summary['next_stage_readiness']['status']}",
            ],
        ),
        (
            "# 2. Asset ID Set Equality",
            [f"[VERIFIED] {json.dumps(summary['id_set_comparison'], ensure_ascii=False)}"],
        ),
        (
            "# 3. Keyframe Layout",
            [f"[VERIFIED] {json.dumps(summary['keyframe_layout'], ensure_ascii=False)}"],
        ),
        (
            "# 4. Selected Video Samples",
            [
                f"[VERIFIED] {record['video_id']}: "
                f"mapping={record['mapping_row_count']}, clip={record['clip_row_count']}, "
                f"keyframes={record['keyframe_image_count']}, "
                f"objects={record['object_json_count']}"
                for record in summary["cross_asset_records"]
            ],
        ),
        (
            "# 5. Row and Filename Alignment",
            [
                f"[VERIFIED] {record['video_id']}: "
                f"keyframe={record['keyframe_filename_contract']}, "
                f"clip={record['clip_row_contract']}, "
                f"object={record['object_filename_contract']}"
                for record in summary["cross_asset_records"]
            ],
        ),
        (
            "# 6. Object JSON Schema",
            [f"[VERIFIED] {json.dumps(summary['object_schema_summary'], ensure_ascii=False)}"],
        ),
        (
            "# 7. Duplicate frame_idx Cases",
            [
                f"[VERIFIED] {json.dumps(case, ensure_ascii=False)}"
                for case in summary["duplicate_frame_idx_case_studies"]
            ]
            or ["[UNKNOWN] No duplicate case was observed in the selected samples."],
        ),
        ("# 8. Verified Contracts", summary["verified_contracts"]),
        ("# 9. Inferred Contracts", summary["inferred_contracts"]),
        ("# 10. Unknown Contracts", summary["unknown_contracts"]),
        (
            "# 11. Issues",
            [f"[VERIFIED] {json.dumps(summary['issues_summary'], ensure_ascii=False)}"],
        ),
        (
            "# 12. Stage 0 Readiness",
            [
                f"[VERIFIED] {summary['next_stage_readiness']['status']}: "
                f"{summary['next_stage_readiness']['reason']}"
            ],
        ),
    ]
    lines: list[str] = []
    for title, content in sections:
        lines.extend([title, "", *[f"- {item}" for item in content], ""])
    return "\n".join(lines)


def write_outputs(result: CrossAssetResult, output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root).expanduser().resolve(strict=False)
    if _is_within(root, KAGGLE_INPUT_ROOT.resolve(strict=False)):
        raise ValueError("Cross-asset survey output must not be written below /kaggle/input")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": root / "cross_asset_survey_v02.json",
        "markdown": root / "cross_asset_survey_v02.md",
        "records": root / "cross_asset_records.jsonl",
        "object_samples": root / "object_schema_samples.jsonl",
        "issues": root / "issues.jsonl",
    }
    paths["summary"].write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["markdown"].write_text(_markdown(result.summary), encoding="utf-8")
    for key, values in (
        ("records", result.records),
        ("object_samples", result.object_samples),
        ("issues", result.issues),
    ):
        with paths[key].open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    return paths


def result_json(result: CrossAssetResult) -> str:
    return json.dumps(result.summary, ensure_ascii=False, indent=2)
