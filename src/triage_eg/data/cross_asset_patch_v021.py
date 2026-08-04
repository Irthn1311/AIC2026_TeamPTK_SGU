"""Focused v0.2.1 patch for Object schema and duplicate asset resolution."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within

SURVEY_VERSION = "0.2.1"
DEFAULT_DATASET_ROOT = Path("/kaggle/input/datasets/nadkli/dataset-aic")
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/cross_asset_survey_v021")
EXPECTED_FIELDS = (
    "detection_boxes",
    "detection_class_entities",
    "detection_class_labels",
    "detection_class_names",
    "detection_scores",
)
SAMPLE_PARTITIONS = {
    "L21_V017": "Keyframes_L21",
    "L24_V010": "Keyframes_L24",
    "L26_V215": "Keyframes_L26_c",
    "L28_V019": "Keyframes_L28",
    "L30_V025": "Keyframes_L30",
}
BASELINE_ORDINALS = {video_id: (1,) for video_id in SAMPLE_PARTITIONS}
DUPLICATE_GROUPS = {
    "L28_V019": (
        {"frame_idx": 480, "n_values": [19, 20]},
        {"frame_idx": 484, "n_values": [22, 23]},
        {"frame_idx": 861, "n_values": [58, 59]},
    ),
    "L30_V025": (
        {"frame_idx": 1814, "n_values": [72, 73]},
        {"frame_idx": 1914, "n_values": [76, 77]},
    ),
}
OUTPUT_NAMES = (
    "patch_summary_v021.json",
    "patch_report_v021.md",
    "object_schema_samples_v021.jsonl",
    "duplicate_case_studies_v021.jsonl",
    "issues_v021.jsonl",
)


@dataclass(frozen=True)
class PatchLimits:
    max_object_json_total: int = 15
    max_object_json_bytes: int = 1_048_576
    max_boxes_per_file: int = 20
    max_boxes_total: int = 100

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_object_json_total > 15:
            raise ValueError("max_object_json_total must not exceed 15")


@dataclass
class PatchResult:
    summary: dict[str, Any]
    object_samples: list[dict[str, Any]] = field(default_factory=list)
    duplicate_cases: list[dict[str, Any]] = field(default_factory=list)
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


def issue(
    severity: str,
    code: str,
    *,
    video_id: str | None,
    ordinal_n: int | None,
    asset_type: str,
    path: str | Path,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "video_id": video_id,
        "ordinal_n": ordinal_n,
        "asset_type": asset_type,
        "path": str(path),
        "message": message,
        "evidence": evidence or {},
    }


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _example(value: Any) -> Any:
    if isinstance(value, str):
        return value[:100]
    if isinstance(value, list):
        numeric = [item for item in value[:6] if isinstance(item, int | float)]
        return {"type": "list", "length": len(value), "numeric_values": numeric}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value)[:10]}
    return value


def _field_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {
            "type": _type_name(value),
            "length": None,
            "item_type_counts": {},
            "item_examples": [_example(value)],
            "null_count": None,
        }
    counts = Counter(_type_name(item) for item in value)
    return {
        "type": "list",
        "length": len(value),
        "item_type_counts": dict(sorted(counts.items())),
        "item_examples": [_example(item) for item in value[:3]],
        "null_count": counts.get("null", 0),
    }


def inspect_parallel_object(
    path: Path,
    *,
    video_id: str,
    ordinal_n: int,
    max_bytes: int,
    max_boxes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect one exact Object JSON without retaining full detection arrays."""

    issues: list[dict[str, Any]] = []
    base: dict[str, Any] = {
        "video_id": video_id,
        "ordinal_n": ordinal_n,
        "path": str(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "inspection_status": "MISSING",
        "top_level_type": None,
        "top_level_keys": [],
        "missing_expected_fields": [],
        "extra_fields": [],
        "field_observations": {},
        "parallel_arrays_valid": False,
        "array_lengths_equal": False,
        "detection_count": None,
        "bbox_observation": {},
        "empty_detection_representation": "NOT_EMPTY_OR_NOT_VALIDATED",
        "is_empty_detection": False,
        "issues": issues,
    }
    if not path.is_file():
        issues.append(
            issue(
                "ERROR",
                "MISSING_TARGET_OBJECT_JSON",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message="Target Object JSON does not exist",
            )
        )
        return base, issues
    if path.stat().st_size > max_bytes:
        base["inspection_status"] = "TOO_LARGE"
        issues.append(
            issue(
                "ERROR",
                "OBJECT_JSON_TOO_LARGE",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message=f"Object JSON exceeds {max_bytes} bytes",
            )
        )
        return base, issues
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        base["inspection_status"] = "TOO_LARGE"
        issues.append(
            issue(
                "ERROR",
                "OBJECT_JSON_TOO_LARGE",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message=f"Object JSON grew beyond {max_bytes} bytes while reading",
            )
        )
        return base, issues
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        base["inspection_status"] = "MALFORMED"
        issues.append(
            issue(
                "ERROR",
                "MALFORMED_OBJECT_JSON",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message=str(error),
            )
        )
        return base, issues
    base["inspection_status"] = "INSPECTED"
    base["top_level_type"] = _type_name(payload)
    if not isinstance(payload, dict):
        issues.append(
            issue(
                "ERROR",
                "OBJECT_TOP_LEVEL_TYPE_MISMATCH",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message="Expected top-level object",
                evidence={"observed_type": _type_name(payload)},
            )
        )
        return base, issues
    keys = sorted(str(key) for key in payload)
    missing = [name for name in EXPECTED_FIELDS if name not in payload]
    extra = [name for name in keys if name not in EXPECTED_FIELDS]
    base.update(
        {
            "top_level_keys": keys,
            "missing_expected_fields": missing,
            "extra_fields": extra,
            "empty_detection_representation": "EMPTY_OBJECT" if not payload else "NOT_EMPTY",
        }
    )
    for name in missing:
        issues.append(
            issue(
                "ERROR",
                "OBJECT_FIELD_MISSING",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message=f"Missing expected field: {name}",
            )
        )
    observations = {name: _field_observation(payload.get(name)) for name in EXPECTED_FIELDS}
    base["field_observations"] = observations
    non_lists = [
        name for name in EXPECTED_FIELDS if name in payload and not isinstance(payload[name], list)
    ]
    for name in non_lists:
        issues.append(
            issue(
                "ERROR",
                "OBJECT_FIELD_TYPE_MISMATCH",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message=f"Expected list field: {name}",
                evidence={"observed_type": _type_name(payload[name])},
            )
        )
    lists_valid = not missing and not non_lists
    lengths = [len(payload[name]) for name in EXPECTED_FIELDS] if lists_valid else []
    equal = bool(lengths) and len(set(lengths)) == 1
    if lists_valid and not equal:
        issues.append(
            issue(
                "ERROR",
                "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message="Expected parallel arrays to have equal lengths",
                evidence={name: len(payload[name]) for name in EXPECTED_FIELDS},
            )
        )
    base["parallel_arrays_valid"] = lists_valid and equal
    base["array_lengths_equal"] = equal
    base["detection_count"] = lengths[0] if lists_valid and equal else None
    if lists_valid and equal and lengths[0] == 0:
        base["empty_detection_representation"] = "FIVE_EMPTY_PARALLEL_ARRAYS"
        base["is_empty_detection"] = True

    boxes = (
        payload.get("detection_boxes") if isinstance(payload.get("detection_boxes"), list) else []
    )
    sampled = boxes[:max_boxes]
    box_types = Counter(_type_name(box) for box in sampled)
    lengths_counter: Counter[str] = Counter()
    coordinates: list[float] = []
    position_values: dict[int, list[float]] = {}
    coordinate_types: Counter[str] = Counter()
    all_numeric = True
    for box_index, box in enumerate(sampled):
        if not isinstance(box, list):
            all_numeric = False
            issues.append(
                issue(
                    "WARNING",
                    "OBJECT_BOX_ITEM_TYPE_MISMATCH",
                    video_id=video_id,
                    ordinal_n=ordinal_n,
                    asset_type="OBJECT",
                    path=path,
                    message=f"Box item {box_index} is not a list",
                )
            )
            continue
        lengths_counter[str(len(box)) if len(box) == 4 else "other"] += 1
        if len(box) != 4:
            issues.append(
                issue(
                    "WARNING",
                    "OBJECT_BOX_LENGTH_UNEXPECTED",
                    video_id=video_id,
                    ordinal_n=ordinal_n,
                    asset_type="OBJECT",
                    path=path,
                    message=f"Expected box length 4; observed {len(box)}",
                )
            )
        for position, value in enumerate(box):
            coordinate_types[_type_name(value)] += 1
            if isinstance(value, int | float) and not isinstance(value, bool):
                numeric = float(value)
                coordinates.append(numeric)
                position_values.setdefault(position, []).append(numeric)
            else:
                all_numeric = False
                issues.append(
                    issue(
                        "WARNING",
                        "OBJECT_COORDINATE_NON_NUMERIC",
                        video_id=video_id,
                        ordinal_n=ordinal_n,
                        asset_type="OBJECT",
                        path=path,
                        message=f"Non-numeric coordinate at box {box_index}, position {position}",
                    )
                )
    scale = "UNKNOWN"
    confidence = "UNKNOWN"
    if coordinates and all(0 <= value <= 1 for value in coordinates):
        scale, confidence = "NORMALIZED_0_1", "MEDIUM"
    elif coordinates and sum(value > 1 for value in coordinates) >= max(2, len(coordinates) // 4):
        scale, confidence = "PIXEL_LIKE", "LOW"
    bbox = {
        "box_item_type_counts": dict(sorted(box_types.items())),
        "box_length_distribution": dict(sorted(lengths_counter.items())),
        "all_sampled_boxes_numeric": all_numeric,
        "sampled_box_count": len(sampled),
        "coordinate_min": min(coordinates) if coordinates else None,
        "coordinate_max": max(coordinates) if coordinates else None,
        "coordinate_value_type_counts": dict(sorted(coordinate_types.items())),
        "per_position_min": [min(position_values[index]) for index in sorted(position_values)],
        "per_position_max": [max(position_values[index]) for index in sorted(position_values)],
        "coordinate_scale_hypothesis": scale,
        "coordinate_scale_confidence": confidence,
        "bbox_order": "UNKNOWN",
    }
    base["bbox_observation"] = bbox

    scores = (
        payload.get("detection_scores") if isinstance(payload.get("detection_scores"), list) else []
    )
    numeric_scores = [
        float(value)
        for value in scores
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    non_numeric_scores = len(scores) - len(numeric_scores)
    if non_numeric_scores:
        issues.append(
            issue(
                "WARNING",
                "OBJECT_SCORE_NON_NUMERIC",
                video_id=video_id,
                ordinal_n=ordinal_n,
                asset_type="OBJECT",
                path=path,
                message="One or more score values are non-numeric",
                evidence={"count": non_numeric_scores},
            )
        )
    observations["detection_scores"].update(
        {
            "numeric_min": min(numeric_scores) if numeric_scores else None,
            "numeric_max": max(numeric_scores) if numeric_scores else None,
            "all_numeric_finite": all(math.isfinite(value) for value in numeric_scores),
            "score_range_hypothesis": (
                "PROBABILITY_LIKE_0_1"
                if numeric_scores and all(0 <= value <= 1 for value in numeric_scores)
                else "UNKNOWN"
            ),
            "score_range_confidence": "MEDIUM" if numeric_scores else "UNKNOWN",
        }
    )
    return base, issues


def aggregate_object_schema(samples: list[dict[str, Any]]) -> dict[str, Any]:
    inspected = [sample for sample in samples if sample["inspection_status"] == "INSPECTED"]
    detection_counts = [
        sample["detection_count"] for sample in inspected if sample["detection_count"] is not None
    ]
    field_types: dict[str, Counter[str]] = {name: Counter() for name in EXPECTED_FIELDS}
    item_types: dict[str, Counter[str]] = {name: Counter() for name in EXPECTED_FIELDS}
    box_lengths: Counter[str] = Counter()
    coordinate_mins: list[float] = []
    coordinate_maxs: list[float] = []
    for sample in inspected:
        for name, observation in sample["field_observations"].items():
            field_types[name][observation["type"]] += 1
            item_types[name].update(observation.get("item_type_counts", {}))
        box_lengths.update(sample["bbox_observation"].get("box_length_distribution", {}))
        minimum = sample["bbox_observation"].get("coordinate_min")
        maximum = sample["bbox_observation"].get("coordinate_max")
        if minimum is not None:
            coordinate_mins.append(minimum)
        if maximum is not None:
            coordinate_maxs.append(maximum)
    all_coordinates_normalized = (
        bool(inspected)
        and all(
            sample["bbox_observation"].get("coordinate_scale_hypothesis")
            in {"NORMALIZED_0_1", "UNKNOWN"}
            for sample in inspected
        )
        and bool(coordinate_mins)
    )
    empty_representations = sorted(
        {
            sample["empty_detection_representation"]
            for sample in inspected
            if sample["is_empty_detection"]
        }
    )
    return {
        "files_requested": len(samples),
        "files_inspected": len(inspected),
        "files_missing": sum(sample["inspection_status"] == "MISSING" for sample in samples),
        "files_malformed": sum(sample["inspection_status"] == "MALFORMED" for sample in samples),
        "files_too_large": sum(sample["inspection_status"] == "TOO_LARGE" for sample in samples),
        "top_level_types": dict(Counter(sample["top_level_type"] for sample in inspected)),
        "top_level_key_sets": sorted({tuple(sample["top_level_keys"]) for sample in inspected}),
        "expected_fields": list(EXPECTED_FIELDS),
        "field_type_summary": {
            name: dict(sorted(counter.items())) for name, counter in field_types.items()
        },
        "parallel_arrays_valid_count": sum(sample["parallel_arrays_valid"] for sample in inspected),
        "parallel_arrays_invalid_count": sum(
            not sample["parallel_arrays_valid"] for sample in inspected
        ),
        "equal_length_count": sum(sample["array_lengths_equal"] for sample in inspected),
        "length_mismatch_count": sum(
            sample["inspection_status"] == "INSPECTED"
            and not sample["array_lengths_equal"]
            and not sample["missing_expected_fields"]
            for sample in samples
        ),
        "detection_count_min": min(detection_counts) if detection_counts else None,
        "detection_count_max": max(detection_counts) if detection_counts else None,
        "detection_count_total_in_sample": sum(detection_counts),
        "empty_detection_files": sum(sample["is_empty_detection"] for sample in inspected),
        "empty_detection_representation": empty_representations,
        "empty_detection_behavior_status": (
            "OBSERVED" if empty_representations else "NOT_OBSERVED_IN_BOUNDED_SAMPLE"
        ),
        "bbox_item_length_distribution": dict(sorted(box_lengths.items())),
        "sampled_coordinate_min": min(coordinate_mins) if coordinate_mins else None,
        "sampled_coordinate_max": max(coordinate_maxs) if coordinate_maxs else None,
        "coordinate_scale_hypothesis": "NORMALIZED_0_1"
        if all_coordinates_normalized
        else "UNKNOWN",
        "coordinate_scale_confidence": "MEDIUM" if all_coordinates_normalized else "UNKNOWN",
        "bbox_order": "UNKNOWN",
        "class_entity_item_types": dict(item_types["detection_class_entities"]),
        "class_label_item_types": dict(item_types["detection_class_labels"]),
        "class_name_item_types": dict(item_types["detection_class_names"]),
        "score_item_types": dict(item_types["detection_scores"]),
    }


def _object_targets(root: Path) -> list[tuple[str, int, Path]]:
    targets: list[tuple[str, int, Path]] = []
    object_root = root / "objects-aic25-b1" / "objects"
    for video_id, ordinals in BASELINE_ORDINALS.items():
        for ordinal in ordinals:
            targets.append((video_id, ordinal, object_root / video_id / f"{ordinal:03d}.json"))
    for video_id, groups in DUPLICATE_GROUPS.items():
        for group in groups:
            for ordinal in group["n_values"]:
                targets.append((video_id, ordinal, object_root / video_id / f"{ordinal:03d}.json"))
    return targets


def resolve_duplicate_cases(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for video_id, groups in DUPLICATE_GROUPS.items():
        partition = SAMPLE_PARTITIONS[video_id]
        keyframe_dir = root / "keyframes" / "keyframes" / partition / "keyframes" / video_id
        object_dir = root / "objects-aic25-b1" / "objects" / video_id
        clip_path = root / "clip-features-32-aic25-b1" / "clip-features-32" / f"{video_id}.npy"
        clip_rows = 0
        if clip_path.is_file():
            clip = np.load(clip_path, mmap_mode="r", allow_pickle=False)
            clip_rows = int(clip.shape[0]) if clip.ndim else 0
        for group in groups:
            related: list[dict[str, Any]] = []
            for ordinal in group["n_values"]:
                stem = f"{ordinal:03d}"
                keyframe_path = keyframe_dir / f"{stem}.jpg"
                object_path = object_dir / f"{stem}.json"
                row_index = ordinal - 1
                keyframe_exists = keyframe_path.is_file()
                object_exists = object_path.is_file()
                row_exists = 0 <= row_index < clip_rows
                if not keyframe_exists:
                    issues.append(
                        issue(
                            "ERROR",
                            "MISSING_DUPLICATE_KEYFRAME",
                            video_id=video_id,
                            ordinal_n=ordinal,
                            asset_type="KEYFRAME",
                            path=keyframe_path,
                            message="Expected zero-padded duplicate keyframe is missing",
                        )
                    )
                if not object_exists:
                    issues.append(
                        issue(
                            "ERROR",
                            "MISSING_DUPLICATE_OBJECT_JSON",
                            video_id=video_id,
                            ordinal_n=ordinal,
                            asset_type="OBJECT",
                            path=object_path,
                            message="Expected zero-padded duplicate Object JSON is missing",
                        )
                    )
                if not row_exists:
                    issues.append(
                        issue(
                            "ERROR",
                            "CLIP_ROW_OUT_OF_RANGE",
                            video_id=video_id,
                            ordinal_n=ordinal,
                            asset_type="CLIP",
                            path=clip_path,
                            message=f"NPY row {row_index} is outside row count {clip_rows}",
                        )
                    )
                related.append(
                    {
                        "n": ordinal,
                        "ordinal_stem": stem,
                        "keyframe": {
                            "path": str(keyframe_path),
                            "exists": keyframe_exists,
                            "size_bytes": keyframe_path.stat().st_size if keyframe_exists else None,
                        },
                        "object": {
                            "path": str(object_path),
                            "exists": object_exists,
                            "size_bytes": object_path.stat().st_size if object_exists else None,
                        },
                        "clip": {"row_index": row_index, "row_exists": row_exists},
                    }
                )
            key_sizes = [item["keyframe"]["size_bytes"] for item in related]
            object_sizes = [item["object"]["size_bytes"] for item in related]
            cases.append(
                {
                    "video_id": video_id,
                    "frame_idx": group["frame_idx"],
                    "n_values": group["n_values"],
                    "related_assets": related,
                    "keyframe_sizes_equal": None not in key_sizes and len(set(key_sizes)) == 1,
                    "object_sizes_equal": None not in object_sizes and len(set(object_sizes)) == 1,
                    "all_expected_files_exist": all(
                        item["keyframe"]["exists"]
                        and item["object"]["exists"]
                        and item["clip"]["row_exists"]
                        for item in related
                    ),
                    "classification": "DUPLICATE_MAPPING_PRESERVED",
                }
            )
    return cases, issues


def _load_v02(path: str | Path | None) -> tuple[bool, list[dict[str, Any]]]:
    if path is None:
        return False, []
    candidate = Path(path)
    if not candidate.is_file():
        return False, [
            issue(
                "WARNING",
                "V02_ARTIFACT_INVALID",
                video_id=None,
                ordinal_n=None,
                asset_type="V02_ARTIFACT",
                path=candidate,
                message="Optional v0.2 artifact was not found; focused constants will be used",
            )
        ]
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, [
            issue(
                "WARNING",
                "V02_ARTIFACT_INVALID",
                video_id=None,
                ordinal_n=None,
                asset_type="V02_ARTIFACT",
                path=candidate,
                message=str(error),
            )
        ]
    valid = payload.get("survey_version") == "0.2.0" and isinstance(
        payload.get("selected_video_ids"), list
    )
    if valid:
        return True, []
    return False, [
        issue(
            "WARNING",
            "V02_ARTIFACT_INVALID",
            video_id=None,
            ordinal_n=None,
            asset_type="V02_ARTIFACT",
            path=candidate,
            message="Artifact does not satisfy the minimal v0.2 summary contract",
        )
    ]


def run_patch(
    dataset_root: str | Path,
    *,
    limits: PatchLimits | None = None,
    v02_summary: str | Path | None = None,
    strict_root: bool = False,
) -> PatchResult:
    started_at = _now()
    selected_limits = limits or PatchLimits()
    root = validate_root(dataset_root, strict_root=strict_root)
    reused, issues = _load_v02(v02_summary)
    targets = _object_targets(root)[: selected_limits.max_object_json_total]
    samples: list[dict[str, Any]] = []
    boxes_remaining = selected_limits.max_boxes_total
    for video_id, ordinal, path in targets:
        per_file = min(selected_limits.max_boxes_per_file, boxes_remaining)
        sample, sample_issues = inspect_parallel_object(
            path,
            video_id=video_id,
            ordinal_n=ordinal,
            max_bytes=selected_limits.max_object_json_bytes,
            max_boxes=max(0, per_file),
        )
        sampled_boxes = sample.get("bbox_observation", {}).get("sampled_box_count", 0)
        boxes_remaining = max(0, boxes_remaining - sampled_boxes)
        samples.append(sample)
        issues.extend(sample_issues)
    schema = aggregate_object_schema(samples)
    duplicate_cases, duplicate_issues = resolve_duplicate_cases(root)
    issues.extend(duplicate_issues)
    verified = [
        "[VERIFIED] Expected Object fields were inspected as bounded parallel arrays.",
        "[VERIFIED] Duplicate ordinals resolve through zero-padded n:03d filenames.",
        "[VERIFIED] CLIP duplicate row index is n-1 and checked against mmap row bounds.",
    ]
    inferred = [
        f"[INFERRED] Coordinate scale hypothesis: {schema['coordinate_scale_hypothesis']} "
        f"({schema['coordinate_scale_confidence']}).",
    ]
    unknown = [
        "[UNKNOWN] Bounding-box coordinate order remains unknown.",
        "[UNKNOWN] Equal file sizes do not establish image or detection semantic equality.",
    ]
    blocking_codes = {
        "OBJECT_FIELD_MISSING",
        "OBJECT_FIELD_TYPE_MISMATCH",
        "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH",
        "MISSING_DUPLICATE_KEYFRAME",
        "MISSING_DUPLICATE_OBJECT_JSON",
        "CLIP_ROW_OUT_OF_RANGE",
    }
    blocking = any(item["code"] in blocking_codes or item["severity"] == "ERROR" for item in issues)
    ready = (
        not blocking
        and schema["files_inspected"] > 0
        and schema["parallel_arrays_invalid_count"] == 0
        and schema["parallel_arrays_valid_count"] == schema["files_inspected"]
        and bool(schema["bbox_item_length_distribution"])
        and bool(schema["class_entity_item_types"])
        and bool(schema["class_label_item_types"])
        and bool(schema["class_name_item_types"])
        and bool(schema["score_item_types"])
        and all(case["all_expected_files_exist"] for case in duplicate_cases)
    )
    summary = {
        "survey_version": SURVEY_VERSION,
        "patch_scope": [
            "OBJECT_PARALLEL_ARRAY_SCHEMA",
            "ZERO_PADDED_DUPLICATE_CASE_LOOKUP",
        ],
        "dataset_root": str(root),
        "started_at": started_at,
        "completed_at": _now(),
        "limits": asdict(selected_limits),
        "reused_v02_artifact": reused,
        "id_set_scan_rerun": False,
        "full_v02_rerun": False,
        "selected_video_ids": list(SAMPLE_PARTITIONS),
        "object_json_paths_requested": [str(path) for _, _, path in targets],
        "object_json_files_inspected": schema["files_inspected"],
        "object_schema_summary": schema,
        "duplicate_case_studies": duplicate_cases,
        "verified_contracts": verified,
        "inferred_contracts": inferred,
        "unknown_contracts": unknown,
        "issues_summary": {
            "total": len(issues),
            "by_code": dict(Counter(item["code"] for item in issues)),
            "by_severity": dict(Counter(item["severity"] for item in issues)),
        },
        "readiness": {
            "status": (
                "READY_FOR_STAGE_0_IMPLEMENTATION_PLANNING"
                if ready
                else "OBJECT_SCHEMA_FOLLOWUP_REQUIRED"
            ),
            "reason": (
                "Parallel arrays and duplicate-related assets passed focused validation"
                if ready
                else "One or more focused schema or duplicate-asset checks remain unresolved"
            ),
        },
        "zip_artifact": {"path": None, "size_bytes": 0, "members": list(OUTPUT_NAMES)},
        "disclaimer": "This is a bounded v0.2.1 schema patch, not a complete Data Audit.",
    }
    return PatchResult(summary, samples, duplicate_cases, issues)


def _markdown(summary: dict[str, Any]) -> str:
    schema = summary["object_schema_summary"]
    sections = [
        ("# 1. Patch Scope", [f"[VERIFIED] {value}" for value in summary["patch_scope"]]),
        (
            "# 2. Files Inspected",
            [
                "[VERIFIED] Requested "
                f"{len(summary['object_json_paths_requested'])}; inspected "
                f"{summary['object_json_files_inspected']}."
            ],
        ),
        (
            "# 3. Parallel-Array Schema",
            [
                f"[VERIFIED] valid={schema['parallel_arrays_valid_count']}, "
                f"invalid={schema['parallel_arrays_invalid_count']}"
            ],
        ),
        (
            "# 4. Field Types and Lengths",
            [f"[VERIFIED] {json.dumps(schema['field_type_summary'], ensure_ascii=False)}"],
        ),
        (
            "# 5. Detection Counts",
            [
                f"[VERIFIED] min={schema['detection_count_min']}, "
                f"max={schema['detection_count_max']}, "
                f"total={schema['detection_count_total_in_sample']}"
            ],
        ),
        (
            "# 6. Bounding-Box Observations",
            [
                f"[INFERRED] scale={schema['coordinate_scale_hypothesis']}; "
                "bbox_order=UNKNOWN; "
                f"lengths={schema['bbox_item_length_distribution']}"
            ],
        ),
        (
            "# 7. Class and Score Observations",
            [
                f"[VERIFIED] entities={schema['class_entity_item_types']}; "
                f"labels={schema['class_label_item_types']}; "
                f"names={schema['class_name_item_types']}; "
                f"scores={schema['score_item_types']}"
            ],
        ),
        (
            "# 8. Empty Detection Behavior",
            [
                f"[UNKNOWN] status={schema['empty_detection_behavior_status']}; "
                f"observed={schema['empty_detection_representation']}"
            ],
        ),
        (
            "# 9. Duplicate frame_idx Asset Resolution",
            [
                f"[VERIFIED] {case['video_id']} frame_idx={case['frame_idx']} "
                f"n={case['n_values']} all_exist={case['all_expected_files_exist']}"
                for case in summary["duplicate_case_studies"]
            ],
        ),
        ("# 10. Verified Contracts", summary["verified_contracts"]),
        ("# 11. Inferred Contracts", summary["inferred_contracts"]),
        ("# 12. Unknown Contracts", summary["unknown_contracts"]),
        (
            "# 13. Issues",
            [f"[VERIFIED] {json.dumps(summary['issues_summary'], ensure_ascii=False)}"],
        ),
        (
            "# 14. Readiness Decision",
            [f"[VERIFIED] {summary['readiness']['status']}: {summary['readiness']['reason']}"],
        ),
        ("# 15. ZIP Artifact", [f"[VERIFIED] {summary['zip_artifact']}"]),
    ]
    lines: list[str] = []
    for title, values in sections:
        lines.extend([title, "", *[f"- {value}" for value in values], ""])
    return "\n".join(lines)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_outputs(result: PatchResult, output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root).expanduser().resolve(strict=False)
    if _is_within(root, KAGGLE_INPUT_ROOT.resolve(strict=False)):
        raise ValueError("Patch output must not be written below /kaggle/input")
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in OUTPUT_NAMES}
    zip_path = root / "cross_asset_survey_v021.zip"
    paths["object_schema_samples_v021.jsonl"] = root / "object_schema_samples_v021.jsonl"
    _write_jsonl(paths["object_schema_samples_v021.jsonl"], result.object_samples)
    _write_jsonl(paths["duplicate_case_studies_v021.jsonl"], result.duplicate_cases)
    _write_jsonl(paths["issues_v021.jsonl"], result.issues)
    result.summary["zip_artifact"].update({"path": str(zip_path), "members": list(OUTPUT_NAMES)})
    for _ in range(4):
        paths["patch_summary_v021.json"].write_text(
            json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        paths["patch_report_v021.md"].write_text(_markdown(result.summary), encoding="utf-8")
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for name in OUTPUT_NAMES:
                archive.write(paths[name], arcname=name)
        size = zip_path.stat().st_size
        if result.summary["zip_artifact"]["size_bytes"] == size:
            break
        result.summary["zip_artifact"]["size_bytes"] = size
    paths["zip"] = zip_path
    return paths


def result_json(result: PatchResult) -> str:
    return json.dumps(result.summary, ensure_ascii=False, indent=2)
