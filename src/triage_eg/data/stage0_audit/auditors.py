"""Bounded-memory asset auditors for Stage 0."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.data.object_numeric_contract import parse_label, parse_numeric
from triage_eg.data.stage0_audit.asset_resolver import relative
from triage_eg.data.stage0_audit.contracts import AUDIT_VERSION, AssetPaths, AuditIssue, issue

MAPPING_COLUMNS = ["n", "pts_time", "fps", "frame_idx"]
OBJECT_FIELDS = [
    "detection_boxes",
    "detection_class_entities",
    "detection_class_labels",
    "detection_class_names",
    "detection_scores",
]
METADATA_FIELDS = [
    "author",
    "channel_id",
    "channel_url",
    "description",
    "keywords",
    "length",
    "publish_date",
    "thumbnail_url",
    "title",
    "watch_url",
]


def audit_mapping(
    path: Path, video_id: str, *, max_rows: int = 1_000_000
) -> tuple[dict[str, Any], list[dict[str, Any]], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        issues.append(
            issue(
                "MISSING_MAPPING",
                "ERROR",
                video_id=video_id,
                asset_type="MAPPING",
                path=path,
                btc=True,
                message="Mapping CSV is missing",
            )
        )
        return (
            {"video_id": video_id, "status": "FAILED", "row_count": 0, "ordinals": []},
            rows,
            issues,
        )
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != MAPPING_COLUMNS:
                issues.append(
                    issue(
                        "MAPPING_COLUMN_MISMATCH",
                        "ERROR",
                        video_id=video_id,
                        asset_type="MAPPING",
                        path=path,
                        btc=True,
                        message="Mapping columns must match exactly",
                        evidence={"observed": reader.fieldnames, "expected": MAPPING_COLUMNS},
                    )
                )
                return (
                    {"video_id": video_id, "status": "FAILED", "row_count": 0, "ordinals": []},
                    rows,
                    issues,
                )
            for index, raw in enumerate(reader):
                if index >= max_rows:
                    issues.append(
                        issue(
                            "MAPPING_ROW_LIMIT_EXCEEDED",
                            "ERROR",
                            video_id=video_id,
                            asset_type="MAPPING",
                            path=path,
                            btc=True,
                        )
                    )
                    break
                parsed: dict[str, int | float] = {}
                specifications = (
                    ("n", int, lambda value: value > 0, "MAPPING_N_INVALID"),
                    ("pts_time", float, lambda value: math.isfinite(value), "MAPPING_PTS_INVALID"),
                    (
                        "fps",
                        float,
                        lambda value: math.isfinite(value) and value > 0,
                        "MAPPING_FPS_INVALID",
                    ),
                    (
                        "frame_idx",
                        int,
                        lambda value: value >= 0,
                        "MAPPING_FRAME_IDX_INVALID",
                    ),
                )
                invalid: tuple[str, str] | None = None
                for field_name, converter, validator, code in specifications:
                    try:
                        value = converter(raw[field_name])
                        if not validator(value):
                            raise ValueError("value is outside the valid contract")
                        parsed[field_name] = value
                    except (TypeError, ValueError) as error:
                        invalid = (code, str(error))
                        break
                if invalid is not None:
                    issues.append(
                        issue(
                            invalid[0],
                            "ERROR",
                            video_id=video_id,
                            asset_type="MAPPING",
                            path=path,
                            btc=True,
                            message=invalid[1],
                            evidence={"row": index + 2},
                        )
                    )
                    continue
                n = int(parsed["n"])
                pts = float(parsed["pts_time"])
                fps = float(parsed["fps"])
                frame_idx = int(parsed["frame_idx"])
                rows.append({"n": n, "pts_time": pts, "fps": fps, "frame_idx": frame_idx})
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        issues.append(
            issue(
                "MAPPING_MALFORMED",
                "ERROR",
                video_id=video_id,
                asset_type="MAPPING",
                path=path,
                btc=True,
                message=str(error),
            )
        )
    ordinals = [row["n"] for row in rows]
    if ordinals != list(range(1, len(rows) + 1)):
        code = "MAPPING_N_INVALID" if ordinals and ordinals[0] != 1 else "MAPPING_N_NON_CONTIGUOUS"
        issues.append(
            issue(
                code,
                "ERROR",
                video_id=video_id,
                asset_type="MAPPING",
                path=path,
                btc=True,
                evidence={"first": ordinals[:5], "count": len(ordinals)},
            )
        )
    pts_values = [row["pts_time"] for row in rows]
    frame_values = [row["frame_idx"] for row in rows]
    if any(
        current < previous for previous, current in zip(pts_values, pts_values[1:], strict=False)
    ):
        issues.append(
            issue(
                "MAPPING_PTS_NON_MONOTONIC",
                "ERROR",
                video_id=video_id,
                asset_type="MAPPING",
                path=path,
                btc=True,
            )
        )
    if any(
        current < previous
        for previous, current in zip(frame_values, frame_values[1:], strict=False)
    ):
        issues.append(
            issue(
                "MAPPING_FRAME_IDX_NON_MONOTONIC",
                "ERROR",
                video_id=video_id,
                asset_type="MAPPING",
                path=path,
                btc=True,
            )
        )
    groups: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        groups[row["frame_idx"]].append(row["n"])
    duplicates = {frame: values for frame, values in groups.items() if len(values) > 1}
    for frame, values in sorted(duplicates.items()):
        issues.append(
            issue(
                "DUPLICATE_FRAME_IDX",
                "WARNING",
                video_id=video_id,
                asset_type="MAPPING",
                path=path,
                message="Duplicate original frame mapping preserved",
                evidence={"frame_idx": frame, "n_values": values},
            )
        )
    status = "VALID" if rows and not any(item.blocks_btc_baseline for item in issues) else "FAILED"
    return (
        {
            "video_id": video_id,
            "status": status,
            "row_count": len(rows),
            "ordinals": ordinals,
            "duplicate_groups": duplicates,
            "fps_values": sorted(set(row["fps"] for row in rows)),
        },
        rows,
        issues,
    )


def _jpg_header_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as stream:
        start = stream.read(2)
        stream.seek(-2, 2)
        end = stream.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def audit_keyframes(
    root: Path, paths: AssetPaths, expected: set[int]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    observed: dict[int, Path] = {}
    directory = paths.keyframe_directory
    if not directory.is_dir():
        issues.append(
            issue(
                "MISSING_KEYFRAME_DIRECTORY",
                "ERROR",
                video_id=paths.video_id,
                asset_type="KEYFRAME",
                path=directory,
                btc=True,
            )
        )
    else:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            if path.suffix.lower() != ".jpg" or not path.stem.isdigit() or len(path.stem) != 3:
                issues.append(
                    issue(
                        "KEYFRAME_FILENAME_INVALID",
                        "WARNING",
                        video_id=paths.video_id,
                        asset_type="KEYFRAME",
                        path=path,
                    )
                )
                continue
            observed[int(path.stem)] = path
    for ordinal in sorted(expected - set(observed)):
        issues.append(
            issue(
                "KEYFRAME_MISSING",
                "ERROR",
                video_id=paths.video_id,
                ordinal_n=ordinal,
                asset_type="KEYFRAME",
                path=directory / f"{ordinal:03d}.jpg",
                btc=True,
            )
        )
    for ordinal in sorted(set(observed) - expected):
        issues.append(
            issue(
                "KEYFRAME_EXTRA",
                "WARNING",
                video_id=paths.video_id,
                ordinal_n=ordinal,
                asset_type="KEYFRAME",
                path=observed[ordinal],
            )
        )
    details: dict[int, dict[str, Any]] = {}
    for ordinal, path in observed.items():
        size = path.stat().st_size
        header = _jpg_header_valid(path)
        if size == 0:
            issues.append(
                issue(
                    "KEYFRAME_FILE_EMPTY",
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="KEYFRAME",
                    path=path,
                    btc=True,
                )
            )
        elif not header:
            issues.append(
                issue(
                    "KEYFRAME_HEADER_INVALID",
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="KEYFRAME",
                    path=path,
                    btc=True,
                )
            )
        details[ordinal] = {
            "path": relative(root, path),
            "exists": True,
            "size_bytes": size,
            "header_valid": header,
        }
    return (
        {
            "video_id": paths.video_id,
            "directory_exists": directory.is_dir(),
            "observed_count": len(observed),
            "ordinals": sorted(observed),
            "ordinal_set_matches_mapping": set(observed) == expected,
        },
        details,
        issues,
    )


def audit_clip(
    root: Path,
    paths: AssetPaths,
    mapping_count: int,
    *,
    expected_dimension: int,
    mode: str,
    chunk_rows: int = 4096,
) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    base = {
        "audit_version": AUDIT_VERSION,
        "video_id": paths.video_id,
        "relative_path": relative(root, paths.clip),
        "shape": [],
        "row_count": 0,
        "dimension": 0,
        "dtype": None,
        "mapping_row_count": mapping_count,
        "row_count_matches_mapping": False,
        "contains_non_finite": None,
        "norm_min": None,
        "norm_max": None,
        "norm_mean": None,
        "validation_mode": mode.upper(),
        "issues": [],
    }
    if not paths.clip.is_file():
        issues.append(
            issue(
                "MISSING_CLIP",
                "ERROR",
                video_id=paths.video_id,
                asset_type="CLIP",
                path=paths.clip,
                btc=True,
            )
        )
        return base, issues
    try:
        matrix = np.load(paths.clip, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        issues.append(
            issue(
                "CLIP_MALFORMED",
                "ERROR",
                video_id=paths.video_id,
                asset_type="CLIP",
                path=paths.clip,
                btc=True,
                message=str(error),
            )
        )
        return base, issues
    base.update({"shape": list(matrix.shape), "dtype": str(matrix.dtype)})
    if matrix.ndim != 2:
        issues.append(
            issue(
                "CLIP_NDIM_MISMATCH",
                "ERROR",
                video_id=paths.video_id,
                asset_type="CLIP",
                path=paths.clip,
                btc=True,
                evidence={"shape": list(matrix.shape)},
            )
        )
        return base, issues
    rows, dimension = map(int, matrix.shape)
    base.update(
        {
            "row_count": rows,
            "dimension": dimension,
            "row_count_matches_mapping": rows == mapping_count,
        }
    )
    if rows != mapping_count:
        issues.append(
            issue(
                "CLIP_ROW_COUNT_MISMATCH",
                "ERROR",
                video_id=paths.video_id,
                asset_type="CLIP",
                path=paths.clip,
                btc=True,
                evidence={"clip": rows, "mapping": mapping_count},
            )
        )
    if dimension != expected_dimension:
        issues.append(
            issue(
                "CLIP_DIMENSION_MISMATCH",
                "ERROR",
                video_id=paths.video_id,
                asset_type="CLIP",
                path=paths.clip,
                btc=True,
                evidence={"observed": dimension, "expected": expected_dimension},
            )
        )
    if mode == "full":
        finite = True
        norm_min = math.inf
        norm_max = -math.inf
        norm_sum = 0.0
        norm_count = 0
        for start in range(0, rows, chunk_rows):
            chunk = np.asarray(matrix[start : start + chunk_rows])
            finite = finite and bool(np.isfinite(chunk).all())
            norms = np.linalg.norm(chunk.astype(np.float64, copy=False), axis=1)
            if len(norms):
                norm_min = min(norm_min, float(norms.min()))
                norm_max = max(norm_max, float(norms.max()))
                norm_sum += float(norms.sum())
                norm_count += len(norms)
        base.update(
            {
                "contains_non_finite": not finite,
                "norm_min": norm_min if norm_count else None,
                "norm_max": norm_max if norm_count else None,
                "norm_mean": norm_sum / norm_count if norm_count else None,
            }
        )
        if not finite:
            issues.append(
                issue(
                    "CLIP_NON_FINITE",
                    "ERROR",
                    video_id=paths.video_id,
                    asset_type="CLIP",
                    path=paths.clip,
                    btc=True,
                )
            )
    base["issues"] = [item.code for item in issues]
    return base, issues


def audit_objects(
    root: Path, paths: AssetPaths, expected: set[int], *, mode: str, max_bytes: int
) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    directory = paths.object_directory
    observed: dict[int, Path] = {}
    if not directory.is_dir():
        issues.append(
            issue(
                "MISSING_OBJECT_DIRECTORY",
                "ERROR",
                video_id=paths.video_id,
                asset_type="OBJECT",
                path=directory,
                btc=True,
            )
        )
    else:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() == ".json" and path.stem.isdigit():
                observed[int(path.stem)] = path
    missing = expected - set(observed)
    extra = set(observed) - expected
    for ordinal in sorted(missing):
        issues.append(
            issue(
                "OBJECT_MISSING",
                "ERROR",
                video_id=paths.video_id,
                ordinal_n=ordinal,
                asset_type="OBJECT",
                path=directory / f"{ordinal:03d}.json",
                btc=True,
            )
        )
    for ordinal in sorted(extra):
        issues.append(
            issue(
                "OBJECT_EXTRA",
                "WARNING",
                video_id=paths.video_id,
                ordinal_n=ordinal,
                asset_type="OBJECT",
                path=observed[ordinal],
            )
        )
    stats: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "video_id": paths.video_id,
        "object_directory": relative(root, directory),
        "expected_file_count": len(expected),
        "observed_file_count": len(observed),
        "ordinal_set_matches_mapping": set(observed) == expected,
        "json_files_parsed": 0,
        "detections_observed": 0,
        "valid_detections": 0,
        "invalid_detections": 0,
        "parallel_array_files_valid": 0,
        "parallel_array_files_invalid": 0,
        "coordinate_valid_count": 0,
        "coordinate_invalid_count": 0,
        "coordinate_min": None,
        "coordinate_max": None,
        "coordinates_outside_zero_one": 0,
        "score_valid_count": 0,
        "score_invalid_count": 0,
        "score_min": None,
        "score_max": None,
        "scores_outside_zero_one": 0,
        "label_valid_count": 0,
        "label_invalid_count": 0,
        "label_min": None,
        "label_max": None,
        "numeric_string_count": 0,
        "native_number_count": 0,
        "trimmed_whitespace_count": 0,
        "bbox_order": "UNKNOWN",
        "issues": [],
    }
    if mode == "filenames":
        stats["issues"] = [item.code for item in issues]
        return stats, issues
    coordinate_values: list[float] = []
    score_values: list[float] = []
    label_values: list[int] = []
    for ordinal in sorted(expected & set(observed)):
        path = observed[ordinal]
        if path.stat().st_size > max_bytes:
            issues.append(
                issue(
                    "OBJECT_JSON_TOO_LARGE",
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="OBJECT",
                    path=path,
                    btc=True,
                )
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            issues.append(
                issue(
                    "OBJECT_JSON_MALFORMED",
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="OBJECT",
                    path=path,
                    btc=True,
                    message=str(error),
                )
            )
            continue
        stats["json_files_parsed"] += 1
        if not isinstance(payload, dict):
            issues.append(
                issue(
                    "OBJECT_TOP_LEVEL_INVALID",
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="OBJECT",
                    path=path,
                    btc=True,
                )
            )
            continue
        missing_fields = [name for name in OBJECT_FIELDS if name not in payload]
        bad_types = [
            name
            for name in OBJECT_FIELDS
            if name in payload and not isinstance(payload[name], list)
        ]
        if missing_fields or bad_types:
            code = "OBJECT_FIELD_MISSING" if missing_fields else "OBJECT_FIELD_TYPE_MISMATCH"
            issues.append(
                issue(
                    code,
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="OBJECT",
                    path=path,
                    btc=True,
                    evidence={"missing": missing_fields, "bad_types": bad_types},
                )
            )
            stats["parallel_array_files_invalid"] += 1
            continue
        lengths = {name: len(payload[name]) for name in OBJECT_FIELDS}
        if len(set(lengths.values())) != 1:
            issues.append(
                issue(
                    "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH",
                    "ERROR",
                    video_id=paths.video_id,
                    ordinal_n=ordinal,
                    asset_type="OBJECT",
                    path=path,
                    btc=True,
                    evidence=lengths,
                )
            )
            stats["parallel_array_files_invalid"] += 1
            continue
        stats["parallel_array_files_valid"] += 1
        stats["detections_observed"] += lengths[OBJECT_FIELDS[0]]
        for index in range(lengths[OBJECT_FIELDS[0]]):
            detection_valid = True
            box = payload["detection_boxes"][index]
            if not isinstance(box, list) or len(box) != 4:
                issues.append(
                    issue(
                        "OBJECT_BOX_LENGTH_INVALID",
                        "ERROR",
                        video_id=paths.video_id,
                        ordinal_n=ordinal,
                        asset_type="OBJECT",
                        path=path,
                        btc=True,
                        evidence={"detection_index": index, "raw_type": type(box).__name__},
                    )
                )
                stats["coordinate_invalid_count"] += 1
                detection_valid = False
            else:
                for position, raw in enumerate(box):
                    parsed = parse_numeric(raw)
                    _count_raw(stats, parsed)
                    if parsed["parse_status"] != "VALID":
                        stats["coordinate_invalid_count"] += 1
                        detection_valid = False
                        issues.append(
                            issue(
                                "OBJECT_NUMERIC_INVALID",
                                "ERROR",
                                video_id=paths.video_id,
                                ordinal_n=ordinal,
                                asset_type="OBJECT",
                                path=path,
                                btc=True,
                                evidence={
                                    "detection_index": index,
                                    "position": position,
                                    "raw_value": raw,
                                    "status": parsed["parse_status"],
                                },
                            )
                        )
                    else:
                        value = float(parsed["parsed_value"])
                        coordinate_values.append(value)
                        stats["coordinate_valid_count"] += 1
                        if not 0 <= value <= 1:
                            stats["coordinates_outside_zero_one"] += 1
                            issues.append(
                                issue(
                                    "OBJECT_COORDINATE_OUTSIDE_ZERO_ONE",
                                    "WARNING",
                                    video_id=paths.video_id,
                                    ordinal_n=ordinal,
                                    asset_type="OBJECT",
                                    path=path,
                                    evidence={
                                        "detection_index": index,
                                        "position": position,
                                        "raw_value": raw,
                                    },
                                )
                            )
            for field_name, parser, valid_key, invalid_key, values, invalid_code, outside_code in (
                (
                    "detection_scores",
                    parse_numeric,
                    "score_valid_count",
                    "score_invalid_count",
                    score_values,
                    "OBJECT_NUMERIC_INVALID",
                    "OBJECT_SCORE_OUTSIDE_ZERO_ONE",
                ),
                (
                    "detection_class_labels",
                    parse_label,
                    "label_valid_count",
                    "label_invalid_count",
                    label_values,
                    "OBJECT_LABEL_INVALID",
                    None,
                ),
            ):
                raw = payload[field_name][index]
                parsed = parser(raw)
                _count_raw(stats, parsed)
                if parsed["parse_status"] != "VALID":
                    stats[invalid_key] += 1
                    detection_valid = False
                    issues.append(
                        issue(
                            invalid_code,
                            "ERROR",
                            video_id=paths.video_id,
                            ordinal_n=ordinal,
                            asset_type="OBJECT",
                            path=path,
                            btc=True,
                            evidence={
                                "detection_index": index,
                                "field": field_name,
                                "raw_value": raw,
                                "status": parsed["parse_status"],
                            },
                        )
                    )
                else:
                    value = parsed["parsed_value"]
                    values.append(value)
                    stats[valid_key] += 1
                    if outside_code and not 0 <= float(value) <= 1:
                        stats["scores_outside_zero_one"] += 1
                        issues.append(
                            issue(
                                outside_code,
                                "WARNING",
                                video_id=paths.video_id,
                                ordinal_n=ordinal,
                                asset_type="OBJECT",
                                path=path,
                                evidence={"detection_index": index, "raw_value": raw},
                            )
                        )
            stats["valid_detections" if detection_valid else "invalid_detections"] += 1
    for prefix, values in (
        ("coordinate", coordinate_values),
        ("score", score_values),
        ("label", label_values),
    ):
        stats[f"{prefix}_min"] = min(values) if values else None
        stats[f"{prefix}_max"] = max(values) if values else None
    stats["issues"] = list(Counter(item.code for item in issues))
    return stats, issues


def _count_raw(stats: dict[str, Any], parsed: dict[str, Any]) -> None:
    if parsed["raw_type"] == "string":
        stats["numeric_string_count"] += 1
        if parsed["raw_value"] != parsed["trimmed_value"]:
            stats["trimmed_whitespace_count"] += 1
    elif parsed["raw_type"] in {"integer", "float"}:
        stats["native_number_count"] += 1


def audit_metadata(root: Path, paths: AssetPaths) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    base = {
        "audit_version": AUDIT_VERSION,
        "video_id": paths.video_id,
        "relative_path": relative(root, paths.metadata),
        "parse_status": "FAILED",
        "top_level_type": None,
        "keys": [],
        "missing_expected_fields": [],
        "null_fields": [],
        "title_length": None,
        "description_length": None,
        "keyword_count": None,
        "issues": [],
    }
    if not paths.metadata.is_file():
        issues.append(
            issue(
                "MISSING_METADATA",
                "WARNING",
                video_id=paths.video_id,
                asset_type="METADATA",
                path=paths.metadata,
            )
        )
        return base, issues
    try:
        payload = json.loads(paths.metadata.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as error:
        issues.append(
            issue(
                "METADATA_ENCODING_ERROR",
                "WARNING",
                video_id=paths.video_id,
                asset_type="METADATA",
                path=paths.metadata,
                message=str(error),
            )
        )
        return base, issues
    except (OSError, json.JSONDecodeError) as error:
        issues.append(
            issue(
                "METADATA_MALFORMED",
                "WARNING",
                video_id=paths.video_id,
                asset_type="METADATA",
                path=paths.metadata,
                message=str(error),
            )
        )
        return base, issues
    if not isinstance(payload, dict):
        issues.append(
            issue(
                "METADATA_TOP_LEVEL_INVALID",
                "WARNING",
                video_id=paths.video_id,
                asset_type="METADATA",
                path=paths.metadata,
            )
        )
        return base, issues
    missing = [name for name in METADATA_FIELDS if name not in payload]
    nulls = [name for name in METADATA_FIELDS if payload.get(name) is None]
    for name in missing:
        issues.append(
            issue(
                "METADATA_FIELD_MISSING",
                "WARNING",
                video_id=paths.video_id,
                asset_type="METADATA",
                path=paths.metadata,
                evidence={"field": name},
            )
        )
    for name in nulls:
        issues.append(
            issue(
                "METADATA_FIELD_NULL",
                "WARNING",
                video_id=paths.video_id,
                asset_type="METADATA",
                path=paths.metadata,
                evidence={"field": name},
            )
        )
    base.update(
        {
            "parse_status": "SUCCESS",
            "top_level_type": "object",
            "keys": sorted(payload),
            "missing_expected_fields": missing,
            "null_fields": nulls,
            "title_length": len(payload["title"])
            if isinstance(payload.get("title"), str)
            else None,
            "description_length": len(payload["description"])
            if isinstance(payload.get("description"), str)
            else None,
            "keyword_count": len(payload["keywords"])
            if isinstance(payload.get("keywords"), list)
            else None,
            "issues": list(Counter(item.code for item in issues)),
        }
    )
    return base, issues


def _fraction(raw: str | None) -> float | None:
    if not raw or raw in {"0/0", "N/A"}:
        return None
    try:
        numerator, denominator = raw.split("/", 1)
        value = float(numerator) / float(denominator)
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(
    root: Path,
    paths: AssetPaths,
    *,
    timeout: int,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], list[AuditIssue]]:
    issues: list[AuditIssue] = []
    base = {
        "audit_version": AUDIT_VERSION,
        "dataset_version": "aic25-b1",
        "video_id": paths.video_id,
        "video_partition": paths.video_partition,
        "relative_video_path": relative(root, paths.video),
        "file_size_bytes": paths.video.stat().st_size if paths.video.is_file() else 0,
        "probe_status": "NOT_RUN",
        "container_format": None,
        "video_codec": None,
        "width": None,
        "height": None,
        "pixel_format": None,
        "avg_frame_rate_raw": None,
        "r_frame_rate_raw": None,
        "avg_fps": None,
        "r_fps": None,
        "time_base": None,
        "start_time_seconds": None,
        "duration_seconds": None,
        "nb_frames": None,
        "has_video_stream": False,
        "has_audio_stream": False,
        "audio_codec": None,
        "cfr_vfr_indicator": "UNKNOWN",
        "issues": [],
    }
    if not paths.video.is_file():
        issues.append(
            issue(
                "MISSING_VIDEO",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
            )
        )
        base["issues"] = [item.code for item in issues]
        return base, issues
    binary = which("ffprobe")
    if binary is None:
        issues.append(
            issue(
                "FFPROBE_NOT_AVAILABLE",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
            )
        )
        base["issues"] = [item.code for item in issues]
        return base, issues
    try:
        completed = run(
            [
                binary,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(paths.video),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        issues.append(
            issue(
                "VIDEO_PROBE_FAILED",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
                message="ffprobe timed out",
            )
        )
        base["probe_status"] = "FAILED"
        return base, issues
    if completed.returncode != 0:
        issues.append(
            issue(
                "VIDEO_PROBE_FAILED",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
                message=completed.stderr[:500],
            )
        )
        base["probe_status"] = "FAILED"
        return base, issues
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        issues.append(
            issue(
                "VIDEO_PROBE_FAILED",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
                message=str(error),
            )
        )
        base["probe_status"] = "FAILED"
        return base, issues
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video_stream is None:
        issues.append(
            issue(
                "VIDEO_STREAM_MISSING",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
            )
        )
        base["probe_status"] = "FAILED"
        return base, issues
    avg_raw = video_stream.get("avg_frame_rate")
    r_raw = video_stream.get("r_frame_rate")
    avg_fps, r_fps = _fraction(avg_raw), _fraction(r_raw)
    duration_raw = video_stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        duration = None
    try:
        nb_frames = (
            int(video_stream["nb_frames"])
            if video_stream.get("nb_frames") not in {None, "N/A"}
            else None
        )
    except (TypeError, ValueError):
        nb_frames = None
    indicator = (
        "UNKNOWN"
        if avg_fps is None or r_fps is None
        else ("CFR_LIKE" if math.isclose(avg_fps, r_fps, rel_tol=1e-6) else "VFR_LIKE")
    )
    base.update(
        {
            "probe_status": "SUCCESS",
            "container_format": payload.get("format", {}).get("format_name"),
            "video_codec": video_stream.get("codec_name"),
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "pixel_format": video_stream.get("pix_fmt"),
            "avg_frame_rate_raw": avg_raw,
            "r_frame_rate_raw": r_raw,
            "avg_fps": avg_fps,
            "r_fps": r_fps,
            "time_base": video_stream.get("time_base"),
            "start_time_seconds": float(video_stream["start_time"])
            if video_stream.get("start_time") not in {None, "N/A"}
            else None,
            "duration_seconds": duration,
            "nb_frames": nb_frames,
            "has_video_stream": True,
            "has_audio_stream": audio_stream is not None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "cfr_vfr_indicator": indicator,
        }
    )
    if nb_frames is None:
        issues.append(
            issue(
                "VIDEO_FRAME_COUNT_UNKNOWN",
                "WARNING",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
            )
        )
    if not duration or duration <= 0:
        issues.append(
            issue(
                "VIDEO_DURATION_INVALID",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
            )
        )
    if not base["width"] or not base["height"]:
        issues.append(
            issue(
                "VIDEO_DIMENSION_INVALID",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
            )
        )
    if avg_fps is None:
        issues.append(
            issue(
                "VIDEO_FPS_INVALID",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
                raw=True,
            )
        )
    if indicator == "VFR_LIKE":
        issues.append(
            issue(
                "VIDEO_VFR_INDICATOR",
                "WARNING",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
            )
        )
    if audio_stream is None:
        issues.append(
            issue(
                "VIDEO_AUDIO_MISSING",
                "INFO",
                video_id=paths.video_id,
                asset_type="VIDEO",
                path=paths.video,
            )
        )
    base["issues"] = [item.code for item in issues]
    return base, issues
