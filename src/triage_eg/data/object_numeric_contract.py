"""Final bounded numeric-string contract validation for Object JSON assets."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from triage_eg.data.cross_asset_patch_v021 import _object_targets
from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within

SURVEY_VERSION = "0.2.2"
DEFAULT_DATASET_ROOT = Path("/kaggle/input/datasets/nadkli/dataset-aic")
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/object_numeric_contract_v022")
OUTPUT_NAMES = (
    "numeric_contract_summary_v022.json",
    "numeric_contract_report_v022.md",
    "file_numeric_results_v022.jsonl",
    "invalid_numeric_samples_v022.jsonl",
    "issues_v022.jsonl",
)
LABEL_PATTERN = re.compile(r"^[+-]?\d+$", re.ASCII)
VALID_EXAMPLE_LIMIT = 20
INVALID_EXAMPLE_LIMIT = 20


@dataclass(frozen=True)
class NumericLimits:
    max_object_json_total: int = 15
    max_object_json_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not 1 <= self.max_object_json_total <= 15:
            raise ValueError("max_object_json_total must be between 1 and 15")
        if self.max_object_json_bytes <= 0:
            raise ValueError("max_object_json_bytes must be greater than zero")


@dataclass
class NumericContractResult:
    summary: dict[str, Any]
    file_results: list[dict[str, Any]] = field(default_factory=list)
    invalid_samples: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _raw_type(value: Any) -> str:
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
    return "other"


def _parse_result(value: Any, *, parsed: int | float | None, status: str) -> dict[str, Any]:
    return {
        "raw_value": value,
        "raw_type": _raw_type(value),
        "trimmed_value": value.strip() if isinstance(value, str) else None,
        "parse_status": status,
        "parsed_value": parsed,
        "is_finite": parsed is not None and math.isfinite(float(parsed)),
    }


def parse_numeric(value: Any) -> dict[str, Any]:
    """Parse a coordinate/score without locale coercion or silent defaults."""
    if value is None:
        return _parse_result(value, parsed=None, status="NULL")
    if isinstance(value, bool):
        return _parse_result(value, parsed=None, status="UNSUPPORTED_TYPE")
    if isinstance(value, int | float):
        parsed = float(value)
        return _parse_result(
            value,
            parsed=parsed if math.isfinite(parsed) else None,
            status="VALID" if math.isfinite(parsed) else "NON_FINITE",
        )
    if not isinstance(value, str):
        return _parse_result(value, parsed=None, status="UNSUPPORTED_TYPE")
    trimmed = value.strip()
    if not trimmed:
        return _parse_result(value, parsed=None, status="EMPTY")
    try:
        parsed = float(trimmed)
    except ValueError:
        return _parse_result(value, parsed=None, status="INVALID_FORMAT")
    if not math.isfinite(parsed):
        return _parse_result(value, parsed=None, status="NON_FINITE")
    return _parse_result(value, parsed=parsed, status="VALID")


def parse_label(value: Any) -> dict[str, Any]:
    """Parse an exact decimal integer label without float-string coercion."""
    if value is None:
        return _parse_result(value, parsed=None, status="NULL")
    if isinstance(value, bool):
        return _parse_result(value, parsed=None, status="UNSUPPORTED_TYPE")
    if isinstance(value, int):
        return _parse_result(value, parsed=value, status="VALID")
    if isinstance(value, float):
        if not math.isfinite(value):
            return _parse_result(value, parsed=None, status="NON_FINITE")
        if not value.is_integer():
            return _parse_result(value, parsed=None, status="INVALID_FORMAT")
        return _parse_result(value, parsed=int(value), status="VALID")
    if not isinstance(value, str):
        return _parse_result(value, parsed=None, status="UNSUPPORTED_TYPE")
    trimmed = value.strip()
    if not trimmed:
        return _parse_result(value, parsed=None, status="EMPTY")
    if not LABEL_PATTERN.fullmatch(trimmed):
        lowered = trimmed.lower()
        status = (
            "NON_FINITE"
            if lowered in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
            else "INVALID_FORMAT"
        )
        return _parse_result(value, parsed=None, status=status)
    return _parse_result(value, parsed=int(trimmed, 10), status="VALID")


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


def _issue(
    code: str,
    *,
    video_id: str,
    ordinal_n: int,
    detection_index: int | None,
    field: str,
    raw_value: Any,
    message: str,
    evidence: dict[str, Any] | None = None,
    severity: str = "ERROR",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "video_id": video_id,
        "ordinal_n": ordinal_n,
        "detection_index": detection_index,
        "field": field,
        "raw_value": raw_value,
        "message": message,
        "evidence": evidence or {},
    }


def _code(prefix: str, result: dict[str, Any], *, label_not_integer: bool = False) -> str:
    if label_not_integer:
        return "LABEL_NOT_INTEGER"
    return f"{prefix}_{result['parse_status']}"


def _empty_accumulator() -> dict[str, Any]:
    return {
        "total": 0,
        "valid": 0,
        "raw_types": Counter(),
        "statuses": Counter(),
        "values": [],
        "position_values": {position: [] for position in range(4)},
        "valid_examples": [],
        "invalid_examples": [],
    }


def _observe(acc: dict[str, Any], parsed: dict[str, Any], context: dict[str, Any]) -> None:
    acc["total"] += 1
    acc["raw_types"][parsed["raw_type"]] += 1
    acc["statuses"][parsed["parse_status"]] += 1
    example = {**context, **parsed}
    if parsed["parse_status"] == "VALID":
        acc["valid"] += 1
        acc["values"].append(parsed["parsed_value"])
        position = context.get("position")
        if position in acc["position_values"]:
            acc["position_values"][position].append(parsed["parsed_value"])
        if len(acc["valid_examples"]) < VALID_EXAMPLE_LIMIT:
            acc["valid_examples"].append(example)
    elif len(acc["invalid_examples"]) < INVALID_EXAMPLE_LIMIT:
        acc["invalid_examples"].append(example)


def _load_object(path: Path, max_bytes: int) -> tuple[Any | None, str, str | None]:
    if not path.is_file():
        return None, "MISSING", "Target Object JSON does not exist"
    if path.stat().st_size > max_bytes:
        return None, "TOO_LARGE", f"Object JSON exceeds {max_bytes} bytes"
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8-sig")), "INSPECTED", None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, "MALFORMED", str(error)


def inspect_file(
    path: Path, *, video_id: str, ordinal_n: int, max_bytes: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payload, status, error = _load_object(path, max_bytes)
    accumulators = {name: _empty_accumulator() for name in ("coordinate", "score", "label")}
    issues: list[dict[str, Any]] = []
    result = {
        "video_id": video_id,
        "ordinal_n": ordinal_n,
        "path": str(path),
        "inspection_status": status,
        "file_normalization_status": "FAILED",
        "detection_count": 0,
        "valid_detection_count": 0,
        "invalid_detection_count": 0,
        "raw_values_preserved": True,
    }
    if status != "INSPECTED":
        code = {
            "MISSING": "MISSING_TARGET_OBJECT_JSON",
            "TOO_LARGE": "OBJECT_JSON_TOO_LARGE",
            "MALFORMED": "MALFORMED_OBJECT_JSON",
        }[status]
        issues.append(
            _issue(
                code,
                video_id=video_id,
                ordinal_n=ordinal_n,
                detection_index=None,
                field="object",
                raw_value=None,
                message=error or code,
            )
        )
        return result, accumulators, issues
    fields = ("detection_boxes", "detection_class_labels", "detection_scores")
    if not isinstance(payload, dict) or any(
        not isinstance(payload.get(name), list) for name in fields
    ):
        issues.append(
            _issue(
                "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH",
                video_id=video_id,
                ordinal_n=ordinal_n,
                detection_index=None,
                field="object",
                raw_value=None,
                message="Required positional fields must be arrays",
            )
        )
        return result, accumulators, issues
    lengths = {name: len(payload[name]) for name in fields}
    if len(set(lengths.values())) != 1:
        issues.append(
            _issue(
                "OBJECT_PARALLEL_ARRAY_LENGTH_MISMATCH",
                video_id=video_id,
                ordinal_n=ordinal_n,
                detection_index=None,
                field="object",
                raw_value=None,
                message="Positional numeric arrays have unequal lengths",
                evidence=lengths,
            )
        )
        return result, accumulators, issues
    detection_count = next(iter(lengths.values()))
    result["detection_count"] = detection_count
    for index in range(detection_count):
        detection_valid = True
        box = payload["detection_boxes"][index]
        if not isinstance(box, list) or len(box) != 4:
            detection_valid = False
            parsed_coordinates = [_parse_result(box, parsed=None, status="UNSUPPORTED_TYPE")]
        else:
            parsed_coordinates = [parse_numeric(value) for value in box]
        for position, parsed in enumerate(parsed_coordinates):
            context = {
                "video_id": video_id,
                "ordinal_n": ordinal_n,
                "detection_index": index,
                "field": "detection_boxes",
                "position": position,
            }
            _observe(accumulators["coordinate"], parsed, context)
            if parsed["parse_status"] != "VALID":
                detection_valid = False
                issues.append(
                    _issue(
                        _code("COORDINATE", parsed),
                        video_id=video_id,
                        ordinal_n=ordinal_n,
                        detection_index=index,
                        field="detection_boxes",
                        raw_value=parsed["raw_value"],
                        message="Coordinate failed fail-closed numeric parsing",
                        evidence={"position": position, "parse_status": parsed["parse_status"]},
                    )
                )
        for kind, field_name, parser in (
            ("score", "detection_scores", parse_numeric),
            ("label", "detection_class_labels", parse_label),
        ):
            raw_value = payload[field_name][index]
            parsed = parser(raw_value)
            context = {
                "video_id": video_id,
                "ordinal_n": ordinal_n,
                "detection_index": index,
                "field": field_name,
            }
            _observe(accumulators[kind], parsed, context)
            if parsed["parse_status"] != "VALID":
                detection_valid = False
                not_integer = (
                    kind == "label"
                    and isinstance(raw_value, float)
                    and math.isfinite(raw_value)
                    and not raw_value.is_integer()
                )
                issues.append(
                    _issue(
                        _code(kind.upper(), parsed, label_not_integer=not_integer),
                        video_id=video_id,
                        ordinal_n=ordinal_n,
                        detection_index=index,
                        field=field_name,
                        raw_value=raw_value,
                        message=f"{kind.title()} failed fail-closed parsing",
                        evidence={"parse_status": parsed["parse_status"]},
                    )
                )
            elif kind == "score" and not 0 <= parsed["parsed_value"] <= 1:
                issues.append(
                    _issue(
                        "SCORE_OUTSIDE_ZERO_ONE",
                        video_id=video_id,
                        ordinal_n=ordinal_n,
                        detection_index=index,
                        field=field_name,
                        raw_value=raw_value,
                        message="Valid finite score lies outside [0,1]",
                        severity="WARNING",
                    )
                )
            elif kind == "label" and parsed["parsed_value"] < 0:
                issues.append(
                    _issue(
                        "LABEL_NEGATIVE",
                        video_id=video_id,
                        ordinal_n=ordinal_n,
                        detection_index=index,
                        field=field_name,
                        raw_value=raw_value,
                        message="Valid integer label is negative",
                        severity="WARNING",
                    )
                )
        result["valid_detection_count" if detection_valid else "invalid_detection_count"] += 1
    result["file_normalization_status"] = (
        "VALID" if result["invalid_detection_count"] == 0 else "FAILED"
    )
    result["numeric_summaries"] = {
        name: _file_accumulator_summary(acc) for name, acc in accumulators.items()
    }
    return result, accumulators, issues


def _file_accumulator_summary(acc: dict[str, Any]) -> dict[str, Any]:
    values = acc["values"]
    return {
        "total": acc["total"],
        "valid": acc["valid"],
        "invalid": acc["total"] - acc["valid"],
        "raw_type_counts": dict(acc["raw_types"]),
        "parse_status_counts": dict(acc["statuses"]),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["total"] += source["total"]
    target["valid"] += source["valid"]
    target["raw_types"].update(source["raw_types"])
    target["statuses"].update(source["statuses"])
    target["values"].extend(source["values"])
    for position in target["position_values"]:
        target["position_values"][position].extend(source["position_values"][position])
    for name, limit in (
        ("valid_examples", VALID_EXAMPLE_LIMIT),
        ("invalid_examples", INVALID_EXAMPLE_LIMIT),
    ):
        target[name].extend(source[name][: max(0, limit - len(target[name]))])


def _coordinate_summary(acc: dict[str, Any]) -> dict[str, Any]:
    values = [float(value) for value in acc["values"]]
    position_values = acc["position_values"]
    valid = len(values)
    within = sum(0 <= value <= 1 for value in values)
    above = sum(value > 1 for value in values)
    negative = sum(value < 0 for value in values)
    if valid == acc["total"] and valid and within == valid:
        hypothesis, confidence = "NORMALIZED_0_1_CANDIDATE", "HIGH"
    elif valid == acc["total"] and above > within:
        hypothesis, confidence = "PIXEL_LIKE_CANDIDATE", "MEDIUM"
    elif valid == acc["total"] and above and within:
        hypothesis, confidence = "MIXED_RANGE", "MEDIUM"
    else:
        hypothesis, confidence = "UNKNOWN", "UNKNOWN"
    return {
        "total_values": acc["total"],
        "valid_count": acc["valid"],
        "invalid_count": acc["total"] - acc["valid"],
        "raw_type_counts": dict(sorted(acc["raw_types"].items())),
        "parse_status_counts": dict(sorted(acc["statuses"].items())),
        "finite_count": acc["valid"],
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "per_position_min": [
            min(position_values[p]) if position_values[p] else None for p in range(4)
        ],
        "per_position_max": [
            max(position_values[p]) if position_values[p] else None for p in range(4)
        ],
        "within_zero_one_count": within,
        "greater_than_one_count": above,
        "negative_count": negative,
        "zero_count": sum(value == 0 for value in values),
        "range_hypothesis": hypothesis,
        "confidence": confidence,
        "valid_examples": acc["valid_examples"],
        "invalid_examples": acc["invalid_examples"],
        "bbox_order": "UNKNOWN",
    }


def _score_summary(acc: dict[str, Any]) -> dict[str, Any]:
    values = [float(value) for value in acc["values"]]
    within = sum(0 <= value <= 1 for value in values)
    return {
        "total_values": acc["total"],
        "valid_count": acc["valid"],
        "invalid_count": acc["total"] - acc["valid"],
        "raw_type_counts": dict(sorted(acc["raw_types"].items())),
        "parse_status_counts": dict(sorted(acc["statuses"].items())),
        "finite_count": acc["valid"],
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": fmean(values) if values and acc["valid"] >= acc["total"] * 0.5 else None,
        "below_zero_count": sum(value < 0 for value in values),
        "within_zero_one_count": within,
        "above_one_count": sum(value > 1 for value in values),
        "outside_zero_one_count": len(values) - within,
        "range_hypothesis": "PROBABILITY_LIKE_0_1"
        if values and within == len(values)
        else "UNKNOWN",
        "valid_examples": acc["valid_examples"],
        "invalid_examples": acc["invalid_examples"],
    }


def _label_summary(acc: dict[str, Any]) -> dict[str, Any]:
    values = [int(value) for value in acc["values"]]
    return {
        "total_values": acc["total"],
        "valid_integer_count": acc["valid"],
        "invalid_count": acc["total"] - acc["valid"],
        "raw_type_counts": dict(sorted(acc["raw_types"].items())),
        "parse_status_counts": dict(sorted(acc["statuses"].items())),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "distinct_count": len(set(values)),
        "negative_count": sum(value < 0 for value in values),
        "valid_examples": acc["valid_examples"],
        "invalid_examples": acc["invalid_examples"],
    }


def run_survey(
    dataset_root: str | Path,
    *,
    limits: NumericLimits | None = None,
    strict_root: bool = False,
    v021_summary: str | Path | None = None,
) -> NumericContractResult:
    selected_limits = limits or NumericLimits()
    root = validate_root(dataset_root, strict_root=strict_root)
    started_at = _now()
    aggregate = {name: _empty_accumulator() for name in ("coordinate", "score", "label")}
    file_results: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for video_id, ordinal_n, path in _object_targets(root)[: selected_limits.max_object_json_total]:
        file_result, file_accumulators, file_issues = inspect_file(
            path,
            video_id=video_id,
            ordinal_n=ordinal_n,
            max_bytes=selected_limits.max_object_json_bytes,
        )
        file_results.append(file_result)
        issues.extend(file_issues)
        for name in aggregate:
            _merge(aggregate[name], file_accumulators[name])
    coordinates = _coordinate_summary(aggregate["coordinate"])
    scores = _score_summary(aggregate["score"])
    labels = _label_summary(aggregate["label"])
    inspected = sum(item["inspection_status"] == "INSPECTED" for item in file_results)
    detections = sum(item["detection_count"] for item in file_results)
    ready = (
        inspected == 15
        and all(item["file_normalization_status"] == "VALID" for item in file_results)
        and coordinates["valid_count"] == coordinates["total_values"] == 6000
        and scores["valid_count"] == scores["total_values"] == 1500
        and labels["valid_integer_count"] == labels["total_values"] == 1500
    )
    invalid_samples = (
        aggregate["coordinate"]["invalid_examples"]
        + aggregate["score"]["invalid_examples"]
        + aggregate["label"]["invalid_examples"]
    )
    summary = {
        "survey_version": SURVEY_VERSION,
        "patch_scope": "OBJECT_NUMERIC_STRING_CONTRACT",
        "dataset_root": str(root),
        "started_at": started_at,
        "completed_at": _now(),
        "limits": asdict(selected_limits),
        "v021_summary_provided": v021_summary is not None,
        "id_set_scan_rerun": False,
        "full_prior_survey_rerun": False,
        "files_requested": len(file_results),
        "files_inspected": inspected,
        "detections_observed": detections,
        "coordinate_summary": coordinates,
        "score_summary": scores,
        "label_summary": labels,
        "normalization_contract": {
            "numeric_strings_supported": True,
            "raw_values_preserved": True,
            "non_finite_rejected": True,
            "invalid_values_rejected": True,
            "silent_defaulting_allowed": False,
            "silent_dropping_allowed": False,
            "bbox_order": "UNKNOWN",
        },
        "verified_contracts": [
            "[VERIFIED] Numeric strings are parsed without locale coercion.",
            "[VERIFIED] Raw values are preserved and invalid detections fail closed.",
        ],
        "inferred_contracts": [
            f"[INFERRED] Coordinate range: {coordinates['range_hypothesis']}.",
            f"[INFERRED] Score range: {scores['range_hypothesis']}.",
        ],
        "unknown_contracts": ["[UNKNOWN] Bounding-box coordinate order remains unknown."],
        "issues_summary": {
            "total": len(issues),
            "by_code": dict(Counter(item["code"] for item in issues)),
            "by_severity": dict(Counter(item["severity"] for item in issues)),
        },
        "readiness": {
            "status": "READY_FOR_STAGE_0_DATA_AUDIT" if ready else "NUMERIC_CONTRACT_BLOCKED",
            "reason": "All bounded numeric values passed fail-closed validation"
            if ready
            else "One or more bounded files or numeric values failed validation",
        },
        "zip_artifact": {"path": None, "size_bytes": 0, "members": list(OUTPUT_NAMES)},
        "disclaimer": "This is the final bounded contract patch before Stage 0 Data Audit.",
    }
    return NumericContractResult(summary, file_results, invalid_samples, issues)


def _markdown(summary: dict[str, Any]) -> str:
    sections = [
        ("# 1. Patch Scope", ["[VERIFIED] OBJECT_NUMERIC_STRING_CONTRACT"]),
        (
            "# 2. Files and Detection Counts",
            [
                f"[VERIFIED] files={summary['files_inspected']}/"
                f"{summary['files_requested']}; "
                f"detections={summary['detections_observed']}"
            ],
        ),
        (
            "# 3. Raw Serialization Types",
            [
                "[VERIFIED] coordinates="
                f"{summary['coordinate_summary']['raw_type_counts']}; "
                f"scores={summary['score_summary']['raw_type_counts']}; "
                f"labels={summary['label_summary']['raw_type_counts']}"
            ],
        ),
        ("# 4. Coordinate Parsing", [f"[VERIFIED] {summary['coordinate_summary']}"]),
        ("# 5. Score Parsing", [f"[VERIFIED] {summary['score_summary']}"]),
        ("# 6. Class Label Parsing", [f"[VERIFIED] {summary['label_summary']}"]),
        ("# 7. Range Observations", summary["inferred_contracts"]),
        (
            "# 8. Fail-Closed Normalization Contract",
            [f"[VERIFIED] {summary['normalization_contract']}"],
        ),
        (
            "# 9. Invalid and Non-Finite Values",
            [f"[VERIFIED] issues={summary['issues_summary']['total']}"],
        ),
        ("# 10. Verified Contracts", summary["verified_contracts"]),
        ("# 11. Inferred Contracts", summary["inferred_contracts"]),
        ("# 12. Unknown Contracts", summary["unknown_contracts"]),
        ("# 13. Issues", [f"[VERIFIED] {summary['issues_summary']}"]),
        ("# 14. Readiness Decision", [f"[VERIFIED] {summary['readiness']}"]),
        ("# 15. ZIP Artifact", [f"[VERIFIED] {summary['zip_artifact']}"]),
    ]
    return "\n".join(
        line
        for title, values in sections
        for line in (title, "", *(f"- {value}" for value in values), "")
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8"
    )


def write_outputs(result: NumericContractResult, output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root).expanduser().resolve(strict=False)
    if _is_within(root, KAGGLE_INPUT_ROOT.resolve(strict=False)):
        raise ValueError("Output must not be written below /kaggle/input")
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in OUTPUT_NAMES}
    zip_path = root / "object_numeric_contract_v022.zip"
    _write_jsonl(paths["file_numeric_results_v022.jsonl"], result.file_results)
    _write_jsonl(paths["invalid_numeric_samples_v022.jsonl"], result.invalid_samples)
    _write_jsonl(paths["issues_v022.jsonl"], result.issues)
    result.summary["zip_artifact"].update({"path": str(zip_path), "members": list(OUTPUT_NAMES)})
    for _ in range(4):
        paths["numeric_contract_summary_v022.json"].write_text(
            json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        paths["numeric_contract_report_v022.md"].write_text(
            _markdown(result.summary), encoding="utf-8"
        )
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for name in OUTPUT_NAMES:
                archive.write(paths[name], arcname=name)
        size = zip_path.stat().st_size
        if result.summary["zip_artifact"]["size_bytes"] == size:
            break
        result.summary["zip_artifact"]["size_bytes"] = size
    paths["zip"] = zip_path
    return paths


def result_json(result: NumericContractResult) -> str:
    return json.dumps(result.summary, ensure_ascii=False, indent=2)
