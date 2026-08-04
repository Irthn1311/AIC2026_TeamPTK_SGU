"""Bounded, read-only dataset layout survey for local and Kaggle use."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
SURVEY_VERSION = "0.1.0"
DEFAULT_DATASET_ROOT = Path("/kaggle/input/datasets/nadkli/dataset-aic")
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/dataset_survey")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}
MAPPING_HINTS = {"map", "mapping", "keyframe", "keyframes", "pts"}
OBJECT_HINTS = {"object", "objects", "detection", "detections", "bbox", "bboxes"}
METADATA_HINTS = {"metadata", "meta", "info", "description", "descriptions"}
FEATURE_HINTS = {"clip", "feature", "features", "embedding", "embeddings"}


@dataclass(frozen=True)
class SurveyLimits:
    """Hard limits preventing an accidental corpus-scale inspection."""

    max_depth: int = 4
    max_listed_per_directory: int = 20
    max_examples_per_group: int = 5
    max_csv_rows: int = 20
    max_json_bytes: int = 1_048_576
    max_npy_rows: int = 5
    max_stat_operations: int = 5_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass
class TraversalBudget:
    """Mutable counter shared by all directory traversal operations."""

    maximum: int
    used: int = 0
    exhausted: bool = False

    def consume(self) -> bool:
        if self.used >= self.maximum:
            self.exhausted = True
            return False
        self.used += 1
        return True


@dataclass(frozen=True)
class Classification:
    asset_type: str
    confidence: str
    heuristic: str
    ambiguous_with: tuple[str, ...] = ()


@dataclass
class SurveyResult:
    """Serializable survey result and inspected-file inventory."""

    summary: dict[str, Any]
    sample_inventory: list[dict[str, Any]] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_dataset_root(
    cli_value: str | Path | None, environ: dict[str, str] | None = None
) -> Path:
    """Resolve CLI, environment, then default dataset root in that order."""

    environment = os.environ if environ is None else environ
    selected = cli_value or environment.get("AIC_DATA_ROOT") or DEFAULT_DATASET_ROOT
    return Path(selected).expanduser().resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_dataset_root(root: str | Path, *, strict_root: bool = False) -> Path:
    """Validate a readable directory and optionally require the Kaggle input boundary."""

    resolved = Path(root).expanduser().resolve(strict=False)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {resolved}")
    if strict_root and not _is_within(resolved, KAGGLE_INPUT_ROOT):
        raise ValueError(f"Strict dataset root must be below {KAGGLE_INPUT_ROOT}: {resolved}")
    if not os.access(resolved, os.R_OK | os.X_OK):
        raise PermissionError(f"Dataset root is not readable/searchable: {resolved}")
    return resolved


def candidate_video_id(path: str | Path) -> str | None:
    """Extract a conservative candidate ID without claiming a canonical convention."""

    item = Path(path)
    stem = item.stem.strip()
    if not stem:
        return None
    if stem.isdigit() and item.parent.name:
        return item.parent.name
    reduced = re.sub(
        r"(?i)(?:[_\-.](?:keyframes?|mapping|map|clip|features?|objects?|metadata))$", "", stem
    )
    return reduced or stem


def classify_asset(path: str | Path) -> Classification:
    """Classify one path using suffix and path-name hints, retaining ambiguity."""

    item = Path(path)
    suffix = item.suffix.lower()
    tokens = set(re.findall(r"[a-z0-9]+", str(item).lower()))
    if suffix in VIDEO_SUFFIXES:
        return Classification("VIDEO", "HIGH", "video suffix")
    if suffix in ARCHIVE_SUFFIXES:
        return Classification("ARCHIVE", "HIGH", "archive suffix")
    if suffix == ".npy":
        confidence = "HIGH" if tokens & FEATURE_HINTS else "MEDIUM"
        return Classification("CLIP_FEATURE", confidence, "NumPy array suffix")
    if suffix in IMAGE_SUFFIXES:
        confidence = "HIGH" if tokens & MAPPING_HINTS else "MEDIUM"
        return Classification("KEYFRAME_IMAGE", confidence, "image suffix")
    if suffix in {".csv", ".tsv"}:
        if tokens & MAPPING_HINTS:
            return Classification("KEYFRAME_MAPPING", "HIGH", "tabular suffix and mapping hint")
        return Classification(
            "METADATA", "LOW", "tabular suffix without mapping hint", ("KEYFRAME_MAPPING",)
        )
    if suffix in {".json", ".jsonl"}:
        if tokens & OBJECT_HINTS:
            return Classification("OBJECT_JSON", "HIGH", "JSON suffix and object hint")
        if tokens & METADATA_HINTS:
            return Classification("METADATA", "HIGH", "JSON suffix and metadata hint")
        return Classification("METADATA", "LOW", "generic JSON suffix", ("OBJECT_JSON",))
    return Classification("UNKNOWN", "LOW", "no recognized suffix or naming hint")


def _safe_child(root: Path, child: Path) -> bool:
    """Reject symlinks and resolved paths outside the survey root."""

    if child.is_symlink():
        return False
    return _is_within(child.resolve(strict=False), root)


def bounded_tree(root: Path, limits: SurveyLimits, budget: TraversalBudget) -> list[dict[str, Any]]:
    """Build a bounded tree without recursive globbing or content reads."""

    def visit(directory: Path, depth: int) -> dict[str, Any]:
        node: dict[str, Any] = {
            "path": "." if directory == root else directory.relative_to(root).as_posix(),
            "type": "directory",
            "depth": depth,
            "entries": [],
            "hidden_entries": 0,
        }
        if depth >= limits.max_depth or budget.exhausted:
            node["truncated"] = True
            return node
        try:
            with os.scandir(directory) as iterator:
                entries = []
                for entry in iterator:
                    if not budget.consume():
                        break
                    entries.append(entry)
                    if len(entries) > limits.max_listed_per_directory:
                        break
        except OSError as error:
            node["error"] = f"{type(error).__name__}: {error}"
            return node
        entries.sort(
            key=lambda value: (not value.is_dir(follow_symlinks=False), value.name.lower())
        )
        shown = entries[: limits.max_listed_per_directory]
        node["hidden_entries"] = max(0, len(entries) - len(shown))
        node["hidden_entries_is_lower_bound"] = bool(node["hidden_entries"])
        for entry in shown:
            child = Path(entry.path)
            if entry.is_symlink():
                node["entries"].append({"name": entry.name, "type": "symlink", "followed": False})
            elif entry.is_dir(follow_symlinks=False) and _safe_child(root, child):
                node["entries"].append(visit(child, depth + 1))
            else:
                node["entries"].append(
                    {"name": entry.name, "type": "file", "suffix": child.suffix.lower()}
                )
        if budget.exhausted:
            node["truncated"] = True
        return node

    return [visit(root, 0)]


def discover_samples(
    root: Path, limits: SurveyLimits, budget: TraversalBudget
) -> tuple[dict[str, list[Path]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Traverse breadth-first within limits and retain bounded samples per asset group."""

    samples: dict[str, list[Path]] = defaultdict(list)
    directories: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    cursor = 0
    while cursor < len(queue) and not budget.exhausted:
        directory, depth = queue[cursor]
        cursor += 1
        counts: Counter[str] = Counter()
        entry_count = 0
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    if not budget.consume():
                        break
                    entry_count += 1
                    child = Path(entry.path)
                    if entry.is_symlink():
                        counts["SYMLINK"] += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        counts["DIRECTORY"] += 1
                        if depth < limits.max_depth and _safe_child(root, child):
                            queue.append((child, depth + 1))
                        continue
                    classification = classify_asset(child)
                    counts[classification.asset_type] += 1
                    group = samples[classification.asset_type]
                    if len(group) < limits.max_examples_per_group:
                        group.append(child)
        except OSError as error:
            issues.append(
                issue("WARNING", "UNREADABLE_DIRECTORY", directory, "UNKNOWN", str(error))
            )
        directories.append(
            {
                "path": "." if directory == root else directory.relative_to(root).as_posix(),
                "depth": depth,
                "observed_entries": entry_count,
                "observed_asset_counts": dict(sorted(counts.items())),
                "count_is_complete": not budget.exhausted,
            }
        )
        if entry_count == 0:
            issues.append(
                issue("INFO", "EMPTY_DIRECTORY", directory, "UNKNOWN", "Directory is empty")
            )
    if budget.exhausted:
        issues.append(
            issue(
                "WARNING",
                "INSPECTION_LIMIT_REACHED",
                root,
                "UNKNOWN",
                f"Stopped after {budget.maximum} file stat operations",
            )
        )
    return dict(samples), directories, issues


def issue(
    severity: str,
    code: str,
    path: str | Path,
    asset_type: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "path": str(path),
        "asset_type": asset_type,
        "message": message,
        "evidence": evidence or {},
    }


def inspect_csv(path: Path, max_rows: int) -> dict[str, Any]:
    """Read a CSV header and at most ``max_rows`` data rows."""

    with path.open("rb") as binary_handle:
        sample = binary_handle.read(65_536)
    encoding = "utf-8"
    try:
        text = sample.decode(encoding)
    except UnicodeDecodeError:
        encoding = "utf-8-sig"
        text = sample.decode(encoding, errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[dict[str, str | None]] = []
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = reader.fieldnames or []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append(dict(row))
    column_observations: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        present = [value for value in values if value not in {None, ""}]
        inferred = "string"
        if present:
            try:
                [int(value) for value in present]
                inferred = "integer"
            except (TypeError, ValueError):
                try:
                    [float(value) for value in present]
                    inferred = "float"
                except (TypeError, ValueError):
                    pass
        column_observations[column] = {
            "inferred_scalar_type": inferred,
            "missing_in_sample": len(values) - len(present),
            "first_observed": present[0] if present else None,
            "last_observed": present[-1] if present else None,
            "duplicate_count_in_sample": len(present) - len(set(present)),
        }
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": columns,
        "rows_inspected": len(rows),
        "column_observations": column_observations,
        "mapping_columns_present": [
            column for column in columns if column.lower() in {"n", "pts_time", "fps", "frame_idx"}
        ],
    }


def inspect_npy(path: Path, max_rows: int, seed: int) -> dict[str, Any]:
    """Memory-map an NPY file and inspect only a few rows."""

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(value) for value in array.shape)
    result: dict[str, Any] = {
        "shape": shape,
        "ndim": int(array.ndim),
        "dtype": str(array.dtype),
        "estimated_bytes": int(array.size * array.dtype.itemsize),
        "clip_model_compatibility": "REQUIRES_LATER_VERIFICATION",
    }
    if array.ndim == 0 or not shape or shape[0] == 0:
        result["rows_inspected"] = 0
        result["zero_rows"] = bool(shape and shape[0] == 0)
        return result
    row_count = min(max_rows, shape[0])
    indices = sorted(random.Random(seed).sample(range(shape[0]), row_count))
    finite = True
    norms: list[float] = []
    for index in indices:
        row = np.asarray(array[index])
        finite = finite and bool(np.isfinite(row).all())
        if np.issubdtype(row.dtype, np.number):
            norms.append(float(np.linalg.norm(row.reshape(-1))))
    result.update(
        {
            "rows_inspected": row_count,
            "sampled_row_indices": indices,
            "sample_all_finite": finite,
            "sample_norm_min": min(norms) if norms else None,
            "sample_norm_max": max(norms) if norms else None,
        }
    )
    return result


def _json_shape(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth >= 2:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        keys = [str(key) for key in list(value)[:30]]
        return {
            "type": "object",
            "keys": keys,
            "nested": {key: _json_shape(value[key], depth + 1) for key in keys[:5]},
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "item_sample": _json_shape(value[0], depth + 1) if value else None,
        }
    if isinstance(value, str):
        return {"type": "string", "sample": value[:200], "truncated": len(value) > 200}
    return {"type": type(value).__name__, "sample": value}


def inspect_json(path: Path, max_bytes: int) -> dict[str, Any]:
    """Inspect JSON only when its complete bytes fit within the configured cap."""

    size = path.stat().st_size
    if size > max_bytes:
        return {"inspection_status": "SKIPPED", "reason": "FILE_TOO_LARGE_TO_INSPECT"}
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        return {"inspection_status": "SKIPPED", "reason": "FILE_TOO_LARGE_TO_INSPECT"}
    encoding = "utf-8"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        encoding = "utf-8-sig"
        text = raw.decode(encoding, errors="replace")
    payload = json.loads(text)
    return {"inspection_status": "INSPECTED", "encoding": encoding, **_json_shape(payload)}


def inspect_keyframe_names(paths: Iterable[Path]) -> dict[str, Any]:
    """Observe numeric image stems without asserting corpus-wide numbering."""

    names = [path.name for path in paths]
    numeric = [int(path.stem) for path in paths if path.stem.isdigit()]
    paddings = sorted({len(path.stem) for path in paths if path.stem.isdigit()})
    conclusion = "AMBIGUOUS"
    if numeric and 0 in numeric:
        conclusion = "ZERO_OBSERVED_IN_SAMPLE"
    elif numeric and 1 in numeric:
        conclusion = "ONE_OBSERVED_BUT_ZERO_NOT_EXCLUDED"
    return {
        "filenames": names,
        "numeric_stem_count": len(numeric),
        "numeric_min": min(numeric) if numeric else None,
        "numeric_max": max(numeric) if numeric else None,
        "zero_padding_widths": paddings,
        "frame_numbering_observation": conclusion,
    }


def _inventory_record(root: Path, path: Path, asset_type: str) -> dict[str, Any]:
    classification = classify_asset(path)
    return {
        "asset_type": asset_type,
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
        "candidate_video_id": candidate_video_id(path),
        "inspection_status": "STAT_ONLY",
        "observations": {
            "classification_heuristic": classification.heuristic,
            "classification_confidence": classification.confidence,
            "ambiguous_with": list(classification.ambiguous_with),
        },
    }


def _inspect_samples(
    root: Path, samples: dict[str, list[Path]], limits: SurveyLimits, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    schemas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    for asset_type in sorted(samples):
        for offset, path in enumerate(samples[asset_type]):
            try:
                record = _inventory_record(root, path, asset_type)
                observations: dict[str, Any] = record["observations"]
                if asset_type == "KEYFRAME_MAPPING":
                    observations.update(inspect_csv(path, limits.max_csv_rows))
                    record["inspection_status"] = "INSPECTED"
                elif asset_type == "CLIP_FEATURE":
                    observations.update(inspect_npy(path, limits.max_npy_rows, seed + offset))
                    record["inspection_status"] = "INSPECTED"
                elif asset_type in {"OBJECT_JSON", "METADATA"} and path.suffix.lower() == ".json":
                    observations.update(inspect_json(path, limits.max_json_bytes))
                    record["inspection_status"] = observations.get("inspection_status", "INSPECTED")
                    if record["inspection_status"] == "SKIPPED":
                        issues.append(
                            issue(
                                "INFO",
                                "FILE_TOO_LARGE_TO_INSPECT",
                                path,
                                asset_type,
                                "JSON exceeds the bounded read limit",
                                {"size_bytes": record["size_bytes"]},
                            )
                        )
                elif asset_type == "VIDEO":
                    observations["media_metadata"] = "NOT_INSPECTED_IN_LAYOUT_SURVEY"
                inventory.append(record)
                if record["inspection_status"] == "INSPECTED":
                    schemas[asset_type].append(
                        {"relative_path": record["relative_path"], **observations}
                    )
            except (OSError, ValueError, csv.Error, json.JSONDecodeError) as error:
                code = {
                    ".npy": "MALFORMED_NPY",
                    ".csv": "MALFORMED_CSV",
                    ".tsv": "MALFORMED_CSV",
                    ".json": "MALFORMED_JSON",
                }.get(path.suffix.lower(), "UNREADABLE_FILE")
                issues.append(issue("WARNING", code, path, asset_type, str(error)))
                try:
                    record = _inventory_record(root, path, asset_type)
                except OSError:
                    continue
                record["inspection_status"] = "SKIPPED"
                record["observations"]["inspection_error"] = type(error).__name__
                inventory.append(record)
    keyframes = [path for path in samples.get("KEYFRAME_IMAGE", [])]
    if keyframes:
        schemas["KEYFRAME_IMAGE"] = [inspect_keyframe_names(keyframes)]
    return inventory, dict(schemas), issues


def _cross_asset_observations(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in inventory:
        identifier = record.get("candidate_video_id")
        if identifier:
            by_id[identifier][record["asset_type"]].append(record["relative_path"])
    rows = []
    for identifier in sorted(by_id):
        groups = by_id[identifier]
        present = sum(bool(groups.get(name)) for name in groups)
        rows.append(
            {
                "candidate_video_id": identifier,
                "video_path": (groups.get("VIDEO") or [None])[0],
                "mapping_path": (groups.get("KEYFRAME_MAPPING") or [None])[0],
                "keyframe_location": (groups.get("KEYFRAME_IMAGE") or [None])[0],
                "clip_path": (groups.get("CLIP_FEATURE") or [None])[0],
                "object_location": (groups.get("OBJECT_JSON") or [None])[0],
                "metadata_match": (groups.get("METADATA") or [None])[0],
                "confidence": "MEDIUM" if present >= 2 else "LOW",
                "notes": (
                    "Sample-only stem-based association; canonical join requires confirmation."
                ),
            }
        )
    return rows


def survey_dataset(
    dataset_root: str | Path,
    *,
    limits: SurveyLimits | None = None,
    strict_root: bool = False,
    seed: int = 2026,
) -> SurveyResult:
    """Run a bounded read-only layout survey and return serializable data."""

    selected_limits = limits or SurveyLimits()
    started_at = utc_now()
    requested_root = Path(dataset_root).expanduser().resolve(strict=False)
    if not requested_root.exists() and not strict_root:
        completed_at = utc_now()
        return SurveyResult(
            {
                "survey_version": SURVEY_VERSION,
                "dataset_root": str(requested_root),
                "started_at": started_at,
                "completed_at": completed_at,
                "limits": asdict(selected_limits),
                "root_exists": False,
                "root_readable": False,
                "root_entries": [],
                "directory_summary": [],
                "asset_groups": {},
                "naming_observations": {"candidate_video_ids": [], "scope": "sample_only"},
                "schema_observations": {},
                "mapping_observations": {
                    "cross_asset_samples": [],
                    "frame_id_policy": "DO_NOT_DERIVE_FROM_TIMESTAMP_TIMES_FPS",
                },
                "sample_issues": [
                    issue(
                        "ERROR",
                        "ROOT_NOT_FOUND",
                        requested_root,
                        "UNKNOWN",
                        "Dataset root does not exist",
                    )
                ],
                "unknowns": ["Dataset layout cannot be surveyed until the root is available."],
                "recommended_contracts_to_confirm": [],
                "stat_operations": {"total": 0, "maximum": selected_limits.max_stat_operations},
                "disclaimer": "This is a bounded layout survey, not a complete Data Audit.",
            }
        )
    root = validate_dataset_root(dataset_root, strict_root=strict_root)
    traversal_budget = TraversalBudget(selected_limits.max_stat_operations)
    tree = bounded_tree(root, selected_limits, traversal_budget)
    samples, directories, discovery_issues = discover_samples(
        root, selected_limits, traversal_budget
    )
    inventory, schemas, inspection_issues = _inspect_samples(root, samples, selected_limits, seed)
    asset_groups = {
        group: {
            "sample_count": len(paths),
            "examples": [path.relative_to(root).as_posix() for path in paths],
            "count_is_complete": False,
        }
        for group, paths in sorted(samples.items())
    }
    unknowns = [
        "Corpus-wide asset counts are not established by this bounded survey.",
        "Canonical video_id and join rules require dataset evidence beyond filename heuristics.",
        "Frame numbering remains unconfirmed unless mapping evidence explicitly defines it.",
        "Video codec, FPS, timebase, duration and decode health were not inspected.",
        "CLIP model compatibility requires later verification.",
    ]
    summary = {
        "survey_version": SURVEY_VERSION,
        "dataset_root": str(root),
        "started_at": started_at,
        "completed_at": utc_now(),
        "limits": asdict(selected_limits),
        "root_exists": True,
        "root_readable": True,
        "root_entries": tree,
        "directory_summary": directories,
        "asset_groups": asset_groups,
        "naming_observations": {
            "candidate_video_ids": sorted(
                {
                    record["candidate_video_id"]
                    for record in inventory
                    if record["candidate_video_id"]
                }
            ),
            "scope": "sample_only",
        },
        "schema_observations": schemas,
        "mapping_observations": {
            "cross_asset_samples": _cross_asset_observations(inventory),
            "frame_id_policy": "DO_NOT_DERIVE_FROM_TIMESTAMP_TIMES_FPS",
        },
        "sample_issues": discovery_issues + inspection_issues,
        "unknowns": unknowns,
        "recommended_contracts_to_confirm": [
            "canonical video_id and cross-asset join keys",
            "zero-based versus one-based keyframe ordinal",
            "original frame index versus keyframe order",
            "map-keyframes columns, types and timebase semantics",
            "metadata missing/null behavior",
            "per-video versus partitioned feature layout",
        ],
        "stat_operations": {
            "total": traversal_budget.used,
            "maximum": selected_limits.max_stat_operations,
        },
        "disclaimer": "This is a bounded layout survey, not a complete Data Audit.",
    }
    return SurveyResult(summary, inventory)


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# TRIAGE-EG Dataset Survey",
        "",
        "> This is a bounded layout survey, not a complete Data Audit.",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Survey version: `{summary['survey_version']}`",
        f"- Completed: `{summary['completed_at']}`",
        "",
        "## Asset groups",
        "",
        "| Group | Sample count | Examples |",
        "|---|---:|---|",
    ]
    for group, details in summary["asset_groups"].items():
        examples = ", ".join(f"`{value}`" for value in details["examples"])
        lines.append(f"| {group} | {details['sample_count']} | {examples} |")
    lines.extend(["", "## Sample issues", ""])
    issues = summary["sample_issues"]
    if issues:
        lines.extend(
            f"- **{item['severity']} {item['code']}** `{item['path']}` — {item['message']}"
            for item in issues
        )
    else:
        lines.append("- No issues observed within the bounded sample.")
    lines.extend(["", "## Unknowns", ""])
    lines.extend(f"- {value}" for value in summary["unknowns"])
    lines.extend(["", "## Contracts to confirm", ""])
    lines.extend(f"- {value}" for value in summary["recommended_contracts_to_confirm"])
    return "\n".join(lines) + "\n"


def write_survey_outputs(
    result: SurveyResult,
    output_root: str | Path,
    *,
    json_output: str = "dataset_survey.json",
    text_output: str = "dataset_survey.md",
) -> dict[str, Path]:
    """Write exactly the three small survey artifacts using safe filenames."""

    root = Path(output_root).expanduser().resolve(strict=False)
    kaggle_input = KAGGLE_INPUT_ROOT.resolve(strict=False)
    if _is_within(root, kaggle_input):
        raise ValueError(f"Survey output must not be written below {KAGGLE_INPUT_ROOT}: {root}")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": root / Path(json_output).name,
        "text": root / Path(text_output).name,
        "inventory": root / "sample_inventory.jsonl",
    }
    paths["json"].write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["text"].write_text(_markdown_report(result.summary), encoding="utf-8")
    with paths["inventory"].open("w", encoding="utf-8") as handle:
        for record in result.sample_inventory:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return paths


def summary_json(result: SurveyResult) -> str:
    """Serialize a report for ``--no-write`` output."""

    return json.dumps(result.summary, ensure_ascii=False, indent=2)
