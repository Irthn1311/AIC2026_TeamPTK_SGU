"""Strict loading and validation of the immutable TCA-1 preparation freeze."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from aic2026_eval.io import sha256_file

from .contracts import (
    EXPECTED_BY_TASK,
    EXPECTED_COUNTS,
    EXPECTED_FAIL_COUNT,
    EXPECTED_UNIT_COUNT,
    FORBIDDEN_REVIEW_FIELDS,
    FROZEN_OVERRIDE_SHA256,
    FROZEN_REVIEW_PROTOCOL,
    FROZEN_REVIEW_SHA256,
    FROZEN_REVIEW_VERSION,
    FROZEN_SOURCE_QC_SHA256,
    FROZEN_ZIP_SHA256,
)

REQUIRED_FILES = (
    "README.md",
    "representation_ceiling_audit.json",
    "translation_blind_review_frozen.jsonl",
    "translation_blind_review_summary.json",
    "translation_fail_overrides.jsonl",
)


@dataclass(frozen=True)
class FrozenReview:
    rows: tuple[dict[str, Any], ...]
    overrides: dict[str, str]
    rows_by_unit: dict[str, dict[str, Any]]
    fail_unit_ids: frozenset[str]
    nonfail_unit_ids: frozenset[str]
    file_hashes: dict[str, str]
    source: str
    source_zip_sha256: str | None
    validation: dict[str, Any]


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_members(source: Path) -> tuple[dict[str, bytes], str | None]:
    if source.is_file():
        digest = sha256_file(source)
        if digest != FROZEN_ZIP_SHA256:
            raise RuntimeError(f"TCA1_FROZEN_ZIP_SHA256_MISMATCH: {digest}")
        with ZipFile(source) as archive:
            names = set(archive.namelist())
            if names != set(REQUIRED_FILES):
                raise RuntimeError(f"TCA1_FROZEN_MEMBER_SET_MISMATCH: {sorted(names)}")
            return {name: archive.read(name) for name in REQUIRED_FILES}, digest
    if not source.is_dir():
        raise FileNotFoundError(source)
    members = {}
    for name in REQUIRED_FILES:
        matches = sorted(source.rglob(name))
        if len(matches) != 1:
            raise RuntimeError(f"TCA1_EXPECTED_ONE_FROZEN_MEMBER: {name}: {matches}")
        members[name] = matches[0].read_bytes()
    return members, None


def _jsonl(value: bytes, name: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in value.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"TCA1_INVALID_FROZEN_JSONL: {name}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"TCA1_INVALID_FROZEN_JSONL_ROWS: {name}")
    return rows


def load_frozen_review(source: str | Path) -> FrozenReview:
    """Load the exact freeze and fail closed on every contract mismatch."""

    path = Path(source).expanduser().resolve(strict=True)
    members, zip_digest = _read_members(path)
    hashes = {name: _hash_bytes(value) for name, value in members.items()}
    if hashes["translation_blind_review_frozen.jsonl"] != FROZEN_REVIEW_SHA256:
        raise RuntimeError("TCA1_FROZEN_REVIEW_SHA256_MISMATCH")
    if hashes["translation_fail_overrides.jsonl"] != FROZEN_OVERRIDE_SHA256:
        raise RuntimeError("TCA1_FROZEN_OVERRIDE_SHA256_MISMATCH")
    rows = _jsonl(members["translation_blind_review_frozen.jsonl"], "review")
    overrides_rows = _jsonl(members["translation_fail_overrides.jsonl"], "overrides")
    summary = json.loads(members["translation_blind_review_summary.json"].decode("utf-8"))
    unit_ids = [str(row.get("unit_id", "")) for row in rows]
    if len(rows) != EXPECTED_UNIT_COUNT or len(set(unit_ids)) != EXPECTED_UNIT_COUNT:
        raise RuntimeError("TCA1_REVIEW_UNIT_COUNT_OR_DUPLICATE_GATE_FAILED")
    if [row.get("review_index") for row in rows] != list(range(1, EXPECTED_UNIT_COUNT + 1)):
        raise RuntimeError("TCA1_REVIEW_ORDER_GATE_FAILED")
    if any(FORBIDDEN_REVIEW_FIELDS & set(row) for row in rows):
        raise RuntimeError("TCA1_REVIEW_GT_OR_OUTCOME_LEAK_GATE_FAILED")
    if any(
        row.get("review_version") != FROZEN_REVIEW_VERSION
        or row.get("review_protocol") != FROZEN_REVIEW_PROTOCOL
        for row in rows
    ):
        raise RuntimeError("TCA1_REVIEW_PROTOCOL_GATE_FAILED")
    counts = Counter(str(row.get("verdict")) for row in rows)
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_task[str(row.get("task"))][str(row.get("verdict"))] += 1
    normalized_by_task = {
        task: {verdict: by_task[task][verdict] for verdict in EXPECTED_COUNTS}
        for task in EXPECTED_BY_TASK
    }
    if dict(counts) != EXPECTED_COUNTS or normalized_by_task != EXPECTED_BY_TASK:
        raise RuntimeError("TCA1_REVIEW_VERDICT_COUNTS_GATE_FAILED")
    fail_ids = {row["unit_id"] for row in rows if row["verdict"] == "FAIL"}
    overrides = {
        str(row.get("unit_id")): str(row.get("reference_en", "")).strip() for row in overrides_rows
    }
    if (
        len(overrides_rows) != EXPECTED_FAIL_COUNT
        or len(overrides) != EXPECTED_FAIL_COUNT
        or set(overrides) != fail_ids
        or any(not value for value in overrides.values())
    ):
        raise RuntimeError("TCA1_EXACT_FAIL_OVERRIDE_SET_GATE_FAILED")
    sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        sources[str(row.get("source_vi", ""))].add(str(row["verdict"]))
    collisions = {
        text: values for text, values in sources.items() if "FAIL" in values and len(values) > 1
    }
    if collisions:
        raise RuntimeError(f"TCA1_CONFLICTING_SOURCE_COLLISION_GATE_FAILED: {collisions}")
    expected_summary = {
        "status": "FROZEN_FOR_TCA1",
        "review_version": FROZEN_REVIEW_VERSION,
        "review_protocol": FROZEN_REVIEW_PROTOCOL,
        "source_translation_blind_qc_sha256": FROZEN_SOURCE_QC_SHA256,
        "review_row_count": EXPECTED_UNIT_COUNT,
        "counts": EXPECTED_COUNTS,
        "by_task": EXPECTED_BY_TASK,
        "fail_override_count": EXPECTED_FAIL_COUNT,
        "review_sha256": FROZEN_REVIEW_SHA256,
        "fail_overrides_sha256": FROZEN_OVERRIDE_SHA256,
        "gt_used_for_verdicts": False,
        "retrieval_rank_or_outcome_used_for_verdicts": False,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise RuntimeError("TCA1_FROZEN_SUMMARY_CONTRACT_GATE_FAILED")
    rows_by_unit = {row["unit_id"]: dict(row) for row in rows}
    validation = {
        "status": "PASS",
        "review_row_count": len(rows),
        "unique_unit_count": len(rows_by_unit),
        "verdict_counts": EXPECTED_COUNTS,
        "by_task": EXPECTED_BY_TASK,
        "fail_override_count": len(overrides),
        "gt_used_for_verdicts": False,
        "retrieval_rank_or_outcome_used_for_verdicts": False,
        "conflicting_source_collision_count": 0,
    }
    return FrozenReview(
        rows=tuple(dict(row) for row in rows),
        overrides=overrides,
        rows_by_unit=rows_by_unit,
        fail_unit_ids=frozenset(fail_ids),
        nonfail_unit_ids=frozenset(set(unit_ids) - fail_ids),
        file_hashes=hashes,
        source=str(path),
        source_zip_sha256=zip_digest,
        validation=validation,
    )


def materialize_frozen_review(source: str | Path, output_root: str | Path) -> FrozenReview:
    """Copy only verified bytes into the diagnostic bundle."""

    frozen = load_frozen_review(source)
    source_path = Path(source).expanduser().resolve(strict=True)
    members, _ = _read_members(source_path)
    target = Path(output_root) / "review"
    target.mkdir(parents=True, exist_ok=True)
    for name, value in members.items():
        (target / name).write_bytes(value)
    return frozen


__all__ = ["FrozenReview", "REQUIRED_FILES", "load_frozen_review", "materialize_frozen_review"]
