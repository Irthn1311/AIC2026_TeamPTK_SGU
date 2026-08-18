"""Pre-GT reader for the immutable BCF-1 preparation artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from aic2026_eval.io import sha256_file

from .contracts import (
    A0_CROSS_SHA256,
    F1_CROSS_SHA256,
    INDEX_FINGERPRINT,
    INDEX_ZIP_SHA256,
    NORM_SHA256,
    POLICY,
    PREPARATION_MEMBER_HASHES,
    PREPARATION_ZIP_SHA256,
    S1_CROSS_SHA256,
    SCA1_ANCHOR_COMMIT,
    VECTOR_SHA256,
)

POST_GT_MEMBER = "bcf1_preparation/POST_GT_DESIGN_SANITY.json"
PRE_GT_MEMBERS = frozenset(PREPARATION_MEMBER_HASHES) - {POST_GT_MEMBER}


@dataclass(frozen=True)
class BCF1Preparation:
    source: str
    source_form: str
    zip_sha256: str
    member_paths: frozenset[str]
    pre_gt_member_hashes: dict[str, str]
    decision_context: dict[str, Any]
    notebook_contract: dict[str, Any]
    a0_cross: list[dict[str, Any]]
    s1_cross: list[dict[str, Any]]
    frozen_f1_cross: list[dict[str, Any]]
    validation: dict[str, Any]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _members(path: Path) -> tuple[set[str], str, str]:
    if path.is_file() and path.suffix.casefold() == ".zip":
        digest = sha256_file(path)
        if digest != PREPARATION_ZIP_SHA256:
            raise RuntimeError(f"BCF1_PREPARATION_ZIP_SHA256_MISMATCH: {digest}")
        with ZipFile(path) as archive:
            names = set(archive.namelist())
        return names, "ORIGINAL_FROZEN_ZIP", "PASS"
    if path.is_dir():
        names = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
        return names, "KAGGLE_EXPANDED_VERIFIED_MEMBERS", "NOT_AVAILABLE_AFTER_EXPANSION"
    raise RuntimeError("BCF1_PREPARATION_FREEZE_SOURCE_UNSUPPORTED")


def _read(path: Path, name: str) -> bytes:
    if path.is_file():
        with ZipFile(path) as archive:
            return archive.read(name)
    return (path / name).read_bytes()


def _jsonl(value: bytes) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in value.decode("utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("BCF1_PREPARATION_JSONL_INVALID")
    return rows


def load_preparation_freeze(source: str | Path) -> BCF1Preparation:
    """Validate only pre-GT-safe members; deliberately leave POST_GT unopened."""

    path = Path(source).expanduser().resolve(strict=True)
    names, source_form, container_gate = _members(path)
    if names != set(PREPARATION_MEMBER_HASHES):
        raise RuntimeError(f"BCF1_PREPARATION_MEMBER_SET_MISMATCH: {sorted(names)}")
    values = {name: _read(path, name) for name in PRE_GT_MEMBERS}
    hashes = {name: _sha256(value) for name, value in values.items()}
    expected = {name: PREPARATION_MEMBER_HASHES[name] for name in PRE_GT_MEMBERS}
    if hashes != expected:
        raise RuntimeError("BCF1_PREPARATION_PRE_GT_MEMBER_SHA256_MISMATCH")
    decision = json.loads(values["bcf1_preparation/decision_context.json"])
    notebook_contract = json.loads(values["bcf1_preparation/notebook_template_contract.json"])
    required = {
        "repo_head_at_sca1_run": SCA1_ANCHOR_COMMIT,
        "sca1_index_zip_sha256": INDEX_ZIP_SHA256,
        "sca1_index_fingerprint": INDEX_FINGERPRINT,
        "sca1_vector_sha256": VECTOR_SHA256,
        "sca1_norm_sha256": NORM_SHA256,
        "a0_cross_prediction_sha256": A0_CROSS_SHA256,
        "s1_cross_prediction_sha256": S1_CROSS_SHA256,
        "f1_cross_prediction_sha256_from_frozen_lists": F1_CROSS_SHA256,
        "sca1_fusion_gate": "OPEN",
    }
    if any(decision.get(key) != value for key, value in required.items()):
        raise RuntimeError("BCF1_PREPARATION_DECISION_CONTRACT_MISMATCH")
    next_policy = decision.get("next_policy", {})
    if (
        next_policy.get("name") != POLICY
        or next_policy.get("production_policy_changed") is not False
    ):
        raise RuntimeError("BCF1_PREPARATION_POLICY_MISMATCH")
    a0 = _jsonl(values["bcf1_preparation/pre_gt_predictions/a0_cross_g1.jsonl"])
    s1 = _jsonl(values["bcf1_preparation/pre_gt_predictions/s1_cross_g1.jsonl"])
    f1 = _jsonl(values["bcf1_preparation/pre_gt_predictions/f1_cross_protected_rrf60.jsonl"])
    if any(len(rows) != 6000 for rows in (a0, s1, f1)):
        raise RuntimeError("BCF1_FROZEN_CROSS_ROW_COUNT_MISMATCH")
    return BCF1Preparation(
        source=str(path),
        source_form=source_form,
        zip_sha256=PREPARATION_ZIP_SHA256,
        member_paths=frozenset(names),
        pre_gt_member_hashes=hashes,
        decision_context=decision,
        notebook_contract=notebook_contract,
        a0_cross=a0,
        s1_cross=s1,
        frozen_f1_cross=f1,
        validation={
            "status": "PASS",
            "source_form": source_form,
            "pre_gt_member_hash_gate": "PASS",
            "post_gt_design_sanity_opened": False,
            "original_zip_container_sha256_gate": container_gate,
            "member_count": len(names),
        },
    )


def load_post_gt_design_sanity(
    preparation: BCF1Preparation, *, finalized_cross_f1_sha256: str
) -> dict[str, Any]:
    if finalized_cross_f1_sha256 != F1_CROSS_SHA256:
        raise RuntimeError("BCF1_POST_GT_OPENED_BEFORE_CROSS_F1_REPRODUCTION")
    path = Path(preparation.source)
    value = _read(path, POST_GT_MEMBER)
    if _sha256(value) != PREPARATION_MEMBER_HASHES[POST_GT_MEMBER]:
        raise RuntimeError("BCF1_POST_GT_DESIGN_SANITY_SHA256_MISMATCH")
    data = json.loads(value)
    if data.get("f1_prediction_sha256") != F1_CROSS_SHA256:
        raise RuntimeError("BCF1_POST_GT_DESIGN_SANITY_CONTRACT_MISMATCH")
    return data


__all__ = [
    "BCF1Preparation",
    "POST_GT_MEMBER",
    "load_post_gt_design_sanity",
    "load_preparation_freeze",
]
