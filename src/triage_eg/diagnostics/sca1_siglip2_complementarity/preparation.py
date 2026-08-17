"""Strict reader for the immutable SCA-1 preparation freeze."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from aic2026_eval.io import sha256_file

from .contracts import (
    EMBEDDING_DIMENSION,
    EXPECTED_A0_PREDICTION_SHA256,
    EXPECTED_OPENAI_CLIP_ID,
    EXPECTED_OPENAI_CLIP_SHA256,
    EXPECTED_ROWS,
    EXPECTED_STAGE1_FINGERPRINT,
    EXPECTED_TRANSLATOR_ID,
    EXPECTED_TRANSLATOR_REVISION,
    FROZEN_PREPARATION_ZIP_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SAFETENSORS_SHA256,
    PREPARATION_MEMBER_HASHES,
    TCA1_ANCHOR_COMMIT,
)


@dataclass(frozen=True)
class PreparationFreeze:
    source: str
    zip_sha256: str
    member_hashes: dict[str, str]
    decision_context: dict[str, Any]
    model_selection: dict[str, Any]
    validation: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_preparation_freeze(source: str | Path) -> PreparationFreeze:
    path = Path(source).expanduser().resolve(strict=True)
    if path.is_file() and path.suffix.casefold() == ".zip":
        zip_digest = sha256_file(path)
        if zip_digest != FROZEN_PREPARATION_ZIP_SHA256:
            raise RuntimeError(f"SCA1_PREPARATION_ZIP_SHA256_MISMATCH: {zip_digest}")
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            if names != set(PREPARATION_MEMBER_HASHES):
                raise RuntimeError(f"SCA1_PREPARATION_MEMBER_SET_MISMATCH: {sorted(names)}")
            members = {name: archive.read(name) for name in PREPARATION_MEMBER_HASHES}
        source_form = "ORIGINAL_FROZEN_ZIP"
        container_hash_gate = "PASS"
    elif path.is_dir():
        actual_paths = {
            item.relative_to(path).as_posix(): item for item in path.rglob("*") if item.is_file()
        }
        if set(actual_paths) != set(PREPARATION_MEMBER_HASHES):
            raise RuntimeError(
                f"SCA1_PREPARATION_EXPANDED_MEMBER_SET_MISMATCH: {sorted(actual_paths)}"
            )
        members = {name: actual_paths[name].read_bytes() for name in PREPARATION_MEMBER_HASHES}
        zip_digest = FROZEN_PREPARATION_ZIP_SHA256
        source_form = "KAGGLE_EXPANDED_VERIFIED_MEMBERS"
        container_hash_gate = "NOT_AVAILABLE_AFTER_KAGGLE_EXPANSION"
    else:
        raise RuntimeError("SCA1_PREPARATION_FREEZE_SOURCE_UNSUPPORTED")
    hashes = {name: _sha256_bytes(value) for name, value in members.items()}
    if hashes != PREPARATION_MEMBER_HASHES:
        raise RuntimeError("SCA1_PREPARATION_MEMBER_SHA256_MISMATCH")
    decision = json.loads(members["sca1_preparation/decision_context.json"])
    selection = json.loads(members["sca1_preparation/model_selection.json"])
    baseline = decision.get("frozen_production_baseline", {})
    model = selection.get("experiment_model", {})
    expected_baseline = {
        "grounding_policy": "G1_COVERAGE_COARSE",
        "stage1_index_fingerprint": EXPECTED_STAGE1_FINGERPRINT,
        "openai_clip_candidate_id": EXPECTED_OPENAI_CLIP_ID,
        "openai_clip_checkpoint_sha256": EXPECTED_OPENAI_CLIP_SHA256,
        "translator_model": EXPECTED_TRANSLATOR_ID,
        "translator_revision": EXPECTED_TRANSLATOR_REVISION,
        "a0_prediction_sha256": EXPECTED_A0_PREDICTION_SHA256,
    }
    if decision.get("source_tca1_git_commit") != TCA1_ANCHOR_COMMIT:
        raise RuntimeError("SCA1_TCA1_ANCHOR_MISMATCH")
    if baseline != expected_baseline:
        raise RuntimeError("SCA1_FROZEN_BASELINE_CONTRACT_MISMATCH")
    expected_model = {
        "model_id": MODEL_ID,
        "hf_revision_pin": MODEL_REVISION,
        "known_model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "image_size": 224,
        "patch_size": 16,
        "embedding_dim": EMBEDDING_DIMENSION,
        "text_max_length": 64,
        "text_padding": "max_length",
        "text_truncation": True,
        "manual_l2_normalization_required_for_get_features": True,
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise RuntimeError("SCA1_FROZEN_MODEL_CONTRACT_MISMATCH")
    isolation = selection.get("isolation", {})
    if (
        isolation.get("frame_bank") != "EXACT_SAME_177321_BTC_KEYFRAME_JPG_ROWS"
        or isolation.get("direct_vietnamese_siglip2") is not False
        or isolation.get("fusion") is not False
        or isolation.get("raw_video_expansion") is not False
    ):
        raise RuntimeError("SCA1_FROZEN_ISOLATION_CONTRACT_MISMATCH")
    return PreparationFreeze(
        source=str(path),
        zip_sha256=zip_digest,
        member_hashes=hashes,
        decision_context=decision,
        model_selection=selection,
        validation={
            "status": "PASS",
            "source_form": source_form,
            "frozen_member_sha256_gate": "PASS",
            "original_zip_container_sha256_gate": container_hash_gate,
            "member_count": len(hashes),
            "expected_catalog_rows": EXPECTED_ROWS,
            "tca1_anchor_commit": TCA1_ANCHOR_COMMIT,
            "a0_prediction_sha256": EXPECTED_A0_PREDICTION_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
    )


__all__ = ["PreparationFreeze", "load_preparation_freeze"]
