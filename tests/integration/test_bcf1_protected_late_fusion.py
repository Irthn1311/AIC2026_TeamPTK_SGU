from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from aic2026_eval.io import sha256_file
from triage_eg.diagnostics.bcf1_protected_late_fusion import (
    fuse_predictions,
    load_post_gt_design_sanity,
    load_preparation_freeze,
)
from triage_eg.diagnostics.bcf1_protected_late_fusion.contracts import (
    F1_CROSS_SHA256,
    PREPARATION_MEMBER_HASHES,
)
from triage_eg.diagnostics.bcf1_protected_late_fusion.io import write_jsonl_lf


def _freeze_source() -> Path:
    override = os.environ.get("AIC_BCF1_FREEZE_SOURCE")
    if override:
        return Path(override)
    return Path("outputs/Task/BCF1_PREPARATION_FREEZE_2026-08-18.zip")


def _queries(predictions: list[dict]) -> list[dict]:
    first: dict[str, dict] = {}
    for row in predictions:
        first.setdefault(str(row["query_id"]), row)
    output = []
    for query_id, row in first.items():
        if "frame_ids" in row:
            output.append(
                {
                    "query_id": query_id,
                    "task": "TRAKE",
                    "query": "frozen pre-GT query surrogate",
                    "event_count": len(row["frame_ids"]),
                }
            )
        elif "answer" in row:
            output.append(
                {
                    "query_id": query_id,
                    "task": "QA",
                    "query": "frozen pre-GT query surrogate",
                    "question": "frozen pre-GT question surrogate",
                }
            )
        else:
            output.append(
                {
                    "query_id": query_id,
                    "task": "KIS",
                    "query": "frozen pre-GT query surrogate",
                }
            )
    return output


def test_exact_frozen_cross_f1_hash_reproduction(tmp_path: Path) -> None:
    source = _freeze_source()
    if not source.exists():
        pytest.skip("BCF-1 preparation freeze is supplied externally at experiment time")
    preparation = load_preparation_freeze(source)
    assert preparation.validation["post_gt_design_sanity_opened"] is False
    queries = _queries(preparation.a0_cross)
    fused, provenance = fuse_predictions(queries, preparation.a0_cross, preparation.s1_cross)
    assert len(fused) == len(provenance) == 6000
    target = write_jsonl_lf(tmp_path / "cross_f1.jsonl", fused)
    assert sha256_file(target) == F1_CROSS_SHA256
    assert fused == preparation.frozen_f1_cross
    provenance_path = write_jsonl_lf(tmp_path / "cross_provenance.jsonl", provenance)
    assert (
        sha256_file(provenance_path)
        == PREPARATION_MEMBER_HASHES["bcf1_preparation/f1_fusion_provenance.jsonl"]
    )
    sanity = load_post_gt_design_sanity(preparation, finalized_cross_f1_sha256=sha256_file(target))
    assert sanity["f1_prediction_sha256"] == F1_CROSS_SHA256


def test_expanded_freeze_has_the_same_pre_gt_contract(tmp_path: Path) -> None:
    source = _freeze_source()
    if not source.is_file():
        pytest.skip("Original BCF-1 ZIP is supplied externally at experiment time")
    expanded = tmp_path / "expanded"
    with ZipFile(source) as archive:
        archive.extractall(expanded)
    preparation = load_preparation_freeze(expanded)
    assert preparation.validation["status"] == "PASS"
    assert preparation.validation["source_form"] == "KAGGLE_EXPANDED_VERIFIED_MEMBERS"
    assert preparation.validation["post_gt_design_sanity_opened"] is False
