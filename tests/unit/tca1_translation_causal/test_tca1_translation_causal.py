from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from triage_eg.diagnostics.tca1_translation_causal import (
    TCA1RuntimeProxy,
    TCA1Settings,
    create_bundle,
    load_frozen_review,
    request_id_to_unit_id,
    validate_pre_gt_integrity,
)
from triage_eg.diagnostics.tca1_translation_causal import review as review_module
from triage_eg.diagnostics.tca1_translation_causal.contracts import (
    EXPECTED_A0_PREDICTION_SHA256,
    EXPECTED_BY_TASK,
    EXPECTED_COUNTS,
)
from triage_eg.diagnostics.tca1_translation_causal.review import FrozenReview
from triage_eg.diagnostics.tca1_translation_causal.runner import run_pre_gt_arm
from triage_eg.retrieval.stage2.contracts import QueryRequest
from triage_eg.retrieval.stage2.language import LanguageResolution
from triage_eg.retrieval.stage2.runtime import EncodedQueryBatch


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frozen_rows() -> tuple[list[dict], list[dict]]:
    rows, overrides = [], []
    index = 0
    for task, counts in EXPECTED_BY_TASK.items():
        for verdict, count in counts.items():
            for _ in range(count):
                index += 1
                query_id = f"CB-{task}-{index:03d}"
                unit_id = f"{query_id}:E1"
                row = {
                    "review_index": index,
                    "unit_id": unit_id,
                    "query_id": query_id,
                    "task": task,
                    "event_id": "E1" if task == "TRAKE" else None,
                    "source_vi": f"nguồn {index}",
                    "opus_en": f"baseline {index}",
                    "verdict": verdict,
                    "reason_tags": [],
                    "reference_en": f"reference {index}" if verdict == "FAIL" else None,
                    "review_protocol": review_module.FROZEN_REVIEW_PROTOCOL,
                    "review_version": review_module.FROZEN_REVIEW_VERSION,
                }
                rows.append(row)
                if verdict == "FAIL":
                    overrides.append({"unit_id": unit_id, "reference_en": f"reference {index}"})
    return rows, overrides


def _write_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, collision=False) -> Path:
    rows, overrides = _frozen_rows()
    if collision:
        fail_source = next(row["source_vi"] for row in rows if row["verdict"] == "FAIL")
        next(row for row in rows if row["verdict"] == "PASS")["source_vi"] = fail_source
    review_bytes = "".join(json.dumps(row) + "\n" for row in rows).encode()
    override_bytes = "".join(json.dumps(row) + "\n" for row in overrides).encode()
    monkeypatch.setattr(review_module, "FROZEN_REVIEW_SHA256", _digest(review_bytes))
    monkeypatch.setattr(review_module, "FROZEN_OVERRIDE_SHA256", _digest(override_bytes))
    summary = {
        "status": "FROZEN_FOR_TCA1",
        "review_version": review_module.FROZEN_REVIEW_VERSION,
        "review_protocol": review_module.FROZEN_REVIEW_PROTOCOL,
        "source_translation_blind_qc_sha256": review_module.FROZEN_SOURCE_QC_SHA256,
        "review_row_count": 100,
        "counts": EXPECTED_COUNTS,
        "by_task": EXPECTED_BY_TASK,
        "fail_override_count": 17,
        "review_sha256": _digest(review_bytes),
        "fail_overrides_sha256": _digest(override_bytes),
        "gt_used_for_verdicts": False,
        "retrieval_rank_or_outcome_used_for_verdicts": False,
    }
    members = {
        "README.md": b"freeze",
        "representation_ceiling_audit.json": b"{}\n",
        "translation_blind_review_frozen.jsonl": review_bytes,
        "translation_blind_review_summary.json": json.dumps(summary).encode(),
        "translation_fail_overrides.jsonl": override_bytes,
    }
    path = tmp_path / "freeze.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    monkeypatch.setattr(
        review_module, "FROZEN_ZIP_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return path


def _small_frozen() -> FrozenReview:
    rows = (
        {
            "unit_id": "CB-KIS-01:E1",
            "query_id": "CB-KIS-01",
            "task": "KIS",
            "event_id": None,
            "source_vi": "nguồn một",
            "opus_en": "baseline one",
            "verdict": "FAIL",
        },
        {
            "unit_id": "CB-QA-01:E1",
            "query_id": "CB-QA-01",
            "task": "QA",
            "event_id": None,
            "source_vi": "nguồn hai",
            "opus_en": "baseline two",
            "verdict": "PASS",
        },
    )
    return FrozenReview(
        rows=rows,
        overrides={"CB-KIS-01:E1": "reference one"},
        rows_by_unit={row["unit_id"]: row for row in rows},
        fail_unit_ids=frozenset({"CB-KIS-01:E1"}),
        nonfail_unit_ids=frozenset({"CB-QA-01:E1"}),
        file_hashes={},
        source="fixture",
        source_zip_sha256=None,
        validation={"status": "PASS"},
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.encoder_inputs: list[list[str]] = []

    def encode_requests(self, requests: list[QueryRequest]) -> EncodedQueryBatch:
        texts = [row.text for row in requests]
        self.encoder_inputs.append(texts)
        embeddings = np.stack(
            [
                np.full(512, (index + 1) / np.sqrt(512), dtype=np.float32)
                for index in range(len(texts))
            ]
        )
        return EncodedQueryBatch(
            embeddings=embeddings,
            resolutions=tuple(
                LanguageResolution(row.language, row.language, "EXPLICIT", "DIRECT_CLIP")
                for row in requests
            ),
            encodings=tuple(
                {
                    "original_query_text": row.text,
                    "resolved_language": row.language,
                    "translation_applied": row.language == "vi",
                    "translated_text": (
                        "baseline one" if row.text == "nguồn một" else "baseline two"
                    )
                    if row.language == "vi"
                    else None,
                    "clip_input_text": (
                        "baseline one" if row.text == "nguồn một" else "baseline two"
                    )
                    if row.language == "vi"
                    else row.text,
                    "clip_candidate_id": "clip",
                    "embedding_dimension": 512,
                    "embedding_finite": True,
                    "embedding_normalized": True,
                }
                for row in requests
            ),
            latencies_ms=tuple({"translation_ms": 1.0} for _ in requests),
            batch_latency_ms=1.0,
        )

    def runtime_manifest(self) -> dict:
        return {"stage1_index_fingerprint": "index", "stage1b": {"candidate_id": "clip"}}


def _snapshots(frozen: FrozenReview):
    def unit(row: dict, clip: str, value: float):
        return SimpleNamespace(
            source_text=row["source_vi"],
            encoding={"clip_input_text": clip},
            embedding=np.full(2, value, dtype=np.float32),
            scores=np.full(3, value, dtype=np.float32),
        )

    a0 = {key: unit(row, row["opus_en"], 1.0) for key, row in frozen.rows_by_unit.items()}
    a1 = {
        key: unit(
            row,
            frozen.overrides.get(key, row["opus_en"]),
            2.0 if key in frozen.fail_unit_ids else 1.0,
        )
        for key, row in frozen.rows_by_unit.items()
    }
    return SimpleNamespace(units=a0), SimpleNamespace(units=a1)


def _run(digest: str) -> dict:
    return {"sha256": digest, "queries": [], "variant": "G1_COVERAGE_COARSE"}


def test_01_frozen_review_schema_count_and_hash_validation(tmp_path, monkeypatch) -> None:
    frozen = load_frozen_review(_write_freeze(tmp_path, monkeypatch))
    assert frozen.validation["status"] == "PASS"
    assert frozen.validation["verdict_counts"] == EXPECTED_COUNTS
    assert len(frozen.fail_unit_ids) == 17


@pytest.mark.parametrize(
    ("request_id", "unit_id"),
    [
        ("CB-KIS-13__grounding", "CB-KIS-13:E1"),
        ("CB-QA-04__grounding", "CB-QA-04:E1"),
        ("CB-TR-13__events__2", "CB-TR-13:E2"),
    ],
)
def test_02_request_id_to_unit_id_mapping(request_id: str, unit_id: str) -> None:
    assert request_id_to_unit_id(request_id) == unit_id


def test_03_unknown_request_id_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="UNKNOWN_REQUEST_ID"):
        request_id_to_unit_id("CB-KIS-01")


def test_04_a1_overrides_exact_fail_only() -> None:
    runtime = _FakeRuntime()
    proxy = TCA1RuntimeProxy(runtime, _small_frozen(), "A1")
    result = proxy.encode_requests(
        [
            QueryRequest("CB-KIS-01__grounding", "nguồn một", "vi"),
            QueryRequest("CB-QA-01__grounding", "nguồn hai", "vi"),
        ]
    )
    assert runtime.encoder_inputs == [["reference one", "nguồn hai"]]
    assert [row["tca1_override_applied"] for row in result.encodings] == [True, False]


def test_05_a0_changes_no_request() -> None:
    runtime = _FakeRuntime()
    TCA1RuntimeProxy(runtime, _small_frozen(), "A0").encode_requests(
        [QueryRequest("CB-KIS-01__grounding", "nguồn một", "vi")]
    )
    assert runtime.encoder_inputs == [["nguồn một"]]


def test_06_conflicting_duplicate_source_fails_closed(tmp_path, monkeypatch) -> None:
    path = _write_freeze(tmp_path, monkeypatch, collision=True)
    with pytest.raises(RuntimeError, match="CONFLICTING_SOURCE_COLLISION"):
        load_frozen_review(path)


def test_07_provenance_patch_is_complete() -> None:
    result = TCA1RuntimeProxy(_FakeRuntime(), _small_frozen(), "A1").encode_requests(
        [QueryRequest("CB-KIS-01__grounding", "nguồn một", "vi")]
    )
    row = result.encodings[0]
    assert row["tca1_unit_id"] == "CB-KIS-01:E1"
    assert row["tca1_original_vi"] == "nguồn một"
    assert row["tca1_baseline_opus_en"] == "baseline one"
    assert row["tca1_reference_en"] == "reference one"


def test_08_runtime_invariants_must_match() -> None:
    frozen = _small_frozen()
    a0, a1 = _snapshots(frozen)
    manifest = {
        "stage1_index_fingerprint": "index",
        "stage1b": {"candidate_id": "clip", "checkpoint_sha256": "sha"},
    }
    with pytest.raises(RuntimeError, match="RUNTIME_INVARIANT"):
        validate_pre_gt_integrity(
            _run(EXPECTED_A0_PREDICTION_SHA256),
            _run("a1"),
            a0,
            a1,
            frozen,
            manifest,
            {**manifest, "stage1_index_fingerprint": "other"},
        )


def test_09_changed_unit_set_and_negative_controls_pass() -> None:
    frozen = _small_frozen()
    a0, a1 = _snapshots(frozen)
    manifest = {
        "stage1_index_fingerprint": "index",
        "stage1b": {"candidate_id": "clip", "checkpoint_sha256": "sha"},
    }
    value = validate_pre_gt_integrity(
        _run(EXPECTED_A0_PREDICTION_SHA256),
        _run("a1"),
        a0,
        a1,
        frozen,
        manifest,
        manifest,
    )
    assert value["changed_clip_input_unit_ids"] == ["CB-KIS-01:E1"]
    assert value["nonfail_exact_score_vector_count"] == 1


def test_09b_full_frozen_contract_changes_exactly_17_units(tmp_path, monkeypatch) -> None:
    frozen = load_frozen_review(_write_freeze(tmp_path, monkeypatch))
    a0, a1 = _snapshots(frozen)
    manifest = {
        "stage1_index_fingerprint": "index",
        "stage1b": {"candidate_id": "clip", "checkpoint_sha256": "sha"},
    }
    value = validate_pre_gt_integrity(
        _run(EXPECTED_A0_PREDICTION_SHA256),
        _run("a1"),
        a0,
        a1,
        frozen,
        manifest,
        manifest,
    )
    assert value["changed_clip_input_unit_count"] == 17
    assert set(value["changed_clip_input_unit_ids"]) == set(frozen.fail_unit_ids)
    assert value["nonfail_exact_score_vector_count"] == 83


def test_10_negative_control_score_change_fails_closed() -> None:
    frozen = _small_frozen()
    a0, a1 = _snapshots(frozen)
    a1.units["CB-QA-01:E1"].scores[0] = 5.0
    with pytest.raises(RuntimeError, match="NEGATIVE_CONTROL_SCORE"):
        validate_pre_gt_integrity(
            _run(EXPECTED_A0_PREDICTION_SHA256),
            _run("a1"),
            a0,
            a1,
            frozen,
            {},
            {},
        )


def test_11_gt_unavailable_gate_is_reused() -> None:
    assert run_pre_gt_arm.__doc__ and "queries only" in run_pre_gt_arm.__doc__


def test_12_bundle_excludes_forbidden_payloads(tmp_path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="FORBIDDEN_MEMBER"):
        create_bundle(root, tmp_path / "bundle.zip")


def test_12b_bundle_excludes_ground_truth_and_sealed_paths(tmp_path) -> None:
    root = tmp_path / "output"
    (root / "diagnostics/sealed_eval").mkdir(parents=True)
    (root / "diagnostics/sealed_eval/gt.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="FORBIDDEN_MEMBER"):
        create_bundle(root, tmp_path / "bundle.zip")


def test_13_bundle_contains_only_bounded_files(tmp_path) -> None:
    root = tmp_path / "output"
    (root / "evaluation").mkdir(parents=True)
    (root / "evaluation/result.json").write_text("{}")
    result = create_bundle(root, tmp_path / "bundle.zip")
    assert result["member_count"] == 1 and result["sha256"]


def test_14_production_features_are_frozen_off() -> None:
    settings = TCA1Settings()
    assert not any(
        (settings.run_m1, settings.use_m2, settings.use_m3, settings.use_graph, settings.use_vlm)
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        TCA1Settings(use_graph=True)


def test_15_a0_hash_mismatch_fails_closed() -> None:
    frozen = _small_frozen()
    a0, a1 = _snapshots(frozen)
    with pytest.raises(RuntimeError, match="A0_G1_REPRODUCTION"):
        validate_pre_gt_integrity(_run("wrong"), _run("a1"), a0, a1, frozen, {}, {})
