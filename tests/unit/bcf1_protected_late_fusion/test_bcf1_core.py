from __future__ import annotations

import inspect
from pathlib import Path
from zipfile import ZipFile

import pytest

from triage_eg.diagnostics.bcf1_protected_late_fusion import (
    BCF1Settings,
    candidate_key,
    create_bundle,
    evaluate_post_gt,
    fuse_predictions,
    normalize_qa_answer,
    promotion_decision,
    run_l21_arm,
    validate_frozen_index,
)
from triage_eg.diagnostics.bcf1_protected_late_fusion import runner as runner_module
from triage_eg.diagnostics.bcf1_protected_late_fusion.contracts import (
    INDEX_DTYPE,
    INDEX_FINGERPRINT,
    INDEX_ROWS,
    INDEX_SHAPE,
    NORM_SHA256,
    VECTOR_SHA256,
)
from triage_eg.diagnostics.bcf1_protected_late_fusion.runner import (
    REQUIRED_BUNDLE_MEMBERS,
    validate_all_hashes_before_gt,
)


def _query(task: str = "KIS", query_id: str = "Q1") -> dict:
    row = {"query_id": query_id, "task": task, "query": "query"}
    if task == "QA":
        row["question"] = "question"
    if task == "TRAKE":
        row["event_count"] = 2
    return row


def _row(task: str, rank: int, *, source: int = 0, query_id: str = "Q1") -> dict:
    base = {"query_id": query_id, "video_id": f"L01_V{source + 1:03d}", "rank": rank}
    if task == "KIS":
        return {**base, "frame_id": source * 1000 + rank}
    if task == "QA":
        return {**base, "frame_id": source * 1000 + rank, "answer": f"Answer {rank}"}
    return {**base, "frame_ids": [source * 1000 + rank, source * 1000 + rank + 10]}


def _top100(task: str, *, source: int = 0) -> list[dict]:
    return [_row(task, rank, source=source) for rank in range(1, 101)]


def test_01_qa_normalization_is_only_casefold_and_whitespace() -> None:
    assert normalize_qa_answer("  Red\t CAR  ") == "red car"
    assert normalize_qa_answer("Red, CAR!") == "red, car!"


def test_02_candidate_keys_follow_submission_semantics() -> None:
    assert candidate_key("KIS", _row("KIS", 1)) == ("L01_V001", 1)
    qa = _row("QA", 1)
    qa["answer"] = "  RED   Car "
    assert candidate_key("QA", qa) == ("L01_V001", 1, "red car")
    assert candidate_key("TRAKE", _row("TRAKE", 1)) == ("L01_V001", (1, 11))


def test_03_exact_rrf_and_protected_prefix() -> None:
    query = _query()
    a0, s1 = _top100("KIS"), _top100("KIS", source=1)
    fused, provenance = fuse_predictions([query], a0, s1)
    assert fused[:5] == a0[:5]
    assert provenance[:5] == [
        {
            "query_id": "Q1",
            "fused_rank": rank,
            "protected_a0_prefix": True,
            "a0_rank": rank,
            "s1_rank": None,
            "rrf_k": 60,
            "rrf_score": None,
            "source": "A0_PROTECTED",
        }
        for rank in range(1, 6)
    ]
    assert provenance[5]["rrf_score"] == pytest.approx(1 / (60 + 1))


def test_04_deterministic_ties_prefer_a0_then_canonical_key() -> None:
    query = _query()
    a0, s1 = _top100("KIS"), _top100("KIS", source=1)
    fused, provenance = fuse_predictions([query], a0, s1)
    a0_tie = next(index for index, row in enumerate(provenance) if row["a0_rank"] == 6)
    s1_tie = next(index for index, row in enumerate(provenance) if row["s1_rank"] == 6)
    assert provenance[a0_tie]["source"] == "A0_ONLY"
    assert provenance[s1_tie]["source"] == "S1_ONLY"
    assert a0_tie < s1_tie
    assert fused[a0_tie]["video_id"] == "L01_V001"


def test_05_dedup_max100_and_strict_renumbering() -> None:
    query = _query("QA")
    a0, s1 = _top100("QA"), _top100("QA", source=1)
    s1[10] = {**a0[10], "rank": 11, "answer": "  ANSWER   11 "}
    fused, provenance = fuse_predictions([query], a0, s1)
    assert len(fused) == len(provenance) == 100
    assert [row["rank"] for row in fused] == list(range(1, 101))
    key = candidate_key("QA", a0[10])
    assert sum(candidate_key("QA", row) == key for row in fused) == 1


def test_06_fusion_has_no_gt_input_or_access() -> None:
    signature = inspect.signature(fuse_predictions)
    assert "ground_truth" not in signature.parameters and "gt" not in signature.parameters
    source = inspect.getsource(fuse_predictions).casefold()
    assert "ground_truth" not in source and "gt.jsonl" not in source


def test_07_gt_fails_closed_before_all_hashes(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="BEFORE_ALL_HASHES"):
        evaluate_post_gt(
            {"A0": {}, "S1": {}, "F1": {}},
            {"A0": {}, "S1": {}, "F1": {}},
            {"status": "FAIL"},
            cross_gt_path=tmp_path / "missing",
            l21_gt_path=tmp_path / "missing",
            output_root=tmp_path / "output",
        )
    with pytest.raises(RuntimeError, match="BEFORE_ALL_HASHES_FINALIZED"):
        validate_all_hashes_before_gt(
            {"A0": {}, "S1": {}, "F1": {}},
            {"A0": {}, "S1": {}, "F1": {}},
            {"status": "PASS", "index_rebuilt": False},
        )


def test_08_exact_index_provenance_and_rebuild_gate(monkeypatch) -> None:
    manifest = {
        "index_fingerprint": INDEX_FINGERPRINT,
        "vector_sha256": VECTOR_SHA256,
        "norm_sha256": NORM_SHA256,
        "rows": INDEX_ROWS,
        "shape": INDEX_SHAPE,
        "dtype": INDEX_DTYPE,
    }
    monkeypatch.setattr(
        runner_module,
        "validate_siglip2_index",
        lambda *args, **kwargs: {"manifest": manifest},
    )
    result = validate_frozen_index("index", stage1_root="stage1")
    assert result["status"] == "PASS" and result["index_rebuilt"] is False
    manifest["index_fingerprint"] = "wrong"
    with pytest.raises(RuntimeError, match="PROVENANCE_MISMATCH"):
        validate_frozen_index("index", stage1_root="stage1")


def test_09_promotion_is_bounded_and_never_automatic() -> None:
    summary = {"final_score": 0.2, "R@1": 0.1, "R@5": 0.1}
    slices = {f"task:{task}": {"final_score": 0.2} for task in ("KIS", "QA", "TRAKE")}
    evaluations = {
        "l21": {
            "arms": {
                "A0": {"summary": summary, "slices": slices},
                "F1": {"summary": dict(summary), "slices": slices},
            },
            "paired": {"task_delta_f1_minus_a0": {task: 0.0 for task in slices}},
        }
    }
    # Use proper task labels instead of slice keys for the paired contract.
    evaluations["l21"]["paired"]["task_delta_f1_minus_a0"] = {
        task: 0.0 for task in ("KIS", "QA", "TRAKE")
    }
    decision = promotion_decision(
        {"status": "PASS", "cross_f1_reproduction_gate": "PASS"}, evaluations
    )
    assert decision["classification"] == "KEEP_FOR_MOCK"
    assert decision["automatic_production_promotion"] is False
    assert decision["production_policy_changed"] is False


def test_10_bundle_contract_forbids_index_and_weights(tmp_path: Path) -> None:
    root = tmp_path / "output"
    for name in REQUIRED_BUNDLE_MEMBERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    bundle = create_bundle(root, tmp_path / "bundle.zip")
    with ZipFile(bundle["path"]) as archive:
        assert set(REQUIRED_BUNDLE_MEMBERS) <= set(archive.namelist())
    forbidden = root / "index/siglip2_vectors.f16.npy"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="FORBIDDEN_MEMBER"):
        create_bundle(root, tmp_path / "bundle2.zip")


def test_11_settings_forbid_sweep_weights_and_production_change() -> None:
    assert BCF1Settings().production_policy_changed is False
    with pytest.raises(ValueError, match="frozen"):
        BCF1Settings(parameter_sweep=True)
    with pytest.raises(ValueError, match="frozen"):
        BCF1Settings(weights=True)
    with pytest.raises(ValueError, match="frozen"):
        BCF1Settings(production_policy_changed=True)


def test_12_l21_uses_frozen_e2eg1_g1_variant(tmp_path: Path, monkeypatch) -> None:
    assert runner_module.run_prediction_variant.__module__ == "triage_eg.e2eg1.runner"
    inference = tmp_path / "inference"
    inference.mkdir()
    (inference / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_run(pipeline, inference_root, benchmark_id, variant, output_root):
        calls.append((pipeline, inference_root, benchmark_id, variant, output_root))
        return {
            "queries": [{"query_id": "Q1", "task": "KIS", "query": "x"}],
            "predictions": [{"query_id": "Q1", "video_id": "L01_V001", "frame_id": 1, "rank": 1}],
            "validation": {"status": "PASS"},
            "variant": variant,
        }

    monkeypatch.setattr(runner_module, "run_prediction_variant", fake_run)
    monkeypatch.setattr(
        runner_module,
        "validate_predictions",
        lambda queries, predictions: ({"status": "PASS"}, []),
    )
    result = run_l21_arm(object(), inference, tmp_path / "output", tmp_path / "temporary", "A0")
    assert calls[0][2:4] == ("DEV_L21_150", "G1_COVERAGE_COARSE")
    assert result["variant"] == "G1_COVERAGE_COARSE"
    assert result["validation"]["status"] == "PASS"
