from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from triage_eg.retrieval.numpy_index import exact_cosine_self_diagnostics
from triage_eg.retrieval.stage1.benchmark import (
    aggregate_self_status,
    classify_self_query,
)
from triage_eg.retrieval.stage1.builder import (
    corpus_readiness_for_self_status,
    resolve_git_commit,
)


def diagnose(
    vectors: np.ndarray,
    query_row: int,
    *,
    top_k: int = 5,
    diagnostic_top_k: int = 100,
    norms: np.ndarray | None = None,
    tie_tolerance: float = 1e-6,
) -> dict[str, object]:
    matrix = np.asarray(vectors, dtype=np.float16)
    stored_norms = (
        np.linalg.norm(matrix.astype(np.float32), axis=1).astype(np.float32)
        if norms is None
        else np.asarray(norms, dtype=np.float32)
    )
    result = exact_cosine_self_diagnostics(
        matrix,
        stored_norms,
        np.asarray([query_row]),
        top_k=top_k,
        diagnostic_top_k=diagnostic_top_k,
        tie_tolerance=tie_tolerance,
        chunk_rows=3,
    )[0]
    result["catalog_round_trip_valid"] = True
    result["classification"] = classify_self_query(
        result, top_k=top_k, self_score_tolerance=1e-5
    )
    return result


def test_unique_vector_self_is_top1_pass() -> None:
    result = diagnose(np.eye(6), 5)
    assert result["classification"] == "PASS_TOP1"
    assert result["actual_deterministic_rank"] == 1
    assert result["tie_equivalent_count"] == 1


def test_self_inside_top5_but_not_top1_passes() -> None:
    result = diagnose(np.ones((2, 2)), 1)
    assert result["included_top_k"] and not result["queried_row_top1"]
    assert result["classification"] == "PASS_TOP_K"


def test_six_identical_vectors_are_tie_saturation() -> None:
    result = diagnose(np.ones((6, 2)), 5)
    assert result["classification"] == "TIE_SATURATION"
    assert result["actual_deterministic_rank"] == 6
    assert result["raw_higher_count"] == 0
    assert result["strictly_better_beyond_tolerance_count"] == 0
    assert result["tie_equivalent_count"] == 6


def test_large_tie_does_not_require_presence_in_diagnostic_top100() -> None:
    result = diagnose(np.ones((121, 2)), 120)
    assert result["classification"] == "TIE_SATURATION"
    assert result["actual_deterministic_rank"] == 121
    assert not result["queried_row_present_in_diagnostic_top_k"]
    assert aggregate_self_status([str(result["classification"])]) == "PASS_WITH_WARNINGS"


def test_near_tie_ranked_out_is_not_a_strict_anomaly() -> None:
    vectors = np.ones((2, 2), dtype=np.float16)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1).astype(np.float32)
    norms[0] *= np.float32(0.999999)
    result = diagnose(vectors, 1, top_k=1, norms=norms)
    assert result["classification"] in {"NEAR_TIE_RANKED_OUT", "TIE_SATURATION"}
    assert result["strictly_better_beyond_tolerance_count"] == 0
    assert float(result["diagnostic_top_candidates"][0]["delta_from_self"]) > 0


def test_strictly_better_candidate_is_failure() -> None:
    vectors = np.ones((2, 2), dtype=np.float16)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1).astype(np.float32)
    norms[0] *= np.float32(0.999998)
    result = diagnose(vectors, 1, top_k=1, norms=norms)
    assert result["classification"] == "STRICTLY_BETTER_VECTOR_ANOMALY"
    assert aggregate_self_status([str(result["classification"])]) == "FAIL"


def test_self_score_far_from_one_is_invalid() -> None:
    vectors = np.eye(2, dtype=np.float16)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1).astype(np.float32)
    norms[1] *= 2
    result = diagnose(vectors, 1, norms=norms)
    assert result["direct_self_score"] == pytest.approx(0.5)
    assert result["classification"] == "SELF_SCORE_INVALID"


@pytest.mark.parametrize(
    "vectors",
    [
        np.asarray([[1, 0], [0, 0]], dtype=np.float16),
        np.asarray([[1, 0], [np.nan, 0]], dtype=np.float16),
    ],
)
def test_zero_or_nonfinite_query_is_invalid(vectors: np.ndarray) -> None:
    result = diagnose(vectors, 1)
    assert result["classification"] == "SELF_SCORE_INVALID"


def test_nonfinite_corpus_vector_is_invalid() -> None:
    result = diagnose(np.asarray([[1, 0], [np.nan, 0]], dtype=np.float16), 0)
    assert result["non_finite_corpus_score_count"] == 1
    assert result["classification"] == "SELF_SCORE_INVALID"


def test_catalog_or_vector_length_mismatch_is_alignment_failure() -> None:
    result = diagnose(np.eye(2), 1)
    result["catalog_round_trip_valid"] = False
    assert (
        classify_self_query(result, top_k=5, self_score_tolerance=1e-5)
        == "INDEX_CATALOG_ALIGNMENT_FAILURE"
    )
    mismatched = exact_cosine_self_diagnostics(
        np.eye(2, dtype=np.float16),
        np.ones(1, dtype=np.float32),
        np.asarray([1]),
    )[0]
    mismatched["catalog_round_trip_valid"] = False
    assert (
        classify_self_query(mismatched, top_k=5, self_score_tolerance=1e-5)
        == "INDEX_CATALOG_ALIGNMENT_FAILURE"
    )


def test_diagnostic_candidate_schema_and_deterministic_order() -> None:
    result = diagnose(np.ones((6, 2)), 5)
    candidates = result["diagnostic_top_candidates"]
    assert [item["global_row"] for item in candidates] == list(range(6))
    assert set(candidates[0]) == {
        "rank",
        "global_row",
        "score",
        "delta_from_self",
        "within_tie_tolerance",
    }


@pytest.mark.parametrize(
    ("classifications", "expected"),
    [
        (["PASS_TOP1", "PASS_TOP_K"], "PASS"),
        (["PASS_TOP1", "TIE_SATURATION"], "PASS_WITH_WARNINGS"),
        (["PASS_TOP1", "SELF_SCORE_INVALID"], "FAIL"),
    ],
)
def test_aggregate_status(classifications: list[str], expected: str) -> None:
    assert aggregate_self_status(classifications) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PASS", "READY"),
        ("PASS_WITH_WARNINGS", "READY_WITH_TIE_WARNINGS"),
        ("FAIL", "BLOCKED_SELF_RETRIEVAL_FAILED"),
    ],
)
def test_stage1_corpus_readiness_mapping(status: str, expected: str) -> None:
    assert corpus_readiness_for_self_status(status) == expected


def test_explicit_commit_has_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIC_RESOLVED_GIT_COMMIT", "environment")
    assert resolve_git_commit(tmp_path, "explicit") == ("explicit", "CLI")


def test_environment_commit_has_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AIC_RESOLVED_GIT_COMMIT", "environment")
    assert resolve_git_commit(tmp_path) == ("environment", "ENV")


def test_git_commit_auto_detect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AIC_RESOLVED_GIT_COMMIT", raising=False)
    seen: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="abc123\n")

    monkeypatch.setattr("triage_eg.retrieval.stage1.builder.subprocess.run", fake_run)
    assert resolve_git_commit(tmp_path) == ("abc123", "GIT_AUTO_DETECT")
    assert seen == [["git", "-C", str(tmp_path), "rev-parse", "HEAD"]]


def test_git_commit_unknown_without_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AIC_RESOLVED_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        "triage_eg.retrieval.stage1.builder.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert resolve_git_commit(tmp_path) == ("UNKNOWN", "UNKNOWN")


def test_notebook_exports_and_passes_resolved_commit_without_stage0_rerun() -> None:
    source = Path("notebooks/06_stage1_btc_retrieval_baseline.ipynb").read_text(
        encoding="utf-8"
    )
    assert 'os.environ[\\"AIC_RESOLVED_GIT_COMMIT\\"] = COMMIT' in source
    assert 'build_git_commit=COMMIT' in source
    assert 'repo_root=REPO_DIR' in source
    assert "run_stage0_data_audit.py" not in source
    assert 'os.environ.get(\\"AIC_STAGE1_REUSE_INDEX\\") is None' not in source
