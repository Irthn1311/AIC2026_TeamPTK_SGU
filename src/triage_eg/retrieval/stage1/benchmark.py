"""Integrity sanity checks and non-semantic latency benchmarks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.retrieval.numpy_index import exact_cosine_self_diagnostics
from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.search import CompactCatalog, load_search_backend

FAILURE_CLASSIFICATIONS = {
    "SELF_SCORE_INVALID",
    "STRICTLY_BETTER_VECTOR_ANOMALY",
    "INDEX_CATALOG_ALIGNMENT_FAILURE",
    "QUERY_ROW_NOT_FOUND",
}
WARNING_CLASSIFICATIONS = {"TIE_SATURATION", "NEAR_TIE_RANKED_OUT"}
CLASSIFICATION_ORDER = (
    "PASS_TOP1",
    "PASS_TOP_K",
    "TIE_SATURATION",
    "NEAR_TIE_RANKED_OUT",
    "SELF_SCORE_INVALID",
    "STRICTLY_BETTER_VECTOR_ANOMALY",
    "INDEX_CATALOG_ALIGNMENT_FAILURE",
    "QUERY_ROW_NOT_FOUND",
)


def classify_self_query(
    diagnostic: dict[str, Any],
    *,
    top_k: int,
    self_score_tolerance: float,
) -> str:
    """Classify one self query after exact full-corpus diagnostics."""

    if not diagnostic.get("corpus_shape_valid"):
        return "INDEX_CATALOG_ALIGNMENT_FAILURE"
    if not diagnostic.get("query_row_resolvable"):
        return "QUERY_ROW_NOT_FOUND"
    if not diagnostic.get("catalog_round_trip_valid"):
        return "INDEX_CATALOG_ALIGNMENT_FAILURE"
    direct_score = diagnostic.get("direct_self_score")
    if (
        not diagnostic.get("self_score_finite")
        or direct_score is None
        or diagnostic.get("query_norm") is None
        or float(diagnostic["query_norm"]) <= 0
        or diagnostic.get("stored_norm") is None
        or float(diagnostic["stored_norm"]) <= 0
        or abs(float(direct_score) - 1.0) > self_score_tolerance
        or not diagnostic.get("search_self_score_finite")
        or not diagnostic.get("search_self_score_consistent")
        or int(diagnostic.get("non_finite_corpus_score_count") or 0) > 0
    ):
        return "SELF_SCORE_INVALID"
    if int(diagnostic.get("strictly_better_beyond_tolerance_count") or 0) > 0:
        return "STRICTLY_BETTER_VECTOR_ANOMALY"
    if diagnostic.get("queried_row_top1"):
        return "PASS_TOP1"
    if diagnostic.get("included_top_k"):
        return "PASS_TOP_K"
    if int(diagnostic.get("tie_equivalent_count") or 0) > top_k:
        return "TIE_SATURATION"
    return "NEAR_TIE_RANKED_OUT"


def aggregate_self_status(classifications: list[str]) -> str:
    if any(value in FAILURE_CLASSIFICATIONS for value in classifications):
        return "FAIL"
    if any(value in WARNING_CLASSIFICATIONS for value in classifications):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def run_self_retrieval(
    stage1_root: Path,
    *,
    samples: int = 100,
    top_k: int = 5,
    seed: int = 2026,
    chunk_rows: int = 16_384,
    diagnostic_top_k: int = 100,
    self_score_tolerance: float = 1e-5,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    index_root = stage1_root / "index"
    vectors = np.load(index_root / "clip_vectors.f16.npy", mmap_mode="r", allow_pickle=False)
    norms = np.load(index_root / "vector_norms.f32.npy", mmap_mode="r", allow_pickle=False)
    manifest = json.loads((index_root / "index_manifest.json").read_text(encoding="utf-8"))
    catalog: CompactCatalog | None = None
    catalog_error: str | None = None
    try:
        catalog = CompactCatalog(index_root)
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        catalog_error = str(error)
    randomizer = np.random.default_rng(seed)
    rows = np.sort(randomizer.choice(len(vectors), size=min(samples, len(vectors)), replace=False))
    diagnostics = exact_cosine_self_diagnostics(
        vectors,
        norms,
        rows,
        top_k=top_k,
        diagnostic_top_k=diagnostic_top_k,
        tie_tolerance=tie_tolerance,
        chunk_rows=chunk_rows,
    )
    catalog_lengths_valid = bool(
        catalog is not None
        and len(vectors) == len(norms) == len(catalog.n)
        and manifest.get("vector_count") == len(vectors)
    )
    issues = []
    for diagnostic in diagnostics:
        mapped: dict[str, Any] | None = None
        if catalog_lengths_valid and catalog is not None and diagnostic["query_row_resolvable"]:
            try:
                mapped = catalog.map_row(int(diagnostic["global_row"]))
            except (IndexError, KeyError, TypeError, ValueError):
                mapped = None
        round_trip = bool(
            mapped is not None
            and mapped["global_row"] == diagnostic["global_row"]
            and mapped["clip_row_index"] == mapped["n"] - 1
        )
        diagnostic["catalog_round_trip_valid"] = round_trip
        diagnostic["video_id"] = mapped["video_id"] if mapped else None
        diagnostic["n"] = mapped["n"] if mapped else None
        diagnostic["original_frame_idx"] = mapped["original_frame_idx"] if mapped else None
        for candidate in diagnostic["diagnostic_top_candidates"]:
            candidate_mapped = None
            if catalog_lengths_valid and catalog is not None:
                try:
                    candidate_mapped = catalog.map_row(int(candidate["global_row"]))
                except (IndexError, KeyError, TypeError, ValueError):
                    candidate_mapped = None
            candidate["video_id"] = candidate_mapped["video_id"] if candidate_mapped else None
            candidate["n"] = candidate_mapped["n"] if candidate_mapped else None
            candidate["original_frame_idx"] = (
                candidate_mapped["original_frame_idx"] if candidate_mapped else None
            )
        classification = classify_self_query(
            diagnostic,
            top_k=top_k,
            self_score_tolerance=self_score_tolerance,
        )
        diagnostic["classification"] = classification
        if classification in WARNING_CLASSIFICATIONS:
            issues.append(
                {
                    "severity": "WARNING",
                    "code": (
                        "SELF_RETRIEVAL_TIE_SATURATION"
                        if classification == "TIE_SATURATION"
                        else "SELF_RETRIEVAL_NEAR_TIE_RANKED_OUT"
                    ),
                    "evidence": {
                        "global_row": diagnostic["global_row"],
                        "actual_deterministic_rank": diagnostic["actual_deterministic_rank"],
                        "direct_self_score": diagnostic["direct_self_score"],
                        "search_self_score": diagnostic["search_self_score"],
                        "rank_higher_count": diagnostic["rank_higher_count"],
                        "tie_equivalent_count": diagnostic["tie_equivalent_count"],
                        "top_candidate_rows": [
                            item["global_row"]
                            for item in diagnostic["diagnostic_top_candidates"][:top_k]
                        ],
                    },
                }
            )
        elif classification == "STRICTLY_BETTER_VECTOR_ANOMALY":
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "SELF_RETRIEVAL_SCORE_ANOMALY",
                    "evidence": {
                        "global_row": diagnostic["global_row"],
                        "actual_deterministic_rank": diagnostic[
                            "actual_deterministic_rank"
                        ],
                        "direct_self_score": diagnostic["direct_self_score"],
                        "search_self_score": diagnostic["search_self_score"],
                        "rank_higher_count": diagnostic["rank_higher_count"],
                        "strictly_better_beyond_tolerance_count": diagnostic[
                            "strictly_better_beyond_tolerance_count"
                        ],
                        "top_candidate_rows": [
                            item["global_row"]
                            for item in diagnostic["diagnostic_top_candidates"][:top_k]
                        ],
                    },
                }
            )
        elif classification in {"INDEX_CATALOG_ALIGNMENT_FAILURE", "QUERY_ROW_NOT_FOUND"}:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "SELF_RETRIEVAL_ALIGNMENT_FAILURE",
                    "evidence": {
                        "global_row": diagnostic["global_row"],
                        "catalog_error": catalog_error,
                        "vector_rows": len(vectors),
                        "stored_norm_rows": len(norms),
                        "catalog_rows": len(catalog.n) if catalog is not None else None,
                        "manifest_vector_count": manifest.get("vector_count"),
                    },
                }
            )

    classifications = [str(item["classification"]) for item in diagnostics]
    observed_counts = Counter(classifications)
    classification_counts = {
        classification: observed_counts[classification]
        for classification in CLASSIFICATION_ORDER
    }
    status = aggregate_self_status(classifications)
    finite_self_scores = [
        float(item["direct_self_score"])
        for item in diagnostics
        if item["direct_self_score"] is not None
        and np.isfinite(float(item["direct_self_score"]))
    ]
    report = {
        "sampled_queries": len(rows),
        "top_k": top_k,
        "diagnostic_top_k": diagnostic_top_k,
        "self_score_tolerance": self_score_tolerance,
        "tie_tolerance": tie_tolerance,
        "queried_row_top1_count": sum(bool(item["queried_row_top1"]) for item in diagnostics),
        "queried_row_topk_count": sum(bool(item["included_top_k"]) for item in diagnostics),
        "self_score_min": min(finite_self_scores) if finite_self_scores else None,
        "self_score_max": max(finite_self_scores) if finite_self_scores else None,
        "classification_counts": classification_counts,
        "tie_saturated_queries": [
            item for item in diagnostics if item["classification"] == "TIE_SATURATION"
        ],
        "warning_queries": [
            item for item in diagnostics if item["classification"] in WARNING_CLASSIFICATIONS
        ],
        "failures": [
            item for item in diagnostics if item["classification"] in FAILURE_CLASSIFICATIONS
        ],
        "query_diagnostics": diagnostics,
        "issues": issues,
        "status": status,
        "semantic_quality_claim": False,
    }
    benchmark_root = stage1_root / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "self_retrieval_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_benchmark(
    stage1_root: Path,
    *,
    random_queries: int = 50,
    self_queries: int = 100,
    top_k: int = 100,
    seed: int = 2026,
    chunk_rows: int = 16_384,
    metric: str = "cosine",
) -> dict[str, Any]:
    if random_queries <= 0 or self_queries <= 0:
        raise ValueError("benchmark query counts must be positive")
    load_started = monotonic()
    config = SearchConfig(
        stage1_root,
        "benchmark",
        top_k=top_k,
        search_chunk_rows=chunk_rows,
        metric=metric,
    )
    backend, catalog = load_search_backend(config)
    load_seconds = monotonic() - load_started
    rng = np.random.default_rng(seed)
    random_started = monotonic()
    random_vectors = rng.normal(size=(random_queries, backend.dimension)).astype(np.float32)
    random_prepare_seconds = monotonic() - random_started
    self_rows = rng.choice(backend.size, size=min(self_queries, backend.size), replace=False)
    self_load_started = monotonic()
    self_vectors = backend.vectors_at(self_rows)
    self_load_seconds = monotonic() - self_load_started
    latencies = []
    formatting_latencies = []
    for query in np.concatenate((random_vectors, self_vectors)):
        started = monotonic()
        _, rows = backend.search(query, top_k)
        latencies.append(monotonic() - started)
        formatting_started = monotonic()
        for row in rows[0]:
            catalog.map_row(int(row))
        formatting_latencies.append(monotonic() - formatting_started)
    report = {
        "index_load_seconds": load_seconds,
        "query_prepare_seconds": {
            "random_vector_generation": random_prepare_seconds,
            "stored_vector_load": self_load_seconds,
        },
        "queries": len(latencies),
        "random_queries": random_queries,
        "self_queries": len(self_vectors),
        "top_k": top_k,
        "chunk_rows": chunk_rows,
        "backend": "numpy_exact",
        "metric": metric,
        "latency_seconds": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "mean": float(np.mean(latencies)),
        },
        "candidate_formatting_latency_seconds": {
            "p50": _percentile(formatting_latencies, 50),
            "p95": _percentile(formatting_latencies, 95),
            "p99": _percentile(formatting_latencies, 99),
            "mean": float(np.mean(formatting_latencies)),
        },
        "throughput_queries_per_second": len(latencies) / sum(latencies),
        "peak_estimated_working_memory_bytes": chunk_rows * backend.dimension * 4 + chunk_rows * 4,
        "ground_truth_metrics_reported": False,
    }
    root = stage1_root / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (root / "benchmark_report.md").write_text(
        "# Stage 1 Exact Search Benchmark\n\n"
        f"- Queries: {report['queries']}\n"
        f"- p50: {report['latency_seconds']['p50']:.6f} s\n"
        f"- p95: {report['latency_seconds']['p95']:.6f} s\n"
        f"- Candidate formatting p95: "
        f"{report['candidate_formatting_latency_seconds']['p95']:.6f} s\n"
        f"- Throughput: {report['throughput_queries_per_second']:.3f} queries/s\n\n"
        "No semantic Recall@K is reported without ground truth.\n",
        encoding="utf-8",
    )
    return report
