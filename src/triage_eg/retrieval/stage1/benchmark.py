"""Integrity sanity checks and non-semantic latency benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.search import load_search_backend


def run_self_retrieval(
    stage1_root: Path,
    *,
    samples: int = 100,
    top_k: int = 5,
    seed: int = 2026,
    chunk_rows: int = 16_384,
) -> dict[str, Any]:
    config = SearchConfig(stage1_root, "self_retrieval", top_k=top_k, search_chunk_rows=chunk_rows)
    backend, catalog = load_search_backend(config)
    randomizer = np.random.default_rng(seed)
    rows = np.sort(randomizer.choice(backend.size, size=min(samples, backend.size), replace=False))
    queries = backend.vectors_at(rows)
    scores, retrieved = backend.search(queries, top_k)
    inclusion = []
    top1 = []
    self_scores = []
    top_score_tie_counts = []
    failures = []
    for index, row in enumerate(rows):
        positions = np.flatnonzero(retrieved[index] == row)
        included = len(positions) > 0
        inclusion.append(included)
        top1.append(int(retrieved[index, 0]) == int(row))
        score = float(scores[index, positions[0]]) if included else None
        self_scores.append(score)
        tie_count = int(np.sum(np.abs(scores[index] - scores[index, 0]) <= 1e-6))
        top_score_tie_counts.append(tie_count)
        mapped = catalog.map_row(int(row))
        if (
            mapped["global_row"] != int(row)
            or mapped["clip_row_index"] != mapped["n"] - 1
            or not included
            or score is None
            or score < 0.999
        ):
            failures.append(
                {"global_row": int(row), "included_top_k": included, "self_score": score}
            )
    report = {
        "sampled_queries": len(rows),
        "top_k": top_k,
        "queried_row_top1_count": sum(top1),
        "queried_row_topk_count": sum(inclusion),
        "self_score_min": min(value for value in self_scores if value is not None),
        "self_score_max": max(value for value in self_scores if value is not None),
        "top_score_tie_tolerance": 1e-6,
        "queries_with_top_score_ties_in_top_k": sum(
            count > 1 for count in top_score_tie_counts
        ),
        "max_top_score_tie_count_in_top_k": max(top_score_tie_counts),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
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
