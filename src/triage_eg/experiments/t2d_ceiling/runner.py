"""Bounded T2-D diagnostic runner and artifact packaging."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.experiments.reference_rt1 import build_video_row_groups
from triage_eg.experiments.reference_rt2 import (
    RT2BenchmarkQuery,
    load_rt2_benchmark,
    resolve_benchmark_identities,
)
from triage_eg.experiments.temporal_t2 import k_best_monotonic_paths
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
)

from .diagnostics import (
    build_t2d_metrics,
    diagnose_source_query,
    validate_expected_t2_reproduction,
)

T2D_VERSION = "0.1.0"
EXPECTED_BENCHMARK_SHA256 = "390176fb748c6bb52692dbbd664f46102700da24608c2fe046cefa699744a453"
EXPECTED_STAGE1_FINGERPRINT = "39ab968d2d957ce111cf8233d10ee08a281868c03b0b7d41ecf39ce5bb2c95b8"
BUNDLE_MEMBERS = (
    "t2d_summary.json",
    "t2d_metrics.json",
    "event_candidate_ceiling.jsonl",
    "forced_event_diagnostics.jsonl",
    "query_oracle_diagnostics.jsonl",
    "k5_failure_analysis.jsonl",
    "run_manifest.json",
    "issues.jsonl",
    "t2d_report.md",
)
HEAVY_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".npy",
    ".npz",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".jpg",
    ".jpeg",
    ".png",
}


@dataclass(frozen=True)
class T2DRunnerConfig:
    stage2: Stage2RuntimeConfig
    benchmark_path: Path
    output_root: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def preflight_t2d(config: T2DRunnerConfig) -> dict[str, Any]:
    benchmark = config.benchmark_path.expanduser().resolve(strict=True)
    benchmark_hash = _sha256(benchmark)
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError(
            f"T2D_BENCHMARK_HASH_MISMATCH: expected {EXPECTED_BENCHMARK_SHA256}, "
            f"found {benchmark_hash}"
        )
    queries = load_rt2_benchmark(benchmark)
    event_count = sum(len(query.events) for query in queries)
    if len(queries) != 24 or event_count != 74:
        raise RuntimeError(f"T2D_BENCHMARK_SHAPE_MISMATCH: {len(queries)}/{event_count}")
    stage1_root = config.stage2.stage1_root.expanduser().resolve(strict=True)
    stage1_summary = _read_json(stage1_root / "stage1_summary.json")
    fingerprint = stage1_summary.get("index_fingerprint")
    if fingerprint != EXPECTED_STAGE1_FINGERPRINT:
        raise RuntimeError(
            f"T2D_STAGE1_FINGERPRINT_MISMATCH: expected {EXPECTED_STAGE1_FINGERPRINT}, "
            f"found {fingerprint}"
        )
    if config.output_root.exists():
        raise FileExistsError(f"T2-D output already exists: {config.output_root}")
    return {
        "status": "READY",
        "benchmark_queries": len(queries),
        "benchmark_events": event_count,
        "benchmark_sha256": benchmark_hash,
        "stage1_index_fingerprint": fingerprint,
        "known_source_video_only": True,
        "score_matrix_computations_per_query": 1,
        "raw_video_required": False,
        "network_required": False,
    }


def _failure_analysis(
    event_ceiling: list[dict[str, Any]],
    forced: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
    t2_join: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ceiling_by_id = {
        (str(row["query_id"]), str(row["event_id"])): row for row in event_ceiling
    }
    forced_by_id = {
        (str(row["query_id"]), str(row["event_id"])): row for row in forced
    }
    oracle_by_query = {str(row["query_id"]): row for row in oracles}
    output = []
    for t2 in t2_join:
        if t2["t2_k5_reachable_6s"]:
            continue
        key = str(t2["query_id"]), str(t2["event_id"])
        d1, d3, d2 = ceiling_by_id[key], forced_by_id[key], oracle_by_query[key[0]]
        output.append(
            {
                "query_id": key[0],
                "event_id": key[1],
                "source_video_id": d1["source_video_id"],
                "event_count": d1["event_count"],
                "best_neighborhood_rank": d1["best_neighborhood_rank"],
                "rank_percentile": d1["rank_percentile"],
                "score_gap_from_top1": d1["score_gap_from_top1"],
                "relative_score_gap_from_top1": d1["relative_score_gap_from_top1"],
                "forced_event_path_feasible": d3["forced_event_path_feasible"],
                "forced_event_relative_score_gap": d3["forced_event_relative_score_gap"],
                "oracle_path_feasible": d2["oracle_path_feasible"],
                "oracle_relative_score_gap": d2["oracle_relative_score_gap"],
                "t2_k5_unique_anchor_count": t2["t2_k5_unique_anchor_count"],
                "t2_k5_minimum_seconds_error": t2["t2_k5_minimum_seconds_error"],
                "t2_k5_anchor_positions": t2["t2_k5_anchor_positions"],
                "root_cause_decision": "NOT_EVALUATED",
            }
        )
    if len(output) != 20:
        raise RuntimeError(f"T2D_K5_FAILURE_COUNT_MISMATCH: expected 20, found {len(output)}")
    return output


def _report(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    reproduced = metrics["T2_REPRODUCTION"]
    diversity = metrics["K5_PATH_DIVERSITY"]
    return "\n".join(
        [
            "# T2-D Candidate + Order-Constrained Ceiling Diagnostic",
            "",
            "This is a known-source-video diagnostic over AI-curated internal pseudo-GT.",
            "It is not end-to-end competition recall and does not assign a root cause.",
            "",
            f"- Status: `{summary['T2D_REAL_STATUS']}`",
            f"- K5 reachable/missed at ±6s: `{reproduced['k5_reachable_6s_count']}` / "
            f"`{reproduced['k5_missed_6s_count']}`",
            "- Single-path all-events K1/K3/K5: "
            f"`{reproduced['single_path_all_events_reachable_6s_counts']}`",
            "- Query-weighted mean anchor diversity: "
            f"`{diversity['QUERY_WEIGHTED_MEAN_ANCHOR_DIVERSITY']:.6f}`",
            "- Event-weighted mean anchor diversity: "
            f"`{diversity['EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY']:.6f}`",
            "- Root-cause decision: `NOT_EVALUATED`",
            "",
        ]
    )


def run_t2d(
    config: T2DRunnerConfig,
    queries: list[RT2BenchmarkQuery],
    *,
    runtime: OperationalRetrievalRuntime | None = None,
) -> dict[str, Any]:
    preflight = preflight_t2d(config)
    frozen = load_rt2_benchmark(config.benchmark_path)
    if [query.as_dict() for query in queries] != [query.as_dict() for query in frozen]:
        raise ValueError("T2-D queries must exactly match the frozen benchmark")
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    active_runtime = runtime or OperationalRetrievalRuntime(
        replace(config.stage2, output_root=output / "_stage2_control")
    )
    owns_runtime = runtime is None
    all_ceiling: list[dict[str, Any]] = []
    all_forced: list[dict[str, Any]] = []
    all_oracles: list[dict[str, Any]] = []
    all_t2_join: list[dict[str, Any]] = []
    all_query_t2: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    experiment_started = monotonic()
    try:
        active_runtime.load()
        if active_runtime.preflight.get("stage1_index_fingerprint") != EXPECTED_STAGE1_FINGERPRINT:
            raise RuntimeError("T2D_RUNTIME_STAGE1_FINGERPRINT_MISMATCH")
        resolve_benchmark_identities(queries, active_runtime.catalog)
        groups = {item.video_id: item for item in build_video_row_groups(active_runtime.catalog)}
        for query in queries:
            query_started = monotonic()
            requests = [
                QueryRequest(f"{query.query_id}__{event.event_id}", event.text, query.language, 1)
                for event in query.events
            ]
            encoding_started = monotonic()
            encoded = active_runtime.encode_requests(requests)
            encoding_ms = (monotonic() - encoding_started) * 1000
            scoring_started = monotonic()
            full_scores = np.asarray(
                active_runtime.backend.score_many_all(encoded.embeddings), dtype=np.float32
            )
            score_matrix_ms = (monotonic() - scoring_started) * 1000
            expected_shape = (len(query.events), active_runtime.backend.size)
            if full_scores.shape != expected_shape or not np.isfinite(full_scores).all():
                raise RuntimeError(f"T2D_SCORE_MATRIX_INVALID: {query.query_id}")
            group = groups[query.source_video_id]
            source_scores = full_scores[:, group.rows]
            t2_paths = k_best_monotonic_paths(source_scores, 5)
            (
                ceiling,
                forced,
                oracle,
                t2_join,
                query_t2,
                diagnostic_timings,
            ) = diagnose_source_query(
                query, source_scores, group.rows, active_runtime.catalog, t2_paths
            )
            all_ceiling.extend(ceiling)
            all_forced.extend(forced)
            all_oracles.append(oracle)
            all_t2_join.extend(t2_join)
            all_query_t2.append(query_t2)
            timings.append(
                {
                    "query_id": query.query_id,
                    "encoding_ms": encoding_ms,
                    "score_matrix_ms": score_matrix_ms,
                    **diagnostic_timings,
                    "total_ms": (monotonic() - query_started) * 1000,
                    "score_matrix_computations": 1,
                }
            )

        metrics = build_t2d_metrics(
            all_ceiling, all_forced, all_oracles, all_t2_join, all_query_t2
        )
        validate_expected_t2_reproduction(metrics)
        failures = _failure_analysis(all_ceiling, all_forced, all_oracles, all_t2_join)
        summary = {
            "experiment": "T2-D",
            "T2D_IMPLEMENTATION_STATUS": "COMPLETE",
            "T2D_REAL_STATUS": "COMPLETE",
            "T2D_DIAGNOSTIC_STATUS": "COMPLETE",
            "ROOT_CAUSE_DECISION": "NOT_EVALUATED",
            "benchmark_queries": len(queries),
            "benchmark_events": len(all_ceiling),
            "k5_failure_count": len(failures),
            "known_source_video_only": True,
            "official_competition_recall_claimed": False,
        }
        manifest = {
            **summary,
            "t2d_version": T2D_VERSION,
            "completed_at": datetime.now(UTC).isoformat(),
            "build_git_commit": config.stage2.build_git_commit,
            "preflight": preflight,
            "stage1_index_fingerprint": EXPECTED_STAGE1_FINGERPRINT,
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "reused_t2_solver": "k_best_monotonic_paths",
            "unconstrained_solver": "dante_monotonic_dp_lambda_0",
            "diagnostics": [
                "D1_EVENT_ONLY_CANDIDATE_CEILING",
                "D2_ALL_EVENT_ORDER_CONSTRAINED_ORACLE",
                "D3_FORCED_EVENT_PATH",
            ],
            "score_matrix_computations_per_query": 1,
            "optional_larger_k": metrics["optional_larger_k"],
            "raw_video_decoding": False,
            "images_created": False,
            "new_retrieval_algorithm": False,
            "new_model": False,
            "network_required": False,
            "timings": {
                "queries": timings,
                "experiment_total_ms": (monotonic() - experiment_started) * 1000,
            },
        }
        write_json(output / "t2d_summary.json", summary)
        write_json(output / "t2d_metrics.json", metrics)
        write_jsonl(output / "event_candidate_ceiling.jsonl", all_ceiling)
        write_jsonl(output / "forced_event_diagnostics.jsonl", all_forced)
        write_jsonl(output / "query_oracle_diagnostics.jsonl", all_oracles)
        write_jsonl(output / "k5_failure_analysis.jsonl", failures)
        write_json(output / "run_manifest.json", manifest)
        write_jsonl(output / "issues.jsonl", issues)
        (output / "t2d_report.md").write_text(_report(summary, metrics), encoding="utf-8")
        return {"summary": summary, "metrics": metrics, "manifest": manifest}
    finally:
        if owns_runtime:
            active_runtime.close()


def create_t2d_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("T2-D ZIP must be outside the output root")
    members = [source / name for name in BUNDLE_MEMBERS]
    missing = [
        name
        for name, path in zip(BUNDLE_MEMBERS, members, strict=True)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing T2-D bundle artifacts: {missing}")
    if any(path.suffix.lower() in HEAVY_SUFFIXES for path in members):
        raise ValueError("T2-D bundle contains a forbidden heavy asset")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, path.relative_to(source).as_posix())
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


__all__ = [
    "EXPECTED_BENCHMARK_SHA256",
    "EXPECTED_STAGE1_FINGERPRINT",
    "T2DRunnerConfig",
    "T2D_VERSION",
    "create_t2d_bundle",
    "preflight_t2d",
    "run_t2d",
]
