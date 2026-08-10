"""Runner and artifact packaging for T2 k-best temporal hypotheses."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.experiments.reference_rt1 import build_video_row_groups, dante_monotonic_dp
from triage_eg.experiments.reference_rt2 import (
    BENCHMARK_TYPE,
    RT2BenchmarkQuery,
    load_rt2_benchmark,
    resolve_benchmark_identities,
)
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
)

from .metrics import (
    build_t2_metrics,
    event_is_reachable,
    query_all_events_reachable,
    validate_recall_monotonicity,
)
from .solver import MAX_BEAM_WIDTH, TemporalPath, k_best_monotonic_paths

T2_VERSION = "0.1.0"
T2_METHOD = "K_BEST_ORDER_ONLY_MONOTONIC_DP"
K_VALUES = (1, 3, 5)
TOLERANCE_SECONDS = (3, 6, 9, 12)
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".npy",
    ".npz",
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
}
BUNDLE_MEMBERS = (
    "t2_summary.json",
    "t2_metrics.json",
    "event_reachability.jsonl",
    "query_results.jsonl",
    "top_paths.jsonl",
    "run_manifest.json",
    "issues.jsonl",
)


@dataclass(frozen=True)
class T2Settings:
    k_values: tuple[int, ...] = K_VALUES
    tolerance_seconds: tuple[int, ...] = TOLERANCE_SECONDS
    beam_width: int = MAX_BEAM_WIDTH

    def __post_init__(self) -> None:
        if self.k_values != K_VALUES or self.tolerance_seconds != TOLERANCE_SECONDS:
            raise ValueError("T2 freezes K=(1,3,5) and tolerances=(3,6,9,12) seconds")
        if self.beam_width != MAX_BEAM_WIDTH:
            raise ValueError("T2 beam width is frozen at 5")


@dataclass(frozen=True)
class T2RunnerConfig:
    stage2: Stage2RuntimeConfig
    benchmark_path: Path
    output_root: Path
    settings: T2Settings = T2Settings()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fps(catalog: Any, rows: np.ndarray) -> float:
    values = np.asarray(catalog.mapping_fps[rows], dtype=np.float64)
    if values.shape != rows.shape or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("T2 source video has invalid Stage 1 mapping FPS")
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
        raise ValueError("T2 source video has inconsistent Stage 1 mapping FPS")
    return float(values[0])


def _path_record(
    query: RT2BenchmarkQuery,
    path: TemporalPath,
    rank: int,
    source_rows: np.ndarray,
    catalog: Any,
) -> dict[str, Any]:
    anchors = []
    for event, position in zip(query.events, path.positions, strict=True):
        global_row = int(source_rows[position])
        mapped = catalog.map_row(global_row)
        anchors.append(
            {
                "event_id": event.event_id,
                "catalog_position": int(position),
                "global_row": global_row,
                "n": int(mapped["n"]),
                "original_frame_idx": int(mapped["original_frame_idx"]),
            }
        )
    return {
        "query_id": query.query_id,
        "source_video_id": query.source_video_id,
        "path_rank": rank,
        "path_score_sum": path.score,
        "positions": list(path.positions),
        "strictly_monotonic": True,
        "anchors": anchors,
    }


def evaluate_source_paths(
    query: RT2BenchmarkQuery,
    paths: tuple[TemporalPath, ...],
    source_rows: np.ndarray,
    catalog: Any,
    *,
    k_values: tuple[int, ...] = K_VALUES,
    tolerances: tuple[int, ...] = TOLERANCE_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("T2 source evaluation requires at least one path")
    fps = _source_fps(catalog, source_rows)
    path_rows = [
        _path_record(query, path, rank, source_rows, catalog)
        for rank, path in enumerate(paths, 1)
    ]
    event_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(query.events):
        by_k = {}
        for k in k_values:
            selected = path_rows[:k]
            positions = [int(item["anchors"][event_index]["catalog_position"]) for item in selected]
            frames = [int(item["anchors"][event_index]["original_frame_idx"]) for item in selected]
            distances = [abs(value - event.reference_original_frame_idx) for value in frames]
            minimum = min(distances)
            by_k[str(k)] = {
                "hypothesis_count": len(selected),
                "anchor_catalog_positions": positions,
                "anchor_original_frame_indices": frames,
                "unique_anchor_position_count": len(set(positions)),
                "minimum_absolute_frame_error": minimum,
                "minimum_absolute_seconds_error": minimum / fps,
                "reachable_seconds": {
                    str(tolerance): event_is_reachable(minimum, fps, tolerance)
                    for tolerance in tolerances
                },
            }
        event_rows.append(
            {
                "query_id": query.query_id,
                "event_id": event.event_id,
                "event_count": len(query.events),
                "source_video_id": query.source_video_id,
                "source_video_fps": fps,
                "reference_catalog_position": event.reference_catalog_position,
                "reference_original_frame_idx": event.reference_original_frame_idx,
                "by_k": by_k,
            }
        )

    query_by_k = {}
    diversity = {}
    for k in k_values:
        selected = path_rows[:k]
        query_by_k[str(k)] = {
            "hypothesis_count": len(selected),
            "all_events_reachable_seconds": {
                str(tolerance): query_all_events_reachable(event_rows, k, tolerance)
                for tolerance in tolerances
            },
        }
        position_diversity = [
            len({int(item["positions"][event_index]) for item in selected})
            for event_index in range(len(query.events))
        ]
        position_paths = [tuple(int(value) for value in item["positions"]) for item in selected]
        diversity[str(k)] = {
            "requested_path_count": k,
            "returned_path_count": len(selected),
            "unique_path_count": len(set(position_paths)),
            "duplicate_path_rate": (
                1.0 - len(set(position_paths)) / len(position_paths) if position_paths else 0.0
            ),
            "anchor_position_diversity_per_event": position_diversity,
        }
    query_row = {
        "query_id": query.query_id,
        "source_video_id": query.source_video_id,
        "event_count": len(query.events),
        "source_video_fps": fps,
        "paths_available": len(path_rows),
        "by_k": query_by_k,
        "path_diversity": diversity,
    }
    return event_rows, query_row, path_rows


def preflight_t2(config: T2RunnerConfig) -> dict[str, Any]:
    benchmark = config.benchmark_path.expanduser().resolve(strict=True)
    queries = load_rt2_benchmark(benchmark)
    event_count = sum(len(query.events) for query in queries)
    if len(queries) != 24 or event_count != 74:
        raise ValueError(
            f"T2 requires the frozen 24-query/74-event RT2 benchmark; found "
            f"{len(queries)}/{event_count}"
        )
    if config.output_root.exists():
        raise FileExistsError(f"T2 output already exists: {config.output_root}")
    return {
        "status": "READY",
        "benchmark_type": BENCHMARK_TYPE,
        "benchmark_queries": len(queries),
        "benchmark_events": event_count,
        "benchmark_sha256": _sha256(benchmark),
        "k_values": list(config.settings.k_values),
        "tolerance_seconds": list(config.settings.tolerance_seconds),
        "raw_video_required": False,
        "network_required": False,
    }


def run_t2(
    config: T2RunnerConfig,
    queries: list[RT2BenchmarkQuery],
    *,
    runtime: OperationalRetrievalRuntime | None = None,
) -> dict[str, Any]:
    preflight = preflight_t2(config)
    frozen_queries = load_rt2_benchmark(config.benchmark_path)
    if [query.as_dict() for query in queries] != [query.as_dict() for query in frozen_queries]:
        raise ValueError("T2 queries must exactly match the frozen benchmark file")
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    stage2_config = replace(config.stage2, output_root=output / "_stage2_control")
    active_runtime = runtime or OperationalRetrievalRuntime(stage2_config)
    owns_runtime = runtime is None
    all_events: list[dict[str, Any]] = []
    all_queries: list[dict[str, Any]] = []
    all_paths: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    timings = []
    started = monotonic()
    try:
        active_runtime.load()
        resolve_benchmark_identities(queries, active_runtime.catalog)
        groups = {item.video_id: item for item in build_video_row_groups(active_runtime.catalog)}
        for query in queries:
            query_started = monotonic()
            requests = [
                QueryRequest(f"{query.query_id}__{event.event_id}", event.text, query.language, 1)
                for event in query.events
            ]
            encoded = active_runtime.encode_requests(requests)
            full_scores = np.asarray(
                active_runtime.backend.score_many_all(encoded.embeddings), dtype=np.float32
            )
            expected_shape = (len(query.events), active_runtime.backend.size)
            if full_scores.shape != expected_shape or not np.isfinite(full_scores).all():
                raise RuntimeError(f"Invalid T2 full Stage 1 score matrix for {query.query_id}")
            source_group = groups[query.source_video_id]
            source_scores = full_scores[:, source_group.rows]
            paths = k_best_monotonic_paths(source_scores, config.settings.beam_width)
            baseline = dante_monotonic_dp(source_scores, distance_lambda=0.0)
            if baseline is None or not paths:
                raise RuntimeError(f"No strict temporal path for {query.query_id}")
            if paths[0].positions != baseline.positions or not np.isclose(
                paths[0].score, baseline.score, rtol=1e-7, atol=1e-7
            ):
                raise RuntimeError(f"T2 K=1 does not reproduce lambda=0 DP for {query.query_id}")
            event_rows, query_row, path_rows = evaluate_source_paths(
                query,
                paths,
                source_group.rows,
                active_runtime.catalog,
                k_values=config.settings.k_values,
                tolerances=config.settings.tolerance_seconds,
            )
            if len(paths) < config.settings.beam_width:
                issues.append(
                    {
                        "severity": "WARNING",
                        "code": "FEWER_THAN_FIVE_UNIQUE_PATHS",
                        "query_id": query.query_id,
                        "evidence": {"paths_available": len(paths)},
                    }
                )
            all_events.extend(event_rows)
            all_queries.append(query_row)
            all_paths.extend(path_rows)
            timings.append(
                {
                    "query_id": query.query_id,
                    "elapsed_ms": (monotonic() - query_started) * 1000,
                    "full_score_matrix_computations": 1,
                }
            )

        metrics = build_t2_metrics(
            all_events,
            all_queries,
            k_values=config.settings.k_values,
            tolerances=config.settings.tolerance_seconds,
        )
        validate_recall_monotonicity(metrics)
        primary = metrics["OVERALL"]["PRIMARY_6_SECONDS"]
        summary = {
            "experiment": "T2",
            "T2_IMPLEMENTATION_STATUS": "COMPLETE",
            "T2_REAL_STATUS": "COMPLETE",
            "T2_QUALITY_DECISION": "NOT_EVALUATED",
            "benchmark_type": BENCHMARK_TYPE,
            "query_count": len(all_queries),
            "event_count": len(all_events),
            "primary_tolerance_seconds": 6,
            "primary_metrics": primary,
            "issues": len(issues),
        }
        manifest = {
            **summary,
            "t2_version": T2_VERSION,
            "completed_at": datetime.now(UTC).isoformat(),
            "build_git_commit": config.stage2.build_git_commit,
            "preflight": preflight,
            "method": T2_METHOD,
            "baseline": "ORDER_ONLY_MONOTONIC_DP_LAMBDA_0_K_1",
            "treatment": "DETERMINISTIC_BOUNDED_K_BEST_MONOTONIC_DP",
            "beam_width_max": MAX_BEAM_WIDTH,
            "k1_exact_dp_parity": True,
            "strict_event_order": True,
            "deterministic_tie_policy": "SCORE_DESC_THEN_LEXICOGRAPHIC_POSITIONS_ASC",
            "distance_lambda": 0.0,
            "stage1_score_matrix_computations_per_query": 1,
            "stage1_index_fingerprint": active_runtime.preflight["stage1_index_fingerprint"],
            "benchmark_sha256": _sha256(config.benchmark_path),
            "fps_source": "STAGE1_CATALOG_FRAME_MAPPING_FPS",
            "raw_video_decoding": False,
            "local_refinement": False,
            "new_model": False,
            "score_perturbation": False,
            "optional_video_metrics": "NOT_RUN_SCOPE_CONTROL",
            "network_required": False,
            "timings": {
                "queries": timings,
                "experiment_total_ms": (monotonic() - started) * 1000,
            },
        }
        write_json(output / "t2_summary.json", summary)
        write_json(output / "t2_metrics.json", metrics)
        write_jsonl(output / "event_reachability.jsonl", all_events)
        write_jsonl(output / "query_results.jsonl", all_queries)
        write_jsonl(output / "top_paths.jsonl", all_paths)
        write_json(output / "run_manifest.json", manifest)
        write_jsonl(output / "issues.jsonl", issues)
        return {"summary": summary, "metrics": metrics, "manifest": manifest}
    finally:
        if owns_runtime:
            active_runtime.close()


def create_t2_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("T2 ZIP must be outside the output root")
    members = [source / name for name in BUNDLE_MEMBERS]
    missing = [
        name
        for name, path in zip(BUNDLE_MEMBERS, members, strict=True)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing T2 bundle artifacts: {missing}")
    if any(path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES for path in members):
        raise ValueError("T2 bundle contains a heavy asset")
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
    "K_VALUES",
    "T2RunnerConfig",
    "T2Settings",
    "T2_METHOD",
    "T2_VERSION",
    "TOLERANCE_SECONDS",
    "create_t2_bundle",
    "evaluate_source_paths",
    "preflight_t2",
    "run_t2",
]
