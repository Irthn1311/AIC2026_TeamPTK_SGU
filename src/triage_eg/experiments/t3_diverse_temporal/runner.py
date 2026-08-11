"""DEV/HOLDOUT runner for T3 coverage-aware diverse temporal hypotheses."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.experiments.reference_rt1 import build_video_row_groups
from triage_eg.experiments.reference_rt2 import (
    RT2BenchmarkQuery,
    load_rt2_benchmark,
    resolve_benchmark_identities,
    split_dev_holdout,
)
from triage_eg.experiments.t2d_ceiling import (
    EXPECTED_BENCHMARK_SHA256,
    EXPECTED_STAGE1_FINGERPRINT,
    reference_neighborhood_mask,
    stable_event_ranking,
)
from triage_eg.experiments.temporal_t2 import k_best_monotonic_paths
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
)

from .hypotheses import (
    FINAL_PATH_LIMIT,
    MAX_RAW_COMBINATIONS,
    POOL_LIMIT,
    REGION_RADIUS_SECONDS,
    DiverseTemporalPath,
    build_diverse_event_pool,
    enumerate_feasible_paths,
    relative_score_gap,
    select_coverage_aware,
    select_score_top_k,
)

T3_VERSION = "0.1.0"
DELTA_GRID = (0.01, 0.03, 0.05)
K_VALUES = (1, 3, 5)
TOLERANCES = (3, 6, 9, 12)
BUNDLE_MEMBERS = (
    "t3_summary.json",
    "t3_metrics.json",
    "dev_delta_sweep.json",
    "event_candidate_pools.jsonl",
    "query_hypotheses.jsonl",
    "event_reachability.jsonl",
    "holdout_comparison.json",
    "run_manifest.json",
    "issues.jsonl",
    "t3_report.md",
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
class T3RunnerConfig:
    stage2: Stage2RuntimeConfig
    benchmark_path: Path
    output_root: Path
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.seed != 2026:
            raise ValueError("T3 reuses the frozen RT2 split seed 2026")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def preflight_t3(config: T3RunnerConfig) -> dict[str, Any]:
    benchmark = config.benchmark_path.expanduser().resolve(strict=True)
    benchmark_hash = _sha256(benchmark)
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError("T3_BENCHMARK_HASH_MISMATCH")
    queries = load_rt2_benchmark(benchmark)
    if len(queries) != 24 or sum(len(query.events) for query in queries) != 74:
        raise RuntimeError("T3_BENCHMARK_SHAPE_MISMATCH")
    stage1_summary = _read_json(
        config.stage2.stage1_root.expanduser().resolve(strict=True) / "stage1_summary.json"
    )
    if stage1_summary.get("index_fingerprint") != EXPECTED_STAGE1_FINGERPRINT:
        raise RuntimeError("T3_STAGE1_FINGERPRINT_MISMATCH")
    dev, holdout = split_dev_holdout(queries, config.seed)
    if len(dev) != 16 or len(holdout) != 8:
        raise RuntimeError(f"T3_RT2_SPLIT_MISMATCH: DEV={len(dev)} HOLDOUT={len(holdout)}")
    if config.output_root.exists():
        raise FileExistsError(f"T3 output already exists: {config.output_root}")
    return {
        "status": "READY",
        "benchmark_sha256": benchmark_hash,
        "stage1_index_fingerprint": EXPECTED_STAGE1_FINGERPRINT,
        "benchmark_queries": len(queries),
        "benchmark_events": 74,
        "dev_queries": len(dev),
        "holdout_queries": len(holdout),
        "pool_limit": POOL_LIMIT,
        "region_radius_seconds": REGION_RADIUS_SECONDS,
        "final_path_limit": FINAL_PATH_LIMIT,
        "maximum_raw_combinations_per_query": MAX_RAW_COMBINATIONS,
        "score_matrix_computations_per_query": 1,
        "known_source_video_only": True,
        "raw_video_required": False,
        "network_required": False,
    }


def _source_arrays(catalog: Any, rows: np.ndarray) -> tuple[np.ndarray, float]:
    frames = np.asarray(catalog.original_idx[rows], dtype=np.int64)
    fps_values = np.asarray(catalog.mapping_fps[rows], dtype=np.float64)
    if (
        frames.shape != rows.shape
        or fps_values.shape != rows.shape
        or not np.isfinite(fps_values).all()
        or np.any(fps_values <= 0)
        or not np.allclose(fps_values, fps_values[0], rtol=0.0, atol=1e-6)
    ):
        raise ValueError("T3 source-video frame/FPS catalog data is invalid")
    return frames, float(fps_values[0])


def _normalize_a0(
    query: RT2BenchmarkQuery,
    paths: tuple[Any, ...],
    scores: np.ndarray,
) -> tuple[DiverseTemporalPath, ...]:
    return tuple(
        DiverseTemporalPath(
            score=path.score,
            positions=path.positions,
            region_ids=tuple(
                f"{event.event_id}:A0:P{position:06d}"
                for event, position in zip(query.events, path.positions, strict=True)
            ),
            event_scores=tuple(
                float(scores[index, position]) for index, position in enumerate(path.positions)
            ),
        )
        for path in paths
    )


def _path_payload(path: DiverseTemporalPath, best_score: float, rank: int) -> dict[str, Any]:
    return {
        "path_rank": rank,
        "path_score": path.score,
        "relative_score_gap": relative_score_gap(best_score, path.score),
        "catalog_positions": list(path.positions),
        "event_region_ids": list(path.region_ids),
        "event_scores": list(path.event_scores),
        "strictly_monotonic": all(
            left < right
            for left, right in zip(path.positions[:-1], path.positions[1:], strict=True)
        ),
    }


def _evaluate_arm(
    query: RT2BenchmarkQuery,
    paths: tuple[DiverseTemporalPath, ...],
    original_frames: np.ndarray,
    fps: float,
    *,
    split: str,
    arm: str,
    delta: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_rows = []
    for event_index, event in enumerate(query.events):
        by_k = {}
        for k in K_VALUES:
            selected = paths[:k]
            positions = [path.positions[event_index] for path in selected]
            errors = [
                abs(int(original_frames[position]) - event.reference_original_frame_idx) / fps
                for position in positions
            ]
            minimum = min(errors) if errors else None
            by_k[str(k)] = {
                "hypothesis_count": len(selected),
                "anchor_positions": list(positions),
                "unique_anchor_count": len(set(positions)),
                "minimum_temporal_error_seconds": minimum,
                "reachable_seconds": {
                    str(tolerance): bool(minimum is not None and minimum <= tolerance)
                    for tolerance in TOLERANCES
                },
            }
        event_rows.append(
            {
                "query_id": query.query_id,
                "event_id": event.event_id,
                "event_count": len(query.events),
                "difficulty_tags": list(query.difficulty_tags),
                "split": split,
                "arm": arm,
                "delta": delta,
                "by_k": by_k,
            }
        )
    by_tolerance = {}
    for tolerance in TOLERANCES:
        union_reachable = all(
            row["by_k"]["5"]["reachable_seconds"][str(tolerance)] for row in event_rows
        )
        single_path_reachable = any(
            all(
                abs(
                    int(original_frames[path.positions[index]])
                    - event.reference_original_frame_idx
                )
                / fps
                <= tolerance
                for index, event in enumerate(query.events)
            )
            for path in paths[:5]
        )
        by_tolerance[str(tolerance)] = {
            "QUERY_ALL_EVENTS_REACHABLE@5": union_reachable,
            "SINGLE_PATH_ALL_EVENTS_REACHABLE@5": single_path_reachable,
        }
    diversities = [row["by_k"]["5"]["unique_anchor_count"] for row in event_rows]
    best_score = paths[0].score if paths else None
    gaps = [relative_score_gap(best_score, path.score) for path in paths] if paths else []
    query_row = {
        "query_id": query.query_id,
        "event_count": len(query.events),
        "difficulty_tags": list(query.difficulty_tags),
        "split": split,
        "arm": arm,
        "delta": delta,
        "path_count": len(paths),
        "unique_path_count": len({path.positions for path in paths}),
        "by_tolerance_seconds": by_tolerance,
        "k5_anchor_diversity_per_event": diversities,
        "best_path_score": best_score,
        "selected_path_scores": [path.score for path in paths],
        "selected_path_relative_score_gaps": gaps,
        "mean_selected_path_relative_score_gap": mean(gaps) if gaps else None,
        "max_selected_path_relative_score_gap": max(gaps) if gaps else None,
    }
    return query_row, event_rows


def _aggregate_arm(
    query_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]], *, slices: bool = False
) -> dict[str, Any]:
    if not query_rows or not event_rows:
        return {"query_count": len(query_rows), "event_count": len(event_rows), "status": "EMPTY"}
    sweep = {}
    for tolerance in TOLERANCES:
        sweep[str(tolerance)] = {
            **{
                f"EVENT_WINDOW_RECALL@{k}": mean(
                    row["by_k"][str(k)]["reachable_seconds"][str(tolerance)]
                    for row in event_rows
                )
                for k in K_VALUES
            },
            "QUERY_ALL_EVENTS_REACHABLE@5": mean(
                row["by_tolerance_seconds"][str(tolerance)][
                    "QUERY_ALL_EVENTS_REACHABLE@5"
                ]
                for row in query_rows
            ),
            "SINGLE_PATH_ALL_EVENTS_REACHABLE@5": mean(
                row["by_tolerance_seconds"][str(tolerance)][
                    "SINGLE_PATH_ALL_EVENTS_REACHABLE@5"
                ]
                for row in query_rows
            ),
        }
    diversity_values = [
        value for row in query_rows for value in row["k5_anchor_diversity_per_event"]
    ]
    query_diversity = [mean(row["k5_anchor_diversity_per_event"]) for row in query_rows]
    errors = [
        row["by_k"]["5"]["minimum_temporal_error_seconds"] for row in event_rows
    ]
    valid_errors = [value for value in errors if value is not None]
    gaps = [
        gap for row in query_rows for gap in row["selected_path_relative_score_gaps"]
    ]
    output = {
        "query_count": len(query_rows),
        "event_count": len(event_rows),
        "PRIMARY_6_SECONDS": sweep["6"],
        "TOLERANCE_SWEEP_SECONDS": sweep,
        "K5_DIVERSITY": {
            "mean_unique_path_count": mean(row["unique_path_count"] for row in query_rows),
            "EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY": mean(diversity_values),
            "QUERY_WEIGHTED_MEAN_ANCHOR_DIVERSITY": mean(query_diversity),
            "event_unique_anchor_distribution": {
                str(unique_count): Counter(diversity_values)[unique_count]
                for unique_count in range(1, FINAL_PATH_LIMIT + 1)
            },
            "fraction_events_with_one_anchor": mean(value == 1 for value in diversity_values),
        },
        "MINIMUM_TEMPORAL_ERROR_SECONDS_AT_5": {
            "mean": mean(valid_errors) if valid_errors else None,
            "median": median(valid_errors) if valid_errors else None,
        },
        "PATH_SCORE_TRADEOFF": {
            "mean_selected_path_relative_score_gap": mean(gaps) if gaps else None,
            "max_selected_path_relative_score_gap": max(gaps) if gaps else None,
        },
    }
    if slices:
        slice_output = {"BY_EVENT_COUNT": {}, "BY_DIFFICULTY_TAG": {}}
        for count in (2, 3, 4):
            selected_queries = [row for row in query_rows if row["event_count"] == count]
            ids = {row["query_id"] for row in selected_queries}
            selected_events = [row for row in event_rows if row["query_id"] in ids]
            value = _aggregate_arm(selected_queries, selected_events)
            value["small_slice_warning"] = len(selected_queries) < 5
            slice_output["BY_EVENT_COUNT"][str(count)] = value
        tags = sorted({tag for row in query_rows for tag in row["difficulty_tags"]})
        for tag in tags:
            selected_queries = [row for row in query_rows if tag in row["difficulty_tags"]]
            ids = {row["query_id"] for row in selected_queries}
            selected_events = [row for row in event_rows if row["query_id"] in ids]
            value = _aggregate_arm(selected_queries, selected_events)
            value["small_slice_warning"] = len(selected_queries) < 5
            slice_output["BY_DIFFICULTY_TAG"][tag] = value
        output["SLICES"] = slice_output
    return output


def _select_delta(dev_sweep: list[dict[str, Any]]) -> float:
    winner = max(
        dev_sweep,
        key=lambda row: (
            row["EVENT_WINDOW_RECALL@5"],
            row["SINGLE_PATH_ALL_EVENTS_REACHABLE@5"],
            row["EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY@5"],
            -row["mean_selected_path_relative_score_gap"],
            -row["delta"],
        ),
    )
    return float(winner["delta"])


def _compare_events(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]]
) -> dict[str, int]:
    baseline_by_id = {
        (row["query_id"], row["event_id"]): row["by_k"]["5"]["reachable_seconds"]["6"]
        for row in baseline
    }
    treatment_by_id = {
        (row["query_id"], row["event_id"]): row["by_k"]["5"]["reachable_seconds"]["6"]
        for row in treatment
    }
    if set(baseline_by_id) != set(treatment_by_id):
        raise RuntimeError("T3 arm event identities do not align")
    return {
        "A0_K5_FAILURES_RECOVERED": sum(
            not baseline_by_id[key] and treatment_by_id[key] for key in baseline_by_id
        ),
        "NEW_REGRESSIONS_VS_A0": sum(
            baseline_by_id[key] and not treatment_by_id[key] for key in baseline_by_id
        ),
    }


def _pool_metrics(pool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(tolerance): {
            "RAW_EVENT_TOP10_WINDOW_RECALL": mean(
                row["pool_ceiling_seconds"][str(tolerance)]["raw_top10_reachable"]
                for row in pool_rows
            ),
            "DIVERSE_POOL_L10_WINDOW_RECALL": mean(
                row["pool_ceiling_seconds"][str(tolerance)]["diverse_pool_reachable"]
                for row in pool_rows
            ),
        }
        for tolerance in TOLERANCES
    }


def validate_a0_reproduction(event_rows: list[dict[str, Any]]) -> int:
    recovered = sum(
        row["by_k"]["5"]["reachable_seconds"]["6"] for row in event_rows
    )
    if len(event_rows) != 74 or recovered != 54:
        raise RuntimeError(f"T3_A0_REPRODUCTION_FAILED: {recovered}/{len(event_rows)}")
    return recovered


def _process_query(
    query: RT2BenchmarkQuery,
    split: str,
    runtime: Any,
    group: Any,
    deltas: tuple[float, ...],
) -> dict[str, Any]:
    started = monotonic()
    requests = [
        QueryRequest(f"{query.query_id}__{event.event_id}", event.text, query.language, 1)
        for event in query.events
    ]
    encoding_started = monotonic()
    encoded = runtime.encode_requests(requests)
    encoding_ms = (monotonic() - encoding_started) * 1000
    scoring_started = monotonic()
    full_scores = np.asarray(runtime.backend.score_many_all(encoded.embeddings), dtype=np.float32)
    score_matrix_ms = (monotonic() - scoring_started) * 1000
    if full_scores.shape != (len(query.events), runtime.backend.size) or not np.isfinite(
        full_scores
    ).all():
        raise RuntimeError(f"T3 score matrix shape is invalid: {query.query_id}")
    scores = full_scores[:, group.rows]
    frames, fps = _source_arrays(runtime.catalog, group.rows)

    pool_started = monotonic()
    pools = tuple(
        build_diverse_event_pool(event.event_id, scores[index], frames, fps)
        for index, event in enumerate(query.events)
    )
    pool_ms = (monotonic() - pool_started) * 1000
    pool_rows = []
    for event_index, (event, pool) in enumerate(zip(query.events, pools, strict=True)):
        ranking = stable_event_ranking(scores[event_index])
        ceilings = {}
        for tolerance in TOLERANCES:
            mask = reference_neighborhood_mask(
                frames, event.reference_original_frame_idx, fps, tolerance
            )
            ceilings[str(tolerance)] = {
                "raw_top10_reachable": any(mask[int(position)] for position in ranking[:10]),
                "diverse_pool_reachable": any(mask[item.catalog_position] for item in pool),
            }
        primary_mask = reference_neighborhood_mask(
            frames, event.reference_original_frame_idx, fps, 6
        )
        pool_rows.append(
            {
                "query_id": query.query_id,
                "event_id": event.event_id,
                "split": split,
                "source_video_id": query.source_video_id,
                "fps": fps,
                "pool_limit": POOL_LIMIT,
                "region_radius_seconds": REGION_RADIUS_SECONDS,
                "retained_count": len(pool),
                "retained_candidates": [
                    {
                        "event_region_id": item.event_region_id,
                        "catalog_position": item.catalog_position,
                        "original_frame_idx": item.original_frame_idx,
                        "similarity": item.similarity,
                    }
                    for item in pool
                ],
                "pool_ceiling_seconds": ceilings,
                **{
                    f"raw_top{cutoff}_reachable_6s": any(
                        primary_mask[int(position)] for position in ranking[:cutoff]
                    )
                    for cutoff in (5, 10, 20)
                },
            }
        )

    enumeration_started = monotonic()
    feasible, raw_count = enumerate_feasible_paths(pools)
    enumeration_ms = (monotonic() - enumeration_started) * 1000
    a0 = _normalize_a0(query, k_best_monotonic_paths(scores, 5), scores)
    a1_started = monotonic()
    a1 = select_score_top_k(feasible)
    a1_ms = (monotonic() - a1_started) * 1000
    a2_started = monotonic()
    a2 = {delta: select_coverage_aware(feasible, delta) for delta in deltas}
    a2_ms = (monotonic() - a2_started) * 1000

    metrics_started = monotonic()
    arm_rows: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    arm_rows["A0"] = _evaluate_arm(
        query, a0, frames, fps, split=split, arm="A0_T2_KBEST_BASELINE", delta=None
    )
    arm_rows["A1"] = _evaluate_arm(
        query, a1, frames, fps, split=split, arm="A1_DIVERSE_EVENT_POOL_SCORE_ONLY", delta=None
    )
    for delta, paths in a2.items():
        arm_rows[f"A2_{delta}"] = _evaluate_arm(
            query,
            paths,
            frames,
            fps,
            split=split,
            arm="A2_COVERAGE_AWARE_DIVERSE_SELECTION",
            delta=delta,
        )
    metrics_ms = (monotonic() - metrics_started) * 1000
    best_feasible = feasible[0].score if feasible else 0.0
    hypothesis_row = {
        "query_id": query.query_id,
        "split": split,
        "source_video_id": query.source_video_id,
        "event_count": len(query.events),
        "difficulty_tags": list(query.difficulty_tags),
        "raw_combination_count": raw_count,
        "feasible_monotonic_path_count": len(feasible),
        "A0": [_path_payload(path, a0[0].score, rank) for rank, path in enumerate(a0, 1)],
        "A1": [
            _path_payload(path, best_feasible, rank) for rank, path in enumerate(a1, 1)
        ],
        "A2": {
            str(delta): [
                _path_payload(path, best_feasible, rank)
                for rank, path in enumerate(paths, 1)
            ]
            for delta, paths in a2.items()
        },
    }
    return {
        "pool_rows": pool_rows,
        "hypothesis_row": hypothesis_row,
        "arm_rows": arm_rows,
        "timing": {
            "query_id": query.query_id,
            "split": split,
            "event_encoding_ms": encoding_ms,
            "score_matrix_ms": score_matrix_ms,
            "candidate_pool_construction_ms": pool_ms,
            "path_enumeration_ms": enumeration_ms,
            "a1_selection_ms": a1_ms,
            "a2_selection_ms": a2_ms,
            "metrics_ms": metrics_ms,
            "total_ms": (monotonic() - started) * 1000,
            "score_matrix_computations": 1,
            "raw_combination_count": raw_count,
            "feasible_monotonic_path_count": len(feasible),
        },
    }


def _report(summary: dict[str, Any], holdout: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# T3 Coverage-Aware Diverse Temporal Hypotheses",
            "",
            "Known-source-video internal pseudo-GT experiment; not competition recall.",
            f"DEV-selected delta: `{summary['selected_delta']}`",
            f"A0 HOLDOUT ±6s: `{holdout['A0']['PRIMARY_6_SECONDS']}`",
            f"A1 HOLDOUT ±6s: `{holdout['A1']['PRIMARY_6_SECONDS']}`",
            f"A2 HOLDOUT ±6s: `{holdout['A2']['PRIMARY_6_SECONDS']}`",
            "Quality decision: `NOT_EVALUATED`",
            "",
        ]
    )


def run_t3(
    config: T3RunnerConfig,
    queries: list[RT2BenchmarkQuery],
    *,
    runtime: OperationalRetrievalRuntime | None = None,
) -> dict[str, Any]:
    preflight = preflight_t3(config)
    frozen = load_rt2_benchmark(config.benchmark_path)
    if [query.as_dict() for query in queries] != [query.as_dict() for query in frozen]:
        raise ValueError("T3 queries must exactly match the frozen benchmark")
    output = config.output_root.expanduser().resolve(strict=False)
    output.mkdir(parents=True)
    active_runtime = runtime or OperationalRetrievalRuntime(
        replace(config.stage2, output_root=output / "_stage2_control")
    )
    owns_runtime = runtime is None
    issues: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    started = monotonic()
    try:
        active_runtime.load()
        if active_runtime.preflight.get("stage1_index_fingerprint") != EXPECTED_STAGE1_FINGERPRINT:
            raise RuntimeError("T3_RUNTIME_STAGE1_FINGERPRINT_MISMATCH")
        resolve_benchmark_identities(queries, active_runtime.catalog)
        groups = {item.video_id: item for item in build_video_row_groups(active_runtime.catalog)}
        dev, holdout = split_dev_holdout(queries, config.seed)

        dev_results = []
        for query in dev:
            result = _process_query(
                query, "DEV", active_runtime, groups[query.source_video_id], DELTA_GRID
            )
            dev_results.append(result)
            pool_rows.extend(result["pool_rows"])
            hypothesis_rows.append(result["hypothesis_row"])
            timings.append(result["timing"])

        dev_sweep = []
        for delta in DELTA_GRID:
            key = f"A2_{delta}"
            delta_query_rows = [result["arm_rows"][key][0] for result in dev_results]
            delta_event_rows = [
                row for result in dev_results for row in result["arm_rows"][key][1]
            ]
            value = _aggregate_arm(delta_query_rows, delta_event_rows)
            dev_sweep.append(
                {
                    "delta": delta,
                    "EVENT_WINDOW_RECALL@5": value["PRIMARY_6_SECONDS"][
                        "EVENT_WINDOW_RECALL@5"
                    ],
                    "SINGLE_PATH_ALL_EVENTS_REACHABLE@5": value["PRIMARY_6_SECONDS"][
                        "SINGLE_PATH_ALL_EVENTS_REACHABLE@5"
                    ],
                    "QUERY_ALL_EVENTS_REACHABLE@5": value["PRIMARY_6_SECONDS"][
                        "QUERY_ALL_EVENTS_REACHABLE@5"
                    ],
                    "EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY@5": value["K5_DIVERSITY"][
                        "EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY"
                    ],
                    "mean_selected_path_relative_score_gap": value["PATH_SCORE_TRADEOFF"][
                        "mean_selected_path_relative_score_gap"
                    ],
                }
            )
        selected_delta = _select_delta(dev_sweep)

        holdout_results = []
        for query in holdout:
            result = _process_query(
                query,
                "HOLDOUT",
                active_runtime,
                groups[query.source_video_id],
                (selected_delta,),
            )
            holdout_results.append(result)
            pool_rows.extend(result["pool_rows"])
            hypothesis_rows.append(result["hypothesis_row"])
            timings.append(result["timing"])

        for result in dev_results:
            for key in ("A0", "A1", *[f"A2_{delta}" for delta in DELTA_GRID]):
                query_row, rows = result["arm_rows"][key]
                query_rows.append(query_row)
                event_rows.extend(rows)
        for result in holdout_results:
            for key in ("A0", "A1", f"A2_{selected_delta}"):
                query_row, rows = result["arm_rows"][key]
                query_rows.append(query_row)
                event_rows.extend(rows)

        def arm_data(results: list[dict[str, Any]], key: str):
            return (
                [result["arm_rows"][key][0] for result in results],
                [row for result in results for row in result["arm_rows"][key][1]],
            )

        holdout_a0_q, holdout_a0_e = arm_data(holdout_results, "A0")
        holdout_a1_q, holdout_a1_e = arm_data(holdout_results, "A1")
        holdout_a2_q, holdout_a2_e = arm_data(
            holdout_results, f"A2_{selected_delta}"
        )
        holdout_comparison = {
            "selected_delta_from_dev": selected_delta,
            "holdout_used_for_delta_selection": False,
            "A0": _aggregate_arm(holdout_a0_q, holdout_a0_e, slices=True),
            "A1": _aggregate_arm(holdout_a1_q, holdout_a1_e, slices=True),
            "A2": _aggregate_arm(holdout_a2_q, holdout_a2_e, slices=True),
            "A1_VS_A0_EVENT_CHANGES": _compare_events(holdout_a0_e, holdout_a1_e),
            "A2_VS_A0_EVENT_CHANGES": _compare_events(holdout_a0_e, holdout_a2_e),
        }

        full_a0_events = [
            row
            for row in event_rows
            if row["arm"] == "A0_T2_KBEST_BASELINE"
        ]
        a0_reproduced = validate_a0_reproduction(full_a0_events)
        pool_coverage = _pool_metrics(pool_rows)
        headroom_counts = {
            str(cutoff): sum(row[f"raw_top{cutoff}_reachable_6s"] for row in pool_rows)
            for cutoff in (5, 10, 20)
        }
        expected_headroom = ((5, 64), (10, 68), (20, 71))
        if any(
            abs(headroom_counts[str(k)] - expected) > 3
            for k, expected in expected_headroom
        ):
            raise RuntimeError(f"T3_EVENT_ONLY_CEILING_SUBSTANTIAL_MISMATCH: {headroom_counts}")
        if any(headroom_counts[str(k)] != expected for k, expected in expected_headroom):
            issues.append(
                {
                    "severity": "WARNING",
                    "code": "T3_EVENT_ONLY_CEILING_MINOR_DIFFERENCE",
                    "evidence": {"actual": headroom_counts, "expected_context": expected_headroom},
                }
            )

        metrics = {
            "benchmark_type": "AI_CURATED_INTERNAL_PSEUDO_GT",
            "known_source_video_only": True,
            "DEV_DELTA_SWEEP": dev_sweep,
            "selected_delta": selected_delta,
            "POOL_COVERAGE": pool_coverage,
            "EVENT_ONLY_REFERENCE_HEADROOM_6S_COUNTS": headroom_counts,
            "FULL_SET_A0_K5_REPRODUCTION": {"recovered": a0_reproduced, "total": 74},
            "HOLDOUT_COMPARISON": holdout_comparison,
        }
        summary = {
            "experiment": "T3",
            "T3_IMPLEMENTATION_STATUS": "COMPLETE",
            "T3_REAL_STATUS": "COMPLETE",
            "T3_DIAGNOSTIC_STATUS": "COMPLETE",
            "T3_QUALITY_DECISION": "NOT_EVALUATED",
            "selected_delta": selected_delta,
            "dev_query_count": len(dev),
            "holdout_query_count": len(holdout),
            "a0_full_set_recovered_6s": a0_reproduced,
        }
        manifest = {
            **summary,
            "t3_version": T3_VERSION,
            "completed_at": datetime.now(UTC).isoformat(),
            "build_git_commit": config.stage2.build_git_commit,
            "preflight": preflight,
            "arms": [
                "A0_T2_KBEST_BASELINE",
                "A1_DIVERSE_EVENT_POOL_SCORE_ONLY",
                "A2_COVERAGE_AWARE_DIVERSE_SELECTION",
            ],
            "fixed_settings": {
                "pool_limit": POOL_LIMIT,
                "region_radius_seconds": REGION_RADIUS_SECONDS,
                "final_k": FINAL_PATH_LIMIT,
                "distance_lambda": 0.0,
            },
            "delta_grid": list(DELTA_GRID),
            "delta_selection_split": "DEV_ONLY",
            "holdout_used_for_selection": False,
            "exact_enumeration": True,
            "internal_beam": False,
            "ground_truth_used_in_runtime_generation": False,
            "score_matrix_computations_per_query": 1,
            "raw_video_decoding": False,
            "new_model": False,
            "network_required": False,
            "timings": {
                "queries": timings,
                "experiment_total_ms": (monotonic() - started) * 1000,
            },
        }
        write_json(output / "t3_summary.json", summary)
        write_json(output / "t3_metrics.json", metrics)
        write_json(
            output / "dev_delta_sweep.json",
            {"rows": dev_sweep, "selected_delta": selected_delta},
        )
        write_jsonl(output / "event_candidate_pools.jsonl", pool_rows)
        write_jsonl(output / "query_hypotheses.jsonl", hypothesis_rows)
        write_jsonl(output / "event_reachability.jsonl", event_rows)
        write_json(output / "holdout_comparison.json", holdout_comparison)
        write_json(output / "run_manifest.json", manifest)
        write_jsonl(output / "issues.jsonl", issues)
        (output / "t3_report.md").write_text(
            _report(summary, holdout_comparison), encoding="utf-8"
        )
        return {"summary": summary, "metrics": metrics, "holdout": holdout_comparison}
    finally:
        if owns_runtime:
            active_runtime.close()


def create_t3_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    source = Path(output_root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("T3 ZIP must be outside output root")
    members = [source / name for name in BUNDLE_MEMBERS]
    missing = [
        name for name, path in zip(BUNDLE_MEMBERS, members, strict=True) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing T3 bundle artifacts: {missing}")
    if any(path.suffix.lower() in HEAVY_SUFFIXES for path in members):
        raise ValueError("T3 bundle contains a forbidden heavy artifact")
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
    "DELTA_GRID",
    "T3RunnerConfig",
    "T3_VERSION",
    "create_t3_bundle",
    "preflight_t3",
    "run_t3",
    "validate_a0_reproduction",
]
