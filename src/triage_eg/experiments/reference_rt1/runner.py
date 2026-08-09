"""Bounded runner and report packaging for reference experiment RT1."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
)

from .contracts import (
    IMPLEMENTATION_TYPE,
    LAMBDA_SOURCE,
    REFERENCE_ALGORITHM,
    RT1_ARMS,
    RT1_VERSION,
    RT1Query,
    RT1Settings,
)
from .scoring import (
    build_video_row_groups,
    rank_dante_dp,
    rank_unordered_event_max,
    top_k_video_overlap,
)
from .visuals import render_rt1_visuals

OVERLAP_CUTOFFS = (1, 5, 10, 20)
FORBIDDEN_BUNDLE_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".mp4"}


@dataclass(frozen=True)
class RT1RunnerConfig:
    stage2: Stage2RuntimeConfig
    dataset_root: Path
    query_suite_path: Path
    output_root: Path
    settings: RT1Settings


def _query_diagnostics(
    whole: list[dict[str, Any]],
    unordered: list[dict[str, Any]],
    dante: list[dict[str, Any]],
    *,
    event_encoding: list[dict[str, Any]],
    performance_ms: dict[str, float],
) -> dict[str, Any]:
    top_unordered = unordered[:20]
    monotonic_count = sum(
        bool(item["independent_argmax_order_is_monotonic"]) for item in top_unordered
    )
    return {
        "quality_status": "EXPLORATORY_NO_GT",
        "structural_metrics_only": True,
        "top_k_video_overlap": {
            "whole_vs_unordered": top_k_video_overlap(whole, unordered, OVERLAP_CUTOFFS),
            "whole_vs_dante": top_k_video_overlap(whole, dante, OVERLAP_CUTOFFS),
            "unordered_vs_dante": top_k_video_overlap(unordered, dante, OVERLAP_CUTOFFS),
        },
        "unordered_top_video_score": (unordered[0]["unordered_score"] if unordered else None),
        "dante_top_video_score": dante[0]["dante_score"] if dante else None,
        "dante_top_chain_span": dante[0]["span_in_keyframes"] if dante else None,
        "top20_unordered_independent_argmax_monotonic": {
            "count": monotonic_count,
            "evaluated": len(top_unordered),
            "percentage": (100.0 * monotonic_count / len(top_unordered) if top_unordered else None),
        },
        "event_encoding": event_encoding,
        "performance_ms": performance_ms,
        "automatic_winner": None,
    }


def _run_query(
    runtime: OperationalRetrievalRuntime,
    config: RT1RunnerConfig,
    query: RT1Query,
    groups: list[Any],
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    started = monotonic()
    query_root = config.output_root / "queries" / query.query_id
    query_root.mkdir(parents=True, exist_ok=False)
    write_json(query_root / "query_spec.json", query.as_dict())
    event_requests = [
        QueryRequest(
            f"{query.query_id}__{event.event_id}",
            event.text,
            query.language,
            1,
        )
        for event in query.events
    ]
    preparation_ms = (monotonic() - started) * 1000

    whole_started = monotonic()
    whole_result = runtime.search_one(
        QueryRequest(
            f"{query.query_id}__whole",
            query.narrative_text,
            query.language,
            config.settings.control_top_k,
        )
    )
    whole_ms = (monotonic() - whole_started) * 1000
    write_jsonl(query_root / "whole_query/ranked_frames.jsonl", whole_result.ranked_frames)
    write_jsonl(query_root / "whole_query/ranked_videos.jsonl", whole_result.ranked_videos)

    encoding_started = monotonic()
    encoded = runtime.encode_requests(event_requests)
    event_encoding_ms = (monotonic() - encoding_started) * 1000
    event_encoding = [
        {
            "event_id": event.event_id,
            "original_event_text": event.text,
            "language_resolution": encoded.resolutions[index].as_dict(),
            **encoded.encodings[index],
        }
        for index, event in enumerate(query.events)
    ]

    scoring_started = monotonic()
    full_scores = np.asarray(runtime.backend.score_many_all(encoded.embeddings), dtype=np.float32)
    full_score_ms = (monotonic() - scoring_started) * 1000
    if full_scores.shape != (len(query.events), runtime.backend.size):
        raise RuntimeError("RT1 full score matrix has an invalid shape")
    if not np.isfinite(full_scores).all():
        raise RuntimeError("RT1 full score matrix contains non-finite values")

    unordered_started = monotonic()
    event_ids = [event.event_id for event in query.events]
    unordered = rank_unordered_event_max(full_scores, event_ids, groups, runtime.catalog)
    unordered_ms = (monotonic() - unordered_started) * 1000
    write_jsonl(query_root / "unordered_event_max/ranked_videos.jsonl", unordered)
    write_jsonl(
        query_root / "unordered_event_max/top_event_anchors.jsonl",
        unordered[: config.settings.chain_export_top_k],
    )

    dante_started = monotonic()
    dante = rank_dante_dp(
        full_scores,
        event_ids,
        groups,
        runtime.catalog,
        distance_lambda=config.settings.dante_lambda,
    )
    dante_ms = (monotonic() - dante_started) * 1000
    write_jsonl(query_root / "dante_dp/ranked_videos.jsonl", dante)
    write_jsonl(
        query_root / "dante_dp/top_chains.jsonl",
        dante[: config.settings.chain_export_top_k],
    )

    visual_started = monotonic()
    _, review_mapping, issues = render_rt1_visuals(
        config.output_root,
        dataset_root=config.dataset_root,
        query_id=query.query_id,
        whole_frames=whole_result.ranked_frames,
        unordered=unordered,
        dante=dante,
        top_k=config.settings.visual_top_k,
        review_seed=config.settings.review_seed,
    )
    visual_ms = (monotonic() - visual_started) * 1000
    performance = {
        "query_preparation_ms": preparation_ms,
        "whole_query_control_ms": whole_ms,
        "event_encoding_ms": event_encoding_ms,
        "full_score_matrix_ms": full_score_ms,
        "unordered_aggregation_ms": unordered_ms,
        "dante_dp_ms": dante_ms,
        "visual_rendering_ms": visual_ms,
        "total_ms": (monotonic() - started) * 1000,
    }
    diagnostics = _query_diagnostics(
        whole_result.ranked_videos,
        unordered,
        dante,
        event_encoding=event_encoding,
        performance_ms=performance,
    )
    write_json(query_root / "comparison/diagnostics.json", diagnostics)
    return diagnostics, review_mapping, issues


def run_reference_rt1(
    config: RT1RunnerConfig,
    queries: list[RT1Query],
    *,
    runtime: OperationalRetrievalRuntime | None = None,
) -> dict[str, Any]:
    output = config.output_root.expanduser().resolve(strict=False)
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        if runtime is None or existing - {"_stage2_control"}:
            raise FileExistsError(f"RT1 output already exists: {output}")
    else:
        output.mkdir(parents=True)
    query_suite_copy = output / "query_suite/reference_rt1_queries.jsonl"
    write_jsonl(query_suite_copy, [query.as_dict() for query in queries])
    stage2_config = replace(config.stage2, output_root=output / "_stage2_control")
    active_runtime = runtime or OperationalRetrievalRuntime(stage2_config)
    owns_runtime = runtime is None
    issues: list[dict[str, Any]] = []
    query_summaries = []
    review_queries = []
    try:
        active_runtime.load()
        groups = build_video_row_groups(active_runtime.catalog)
        catalog_diagnostics = {
            "video_count": len(groups),
            "row_count": int(active_runtime.backend.size),
            "source_contiguous_video_groups": sum(group.source_was_contiguous for group in groups),
            "canonical_temporal_order": "N_ASCENDING_THEN_GLOBAL_ROW",
            "catalog_position_base": 0,
        }
        for query in queries:
            diagnostics, mapping, query_issues = _run_query(
                active_runtime,
                replace(config, dataset_root=dataset, output_root=output),
                query,
                groups,
            )
            query_summaries.append(
                {
                    "query_id": query.query_id,
                    "source": query.source,
                    "event_count": len(query.events),
                    "quality_status": "EXPLORATORY_NO_GT",
                    "performance_ms": diagnostics["performance_ms"],
                    "top_k_video_overlap": diagnostics["top_k_video_overlap"],
                }
            )
            review_queries.append({"query_id": query.query_id, **mapping})
            issues.extend(query_issues)
        write_json(
            output / "visuals/review_key.json",
            {
                "seed": config.settings.review_seed,
                "blinded_methods": ["METHOD_A", "METHOD_B"],
                "queries": review_queries,
            },
        )
        write_jsonl(output / "issues.jsonl", issues)
        manifest = {
            "reference_experiment": "RT1",
            "rt1_version": RT1_VERSION,
            "status": "COMPLETE",
            "completed_at": datetime.now(UTC).isoformat(),
            "build_git_commit": config.stage2.build_git_commit,
            "arms": list(RT1_ARMS),
            "reference_algorithm": REFERENCE_ALGORITHM,
            "implementation_type": IMPLEMENTATION_TYPE,
            "dante": {
                "lambda": config.settings.dante_lambda,
                "lambda_source": LAMBDA_SOURCE,
                "strict_order": True,
                "complexity": "O(EVENTS_TIMES_CORPUS_ROWS)",
            },
            "stage1_index_fingerprint": active_runtime.preflight["stage1_index_fingerprint"],
            "stage2a_control_manifest": active_runtime.runtime_manifest(),
            "catalog": catalog_diagnostics,
            "queries_completed": len(query_summaries),
            "ground_truth_available": False,
            "quality_decision": "EXPLORATORY_NO_GT",
            "network_required": False,
            "no_stage1_rebuild": True,
            "no_model_change": True,
            "forbidden_modules_used": False,
        }
        write_json(output / "run_manifest.json", manifest)
        summary = {
            "reference_experiment": "RT1",
            "status": "COMPLETE",
            "real_experiment_status": "COMPLETE",
            "quality_decision": "EXPLORATORY_NO_GT",
            "queries": query_summaries,
            "issues": {
                "total": len(issues),
                "by_code": {
                    code: sum(item.get("code") == code for item in issues)
                    for code in sorted({str(item.get("code")) for item in issues})
                },
            },
            "non_claims": [
                "No ground truth was available",
                "Structural overlap is not semantic quality",
                "RT1 does not select KEEP or DROP automatically",
                "This is not an exact reproduction of the complete DANTE system",
            ],
        }
        write_json(output / "experiment_summary.json", summary)
        return summary
    finally:
        if owns_runtime:
            active_runtime.close()


def create_rt1_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("RT1 ZIP must be outside the experiment output root")
    required = (
        "run_manifest.json",
        "experiment_summary.json",
        "query_suite/reference_rt1_queries.jsonl",
        "visuals/review_key.json",
        "issues.jsonl",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RT1 bundle artifacts: {missing}")
    members = [source / name for name in required]
    members.extend(path for path in (source / "queries").rglob("*") if path.is_file())
    members.extend(path for path in (source / "visuals").rglob("*.jpg") if path.is_file())
    relative = [path.relative_to(source).as_posix() for path in members]
    if any(
        Path(name).suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES
        or name.startswith(("_stage2_control/", "cache/", "logs/"))
        for name in relative
    ):
        raise ValueError("RT1 bundle contains a forbidden artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    if staging.exists():
        staging.unlink()
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path, name in sorted(zip(members, relative, strict=True), key=lambda x: x[1]):
                archive.write(path, arcname=name)
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


__all__ = ["RT1RunnerConfig", "create_rt1_bundle", "run_reference_rt1"]
