"""Offline, target-aware stage analysis for L21-150 QA and TRAKE runs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .l21_150_schema import (
    L21150Benchmark,
    L21150QAQuery,
    L21150TRAKEQuery,
)


class L21150StageAnalysisError(ValueError):
    """A run artifact cannot be interpreted without guessing."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise L21150StageAnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise L21150StageAnalysisError(f"{path} must not contain a UTF-8 BOM")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise L21150StageAnalysisError(f"cannot load {path}: {exc}") from exc
    if type(value) is not dict:
        raise L21150StageAnalysisError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise L21150StageAnalysisError(f"{path} must not contain a UTF-8 BOM")
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise L21150StageAnalysisError(f"{path} must be valid UTF-8") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as exc:
            raise L21150StageAnalysisError(
                f"{path}:{line_number} contains invalid JSON: {exc}"
            ) from exc
        if type(value) is not dict:
            raise L21150StageAnalysisError(
                f"{path}:{line_number} must contain a JSON object"
            )
        records.append(value)
    return records


def _request_directories(run_directory: Path, manifest_name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(run_directory.rglob(manifest_name), key=lambda item: item.as_posix()):
        payload = _load_json(path)
        query_id = payload.get("query_id")
        if type(query_id) is not str or not query_id:
            raise L21150StageAnalysisError(f"{path} has no valid query_id")
        if query_id in result:
            raise L21150StageAnalysisError(
                f"multiple {manifest_name} artifacts found for query {query_id}"
            )
        result[query_id] = path.parent
    return result


def _aggregate_predictions(run_directory: Path) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    path = run_directory / "predictions.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return grouped, False
    for record in _load_jsonl(path):
        query_id = record.get("query_id")
        if type(query_id) is str:
            grouped[query_id].append(record)
    return grouped, True


def _successful_query_ids(run_directory: Path) -> set[str]:
    path = run_directory / "experiment_manifest.json"
    if not path.is_file():
        return set()
    payload = _load_json(path)
    rows = payload.get("queries")
    if type(rows) is not list:
        return set()
    return {
        row["query_id"]
        for row in rows
        if type(row) is dict
        and type(row.get("query_id")) is str
        and row.get("status") == "SUCCESS"
    }


def _stage_summary(
    selected: Sequence[Any],
    observations: Mapping[str, tuple[bool, bool, bool]],
    *,
    target_applicable: bool = True,
) -> dict[str, Any]:
    selected_ids = [query.query_id for query in selected]
    available_ids = [
        query_id
        for query_id in selected_ids
        if observations.get(query_id, (False, False, False))[0]
    ]
    non_empty_ids = [
        query_id for query_id in available_ids if observations[query_id][1]
    ]
    hit_ids = [
        query_id for query_id in available_ids if observations[query_id][2]
    ]
    if not available_ids:
        status = "UNAVAILABLE"
    elif len(available_ids) == len(selected_ids):
        status = "AVAILABLE"
    else:
        status = "PARTIALLY_AVAILABLE"
    all_selected_recall = None
    non_empty_recall = None
    if target_applicable and len(available_ids) == len(selected_ids) and selected_ids:
        all_selected_recall = len(hit_ids) / len(selected_ids)
    if target_applicable and non_empty_ids:
        non_empty_recall = len(hit_ids) / len(non_empty_ids)
    return {
        "status": status,
        "selected_query_count": len(selected_ids),
        "stage_available_query_count": len(available_ids),
        "stage_non_empty_query_count": len(non_empty_ids),
        "target_video_hit_query_count": len(hit_ids) if target_applicable else None,
        "target_video_recall_among_all_selected_queries": all_selected_recall,
        "target_video_recall_among_non_empty_stage_outputs": non_empty_recall,
        "target_video_hit_query_ids": hit_ids if target_applicable else None,
    }


def _candidate_observation(
    records: Any,
    *,
    target_video_id: str,
) -> tuple[bool, bool, bool]:
    if type(records) is not list:
        return False, False, False
    video_ids = [
        record.get("video_id")
        for record in records
        if type(record) is dict and type(record.get("video_id")) is str
    ]
    return True, bool(video_ids), target_video_id in video_ids


def _video_id_observation(
    video_ids: Any,
    *,
    target_video_id: str,
) -> tuple[bool, bool, bool]:
    if type(video_ids) is not list or any(type(item) is not str for item in video_ids):
        return False, False, False
    return True, bool(video_ids), target_video_id in video_ids


def _refinement_success_observation(
    records: Any,
    *,
    target_video_id: str,
) -> tuple[bool, bool, bool]:
    if type(records) is not list:
        return False, False, False
    successful = [
        record
        for record in records
        if type(record) is dict
        and record.get("status") == "REFINED"
        and type(record.get("refined_frame_id")) is int
        and record["refined_frame_id"] >= 0
    ]
    return _candidate_observation(successful, target_video_id=target_video_id)


def _final_observation(
    query_id: str,
    target_video_id: str,
    request_directory: Path | None,
    request_filename: str,
    aggregate: Mapping[str, list[dict[str, Any]]],
    aggregate_available: bool,
    successful_ids: set[str],
) -> tuple[bool, bool, bool]:
    if request_directory is not None:
        path = request_directory / request_filename
        if path.is_file():
            records = _load_jsonl(path)
            return _candidate_observation(records, target_video_id=target_video_id)
    if aggregate_available and query_id in successful_ids:
        return _candidate_observation(
            aggregate.get(query_id, []), target_video_id=target_video_id
        )
    return False, False, False


def compare_partial_chain_and_zero_output(
    partial_chain_query_ids: Iterable[str],
    zero_output_query_ids: Iterable[str],
) -> dict[str, Any]:
    """Compare identity sets; equal counts alone never establish equivalence."""

    partial = sorted(set(partial_chain_query_ids))
    zero = sorted(set(zero_output_query_ids))
    return {
        "status": "ESTABLISHED",
        "sets_equal": partial == zero,
        "partial_chain_query_ids": partial,
        "zero_output_query_ids": zero,
        "partial_chain_only_query_ids": sorted(set(partial) - set(zero)),
        "zero_output_only_query_ids": sorted(set(zero) - set(partial)),
    }


def _partial_chain_comparison(
    error_analysis_path: Path | None,
    final_observations: Mapping[str, tuple[bool, bool, bool]],
) -> dict[str, Any]:
    if error_analysis_path is None:
        return {
            "status": "NOT_ESTABLISHED",
            "reason": "no error-analysis artifact was supplied for query-id comparison",
        }
    if any(not observation[0] for observation in final_observations.values()):
        return {
            "status": "NOT_ESTABLISHED",
            "reason": "final output availability is incomplete",
        }
    payload = _load_json(error_analysis_path)
    rows = payload.get("query_errors")
    if type(rows) is not list:
        return {
            "status": "NOT_ESTABLISHED",
            "reason": "error-analysis artifact has no query_errors array",
        }
    partial = [
        row["query_id"]
        for row in rows
        if type(row) is dict
        and type(row.get("query_id")) is str
        and type(row.get("categories")) is list
        and "PARTIAL_CHAIN" in row["categories"]
    ]
    zero = [query_id for query_id, observation in final_observations.items() if not observation[1]]
    return compare_partial_chain_and_zero_output(partial, zero)


def analyze_l21_150_stages(
    benchmark: L21150Benchmark,
    run_directory: Path,
    *,
    split: str = "DEV",
    error_analysis_path: Path | None = None,
) -> dict[str, Any]:
    """Compare target-agnostic runtime traces to internal target video IDs offline."""

    if split not in {"DEV", "HOLDOUT"}:
        raise L21150StageAnalysisError("split must be DEV or HOLDOUT")
    run_dir = Path(run_directory)
    qa_queries = [
        query
        for query in benchmark.queries
        if isinstance(query, L21150QAQuery) and query.split == split
    ]
    trake_queries = [
        query
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == split
    ]
    qa_dirs = _request_directories(run_dir, "qa_request_manifest.json")
    trake_dirs = _request_directories(run_dir, "trake_request_manifest.json")
    aggregate, aggregate_available = _aggregate_predictions(run_dir)
    successful_ids = _successful_query_ids(run_dir)

    qa_stages: dict[str, dict[str, Any]] = {}
    qa_supported: dict[str, tuple[bool, bool, bool]] = {}
    qa_candidate_stages: dict[str, dict[str, tuple[bool, bool, bool]]] = {
        "RETRIEVAL_FUSED": {},
        "NOMINATED_VIDEO": {},
        "GROUNDING_CANDIDATE": {},
        "REFINEMENT_SELECTED": {},
        "REFINEMENT_SUCCESS": {},
        "KEYFRAME_EVIDENCE": {},
        "RAW_REFINED_EVIDENCE": {},
        "TEMPORAL_SEED": {},
        "TEMPORAL_REFINEMENT_SUCCESS": {},
        "TEMPORAL_EVIDENCE": {},
        "PROVIDER_EVIDENCE": {},
        "REFINED": {},
        "USABLE_EVIDENCE": {},
        "FINAL_OUTPUT": {},
    }
    qa_keys = {
        "RETRIEVAL_FUSED": "fused_retrieval_candidates",
        "GROUNDING_CANDIDATE": "grounding_candidates",
        "REFINEMENT_SELECTED": "refinement_selected_candidates",
        "KEYFRAME_EVIDENCE": "keyframe_evidence_candidates",
        "RAW_REFINED_EVIDENCE": "raw_refined_evidence_candidates",
        "TEMPORAL_SEED": "temporal_seed_candidates",
        "TEMPORAL_REFINEMENT_SUCCESS": "temporal_refined_evidence_candidates",
        "TEMPORAL_EVIDENCE": "temporal_evidence_candidates",
        "PROVIDER_EVIDENCE": "provider_evidence_candidates",
    }
    for query in qa_queries:
        request_dir = qa_dirs.get(query.query_id)
        evidence = None
        manifest = None
        if request_dir is not None:
            evidence_path = request_dir / "qa_evidence.json"
            manifest_path = request_dir / "qa_request_manifest.json"
            if evidence_path.is_file():
                evidence = _load_json(evidence_path)
            if manifest_path.is_file():
                manifest = _load_json(manifest_path)
        question_type = None
        if evidence is not None:
            question_type = evidence.get("question_type")
        if question_type is None and manifest is not None:
            question_type = manifest.get("question_type")
        qa_supported[query.query_id] = (
            type(question_type) is str,
            type(question_type) is str and question_type != "UNSUPPORTED",
            False,
        )
        for stage, key in qa_keys.items():
            qa_candidate_stages[stage][query.query_id] = _candidate_observation(
                evidence.get(key) if evidence is not None else None,
                target_video_id=query.video_id,
            )
        selected_video_ids = (
            evidence.get("selected_video_ids") if evidence is not None else None
        )
        qa_candidate_stages["NOMINATED_VIDEO"][query.query_id] = (
            _video_id_observation(
                selected_video_ids,
                target_video_id=query.video_id,
            )
        )
        refined_records = (
            evidence.get("refined_candidates") if evidence is not None else None
        )
        success_observation = _refinement_success_observation(
            refined_records,
            target_video_id=query.video_id,
        )
        qa_candidate_stages["REFINEMENT_SUCCESS"][query.query_id] = (
            success_observation
        )
        # Backward-compatible aliases now use truthful provider-neutral semantics.
        qa_candidate_stages["REFINED"][query.query_id] = success_observation
        provider_observation = qa_candidate_stages["PROVIDER_EVIDENCE"][
            query.query_id
        ]
        if not provider_observation[0]:
            provider_observation = _candidate_observation(
                evidence.get("usable_evidence_candidates")
                if evidence is not None
                else None,
                target_video_id=query.video_id,
            )
        qa_candidate_stages["USABLE_EVIDENCE"][query.query_id] = provider_observation
        qa_candidate_stages["FINAL_OUTPUT"][query.query_id] = _final_observation(
            query.query_id,
            query.video_id,
            request_dir,
            "qa_predictions.jsonl",
            aggregate,
            aggregate_available,
            successful_ids,
        )
    qa_stages["SUPPORTED_QUERY"] = _stage_summary(
        qa_queries, qa_supported, target_applicable=False
    )
    for stage, observations in qa_candidate_stages.items():
        qa_stages[stage] = _stage_summary(qa_queries, observations)

    trake_stages: dict[str, dict[str, Any]] = {}
    max_event_count = max((len(query.events) for query in trake_queries), default=0)
    event_observations: dict[int, dict[str, tuple[bool, bool, bool]]] = {
        event_index: {} for event_index in range(max_event_count)
    }
    any_event_observations: dict[str, tuple[bool, bool, bool]] = {}
    all_event_observations: dict[str, tuple[bool, bool, bool]] = {}
    c1_observations: dict[str, tuple[bool, bool, bool]] = {}
    trake_final_observations: dict[str, tuple[bool, bool, bool]] = {}

    for query in trake_queries:
        request_dir = trake_dirs.get(query.query_id)
        candidate_payload = None
        refinement_payload = None
        if request_dir is not None:
            candidate_path = request_dir / "trake_event_candidates.json"
            refinement_path = request_dir / "trake_refinement.json"
            if candidate_path.is_file():
                candidate_payload = _load_json(candidate_path)
            if refinement_path.is_file():
                refinement_payload = _load_json(refinement_path)
        pools = candidate_payload.get("event_candidates") if candidate_payload else None
        pool_observations: list[tuple[bool, bool, bool]] = []
        for event_index in range(len(query.events)):
            records = None
            if type(pools) is list:
                matches = [
                    pool
                    for pool in pools
                    if type(pool) is dict and pool.get("event_index") == event_index
                ]
                if len(matches) == 1:
                    records = matches[0].get("candidates")
            observation = _candidate_observation(
                records, target_video_id=query.video_id
            )
            event_observations[event_index][query.query_id] = observation
            pool_observations.append(observation)
        all_available = bool(pool_observations) and all(item[0] for item in pool_observations)
        any_non_empty = any(item[1] for item in pool_observations) if all_available else False
        any_hit = any(item[2] for item in pool_observations) if all_available else False
        all_hit = all(item[2] for item in pool_observations) if all_available else False
        any_event_observations[query.query_id] = (
            all_available,
            any_non_empty,
            any_hit,
        )
        all_event_observations[query.query_id] = (
            all_available,
            all_available and all(item[1] for item in pool_observations),
            all_hit,
        )

        c1_records = None
        if refinement_payload is not None:
            c1_records = refinement_payload.get("c1_paths")
            if type(c1_records) is not list:
                path_records = refinement_payload.get("path_diagnostics")
                if type(path_records) is list:
                    c1_records = [
                        {
                            "rank": row.get("c1_rank"),
                            "video_id": row.get("video_id"),
                            "frame_ids": row.get("original_frame_ids"),
                        }
                        for row in path_records
                        if type(row) is dict
                    ]
        c1_observations[query.query_id] = _candidate_observation(
            c1_records, target_video_id=query.video_id
        )
        trake_final_observations[query.query_id] = _final_observation(
            query.query_id,
            query.video_id,
            request_dir,
            "trake_predictions.jsonl",
            aggregate,
            aggregate_available,
            successful_ids,
        )

    for event_index, observations in event_observations.items():
        relevant = [query for query in trake_queries if len(query.events) > event_index]
        trake_stages[f"EVENT_{event_index + 1}_POOL"] = _stage_summary(
            relevant, observations
        )
    trake_stages["ANY_EVENT_POOL_CONTAINS_TARGET_VIDEO"] = _stage_summary(
        trake_queries, any_event_observations
    )
    trake_stages["ALL_EVENT_POOLS_CONTAIN_TARGET_VIDEO"] = _stage_summary(
        trake_queries, all_event_observations
    )
    trake_stages["C1_PLANNER"] = _stage_summary(trake_queries, c1_observations)
    trake_stages["FINAL_OUTPUT"] = _stage_summary(
        trake_queries, trake_final_observations
    )

    return {
        "schema_version": 1,
        "benchmark_id": benchmark.benchmark_id,
        "analysis_role": "OFFLINE_STAGE_WISE_TARGET_VIDEO_DIAGNOSTIC",
        "semantic_gt_authority": "SOURCE_PROPOSED_INTERNAL",
        "official_competition_claim": False,
        "split": split,
        "qa": {
            "selected_query_count": len(qa_queries),
            "stages": qa_stages,
        },
        "trake": {
            "selected_query_count": len(trake_queries),
            "stages": trake_stages,
            "partial_chain_vs_zero_output": _partial_chain_comparison(
                error_analysis_path,
                trake_final_observations,
            ),
        },
    }
