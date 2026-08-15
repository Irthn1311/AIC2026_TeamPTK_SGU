"""GT-isolated D1 orchestration, summaries, manifests, and packaging."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from aic2026_eval.io import read_jsonl, sha256_file, write_json, write_jsonl
from aic2026_eval.scoring import evaluate
from aic2026_eval.validation import validate_predictions
from triage_eg.e2e1.contracts import FORBIDDEN_INFERENCE_FIELDS
from triage_eg.e2eg1.pipeline import SafeCoveragePipeline, is_opaque_machine_id
from triage_eg.e2eg1.runner import run_prediction_variant

from .contracts import (
    SELECTED_GROUNDING_POLICY,
    D1Settings,
    InferenceSnapshot,
    SemanticUnitSnapshot,
)
from .single_event import audit_single_event, summarize_single_events
from .trake import audit_trake_query, summarize_trake
from .translation import (
    blind_translation_rows,
    translation_provenance_rows,
    translation_review_instructions,
    translation_surface_summary,
)

EXPECTED_E2EG1_SHA256 = "662e274130b1de22cf96af2fefefcc6eca91515adf15c640eb197c0da12c001e"
EXPECTED_E2EG1_CROSS_G1_SHA256 = "8a774e25aae0d4e23eafa905e468b25baeabc0b2ed74ba16491a1138b099ef9e"


def run_g1_reproduction(
    pipeline: SafeCoveragePipeline,
    inference_root: str | Path,
    output_root: str | Path,
    temporary_root: str | Path,
) -> dict[str, Any]:
    """Run only selected G1 while the physical inference root contains queries only."""

    inference = Path(inference_root).resolve(strict=True)
    if {path.name for path in inference.iterdir()} != {"queries.jsonl"}:
        raise RuntimeError("D1_GT_UNAVAILABLE_DURING_PREDICTION_GATE_FAILED")
    temporary = Path(temporary_root).resolve(strict=False)
    if temporary.exists():
        shutil.rmtree(temporary)
    run = run_prediction_variant(
        pipeline,
        inference,
        "DEV_CROSS_60",
        SELECTED_GROUNDING_POLICY,
        temporary,
    )
    target = Path(output_root) / "predictions/cross_g1_reproduction.jsonl"
    write_jsonl(target, run["predictions"])
    digest = sha256_file(target)
    validation, issues = validate_predictions(run["queries"], run["predictions"])
    if validation["status"] != "PASS" or issues or not digest:
        raise RuntimeError("D1_G1_REPRODUCTION_VALIDATION_FAILED")
    opaque = [
        row
        for row in run["predictions"]
        if "answer" in row and is_opaque_machine_id(str(row["answer"]))
    ]
    if opaque:
        raise RuntimeError(f"D1_QA_OPAQUE_MACHINE_ID_OUTPUT: {opaque[:5]}")
    return {
        **run,
        "prediction_path": target,
        "sha256": digest,
        "validation": validation,
        "qa_opaque_machine_id_output_count": 0,
    }


def _frozen_array(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=np.float32, copy=True)
    output.flags.writeable = False
    return output


def capture_inference_snapshot(
    pipeline: SafeCoveragePipeline, run: dict[str, Any]
) -> InferenceSnapshot:
    """Freeze all score/provenance state only after the prediction hash exists."""

    if not run.get("sha256") or run.get("validation", {}).get("status") != "PASS":
        raise RuntimeError("D1_PREDICTIONS_MUST_BE_HASHED_BEFORE_SCORE_SNAPSHOT")
    if any(FORBIDDEN_INFERENCE_FIELDS & set(query) for query in run["queries"]):
        raise RuntimeError("D1_GT_FIELD_LEAKED_INTO_INFERENCE_QUERY")
    results = {result.query_plan["query_id"]: result for result in run["results"]}
    units: dict[str, SemanticUnitSnapshot] = {}
    unit_ids_by_query: dict[str, tuple[str, ...]] = {}
    single_pools: dict[str, tuple[dict[str, Any], ...]] = {}
    allocations: dict[str, tuple[dict[str, Any], ...]] = {}
    trake_chains: dict[str, tuple[dict[str, Any], ...]] = {}
    for plan in run["plans"]:
        definitions = list(plan.events) if plan.task == "TRAKE" else [("E1", plan.grounding_text)]
        query_unit_ids = []
        for event_id, text in definitions:
            unit_id = f"{plan.query_id}:{event_id}"
            key = (plan.language, text)
            if key not in pipeline._encoded_text or key not in pipeline._score_cache:
                raise RuntimeError(f"D1_FROZEN_SCORE_MISSING: {unit_id}")
            embedding, encoding = pipeline._encoded_text[key]
            encoding = dict(encoding)
            translator_contract = (
                pipeline.runtime.inputs.get("language_contract", {})
                .get("vietnamese_path", {})
                .get("translator", {})
            )
            encoding["translator_route"] = (
                "VI_TO_EN_THEN_CLIP" if encoding.get("translation_applied") else "DIRECT_CLIP"
            )
            encoding["translator_model"] = (
                translator_contract.get("model_id") if encoding.get("translation_applied") else None
            )
            encoding["translator_revision"] = (
                translator_contract.get("exact_revision")
                if encoding.get("translation_applied")
                else None
            )
            units[unit_id] = SemanticUnitSnapshot(
                unit_id=unit_id,
                query_id=plan.query_id,
                task=plan.task,
                event_id=event_id if plan.task == "TRAKE" else None,
                source_language=plan.language,
                source_text=text,
                embedding=_frozen_array(embedding),
                scores=_frozen_array(pipeline._score_cache[key]),
                encoding=encoding,
            )
            query_unit_ids.append(unit_id)
        unit_ids_by_query[plan.query_id] = tuple(query_unit_ids)
        result = results[plan.query_id]
        if plan.task in {"KIS", "QA"}:
            cache_key = (plan.query_id, plan.language, plan.grounding_text)
            if cache_key not in pipeline._single_pool_cache:
                raise RuntimeError(f"D1_FROZEN_SINGLE_POOL_MISSING: {plan.query_id}")
            single_pools[plan.query_id] = tuple(
                dict(row) for row in pipeline._single_pool_cache[cache_key]
            )
            allocations[plan.query_id] = tuple(
                dict(row)
                for row in result.diagnostics
                if row.get("diagnostic_type") == "coverage_allocation"
            )
        else:
            trake_chains[plan.query_id] = tuple(dict(row) for row in result.predictions)
    snapshot = InferenceSnapshot(
        prediction_sha256=run["sha256"],
        units=units,
        unit_ids_by_query=unit_ids_by_query,
        single_event_pools=single_pools,
        g1_allocations=allocations,
        trake_chains=trake_chains,
    )
    if any(unit.scores.flags.writeable for unit in snapshot.units.values()):
        raise RuntimeError("D1_SCORE_SNAPSHOT_IS_MUTABLE")
    return snapshot


def write_blind_translation_artifacts(
    snapshot: InferenceSnapshot, output_root: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = blind_translation_rows(snapshot.units)
    provenance = translation_provenance_rows(snapshot.units)
    summary = translation_surface_summary(rows)
    root = Path(output_root)
    write_jsonl(root / "diagnostics/translation_blind_qc.jsonl", rows)
    write_jsonl(root / "diagnostics/translation_provenance.jsonl", provenance)
    write_json(root / "diagnostics/translation_surface_summary.json", summary)
    (root / "AI_TRANSLATION_REVIEW_INSTRUCTIONS.md").write_text(
        translation_review_instructions(), encoding="utf-8"
    )
    return rows, summary


def _historical_rows(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source.is_file():
        digest = sha256_file(source)
        if digest != EXPECTED_E2EG1_SHA256:
            raise RuntimeError(f"D1_HISTORICAL_E2EG1_SHA256_MISMATCH: {digest}")
        with ZipFile(source) as archive:
            rows = [
                json.loads(line)
                for line in archive.read("predictions/dev_cross_60_g1.jsonl")
                .decode("utf-8")
                .splitlines()
                if line
            ]
        return rows, {"source_kind": "ZIP", "artifact_sha256_verified": True}
    prediction = source / "predictions/dev_cross_60_g1.jsonl"
    if not prediction.is_file() or sha256_file(prediction) != EXPECTED_E2EG1_CROSS_G1_SHA256:
        raise RuntimeError("D1_HISTORICAL_E2EG1_EXTRACTED_G1_HASH_MISMATCH")
    return read_jsonl(prediction), {
        "source_kind": "EXTRACTED_ROOT",
        "artifact_sha256_verified": False,
        "g1_prediction_sha256_verified": True,
    }


def _grounding_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    frames = row.get("frame_ids", [row.get("frame_id")])
    return row["query_id"], int(row["rank"]), row["video_id"], tuple(int(v) for v in frames)


def verify_historical_reproduction(
    run: dict[str, Any], historical_source: str | Path | None, output_root: str | Path
) -> dict[str, Any]:
    if historical_source is None:
        report = {
            "status": "NOT_CHECKED",
            "reason": "OPTIONAL_ARTIFACT_NOT_MOUNTED",
            "current_g1_finalized": True,
            "current_prediction_sha256": run["sha256"],
            "current_validation_status": run["validation"]["status"],
        }
    else:
        historical, source = _historical_rows(Path(historical_source).resolve(strict=True))
        expected = [_grounding_identity(row) for row in historical]
        actual = [_grounding_identity(row) for row in run["predictions"]]
        report = {
            "status": "PASS" if actual == expected else "FAIL",
            "current_g1_finalized": True,
            "current_prediction_sha256": run["sha256"],
            "current_validation_status": run["validation"]["status"],
            "grounding_tuple_count": len(actual),
            "historical_grounding_tuple_count": len(expected),
            "comparison_fields": ["query_id", "rank", "video_id", "frame_id_or_frame_ids"],
            "ignored_non_grounding_metadata": True,
            **source,
        }
    write_json(Path(output_root) / "diagnostics/reproduction_check.json", report)
    if report["status"] == "FAIL":
        raise RuntimeError("D1_G1_REPRODUCTION=FAIL")
    return report


def _primary_bottleneck(
    single_rows: list[dict[str, Any]], trake_rows: list[dict[str, Any]]
) -> tuple[str, dict[str, int]]:
    single_counts = Counter(row["primary_failure_reason"] for row in single_rows)
    trake_counts = Counter(row["primary_failure_reason"] for row in trake_rows)
    measured = {
        "REPRESENTATION": single_counts["BTC_REPRESENTATION_GAP"]
        + trake_counts["BTC_EVENT_REPRESENTATION_GAP"],
        "SEMANTIC_SCORING": single_counts["TARGET_SEMANTIC_SCORE_WEAK"]
        + trake_counts["EVENT_SEMANTIC_SCORE_GAP"],
        "T3_POOL": single_counts["T3_REGION_REPRESENTATIVE_GAP"]
        + trake_counts["T3_EVENT_POOL_GAP"],
        "GLOBAL_RANKING": single_counts["GLOBAL_VIDEO_RANKING_GAP"]
        + trake_counts["GLOBAL_CHAIN_RANKING_GAP"],
        "G1_ALLOCATION": single_counts["G1_ALLOCATION_GAP"],
        "MONOTONIC_COMPOSITION": trake_counts["MONOTONIC_COMPOSITION_GAP"],
    }
    maximum = max(measured.values(), default=0)
    leaders = [key for key, value in measured.items() if value == maximum and value > 0]
    if len(leaders) != 1:
        return "MIXED", measured
    mapping = {
        "REPRESENTATION": "BTC_REPRESENTATION",
        "SEMANTIC_SCORING": "SEMANTIC_SCORING_OR_TRANSLATION",
        "T3_POOL": "T3_POOL",
        "GLOBAL_RANKING": "GLOBAL_RANKING",
        "MONOTONIC_COMPOSITION": "MONOTONIC_COMPOSITION",
        "G1_ALLOCATION": "MIXED",
    }
    return mapping[leaders[0]], measured


def run_post_gt_attribution(
    snapshot: InferenceSnapshot,
    run: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    pipeline: SafeCoveragePipeline,
    output_root: str | Path,
    *,
    settings: D1Settings | None = None,
    translation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Consume finalized state plus GT; never call prediction, encoding, or scoring APIs."""

    if snapshot.prediction_sha256 != run.get("sha256"):
        raise RuntimeError("D1_FINALIZED_PREDICTION_HASH_CHANGED")
    settings = settings or D1Settings()
    gt = {row["query_id"]: row for row in ground_truth}
    query_map = {row["query_id"]: row for row in run["queries"]}
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["predictions"]:
        predictions[row["query_id"]].append(row)
    _, official_per_query, _, _ = evaluate(
        run["queries"], run["predictions"], ground_truth, metadata={"diagnostic": "D1"}
    )
    official = {row["query_id"]: row for row in official_per_query}
    single_rows: list[dict[str, Any]] = []
    trake_event_rows: list[dict[str, Any]] = []
    trake_query_rows: list[dict[str, Any]] = []
    for query_id in sorted(query_map):
        query = query_map[query_id]
        units = [snapshot.units[value] for value in snapshot.unit_ids_by_query[query_id]]
        if query["task"] in {"KIS", "QA"}:
            row = audit_single_event(
                units[0],
                ground_truth=gt[query_id],
                predictions=predictions[query_id],
                source_pool=snapshot.single_event_pools[query_id],
                g1_allocation=snapshot.g1_allocations[query_id],
                groups=pipeline.groups,
                group_by_video=pipeline.group_by_video,
                catalog=pipeline.runtime.catalog,
                settings=settings,
                full_qa_correct=(
                    bool(official[query_id]["R@100"]) if query["task"] == "QA" else None
                ),
            )
            single_rows.append(row)
        else:
            events, query_row = audit_trake_query(
                units,
                ground_truth=gt[query_id],
                predictions=predictions[query_id],
                groups=pipeline.groups,
                group_by_video=pipeline.group_by_video,
                catalog=pipeline.runtime.catalog,
                settings=settings,
            )
            trake_event_rows.extend(events)
            trake_query_rows.append(query_row)
    root = Path(output_root)
    single_summary = {
        "KIS": summarize_single_events(single_rows, "KIS"),
        "QA": summarize_single_events(single_rows, "QA"),
    }
    trake_summary = summarize_trake(trake_event_rows, trake_query_rows)
    observed_counts = {
        "KIS": single_summary["KIS"]["query_count"],
        "QA": single_summary["QA"]["query_count"],
        "TRAKE": trake_summary["query_count"],
    }
    if observed_counts != {"KIS": 20, "QA": 20, "TRAKE": 20}:
        raise RuntimeError(f"D1_DEV_CROSS_60_QUERY_COUNT_MISMATCH: {observed_counts}")
    primary, measured = _primary_bottleneck(single_rows, trake_query_rows)
    attribution_summary = {
        "primary_measured_bottleneck": primary,
        "measured_primary_counts": measured,
        "REPRESENTATION": measured["REPRESENTATION"],
        "SEMANTIC_SCORING": measured["SEMANTIC_SCORING"],
        "T3_POOL": measured["T3_POOL"],
        "GLOBAL_RANKING": measured["GLOBAL_RANKING"],
        "G1_ALLOCATION": measured["G1_ALLOCATION"],
        "MONOTONIC_COMPOSITION": measured["MONOTONIC_COMPOSITION"],
        "TRANSLATION_SURFACE_ANOMALY": int(
            (translation_summary or {}).get("translation_surface_anomaly_count", 0)
        ),
        "next_decision_requires_ai_translation_review": True,
        "translation_causal_status": "NOT_ESTABLISHED",
        "production_policy_changed": False,
    }
    write_jsonl(root / "diagnostics/single_event_attribution.jsonl", single_rows)
    write_json(root / "diagnostics/single_event_summary.json", single_summary)
    write_jsonl(root / "diagnostics/trake_event_attribution.jsonl", trake_event_rows)
    write_jsonl(root / "diagnostics/trake_query_attribution.jsonl", trake_query_rows)
    write_json(root / "diagnostics/trake_summary.json", trake_summary)
    write_json(root / "diagnostics/attribution_summary.json", attribution_summary)
    return {
        "single_rows": single_rows,
        "single_summary": single_summary,
        "trake_event_rows": trake_event_rows,
        "trake_query_rows": trake_query_rows,
        "trake_summary": trake_summary,
        "attribution_summary": attribution_summary,
    }


def write_manifests(
    output_root: str | Path,
    *,
    pipeline: SafeCoveragePipeline,
    settings: D1Settings,
    dataset_root: str | Path,
    team_eval_bundle: str | Path,
    historical_e2eg1_source: str | Path | None,
    branch: str,
    git_commit: str,
    reproduction: dict[str, Any],
    run: dict[str, Any],
) -> None:
    root = Path(output_root)
    runtime = pipeline.runtime.runtime_manifest()
    stage1b, translator = runtime.get("stage1b", {}), runtime.get("translator", {})
    write_json(root / "config_snapshot.json", settings.as_dict())
    write_json(
        root / "prediction_hashes.json",
        {"DEV_CROSS_60": {SELECTED_GROUNDING_POLICY: run["sha256"]}},
    )
    write_json(
        root / "run_manifest.json",
        {
            "experiment": "TRIAGE_E2ED1",
            "experiment_version": "0.1",
            "branch": branch,
            "git_commit": git_commit,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(Path(dataset_root).resolve()),
            "team_eval_sha256": sha256_file(team_eval_bundle),
            "historical_e2eg1_sha256": (
                sha256_file(historical_e2eg1_source)
                if historical_e2eg1_source is not None and Path(historical_e2eg1_source).is_file()
                else None
            ),
            "historical_e2eg1_expected_sha256": EXPECTED_E2EG1_SHA256,
            "historical_reproduction_status": reproduction["status"],
            "selected_grounding_policy": SELECTED_GROUNDING_POLICY,
            "clip_model": stage1b.get("candidate_id"),
            "clip_checkpoint_sha256": stage1b.get("checkpoint_sha256"),
            "translator_model": translator.get("model_id"),
            "translator_device": runtime.get("devices", {}).get("translator"),
            "stage1_exact": True,
            "t3_pool_limit": settings.t3_pool_limit,
            "t3_region_radius_seconds": settings.t3_region_radius_seconds,
            "gt_available_to_inference": False,
            "translation_qc_blinded": True,
            "m1_used": False,
            "m2_used": False,
            "m3_used": False,
            "graph_used": False,
            "vlm_used": False,
            "agent_used": False,
            "sealed_accessed": False,
        },
    )
    (root / "README.md").write_text(
        "# TRIAGE-EG D1\n\n"
        "Diagnostic-only attribution of BTC representation, frozen CLIP scoring, T3 pools, "
        "global ranking, G1 allocation, TRAKE monotonic composition, and blinded translation "
        "surface anomalies. Production remains G1_COVERAGE_COARSE; D1 performs no tuning.\n",
        encoding="utf-8",
    )


def create_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(output_root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    required = {
        "README.md",
        "run_manifest.json",
        "config_snapshot.json",
        "prediction_hashes.json",
        "predictions/cross_g1_reproduction.jsonl",
        "diagnostics/translation_blind_qc.jsonl",
        "diagnostics/translation_provenance.jsonl",
        "diagnostics/translation_surface_summary.json",
        "diagnostics/single_event_attribution.jsonl",
        "diagnostics/single_event_summary.json",
        "diagnostics/trake_event_attribution.jsonl",
        "diagnostics/trake_query_attribution.jsonl",
        "diagnostics/trake_summary.json",
        "diagnostics/attribution_summary.json",
        "diagnostics/reproduction_check.json",
        "AI_TRANSLATION_REVIEW_INSTRUCTIONS.md",
    }
    missing = sorted(member for member in required if not (root / member).is_file())
    if missing:
        raise RuntimeError(f"D1_BUNDLE_REQUIRED_MEMBERS_MISSING: {missing}")
    review_files = sorted((root / "review").glob("*.jpg"))
    montage_files = sorted((root / "montages").glob("*.jpg"))
    if not 1 <= len(review_files) <= 18 or not 1 <= len(montage_files) <= 3:
        raise RuntimeError(
            f"D1_REVIEW_BUNDLE_INVALID: review={len(review_files)} montages={len(montage_files)}"
        )
    forbidden_suffixes = (".mp4", ".npy", ".npz", ".pt", ".pth", ".bin")
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(root).as_posix().casefold()
        if "sealed" in relative or relative.endswith(forbidden_suffixes):
            raise RuntimeError(f"FORBIDDEN_D1_BUNDLE_MEMBER: {relative}")
        if relative.endswith(("gt.jsonl", "queries.jsonl", "annotation_audit.jsonl")):
            raise RuntimeError(f"RAW_BENCHMARK_COPY_FORBIDDEN: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    return target


def formal_report_lines(
    *,
    git_commit: str,
    reproduction: dict[str, Any],
    translation_summary: dict[str, Any],
    audit: dict[str, Any],
    zip_path: str | Path,
) -> list[str]:
    kis, qa = audit["single_summary"]["KIS"], audit["single_summary"]["QA"]
    trake = audit["trake_summary"]
    k_counts, q_counts = kis["primary_failure_counts"], qa["primary_failure_counts"]
    t_counts = trake["primary_failure_counts"]
    return [
        f"HEAD={git_commit}",
        "D1_IMPLEMENTATION=COMPLETE",
        f"G1_REPRODUCTION={reproduction['status']}",
        "GT_LEAKAGE_GATE=PASS",
        "SEALED_ACCESS_GATE=PASS",
        "TRANSLATION_BLIND_QC=READY",
        f"TRANSLATION_UNIT_COUNT={translation_summary['translation_unit_count']}",
        f"TRANSLATION_SURFACE_ANOMALY_COUNT={translation_summary['translation_surface_anomaly_count']}",
        f"KIS_QUERY_COUNT={kis['query_count']}",
        f"KIS_G1_SUCCESS={k_counts.get('SUCCESS_G1_TARGET_HIT', 0)}",
        f"KIS_BTC_REPRESENTATION_GAP={k_counts.get('BTC_REPRESENTATION_GAP', 0)}",
        f"KIS_TARGET_SEMANTIC_SCORE_WEAK={k_counts.get('TARGET_SEMANTIC_SCORE_WEAK', 0)}",
        f"KIS_T3_REGION_REPRESENTATIVE_GAP={k_counts.get('T3_REGION_REPRESENTATIVE_GAP', 0)}",
        f"KIS_GLOBAL_VIDEO_RANKING_GAP={k_counts.get('GLOBAL_VIDEO_RANKING_GAP', 0)}",
        f"KIS_G1_ALLOCATION_GAP={k_counts.get('G1_ALLOCATION_GAP', 0)}",
        f"KIS_UNCLASSIFIED={k_counts.get('UNCLASSIFIED_SINGLE_EVENT', 0)}",
        f"KIS_MEDIAN_CORRECT_VIDEO_RANK={kis['correct_video_rank']['median']}",
        f"KIS_MEDIAN_TARGET_WITHIN_VIDEO_RANK={kis['best_target_within_video_rank']['median']}",
        f"KIS_MEDIAN_TARGET_GLOBAL_RANK={kis['best_target_global_rank']['median']}",
        f"QA_QUERY_COUNT={qa['query_count']}",
        f"QA_G1_GROUNDING_SUCCESS={q_counts.get('SUCCESS_G1_TARGET_HIT', 0)}",
        f"QA_BTC_REPRESENTATION_GAP={q_counts.get('BTC_REPRESENTATION_GAP', 0)}",
        f"QA_TARGET_SEMANTIC_SCORE_WEAK={q_counts.get('TARGET_SEMANTIC_SCORE_WEAK', 0)}",
        f"QA_T3_REGION_REPRESENTATIVE_GAP={q_counts.get('T3_REGION_REPRESENTATIVE_GAP', 0)}",
        f"QA_GLOBAL_VIDEO_RANKING_GAP={q_counts.get('GLOBAL_VIDEO_RANKING_GAP', 0)}",
        f"QA_G1_ALLOCATION_GAP={q_counts.get('G1_ALLOCATION_GAP', 0)}",
        f"QA_UNCLASSIFIED={q_counts.get('UNCLASSIFIED_SINGLE_EVENT', 0)}",
        f"TRAKE_QUERY_COUNT={trake['query_count']}",
        f"TRAKE_EVENT_COUNT={trake['event_count_total']}",
        f"TRAKE_EVENTS_WITH_BTC_TARGET={trake['events_with_btc_target_keyframes']}",
        f"TRAKE_EVENTS_WITH_T3_TARGET={trake['events_with_t3_target_hit']}",
        f"TRAKE_BTC_TARGET_CHAIN_EXISTS={trake['queries_with_btc_target_chain']}/20",
        f"TRAKE_T3_TARGET_CHAIN_EXISTS={trake['queries_with_t3_target_chain']}/20",
        f"TRAKE_CORRECT_VIDEO_FEASIBLE_CHAIN_EXISTS={trake['queries_with_correct_video_feasible_t3_chain']}/20",
        f"TRAKE_CORRECT_VIDEO_TOP100={trake['queries_with_correct_video_in_top100']}/20",
        f"TRAKE_FULL_TARGET_CHAIN_TOP100={trake['queries_with_full_target_chain_in_top100']}/20",
        f"TRAKE_SUCCESS_FULL_CHAIN={t_counts.get('SUCCESS_FULL_CHAIN', 0)}",
        f"TRAKE_BTC_EVENT_REPRESENTATION_GAP={t_counts.get('BTC_EVENT_REPRESENTATION_GAP', 0)}",
        f"TRAKE_EVENT_SEMANTIC_SCORE_GAP={t_counts.get('EVENT_SEMANTIC_SCORE_GAP', 0)}",
        f"TRAKE_T3_EVENT_POOL_GAP={t_counts.get('T3_EVENT_POOL_GAP', 0)}",
        f"TRAKE_MONOTONIC_COMPOSITION_GAP={t_counts.get('MONOTONIC_COMPOSITION_GAP', 0)}",
        f"TRAKE_GLOBAL_CHAIN_RANKING_GAP={t_counts.get('GLOBAL_CHAIN_RANKING_GAP', 0)}",
        f"TRAKE_UNCLASSIFIED={t_counts.get('UNCLASSIFIED_TRAKE', 0)}",
        f"PRIMARY_MEASURED_BOTTLENECK={audit['attribution_summary']['primary_measured_bottleneck']}",
        "AI_TRANSLATION_REVIEW_REQUIRED=YES",
        "PRODUCTION_POLICY_CHANGED=NO",
        "NEW_MODEL_ADDED=NO",
        "PARAMETER_TUNING_PERFORMED=NO",
        f"OUTPUT_ZIP={Path(zip_path)}",
        "RETURN_FOR_AI_REVIEW=YES",
    ]


__all__ = [
    "EXPECTED_E2EG1_CROSS_G1_SHA256",
    "EXPECTED_E2EG1_SHA256",
    "capture_inference_snapshot",
    "create_bundle",
    "formal_report_lines",
    "run_g1_reproduction",
    "run_post_gt_attribution",
    "verify_historical_reproduction",
    "write_blind_translation_artifacts",
    "write_manifests",
]
