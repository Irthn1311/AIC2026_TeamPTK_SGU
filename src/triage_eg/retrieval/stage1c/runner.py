"""Stage 1C orchestration over the verified Stage 1B encoder and Stage 1A exact index."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.data.dataset_survey import KAGGLE_INPUT_ROOT, _is_within
from triage_eg.retrieval.stage1.builder import resolve_git_commit
from triage_eg.retrieval.stage1.contracts import SearchConfig
from triage_eg.retrieval.stage1.search import load_search_backend
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    materialize_kaggle_expanded_tokenizer,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.assets import AdapterFactory, load_multimodal_encoder
from triage_eg.retrieval.stage1b.contracts import CandidateContract
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1c.artifacts import (
    review_instructions,
    write_query_artifacts,
    write_review_template,
)
from triage_eg.retrieval.stage1c.contracts import (
    QUERY_SUITE_VERSION,
    STAGE1C_VERSION,
    QueryRecord,
    Stage1CConfig,
    Stage1CResult,
)
from triage_eg.retrieval.stage1c.metrics import (
    numeric_summary,
    paired_language_diagnostic,
    query_diagnostics,
)
from triage_eg.retrieval.stage1c.query_suite import filter_query_suite, load_query_suite


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _issue(
    severity: str,
    code: str,
    *,
    query_id: str | None = None,
    message: str | None = None,
    **evidence: Any,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "query_id": query_id,
        "global_row": None,
        "path": None,
        "message": message or code.replace("_", " ").title(),
        "evidence": evidence,
    }


def _validate_inputs(config: Stage1CConfig) -> dict[str, Any]:
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    stage0 = config.stage0_root.expanduser().resolve(strict=True)
    stage1 = config.stage1_root.expanduser().resolve(strict=True)
    stage1b = config.stage1b_root.expanduser().resolve(strict=True)
    output = config.output_root.expanduser().resolve(strict=False)
    if config.strict_root and not _is_within(dataset, KAGGLE_INPUT_ROOT):
        raise ValueError("Strict dataset root must be below /kaggle/input")
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError("Stage 1C output root is too broad")
    if any(_is_within(output, root) for root in (dataset, stage0, stage1, stage1b)):
        raise ValueError("Stage 1C output must not write into any input root")

    for name in ("audit_summary.json", "run_manifest.json"):
        if not (stage0 / name).is_file():
            raise FileNotFoundError(f"Missing Stage 0 artifact: {name}")
    stage0_manifest = _json(stage0 / "run_manifest.json")
    if stage0_manifest.get("status") != "COMPLETE":
        raise ValueError("Stage 0 required manifest is not COMPLETE")

    stage1_summary = _json(stage1 / "stage1_summary.json")
    stage1_manifest = _json(stage1 / "index/index_manifest.json")
    stage1_run_manifest = _json(stage1 / "run_manifest.json")
    if (
        stage1_summary.get("index_status") != "COMPLETE"
        or stage1_manifest.get("status") != "COMPLETE"
    ):
        raise ValueError("STAGE1_INDEX_NOT_READY: Stage 1A index is not COMPLETE")
    if stage1_summary.get("next_stage_readiness", {}).get("corpus_index") not in {
        "READY",
        "READY_WITH_TIE_WARNINGS",
    }:
        raise ValueError("STAGE1_INDEX_NOT_READY: corpus index readiness is blocked")

    stage1b_summary = _json(stage1b / "stage1b_summary.json")
    selected_contract = _json(stage1b / "encoder/selected_encoder_contract.json")
    runtime_manifest = _json(stage1b / "encoder/runtime_adapter_manifest.json")
    if stage1b_summary.get("evaluation_status") != "COMPLETE":
        raise ValueError("STAGE1B_ENCODER_NOT_VERIFIED: evaluation is not COMPLETE")
    if (
        selected_contract.get("compatibility_status") != "VERIFIED"
        or not selected_contract.get("selected_candidate_id")
        or stage1b_summary.get("readiness", {}).get("text_retrieval")
        != "READY_FOR_QUALITATIVE_TESTING"
    ):
        raise ValueError("STAGE1B_ENCODER_NOT_VERIFIED: no verified selected candidate")
    if runtime_manifest.get("model_space_status") != "MODEL_SPACE_VERIFIED":
        raise ValueError("STAGE1B_MODEL_SPACE_NOT_VERIFIED")
    stage1_fingerprint = stage1_summary.get("index_fingerprint")
    if (
        not stage1_fingerprint
        or stage1_run_manifest.get("index_fingerprint") != stage1_fingerprint
        or stage1b_summary.get("stage1_index_fingerprint") != stage1_fingerprint
    ):
        raise ValueError("STAGE1_INDEX_NOT_READY: Stage 1A manifests/Stage 1B fingerprint mismatch")
    if not selected_contract.get("checkpoint_sha256"):
        raise ValueError("STAGE1B_ENCODER_NOT_VERIFIED: checkpoint provenance is missing")

    search_config = SearchConfig(stage1, "stage1c_preflight", top_k=1)
    backend, catalog = load_search_backend(search_config)
    return {
        "dataset": dataset,
        "stage0": stage0,
        "stage1": stage1,
        "stage1b": stage1b,
        "output": output,
        "stage0_manifest": stage0_manifest,
        "stage1_summary": stage1_summary,
        "stage1_manifest": stage1_manifest,
        "stage1_run_manifest": stage1_run_manifest,
        "stage1b_summary": stage1b_summary,
        "selected_contract": selected_contract,
        "stage1b_runtime": runtime_manifest,
        "backend": backend,
        "catalog": catalog,
    }


def preflight_stage1c(config: Stage1CConfig) -> dict[str, Any]:
    inputs = _validate_inputs(config)
    records, suite_manifest = load_query_suite(config.query_suite)
    selected = filter_query_suite(
        records,
        query_ids=config.query_ids,
        languages=config.languages,
        categories=config.categories,
    )
    return {
        "stage0_status": inputs["stage0_manifest"]["status"],
        "stage1_index_status": inputs["stage1_summary"]["index_status"],
        "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
        "stage1b_encoder_status": inputs["selected_contract"]["compatibility_status"],
        "model_space_status": inputs["stage1b_runtime"]["model_space_status"],
        "checkpoint_sha256": inputs["selected_contract"]["checkpoint_sha256"],
        "query_suite_status": "VALID",
        "query_suite_fingerprint": suite_manifest["fingerprint"],
        "queries_selected": len(selected),
    }


def _runtime_candidate(
    config: Stage1CConfig,
    selected_contract: dict[str, Any],
    staging: Path,
) -> CandidateContract:
    candidate = CandidateContract.from_dict(selected_contract)
    if candidate.compatibility_status != "VERIFIED":
        raise ValueError("STAGE1B_ENCODER_NOT_VERIFIED")
    if candidate.implementation == "openai_clip":
        paths = resolve_official_asset_paths(config.encoder_asset_root)
        runtime_source, _ = materialize_kaggle_expanded_tokenizer(
            paths.source_root, staging.parent / "triage_eg_openai_clip_runtime_source"
        )
        candidate = replace(
            candidate,
            source_root=str(runtime_source),
            checkpoint_path=str(paths.checkpoint_path),
            asset_manifest_path=str(paths.asset_manifest_path),
            device=config.device,
            batch_size=config.batch_size,
        )
    else:
        candidate = replace(candidate, device=config.device, batch_size=config.batch_size)
    return candidate


def _encoding_records(
    queries: list[QueryRecord], encoder: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    started = monotonic()
    values = np.asarray(encoder.encode_text([item.text for item in queries]), dtype=np.float32)
    elapsed = monotonic() - started
    if values.shape != (len(queries), 512):
        raise ValueError("QUERY_ENCODING_FAILED: expected normalized 512-D embeddings")
    if not np.isfinite(values).all():
        raise ValueError("QUERY_ENCODING_FAILED: non-finite embedding")
    returned_norms = np.linalg.norm(values, axis=1)
    if np.any(returned_norms == 0):
        raise ValueError("QUERY_ENCODING_FAILED: zero-norm embedding")
    normalized = values / returned_norms[:, None]
    runtime_manifest = getattr(encoder, "runtime_manifest", None)
    runtime = runtime_manifest() if callable(runtime_manifest) else {}
    execution = runtime.get("text_execution", {})
    raw_norms = execution.get("raw_norms", returned_norms.tolist())
    final_norms = execution.get("normalized_norms", np.linalg.norm(normalized, axis=1).tolist())
    latencies = execution.get("latency_seconds", [elapsed / len(queries)] * len(queries))
    tokenization = execution.get("tokenization_status", ["SUCCESS"] * len(queries))
    truncated = execution.get("text_was_truncated", [False] * len(queries))
    records = []
    for index, query in enumerate(queries):
        records.append(
            {
                "query_id": query.query_id,
                "text": query.text,
                "language": query.language,
                "embedding_dimension": 512,
                "embedding_finite": True,
                "embedding_norm_before_normalization": float(raw_norms[index]),
                "embedding_norm_after_normalization": float(final_norms[index]),
                "tokenization_status": str(tokenization[index]),
                "text_was_truncated": bool(truncated[index]),
                "encode_latency_ms": float(latencies[index]) * 1000,
            }
        )
    return normalized.astype(np.float32, copy=False), records


def _frame_records(
    query: QueryRecord,
    scores: np.ndarray,
    rows: np.ndarray,
    catalog: Any,
) -> list[dict[str, Any]]:
    output = []
    previous_score: float | None = None
    for rank, (score, global_row) in enumerate(zip(scores, rows, strict=True), start=1):
        mapped = catalog.map_row(int(global_row))
        current_score = float(score)
        output.append(
            {
                "query_id": query.query_id,
                "rank": rank,
                "global_row": int(global_row),
                "video_id": mapped["video_id"],
                "n": mapped["n"],
                "original_frame_idx": mapped["original_frame_idx"],
                "score": current_score,
                "keyframe_relative_path": mapped["keyframe_relative_path"],
                "is_initial_frame": (
                    mapped["n"] == 1 and mapped["original_frame_idx"] == 0
                ),
                "same_score_as_previous": (
                    previous_score is not None and current_score == previous_score
                ),
            }
        )
        previous_score = current_score
    return output


def _structural_issues(
    query_id: str, diagnostics: dict[str, Any], config: Stage1CConfig
) -> list[dict[str, Any]]:
    rules = (
        (
            "HIGH_INITIAL_FRAME_CONCENTRATION",
            "initial_frame_rate_top20",
            config.structural_flags.initial_frame_rate_top20_warn,
        ),
        (
            "HIGH_SINGLE_VIDEO_CONCENTRATION",
            "top_video_share_top20",
            config.structural_flags.top_video_share_top20_warn,
        ),
        (
            "HIGH_EXACT_VECTOR_DUPLICATION",
            "exact_duplicate_rate_top20",
            config.structural_flags.exact_duplicate_rate_top20_warn,
        ),
    )
    return [
        _issue(
            "WARNING",
            code,
            query_id=query_id,
            metric=metric,
            value=diagnostics[metric],
            threshold=threshold,
            policy="PROJECT_REVIEW_HEURISTIC",
        )
        for code, metric, threshold in rules
        if diagnostics[metric] >= threshold
    ]


def _aggregate_structural(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "initial_frame": {
            "initial_frame_rate_top20": numeric_summary(
                [item["initial_frame_rate_top20"] for item in diagnostics]
            )
        },
        "same_video_concentration": {
            "top_video_share_top20": numeric_summary(
                [item["top_video_share_top20"] for item in diagnostics]
            )
        },
        "exact_vector_duplication": {
            "exact_duplicate_rate_top20": numeric_summary(
                [item["exact_duplicate_rate_top20"] for item in diagnostics]
            )
        },
    }


def _paired_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs_completed": len(pairs),
        "embedding_cosine_summary": numeric_summary(
            [item["text_embedding_cosine_en_vi"] for item in pairs]
        ),
        "frame_overlap_summary": {
            "top20_global_row_jaccard": numeric_summary(
                [item["top20_global_row_jaccard"] for item in pairs]
            )
        },
        "video_overlap_summary": {
            "top20_video_jaccard": numeric_summary(
                [item["top20_video_jaccard"] for item in pairs]
            )
        },
    }


def _report(
    summary: dict[str, Any],
    queries: list[QueryRecord],
    diagnostics_by_query: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> str:
    pair_by_query: dict[str, dict[str, Any]] = {}
    for item in pairs:
        pair_by_query[item["en_query_id"]] = item
        pair_by_query[item["vi_query_id"]] = item
    lines = [
        "# Stage 1C Qualitative Retrieval Evaluation",
        "",
        "RETRIEVAL_QUALITY_STATUS = NOT_REVIEWED",
        "",
        "## Provenance",
        "",
        f"- Build commit: {provenance['build_git_commit']}",
        f"- Stage 1C version: {STAGE1C_VERSION}",
        "",
        "## Stage 1A Index",
        "",
        f"- Fingerprint: {summary['stage1_index_fingerprint']}",
        "- Reused without rebuild: true",
        "",
        "## Stage 1B Verified Encoder",
        "",
        f"- Candidate: {summary['stage1b_encoder']['candidate_id']}",
        f"- Compatibility: {summary['stage1b_encoder']['compatibility_status']}",
        f"- Model space: {summary['stage1b_encoder']['model_space_status']}",
        f"- Checkpoint SHA-256: {summary['stage1b_encoder']['checkpoint_sha256']}",
        "",
        "## Query Suite",
        "",
        f"- Version: {QUERY_SUITE_VERSION}",
        f"- Fingerprint: {summary['query_suite']['fingerprint']}",
        f"- Queries: {summary['query_suite']['query_count']}",
        "",
        "## Retrieval Execution",
        "",
        f"- Raw frame Top-K: {summary['retrieval']['raw_frame_top_k']}",
        f"- Internal KIS Top-K: {summary['retrieval']['kis_export_top_k']}",
        "- Ranking: raw Stage 1A exact cosine; no reranking or diversification",
        "",
        "## Structural Diagnostics",
        "",
        "Structural flags are PROJECT_REVIEW_HEURISTIC warnings, never quality gates.",
        "",
        "## Initial-Frame Analysis",
        "",
        json.dumps(summary["structural_diagnostics"]["initial_frame"], ensure_ascii=False),
        "",
        "## Same-Video Concentration",
        "",
        json.dumps(
            summary["structural_diagnostics"]["same_video_concentration"],
            ensure_ascii=False,
        ),
        "",
        "## Exact-Vector Duplicate Analysis",
        "",
        json.dumps(
            summary["structural_diagnostics"]["exact_vector_duplication"],
            ensure_ascii=False,
        ),
        "",
        "## English/Vietnamese Pair Diagnostics",
        "",
        json.dumps(summary["paired_language_diagnostics"], ensure_ascii=False),
        "",
        "## Human Review Status",
        "",
        f"- Status: {summary['human_review']['status']}",
        f"- Expected judgments: {summary['human_review']['judgments_expected']}",
        "- Human review is the source of truth for semantic relevance.",
        "",
        "## Query-by-Query Summary",
        "",
        "| Query | Lang | Category | Text | Top-1 | Videos@20 | Initial@20 | "
        "Exact duplicates@20 | Paired frame Jaccard@20 | Review |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for query in queries:
        diagnostics = diagnostics_by_query[query.query_id]
        pair = pair_by_query.get(query.query_id, {})
        lines.append(
            f"| {query.query_id} | {query.language} | {query.category} | {query.text} | "
            f"{diagnostics['top1_score']:.4f} | {diagnostics['unique_videos_top20']} | "
            f"{diagnostics['initial_frame_count_top20']} | "
            f"{diagnostics['exact_duplicate_rows_top20']} | "
            f"{pair.get('top20_global_row_jaccard', 'N/A')} | NOT_REVIEWED |"
        )
    lines.extend(
        [
            "",
            "## Review Instructions",
            "",
            "Open each contact sheet and fill review/review_template.csv using only "
            "RELEVANT, PARTIAL, IRRELEVANT, or UNCERTAIN.",
            "",
            "## Non-Claims",
            "",
            *[f"- {item}" for item in summary["non_claims"]],
            "",
            "## Next Decision Gate",
            "",
            "Do not choose a retrieval optimization until human review is complete.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_stage1c(
    config: Stage1CConfig,
    *,
    adapter_factory: AdapterFactory | None = None,
) -> Stage1CResult:
    inputs = _validate_inputs(config)
    output: Path = inputs["output"]
    if config.reuse_results and (output / "stage1c_summary.json").is_file():
        summary = _json(output / "stage1c_summary.json")
        if summary.get("evaluation_status") not in {"COMPLETE", "COMPLETE_WITH_WARNINGS"}:
            raise ValueError("Existing Stage 1C result is not reusable")
        all_queries, suite_manifest = load_query_suite(config.query_suite)
        selected = filter_query_suite(
            all_queries,
            query_ids=config.query_ids,
            languages=config.languages,
            categories=config.categories,
        )
        saved_suite = _json(output / "query_suite/query_suite_manifest.json")
        expected_ids = [item.query_id for item in selected]
        if (
            summary.get("stage1c_version") != STAGE1C_VERSION
            or summary.get("query_suite", {}).get("fingerprint")
            != suite_manifest["fingerprint"]
            or saved_suite.get("selected_query_ids") != expected_ids
            or summary.get("retrieval", {}).get("raw_frame_top_k") != config.frame_top_k
            or summary.get("retrieval", {}).get("kis_export_top_k") != config.kis_top_k
            or summary.get("retrieval", {}).get("review_top_k") != config.review_top_k
        ):
            raise ValueError("Existing Stage 1C result does not match the requested configuration")
        return Stage1CResult(output, summary, True)
    if output.exists():
        if not config.overwrite:
            raise FileExistsError(f"Stage 1C output exists: {output}")
        shutil.rmtree(output)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat()
    encoder = None
    try:
        all_queries, suite_manifest = load_query_suite(config.query_suite)
        queries = filter_query_suite(
            all_queries,
            query_ids=config.query_ids,
            languages=config.languages,
            categories=config.categories,
        )
        candidate = _runtime_candidate(config, inputs["selected_contract"], staging)
        try:
            encoder = load_multimodal_encoder(candidate, adapter_factory=adapter_factory)
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"ENCODER_LOAD_FAILED: {error}") from error
        runtime_manifest = getattr(encoder, "runtime_manifest", None)
        if callable(runtime_manifest):
            runtime = runtime_manifest()
            runtime_sha = runtime.get("checkpoint_sha256")
            expected_sha = inputs["selected_contract"]["checkpoint_sha256"]
            if runtime_sha and runtime_sha != expected_sha:
                raise RuntimeError("ENCODER_LOAD_FAILED: selected checkpoint SHA mismatch")
            runtime_source_commit = runtime.get("source_commit")
            expected_source_commit = inputs["selected_contract"].get(
                "asset_provenance", {}
            ).get("source_commit")
            if (
                runtime_source_commit
                and expected_source_commit
                and runtime_source_commit != expected_source_commit
            ):
                raise RuntimeError("ENCODER_LOAD_FAILED: selected source commit mismatch")
        try:
            embeddings, encoding_records = _encoding_records(queries, encoder)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            code = (
                "QUERY_TOKENIZATION_FAILED"
                if "TOKEN" in str(error).upper() or "CONTEXT_LENGTH" in str(error).upper()
                else "QUERY_ENCODING_FAILED"
            )
            raise RuntimeError(f"{code}: {error}") from error
        backend, catalog = inputs["backend"], inputs["catalog"]
        search_started = monotonic()
        try:
            scores, rows = backend.search(embeddings, config.kis_top_k)
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"QUERY_SEARCH_FAILED: {error}") from error
        search_elapsed = monotonic() - search_started

        query_suite_root = staging / "query_suite"
        write_jsonl(query_suite_root / "query_suite.jsonl", [asdict(item) for item in queries])
        selected_suite_manifest = {
            **suite_manifest,
            "version": QUERY_SUITE_VERSION,
            "selected_query_count": len(queries),
            "selected_query_ids": [item.query_id for item in queries],
        }
        write_json(query_suite_root / "query_suite_manifest.json", selected_suite_manifest)

        encoding_by_query = {item["query_id"]: item for item in encoding_records}
        frames_by_query: dict[str, list[dict[str, Any]]] = {}
        diagnostics_by_query: dict[str, dict[str, Any]] = {}
        issues: list[dict[str, Any]] = []
        for index, query in enumerate(queries):
            internal_frames = _frame_records(query, scores[index], rows[index], catalog)
            raw_frames = internal_frames[: config.frame_top_k]
            stored = backend.vectors_at(rows[index, : len(raw_frames)])
            diagnostics = {
                **query_diagnostics(raw_frames, stored),
                "raw_frame_top_k": len(raw_frames),
                "internal_kis_search_top_k": len(internal_frames),
                "search_latency_ms_amortized": search_elapsed / len(queries) * 1000,
                "diagnostic_only": True,
            }
            frames_by_query[query.query_id] = raw_frames
            diagnostics_by_query[query.query_id] = diagnostics
            _, contact_issues = write_query_artifacts(
                staging,
                query,
                encoding_by_query[query.query_id],
                raw_frames,
                internal_frames,
                diagnostics,
                inputs["dataset"],
                contact_sheet_top_k=config.contact_sheet_top_k,
                skip_contact_sheets=config.skip_contact_sheets,
            )
            issues.extend(contact_issues)
            issues.extend(_structural_issues(query.query_id, diagnostics, config))

        pair_members: defaultdict[str, dict[str, QueryRecord]] = defaultdict(dict)
        query_index = {item.query_id: index for index, item in enumerate(queries)}
        for query in queries:
            pair_members[query.pair_id][query.language] = query
        pairs = []
        for pair_id, members in sorted(pair_members.items()):
            if set(members) != {"en", "vi"}:
                continue
            en, vi = members["en"], members["vi"]
            pairs.append(
                paired_language_diagnostic(
                    pair_id,
                    embeddings[query_index[en.query_id]],
                    embeddings[query_index[vi.query_id]],
                    frames_by_query[en.query_id],
                    frames_by_query[vi.query_id],
                )
            )
        write_jsonl(staging / "pairs/pair_diagnostics.jsonl", pairs)

        judgments = write_review_template(
            staging / "review/review_template.csv",
            queries,
            frames_by_query,
            config.review_top_k,
        )
        (staging / "review/review_instructions.md").write_text(
            review_instructions(), encoding="utf-8"
        )
        evaluation_status = "COMPLETE_WITH_WARNINGS" if issues else "COMPLETE"
        summary = {
            "stage1c_version": STAGE1C_VERSION,
            "evaluation_status": evaluation_status,
            "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
            "stage1b_encoder": {
                "candidate_id": candidate.candidate_id,
                "implementation": candidate.implementation,
                "architecture": candidate.architecture,
                "pretrained": candidate.pretrained,
                "compatibility_status": candidate.compatibility_status,
                "model_space_status": inputs["stage1b_runtime"]["model_space_status"],
                "checkpoint_sha256": inputs["selected_contract"]["checkpoint_sha256"],
                "source_commit": inputs["selected_contract"].get(
                    "asset_provenance", {}
                ).get("source_commit"),
            },
            "query_suite": {
                "version": QUERY_SUITE_VERSION,
                "fingerprint": suite_manifest["fingerprint"],
                "query_count": len(queries),
                "pair_count": len(pairs),
                "languages": sorted({item.language for item in queries}),
            },
            "retrieval": {
                "raw_frame_top_k": config.frame_top_k,
                "kis_export_top_k": config.kis_top_k,
                "review_top_k": config.review_top_k,
                "queries_completed": len(queries),
                "queries_failed": 0,
                "ranking_policy": "RAW_STAGE1A_EXACT_COSINE_NO_RERANKING",
            },
            "structural_diagnostics": _aggregate_structural(
                list(diagnostics_by_query.values())
            ),
            "paired_language_diagnostics": _paired_summary(pairs),
            "human_review": {
                "status": "NOT_REVIEWED",
                "judgments_expected": judgments,
                "judgments_completed": 0,
            },
            "retrieval_quality_status": "NOT_REVIEWED",
            "issues": {
                "total": len(issues),
                "by_code": dict(sorted(Counter(item["code"] for item in issues).items())),
                "by_severity": dict(
                    sorted(Counter(item["severity"] for item in issues).items())
                ),
            },
            "non_claims": [
                "No competition Recall@K is measured",
                "Structural overlap does not prove semantic relevance",
                "English/Vietnamese overlap does not prove language quality",
                "No retrieval optimization is performed in Stage 1C",
            ],
        }
        build_git_commit, commit_source = resolve_git_commit(
            config.repo_root, config.build_git_commit
        )
        manifest = {
            "stage1c_version": STAGE1C_VERSION,
            "status": evaluation_status,
            "build_git_commit": build_git_commit,
            "build_git_commit_source": commit_source,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "stage0_root": str(inputs["stage0"]),
            "stage1_root": str(inputs["stage1"]),
            "stage1b_root": str(inputs["stage1b"]),
            "dataset_root": str(inputs["dataset"]),
            "query_suite_fingerprint": suite_manifest["fingerprint"],
            "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
            "no_stage0_rerun": True,
            "no_stage1_rebuild": True,
            "stage1b_compatibility_logic_unchanged": True,
            "no_model_download": True,
            "no_translation": True,
            "no_query_expansion": True,
            "no_reranking": True,
            "no_multilingual_model": True,
            "human_relevance_auto_labeling": False,
            "structural_flag_policy": "PROJECT_REVIEW_HEURISTIC",
        }
        write_json(staging / "run_manifest.json", manifest)
        write_json(staging / "stage1c_summary.json", summary)
        write_jsonl(staging / "issues.jsonl", issues)
        (staging / "stage1c_report.md").write_text(
            _report(
                summary,
                queries,
                diagnostics_by_query,
                pairs,
                {"build_git_commit": build_git_commit},
            ),
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return Stage1CResult(output, summary, False)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
