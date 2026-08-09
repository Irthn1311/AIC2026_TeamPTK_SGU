"""Stage 1D orchestration with frozen Stage 1C baselines and translated-only search."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from triage_eg.data.dataset_survey import _is_within
from triage_eg.retrieval.stage1.builder import resolve_git_commit
from triage_eg.retrieval.stage1b.assets import AdapterFactory, load_multimodal_encoder
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1c.contracts import QueryRecord, Stage1CConfig
from triage_eg.retrieval.stage1c.metrics import query_diagnostics
from triage_eg.retrieval.stage1c.runner import (
    _aggregate_structural,
    _encoding_records,
    _frame_records,
    _runtime_candidate,
    _structural_issues,
    _validate_inputs,
)

from .artifacts import (
    create_stage1d_bundle,
    write_pair_comparison_artifacts,
    write_translated_query_artifacts,
)
from .contracts import (
    ARMS,
    STAGE1D_VERSION,
    TRANSLATOR_ARCHITECTURE,
    Stage1DConfig,
    Stage1DResult,
)
from .inputs import FrozenBaseline, load_frozen_baseline, read_json, validate_translator_asset
from .metrics import aggregate_comparisons, extended_numeric_summary, pair_comparison
from .report import build_stage1d_report
from .review import write_blinded_review
from .translator import OfflineViEnTranslator, translator_dependency_versions

TranslatorFactory = Callable[[Path, Any, Any], Any]


def _stage1c_runtime_config(config: Stage1DConfig, query_suite: Path) -> Stage1CConfig:
    return Stage1CConfig(
        repo_root=config.repo_root,
        dataset_root=config.dataset_root,
        stage0_root=config.stage0_root,
        stage1_root=config.stage1_root,
        stage1b_root=config.stage1b_root,
        encoder_asset_root=config.clip_asset_root,
        query_suite=query_suite,
        output_root=config.output_root,
        frame_top_k=config.retrieval.frame_top_k,
        kis_top_k=config.retrieval.kis_top_k,
        review_top_k=config.review.top_k,
        contact_sheet_top_k=config.retrieval.contact_sheet_top_k,
        device=config.clip_device,
        batch_size=max(16, len(config.pair_ids) * 3),
        strict_root=config.strict_root,
        structural_flags=config.structural_flags,
    )


def _jsonable_asset_validation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(item) if isinstance(item, Path) else item
        for key, item in value.items()
        if key not in {"model_root"}
    }


def _validate_stage1d_inputs(config: Stage1DConfig) -> dict[str, Any]:
    stage1c = config.stage1c_root.expanduser().resolve(strict=True)
    query_suite = (
        config.query_suite.expanduser().resolve(strict=True)
        if config.query_suite is not None
        else stage1c / "query_suite/query_suite.jsonl"
    )
    stage1c_config = _stage1c_runtime_config(config, query_suite)
    inputs = _validate_inputs(stage1c_config)
    output = config.output_root.expanduser().resolve(strict=False)
    translator_root = config.translator_asset_root.expanduser().resolve(strict=True)
    if any(_is_within(output, root) for root in (stage1c, translator_root)):
        raise ValueError("Stage 1D output must not write into Stage 1C or translator inputs")
    baseline = load_frozen_baseline(
        stage1c,
        stage1_fingerprint=inputs["stage1_summary"]["index_fingerprint"],
        stage1b_contract=inputs["selected_contract"],
        stage1b_runtime=inputs["stage1b_runtime"],
        explicit_query_suite=config.query_suite,
        expected_query_count=config.expected_query_count,
        expected_pair_count=config.expected_pair_count,
        pair_ids=config.pair_ids,
    )
    asset_validation = validate_translator_asset(translator_root)
    return {
        **inputs,
        "stage1c": stage1c,
        "baseline": baseline,
        "translator_asset": asset_validation,
        "stage1c_config": stage1c_config,
    }


def preflight_stage1d(
    config: Stage1DConfig,
    *,
    translator_module_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    inputs = _validate_stage1d_inputs(config)
    dependencies = translator_dependency_versions(
        translator_module_loader
    ) if translator_module_loader is not None else translator_dependency_versions()
    baseline: FrozenBaseline = inputs["baseline"]
    return {
        "stage0_status": inputs["stage0_manifest"]["status"],
        "stage1_index_status": inputs["stage1_summary"]["index_status"],
        "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
        "stage1b_encoder_status": inputs["selected_contract"]["compatibility_status"],
        "model_space_status": inputs["stage1b_runtime"]["model_space_status"],
        "stage1c_frozen_baseline_status": "VALID",
        "stage1c_query_suite_fingerprint": baseline.query_suite_fingerprint,
        "baseline_regenerated": False,
        "translator_asset_status": inputs["translator_asset"]["status"],
        "translator_revision": inputs["translator_asset"]["exact_revision"],
        "translator_hash_status": inputs["translator_asset"]["hash_verification"],
        "translator_dependencies": dependencies,
        "translator_device": config.translator.device,
        "offline_only": True,
        "pairs_selected": len(baseline.pairs),
    }


def _load_reusable_translations(
    output: Path,
    baseline: FrozenBaseline,
    config: Stage1DConfig,
) -> list[dict[str, Any]] | None:
    path = output / "translations/translations.jsonl"
    if not config.reuse_translations or not path.is_file():
        return None
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_pair = {item.get("pair_id"): item for item in records}
    if set(by_pair) != set(baseline.pairs):
        raise ValueError("Reusable translations do not cover selected pairs")
    expected_generation = config.generation.as_generate_kwargs()
    for pair_id, members in baseline.pairs.items():
        record = by_pair[pair_id]
        if (
            record.get("original_vi_text") != members["vi"].text
            or record.get("en_reference_text") != members["en"].text
            or record.get("translator", {}).get("exact_revision")
            != config.translator.exact_revision
            or record.get("generation") != expected_generation
            or record.get("status") != "SUCCESS"
            or not str(record.get("translated_text_for_clip", "")).strip()
        ):
            raise ValueError(f"Reusable translation mismatch for {pair_id}")
    return [by_pair[pair_id] for pair_id in sorted(baseline.pairs)]


def _translation_records(
    baseline: FrozenBaseline,
    config: Stage1DConfig,
    translator: Any,
) -> list[dict[str, Any]]:
    ordered_pairs = sorted(baseline.pairs)
    vi_texts = [baseline.pairs[pair_id]["vi"].text for pair_id in ordered_pairs]
    results = translator.translate(vi_texts)
    records = []
    for pair_id, result in zip(ordered_pairs, results, strict=True):
        members = baseline.pairs[pair_id]
        records.append(
            {
                "pair_id": pair_id,
                "vi_query_id": members["vi"].query_id,
                "en_reference_query_id": members["en"].query_id,
                "original_vi_text": members["vi"].text,
                "en_reference_text": members["en"].text,
                "translated_text_raw": result["translated_text_raw"],
                "translated_text_for_clip": result["translated_text_for_clip"],
                "translator": {
                    "model_id": config.translator.model_id,
                    "exact_revision": config.translator.exact_revision,
                    "architecture": "MarianMT",
                    "device": translator.device,
                },
                "generation": config.generation.as_generate_kwargs(),
                "translation_latency_ms": result["translation_latency_ms"],
                "status": "SUCCESS",
            }
        )
    return records


def _translated_queries(
    baseline: FrozenBaseline,
    translations: list[dict[str, Any]],
) -> list[QueryRecord]:
    return [
        QueryRecord(
            query_id=f"{item['pair_id']}_vi_translated_en",
            pair_id=item["pair_id"],
            language="en",
            category=baseline.pairs[item["pair_id"]]["vi"].category,
            difficulty=baseline.pairs[item["pair_id"]]["vi"].difficulty,
            text=item["translated_text_for_clip"],
            notes="Deterministic OPUS-MT vi-to-en output; strip-only CLIP text.",
        )
        for item in translations
    ]


def _issue_counts(issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(issues),
        "by_code": dict(sorted(Counter(item["code"] for item in issues).items())),
        "by_severity": dict(
            sorted(Counter(item["severity"] for item in issues).items())
        ),
    }


def run_stage1d(
    config: Stage1DConfig,
    *,
    translator_factory: TranslatorFactory | None = None,
    clip_adapter_factory: AdapterFactory | None = None,
) -> Stage1DResult:
    inputs = _validate_stage1d_inputs(config)
    output: Path = inputs["output"]
    baseline: FrozenBaseline = inputs["baseline"]
    if config.reuse_results and (output / "stage1d_summary.json").is_file():
        summary = read_json(output / "stage1d_summary.json")
        if (
            summary.get("stage1d_version") != STAGE1D_VERSION
            or summary.get("execution_status")
            not in {"COMPLETE", "COMPLETE_WITH_WARNINGS"}
            or summary.get("stage1c_frozen_baseline", {}).get(
                "query_suite_fingerprint"
            )
            != baseline.query_suite_fingerprint
            or summary.get("translator", {}).get("exact_revision")
            != config.translator.exact_revision
        ):
            raise ValueError("Existing Stage 1D result is not reusable")
        return Stage1DResult(output, summary, True)
    reusable_translations = _load_reusable_translations(output, baseline, config)
    if output.exists():
        if not config.overwrite:
            raise FileExistsError(f"Stage 1D output exists: {output}")
        shutil.rmtree(output)
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started_at = datetime.now(UTC).isoformat()
    translator = None
    encoder = None
    try:
        if reusable_translations is None:
            factory = translator_factory or (
                lambda model_path, translator_config, generation: OfflineViEnTranslator(
                    model_path, translator_config, generation
                )
            )
            translator = factory(
                inputs["translator_asset"]["model_root"],
                config.translator,
                config.generation,
            )
            translator.load()
            translations = _translation_records(baseline, config, translator)
            translator_runtime = translator.runtime_manifest()
        else:
            translations = reusable_translations
            translator_runtime = {
                "status": "TRANSLATIONS_REUSED",
                "model_id": config.translator.model_id,
                "exact_revision": config.translator.exact_revision,
                "local_files_only": True,
                "device": config.translator.device,
                "effective_generation_config": config.generation.as_generate_kwargs(),
            }
        write_jsonl(staging / "translations/translations.jsonl", translations)
        write_json(
            staging / "translator/translator_contract.json",
            {
                "translator": asdict(config.translator),
                "generation": asdict(config.generation),
                "input_text_policy": "EXACT_STAGE1C_VI_TEXT",
                "clip_text_policy": "STRIP_ONLY",
                "remote_model_id_used_at_runtime": False,
            },
        )
        write_json(staging / "translator/translator_runtime_manifest.json", translator_runtime)
        write_json(
            staging / "translator/asset_validation.json",
            _jsonable_asset_validation(inputs["translator_asset"]),
        )

        stage1c_config: Stage1CConfig = inputs["stage1c_config"]
        candidate = _runtime_candidate(stage1c_config, inputs["selected_contract"], staging)
        try:
            encoder = load_multimodal_encoder(
                candidate, adapter_factory=clip_adapter_factory
            )
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"STAGE1B_ENCODER_NOT_VERIFIED: {error}") from error
        runtime_method = getattr(encoder, "runtime_manifest", None)
        clip_runtime = runtime_method() if callable(runtime_method) else {}
        expected_sha = inputs["selected_contract"]["checkpoint_sha256"]
        if clip_runtime.get("checkpoint_sha256") not in {None, expected_sha}:
            raise RuntimeError("STAGE1B_ENCODER_NOT_VERIFIED: checkpoint SHA mismatch")

        translated_queries = _translated_queries(baseline, translations)
        en_queries = [baseline.pairs[pair_id]["en"] for pair_id in sorted(baseline.pairs)]
        vi_queries = [baseline.pairs[pair_id]["vi"] for pair_id in sorted(baseline.pairs)]
        diagnostic_queries = en_queries + vi_queries + translated_queries
        try:
            embeddings, encoding_records = _encoding_records(diagnostic_queries, encoder)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"CLIP_TEXT_ENCODING_FAILED: {error}") from error
        embedding_by_id = {
            query.query_id: embeddings[index]
            for index, query in enumerate(diagnostic_queries)
        }
        encoding_by_id = {item["query_id"]: item for item in encoding_records}
        translated_matrix = np.stack(
            [embedding_by_id[query.query_id] for query in translated_queries]
        )
        search_started = monotonic()
        try:
            scores, rows = inputs["backend"].search(
                translated_matrix, config.retrieval.kis_top_k
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"TRANSLATED_QUERY_SEARCH_FAILED: {error}") from error
        search_elapsed = monotonic() - search_started

        translations_by_pair = {item["pair_id"]: item for item in translations}
        translated_frames: dict[str, list[dict[str, Any]]] = {}
        translated_diagnostics: dict[str, dict[str, Any]] = {}
        issues: list[dict[str, Any]] = []
        for index, query in enumerate(translated_queries):
            internal = _frame_records(query, scores[index], rows[index], inputs["catalog"])
            raw = internal[: config.retrieval.frame_top_k]
            stored = inputs["backend"].vectors_at(rows[index, : len(raw)])
            diagnostics = {
                **query_diagnostics(raw, stored),
                "raw_frame_top_k": len(raw),
                "internal_kis_search_top_k": len(internal),
                "search_latency_ms_amortized": (
                    search_elapsed / len(translated_queries) * 1000
                ),
                "diagnostic_only": True,
            }
            translated_frames[query.pair_id] = raw
            translated_diagnostics[query.pair_id] = diagnostics
            _, artifact_issues = write_translated_query_artifacts(
                staging,
                query.pair_id,
                query,
                encoding_by_id[query.query_id],
                translations_by_pair[query.pair_id],
                raw,
                internal,
                diagnostics,
                inputs["dataset"],
                contact_sheet_top_k=config.retrieval.contact_sheet_top_k,
                skip_contact_sheets=config.skip_contact_sheets,
            )
            issues.extend(artifact_issues)
            issues.extend(_structural_issues(query.query_id, diagnostics, stage1c_config))

        comparisons = []
        frames_by_pair_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for pair_id, members in baseline.pairs.items():
            en, vi = members["en"], members["vi"]
            en_frames = baseline.frames_by_query[en.query_id]
            vi_frames = baseline.frames_by_query[vi.query_id]
            mt_frames = translated_frames[pair_id]
            frames_by_pair_arm[pair_id] = {
                "EN_DIRECT": en_frames,
                "VI_DIRECT": vi_frames,
                "VI_TRANSLATED_EN": mt_frames,
            }
            comparison = pair_comparison(
                pair_id=pair_id,
                category=en.category,
                difficulty=en.difficulty,
                en_text=en.text,
                vi_text=vi.text,
                translated_text=translations_by_pair[pair_id][
                    "translated_text_for_clip"
                ],
                en_embedding=embedding_by_id[en.query_id],
                vi_embedding=embedding_by_id[vi.query_id],
                translated_embedding=embedding_by_id[
                    f"{pair_id}_vi_translated_en"
                ],
                en_frames=en_frames,
                vi_frames=vi_frames,
                translated_frames=mt_frames,
                structural={
                    "EN_DIRECT": baseline.diagnostics_by_query[en.query_id],
                    "VI_DIRECT": baseline.diagnostics_by_query[vi.query_id],
                    "VI_TRANSLATED_EN": translated_diagnostics[pair_id],
                },
            )
            comparisons.append(comparison)
            issues.extend(
                write_pair_comparison_artifacts(
                    staging,
                    pair_id,
                    en_frames=en_frames,
                    vi_frames=vi_frames,
                    translated_frames=mt_frames,
                    dataset_root=inputs["dataset"],
                    en_text=en.text,
                    vi_text=vi.text,
                    translated_text=translations_by_pair[pair_id][
                        "translated_text_for_clip"
                    ],
                    stage1c_root=baseline.root,
                    query_suite_fingerprint=baseline.query_suite_fingerprint,
                    skip_contact_sheets=config.skip_contact_sheets,
                )
            )
        write_jsonl(staging / "comparisons/pair_comparisons.jsonl", comparisons)
        judgments = write_blinded_review(
            staging,
            pairs=baseline.pairs,
            frames_by_pair_arm=frames_by_pair_arm,
            config=config.review,
        )
        comparison_summary = aggregate_comparisons(comparisons)
        structural_summary = _aggregate_structural(
            list(translated_diagnostics.values())
        )
        execution_status = "COMPLETE_WITH_WARNINGS" if issues else "COMPLETE"
        build_commit, commit_source = resolve_git_commit(
            config.repo_root, config.build_git_commit
        )
        translation_latencies = [
            float(item["translation_latency_ms"]) for item in translations
        ]
        summary = {
            "stage1d_version": STAGE1D_VERSION,
            "execution_status": execution_status,
            "build_git_commit": build_commit,
            "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
            "stage1b_encoder": {
                "candidate_id": candidate.candidate_id,
                "compatibility_status": candidate.compatibility_status,
                "model_space_status": inputs["stage1b_runtime"]["model_space_status"],
                "checkpoint_sha256": expected_sha,
            },
            "stage1c_frozen_baseline": {
                "status": "VALID",
                "query_suite_fingerprint": baseline.query_suite_fingerprint,
                "query_count": config.expected_query_count,
                "pair_count": config.expected_pair_count,
                "pairs_selected": len(baseline.pairs),
                "baseline_regenerated": False,
            },
            "translator": {
                "status": "READY",
                "model_id": config.translator.model_id,
                "exact_revision": config.translator.exact_revision,
                "architecture": TRANSLATOR_ARCHITECTURE,
                "asset_status": "VALID",
                "local_only": True,
                "device": translator_runtime.get("device", config.translator.device),
                "asset_materialization": inputs["translator_asset"].get(
                    "asset_materialization", config.translator_asset_materialization
                ),
            },
            "translation": {
                "queries_requested": len(baseline.pairs),
                "queries_completed": len(translations),
                "queries_failed": 0,
                "latency_ms": extended_numeric_summary(translation_latencies),
                "reused": reusable_translations is not None,
            },
            "retrieval": {
                "status": "GENERATED",
                "translated_queries_completed": len(translated_queries),
                "translated_queries_failed": 0,
                "frame_top_k": config.retrieval.frame_top_k,
                "kis_top_k": config.retrieval.kis_top_k,
                "ranking_policy": "RAW_STAGE1A_EXACT_COSINE_NO_RERANKING",
                "baseline_retrieval_source": "FROZEN_STAGE1C_ARTIFACTS",
                "diagnostic_text_embedding_source": (
                    "STAGE1D_REENCODED_WITH_VERIFIED_CLIP"
                ),
            },
            "comparison": {
                "arms": list(ARMS),
                **comparison_summary,
            },
            "structural_diagnostics": structural_summary,
            "human_review": {
                "status": "NOT_REVIEWED",
                "review_top_k": config.review.top_k,
                "judgments_expected": judgments,
                "judgments_completed": 0,
                "blinded": True,
            },
            "language_bridge_quality_status": "NOT_REVIEWED",
            "issues": _issue_counts(issues),
            "non_claims": [
                "Translation smoke quality does not prove retrieval quality",
                "Ranking overlap does not prove semantic relevance",
                "No competition Recall@K is measured",
                "Stage 1C EN/VI direct rankings are frozen artifacts",
                "No reranking or diversification is applied",
            ],
        }
        manifest = {
            "stage1d_version": STAGE1D_VERSION,
            "status": execution_status,
            "build_git_commit": build_commit,
            "build_git_commit_source": commit_source,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(inputs["dataset"]),
            "stage0_root": str(inputs["stage0"]),
            "stage1_root": str(inputs["stage1"]),
            "stage1b_root": str(inputs["stage1b"]),
            "stage1c_root": str(baseline.root),
            "stage1c_materialization": config.stage1c_materialization,
            "translator_asset_root": str(inputs["translator_asset"]["asset_root"]),
            "translator_asset_materialization": config.translator_asset_materialization,
            "stage1_index_fingerprint": inputs["stage1_summary"]["index_fingerprint"],
            "query_suite_fingerprint": baseline.query_suite_fingerprint,
            "no_stage0_rerun": True,
            "no_stage1_rebuild": True,
            "stage1b_compatibility_logic_unchanged": True,
            "stage1c_baseline_regenerated": False,
            "no_model_download": True,
            "no_translation_api": True,
            "no_reranking": True,
            "no_diversification": True,
            "no_query_expansion": True,
            "no_multilingual_clip": True,
            "human_relevance_auto_labeling": False,
        }
        write_json(staging / "run_manifest.json", manifest)
        write_json(staging / "stage1d_summary.json", summary)
        write_jsonl(staging / "issues.jsonl", issues)
        (staging / "stage1d_report.md").write_text(
            build_stage1d_report(summary, comparisons, translations),
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return Stage1DResult(output, summary, False)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
        close = getattr(translator, "close", None)
        if callable(close):
            close()


__all__ = [
    "create_stage1d_bundle",
    "preflight_stage1d",
    "run_stage1d",
]
