"""Run fresh blind R5-QE inference on the exact SOTUYEN1 query package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from run_prelim1_team_candidates import (
    _apply_trake_graph_tail,
    _asset_provenance,
    _contact_panels,
    _csv,
    _find_one,
    _jsonl,
    _sha256,
    _xclip_audit,
)
from run_prelim_r5_final import (
    E5_EXACT_REVISION,
    _asr_search_all_views,
    _runtime,
    _translate_sources,
    _views_and_visual_retrieval,
)

from triage_eg.diagnostics.bcf1_protected_late_fusion import BCF1Settings
from triage_eg.diagnostics.bcf1_protected_late_fusion.fusion import fuse_query
from triage_eg.diagnostics.sca1_siglip2_complementarity import (
    Siglip2ExactBackend,
    Siglip2GroundingPipeline,
    Siglip2OfflineEncoder,
    local_only_load_smoke,
    validate_offline_asset,
)
from triage_eg.e2eg1 import SafeCoveragePipeline
from triage_eg.external_multimodal_v3.trial_smoke import OnnxE5QueryEncoder
from triage_eg.prelim1_team.actual import (
    ACTUAL_SYSTEM,
    build_primary_rows,
    confidence_bucket,
    select_review_rows,
    validate_actual_results,
    write_results_report,
)
from triage_eg.prelim1_team.packet import (
    CatalogResolver,
    export_candidate_embeddings,
    write_contact_sheet,
    write_json,
)
from triage_eg.prelim1_team.parser import parse_prelim1_zip
from triage_eg.prelim1_team.ranking import (
    build_qa_review_rows,
    fuse_team_chains,
    fuse_team_frames,
)
from triage_eg.prelim_r5.evidence import fuse_asr_multiview
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    materialize_kaggle_expanded_tokenizer,
    resolve_official_asset_paths,
)
from triage_eg.submission.aic26_prelim import create_submission_zip
from triage_eg.trial_p1.asr_v12_loader import (
    ASR_EXTERNAL_V3_SOURCE_TYPE,
    load_asr_evidence,
)
from triage_eg.trial_p1.multimodal_dryrun import build_external_parquet_evidence


def _specificity_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    tiers = {
        "TIER_A_DIRECT": "HIGH",
        "TIER_B_CORROBORATED": "MEDIUM",
        "TIER_C_WEAK": "LOW",
    }
    for fallback_rank, item in enumerate(result.get("gated_candidates", []), 1):
        representative = dict(item.get("representative", {}))
        if representative.get("frame_id") is None:
            continue
        spans = item.get("source_spans", [])
        evidence_text = " | ".join(
            str(span.get("text") or span.get("asr_span", {}).get("text", ""))
            for span in spans[:3]
        )
        output.append(
            {
                **representative,
                "rank": int(item.get("rank", fallback_rank)),
                "specificity_tier": tiers.get(str(item.get("tier")), "LOW"),
                "text": evidence_text,
                "specificity_provenance": {
                    key: item.get(key)
                    for key in (
                        "tier",
                        "high_anchor_match_count",
                        "matched_high_phrases",
                        "lexical_e5_agreement",
                        "visual_support",
                        "generic_view_only",
                    )
                },
            }
        )
    return output


def _baseline_equivalent(
    queries: list[dict[str, Any]], visual: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions, provenance, failures = [], [], []
    for query in queries:
        query_id = str(query["query_id"])
        fusion_query = {**query, "task": "KIS"} if query["task"] == "QA" else query
        try:
            rows, sources = fuse_query(
                fusion_query,
                visual[query_id]["a0"],
                visual[query_id]["s1"],
                settings=BCF1Settings(),
            )
            for row in rows:
                row.update(
                    {
                        "query_id": query_id,
                        "task_type": query["task"],
                        "source_system": "MY_PRELIM1_BCF1_EQUIVALENT",
                    }
                )
                if query["task"] == "QA":
                    row["answer"] = ""
            predictions.extend(rows)
            provenance.extend({**row, "task_type": query["task"]} for row in sources)
        except Exception as error:  # optional comparison arm must not block R5 inference
            failures.append({"query_id": query_id, "error": f"{type(error).__name__}: {error}"})
    status = "ACTIVE" if not failures else "DISABLED_INTEGRATION_FAILED"
    return predictions, provenance, {"status": status, "failures": failures}


def _canonicalize_rows(rows: list[dict[str, Any]], resolver: CatalogResolver) -> None:
    for row in rows:
        if row["task_type"] == "TRAKE":
            canonical = [
                resolver.map_coordinate(str(row["video_id"]), int(frame))
                for frame in row["frame_ids"]
            ]
            row["frame_ids"] = [int(value["original_frame_idx"]) for value in canonical]
            row["global_rows"] = [int(value["global_row"]) for value in canonical]
        else:
            mapped = resolver.map_coordinate(str(row["video_id"]), int(row["frame_id"]))
            row["frame_id"] = int(mapped["original_frame_idx"])
            row["video_time_sec"] = float(mapped["pts_time"])
            row["global_row"] = int(mapped["global_row"])


def _draft_predictions(
    queries: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for query in queries:
        selected = [row for row in rows if row["query_id"] == query["query_id"]]
        if query["task"] == "QA":
            selected = [row for row in selected[:5] if row.get("answer")]
        for rank, row in enumerate(selected[:100], 1):
            output.append({**row, "rank": rank})
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    repo = Path(args.repo_dir).expanduser().resolve(strict=True)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    query_zip = Path(args.query_zip).expanduser().resolve(strict=True)
    dataset_root = Path(args.dataset_root).expanduser().resolve(strict=True)
    stage1_root = Path(args.stage1_root).expanduser().resolve(strict=True)
    roots = {
        "stage1": stage1_root,
        "stage1b": Path(args.stage1b_root).expanduser().resolve(strict=True),
        "stage1e": Path(args.stage1e_root).expanduser().resolve(strict=True),
        "clip": Path(args.clip_root).expanduser().resolve(strict=True),
        "opus": Path(args.opus_root).expanduser().resolve(strict=True),
        "siglip": Path(args.siglip_root).expanduser().resolve(strict=True),
    }
    siglip_index = Path(args.siglip_index_root).expanduser().resolve(strict=True)
    asr_root = Path(args.asr_root).expanduser().resolve(strict=True)
    e5_root = Path(args.e5_root).expanduser().resolve(strict=True)
    external_root = Path(args.external_evidence_root).expanduser().resolve(strict=True)
    xclip_root = (
        Path(args.xclip_root).expanduser().resolve(strict=True) if args.xclip_root else None
    )
    manifest = parse_prelim1_zip(
        query_zip,
        expected_sha256="" if args.allow_repacked_query_zip else None,
    ) if args.allow_repacked_query_zip else parse_prelim1_zip(query_zip)
    queries = list(manifest["queries"])
    write_json(output / "query_manifest.json", manifest)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    work = output.parent / "prelim1_actual_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    clip_paths = resolve_official_asset_paths(roots["clip"])
    shared_clip_source, _ = materialize_kaggle_expanded_tokenizer(
        clip_paths.source_root, work / "shared_openai_clip_source"
    )
    os.environ["AIC_OPENAI_CLIP_SOURCE_ROOT"] = str(shared_clip_source)
    validate_offline_asset(roots["siglip"])
    local_only_load_smoke(roots["siglip"])
    siglip_encoder = Siglip2OfflineEncoder(
        roots["siglip"], device=args.device, batch_size=args.siglip_batch_size
    ).load()
    a0_runtime = _runtime(repo, head, "prelim1_actual_a0", roots, work)
    s1_runtime = _runtime(repo, head, "prelim1_actual_s1", roots, work)
    a0_pipeline = SafeCoveragePipeline(a0_runtime, dataset_root)
    s1_pipeline = Siglip2GroundingPipeline(
        s1_runtime,
        dataset_root,
        grounding_encoder=siglip_encoder,
        grounding_backend=Siglip2ExactBackend(siglip_index, stage1_root=stage1_root),
    )
    asset_status: dict[str, Any] = {
        "A0_OPENAI_CLIP": {"status": "ACTIVE"},
        "S1_SIGLIP2": {"status": "ACTIVE"},
        "ASR_EXTERNAL_V3": {"status": "ACTIVE", "root": str(asr_root)},
        "E5_QUERY": {"status": "PENDING"},
        "OCR_EXTERNAL_V3": {"status": "PENDING"},
        "OBJECT_EXTERNAL_V3": {"status": "PENDING"},
        "XCLIP": {"status": "DISABLED_MISSING_OPTIONAL_ASSET"},
        "T3_STAGE2A": {"status": "ACTIVE_FROZEN_COVERAGE_AWARE_CHAINS"},
        "EVENT_GRAPH": {"status": "PENDING_TRAKE_XCLIP_TAIL"},
        "MY_PRELIM1_BCF1_EQUIVALENT": {"status": "PENDING_OPTIONAL"},
        "CONTACT_SHEETS": {"status": "OPTIONAL_NOT_REQUESTED"},
        "CANDIDATE_EMBEDDINGS": {"status": "OPTIONAL_NOT_REQUESTED"},
        "WHISPER": {"status": "PROHIBITED_NOT_RUN"},
        "COMPLETION_V11_EVIDENCE": {"status": "NOT_REQUIRED_NONEXISTENT"},
    }
    try:
        translation = _translate_sources({"prelim1": queries}, a0_runtime)
        visual, view_rows, visual_provenance = _views_and_visual_retrieval(
            queries, translation, a0_pipeline, s1_pipeline
        )
        asr_loader = load_asr_evidence(asr_root, ASR_EXTERNAL_V3_SOURCE_TYPE)
        e5_encoder = OnnxE5QueryEncoder(e5_root, exact_revision=E5_EXACT_REVISION)
        asr_search = _asr_search_all_views(queries, visual, asr_loader, e5_encoder)
        asset_status["E5_QUERY"] = {"status": "ACTIVE", "provenance": e5_encoder.provenance}
        ocr_path = _find_one(external_root, "ocr_records_external_v3.parquet")
        object_path = _find_one(external_root, "object_records_external_v3.parquet")
        if ocr_path is None or object_path is None:
            raise RuntimeError("PRELIM1_REQUIRED_EXTERNAL_EVIDENCE_MISSING")
        ocr = build_external_parquet_evidence(queries, ocr_path, "ocr", limit=200)
        objects = build_external_parquet_evidence(queries, object_path, "object", limit=200)
        asset_status["OCR_EXTERNAL_V3"] = {
            "status": "ACTIVE_CORROBORATION_ONLY",
            "path": str(ocr_path),
            "sha256": _sha256(ocr_path),
        }
        asset_status["OBJECT_EXTERNAL_V3"] = {
            "status": "ACTIVE_WEAK_CORROBORATION_ONLY",
            "path": str(object_path),
            "sha256": _sha256(object_path),
        }
        baseline, baseline_provenance, baseline_status = _baseline_equivalent(queries, visual)
        asset_status["MY_PRELIM1_BCF1_EQUIVALENT"] = baseline_status
        if baseline_status["status"] == "ACTIVE":
            _jsonl(output / "baseline_bcf1_equivalent_top100.jsonl", baseline)

        resolver = CatalogResolver(stage1_root)
        candidates: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = [
            *({"record_type": "QUERY_VIEW", **row} for row in view_rows),
            *({"record_type": "VISUAL_VIEW_FUSION", **row} for row in visual_provenance),
            *({"record_type": "BCF1_EQUIVALENT", **row} for row in baseline_provenance),
        ]
        qa_hypotheses: list[dict[str, Any]] = []
        for query in queries:
            query_id, task = str(query["query_id"]), str(query["task"])
            if task == "TRAKE":
                rows = fuse_team_chains(
                    query, a0=visual[query_id]["a0"], s1=visual[query_id]["s1"], limit=20
                )
            else:
                lexical = [
                    row for values in asr_search[query_id]["lexical"].values() for row in values
                ]
                e5 = [row for values in asr_search[query_id]["e5"].values() for row in values]
                specificity_query = {**query, "task": "KIS"} if task == "QA" else query
                specificity_result = fuse_asr_multiview(
                    specificity_query,
                    visual[query_id]["views"],
                    asr_search[query_id]["lexical"],
                    asr_search[query_id]["e5"],
                    a0_multiview=visual[query_id]["a0"],
                    s1_multiview=visual[query_id]["s1"],
                    fallback_rows=visual[query_id]["a0"],
                    safe_r4_rows=visual[query_id]["s1"],
                    ocr_rows=ocr[query_id],
                )
                specificity = _specificity_rows(specificity_result)
                evidence_rows.extend(
                    {"record_type": "ASR_SPECIFICITY", **row}
                    for row in specificity_result["provenance"]
                )
                provenance = [
                    row for row in visual_provenance if str(row["query_id"]) == query_id
                ]
                contexts, audit = fuse_team_frames(
                    query,
                    a0=visual[query_id]["a0"],
                    s1=visual[query_id]["s1"],
                    a0_provenance=[row for row in provenance if row["branch"] == "A0"],
                    s1_provenance=[row for row in provenance if row["branch"] == "S1"],
                    asr_lexical=lexical,
                    asr_e5=e5,
                    ocr=ocr[query_id],
                    objects=objects[query_id],
                    resolver=resolver,
                    asr_specificity=specificity,
                    limit=100,
                )
                evidence_rows.extend({"record_type": "FINAL_FUSION", **row} for row in audit)
                if task == "QA":
                    qa_rows, qa_audit = build_qa_review_rows(
                        query, contexts, asr_rows=[*lexical, *e5], ocr_rows=ocr[query_id]
                    )
                    qa_hypotheses.extend(qa_rows)
                    evidence_rows.extend({"record_type": "QA_EVIDENCE", **row} for row in qa_audit)
                    rows = [*qa_rows, *contexts[5:]]
                    for row in rows[5:]:
                        row.update(
                            {
                                "answer": "",
                                "status": "CONTEXT_RETRIEVAL_ONLY",
                                "support_spans": [],
                            }
                        )
                else:
                    rows = contexts
            for row in rows:
                row["source_system"] = ACTUAL_SYSTEM
                row["confidence_bucket"] = confidence_bucket(row)
            candidates.extend(rows)

        xclip_rows, xclip_status = _xclip_audit(
            queries, candidates, dataset_root, xclip_root
        )
        asset_status["XCLIP"] = xclip_status
        evidence_rows.extend({"record_type": "XCLIP", **row} for row in xclip_rows)
        graph_rows = _apply_trake_graph_tail(candidates, xclip_rows)
        evidence_rows.extend({"record_type": "EVENT_GRAPH", **row} for row in graph_rows)
        asset_status["EVENT_GRAPH"] = {
            "status": (
                "ACTIVE_XCLIP_QUERY_LOCAL_GRAPH_OVER_T3"
                if xclip_rows
                else "DEGRADED_T3_WITHOUT_OPTIONAL_XCLIP"
            ),
            "chain_count": len(graph_rows),
            "event_count": next(
                int(query["event_count"]) for query in queries if query["task"] == "TRAKE"
            ),
        }
        _canonicalize_rows(candidates, resolver)
        candidates.sort(
            key=lambda row: (
                next(i for i, query in enumerate(queries) if query["query_id"] == row["query_id"]),
                int(row["candidate_rank"]),
            )
        )
        top5 = select_review_rows(queries, candidates, 5)
        top10 = select_review_rows(queries, candidates, 10)
        primary = build_primary_rows(queries, candidates)
        trake_top20 = [row for row in candidates if row["task_type"] == "TRAKE"]

        if args.with_contact_sheets:
            for query in queries:
                if query["task"] == "TRAKE":
                    continue
                selected = [row for row in top5 if row["query_id"] == query["query_id"]]
                write_contact_sheet(
                    output / "optional_contact_sheets" / f"{query['query_id']}.jpg",
                    str(query["normalized_text"]),
                    _contact_panels(selected, resolver, dataset_root),
                )
            asset_status["CONTACT_SHEETS"] = {"status": "ACTIVE_OPTIONAL"}
        embedding_summary = None
        if args.with_embeddings:
            embedding_summary = export_candidate_embeddings(
                top5,
                resolver,
                siglip_index,
                output / "optional_candidate_embeddings.npz",
                output / "optional_candidate_embedding_index.csv",
            )
            asset_status["CANDIDATE_EMBEDDINGS"] = {
                "status": "ACTIVE_OPTIONAL",
                **embedding_summary,
            }

        _csv(output / "my_system_primary.csv", primary)
        _csv(output / "my_system_top5.csv", top5)
        _csv(output / "my_system_top10.csv", top10)
        _jsonl(output / "my_system_top100.jsonl", candidates)
        _jsonl(output / "candidate_provenance.jsonl", evidence_rows)
        _csv(output / "qa_hypotheses.csv", qa_hypotheses)
        write_json(output / "trake_top20.json", trake_top20)
        write_json(output / "asset_status.json", asset_status)
        write_results_report(output / "MY_SYSTEM_RESULTS.md", queries, candidates)

        primary_qa = [row for row in primary if row["task_type"] == "QA"]
        draft_ready = len(primary_qa) == 4 and all(row.get("answer") for row in primary_qa)
        draft_path = output / "prelim1_MY_PRELIM1_R5_QE_DRAFT_submission.zip"
        if draft_ready:
            create_submission_zip(queries, _draft_predictions(queries, candidates), draft_path)
        provenance = {
            "HEAD": head,
            "source_system": ACTUAL_SYSTEM,
            "query_package_sha256": manifest["official_package_sha256"],
            "resolved_archive_sha256": manifest["source_zip_sha256"],
            "query_content_sha256": manifest["content_sha256"],
            "elapsed_seconds": time.monotonic() - started,
            "asset_provenance": [_asset_provenance(name, path) for name, path in roots.items()],
            "siglip_index_root": str(siglip_index),
            "external_asr_provenance_sha256": _sha256(
                _find_one(asr_root, "asr_external_v3_provenance.json")
            ),
            "ground_truth_opened": False,
            "sealed_final_30_opened": False,
            "leaderboard_used": False,
            "whisper_run": False,
            "submission_uploaded": False,
            "model_download": False,
            "oj_draft_status": "OJ_DRAFT_READY" if draft_ready else "OJ_DRAFT_NOT_READY",
            "optional_embedding_summary": embedding_summary,
        }
        write_json(output / "run_provenance.json", provenance)
        validation = validate_actual_results(
            output, manifest, candidates, qa_hypotheses, trake_top20
        )
        write_json(output / "inference_validation.json", validation)
        if validation["status"] != "PASS":
            raise RuntimeError(f"PRELIM1_ACTUAL_VALIDATION_FAILED:{validation['issues']}")
        bundle = Path(args.output_zip).expanduser().resolve()
        bundle.unlink(missing_ok=True)
        with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
        return {
            "Git commit SHA": head,
            "query package SHA": manifest["official_package_sha256"],
            "parsed counts": manifest["task_counts"],
            "active/disabled branches": asset_status,
            "KIS completion status": "20/20",
            "QA completion status": "4/4",
            "TRAKE completion status": "1/1",
            "core output file paths": {
                name: str(output / name)
                for name in (
                    "MY_SYSTEM_RESULTS.md",
                    "my_system_primary.csv",
                    "my_system_top5.csv",
                    "my_system_top10.csv",
                    "my_system_top100.jsonl",
                    "qa_hypotheses.csv",
                    "trake_top20.json",
                )
            },
            "output bundle": str(bundle),
            "bundle SHA-256": _sha256(bundle),
            "inference status": "INFERENCE_RESULTS_READY",
            "OJ draft status": provenance["oj_draft_status"],
            "no GT": True,
            "no leaderboard": True,
            "no auto-submit": True,
        }
    finally:
        a0_pipeline.close()
        s1_pipeline.close()
        siglip_encoder.close()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-zip", required=True)
    parser.add_argument("--allow-repacked-query-zip", action="store_true")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage1-root", required=True)
    parser.add_argument("--stage1b-root", required=True)
    parser.add_argument("--stage1e-root", required=True)
    parser.add_argument("--clip-root", required=True)
    parser.add_argument("--opus-root", required=True)
    parser.add_argument("--siglip-root", required=True)
    parser.add_argument("--siglip-index-root", required=True)
    parser.add_argument("--asr-root", required=True)
    parser.add_argument("--e5-root", required=True)
    parser.add_argument("--external-evidence-root", required=True)
    parser.add_argument("--xclip-root")
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="outputs/prelim1_actual_inference")
    parser.add_argument("--output-zip", default="outputs/prelim1_actual_inference_bundle.zip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--siglip-batch-size", type=int, default=64)
    parser.add_argument("--with-contact-sheets", action="store_true")
    parser.add_argument("--with-embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(arguments()), ensure_ascii=False, indent=2))
