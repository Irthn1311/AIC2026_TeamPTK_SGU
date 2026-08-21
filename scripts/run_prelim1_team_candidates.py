"""Run blind SOTUYEN1 inference and build a review/consensus packet; never open GT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from run_prelim_r5_final import (  # Reuse the exact R5 retrieval/runtime path.
    E5_EXACT_REVISION,
    _asr_search_all_views,
    _runtime,
    _translate_sources,
    _views_and_visual_retrieval,
)

from triage_eg.diagnostics.sca1_siglip2_complementarity import (
    Siglip2ExactBackend,
    Siglip2GroundingPipeline,
    Siglip2OfflineEncoder,
    local_only_load_smoke,
    validate_offline_asset,
)
from triage_eg.e2eg1 import SafeCoveragePipeline
from triage_eg.external_multimodal_v3.trial_smoke import OnnxE5QueryEncoder
from triage_eg.prelim1_team.packet import (
    CatalogResolver,
    export_candidate_embeddings,
    validate_team_packet,
    write_contact_sheet,
    write_json,
)
from triage_eg.prelim1_team.parser import parse_prelim1_zip
from triage_eg.prelim1_team.ranking import (
    build_qa_review_rows,
    fuse_team_chains,
    fuse_team_frames,
)
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


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"PRELIM1_EMPTY_CSV:{path.name}")
    fieldnames = fields or list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            row = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, list | dict | tuple)
                else value
                for key, value in raw.items()
            }
            writer.writerow(row)


def _find_one(root: Path, filename: str, *, optional: bool = False) -> Path | None:
    candidates = sorted({path.resolve() for path in root.rglob(filename) if path.is_file()})
    if optional and not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(f"PRELIM1_EXPECTED_ONE_FILE:{filename}:{candidates}")
    return candidates[0]


def _asset_provenance(label: str, root: Path) -> dict[str, Any]:
    manifests = sorted(path for path in root.rglob("*.json") if "manifest" in path.name.casefold())
    return {
        "label": label,
        "root": str(root),
        "manifest_hashes": {str(path.relative_to(root)): _sha256(path) for path in manifests[:20]},
    }


def _contact_panels(
    rows: list[dict[str, Any]], resolver: CatalogResolver, dataset_root: Path
) -> list[tuple[Path, str]]:
    output = []
    for row in rows:
        global_row = resolver.nearest_row(str(row["video_id"]), int(row["frame_id"]))
        path = resolver.image_path(dataset_root, global_row)
        label = (
            f"#{row['candidate_rank']} {row['video_id']} f={row['frame_id']} "
            f"t={float(row.get('video_time_sec') or 0.0):.1f}s"
        )
        output.append((path, label))
    return output


def _xclip_audit(
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    dataset_root: Path,
    xclip_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if xclip_root is None:
        return [], {"status": "DISABLED_MISSING_ASSET", "records": 0}
    from triage_eg.data.stage0_audit.asset_resolver import discover_layout, resolve_assets
    from triage_eg.fs1_v11.xclip import XClipAdapter, uniform_indices
    from triage_eg.video import OpenCVRawVideoDecoder

    by_id = {str(query["query_id"]): query for query in queries}
    trake = [row for row in candidates if row["task_type"] == "TRAKE"]
    video_parts, keyframe_parts = discover_layout(dataset_root)
    adapter = XClipAdapter(xclip_root)
    records, failures = [], []
    try:
        adapter.load()
        for row in trake:
            query = by_id[str(row["query_id"])]
            for event_index, frame_id in enumerate(row["frame_ids"]):
                try:
                    assets = resolve_assets(
                        dataset_root, str(row["video_id"]), video_parts, keyframe_parts
                    )
                    decoder = OpenCVRawVideoDecoder(str(row["video_id"]), assets.video)
                    radius = max(1, int(decoder.info.fps * 3))
                    indices = uniform_indices(
                        max(0, int(frame_id) - radius),
                        min(decoder.info.total_frames - 1, int(frame_id) + radius),
                    )
                    frames = [value.image for value in decoder.decode_indices(indices)]
                    decoder.close()
                    result = adapter.score(
                        str(query["event_descriptions"][event_index]["description"]), frames
                    )
                    records.append(
                        {
                            "query_id": row["query_id"],
                            "candidate_rank": row["candidate_rank"],
                            "event_index": event_index,
                            "video_id": row["video_id"],
                            "frame_id": int(frame_id),
                            "source": "XCLIP_EXISTING_PATH",
                            **result,
                        }
                    )
                except Exception as error:  # optional bounded branch
                    failures.append(
                        {
                            "query_id": row["query_id"],
                            "candidate_rank": row["candidate_rank"],
                            "event_index": event_index,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    except Exception as error:  # optional bounded branch
        return [], {"status": "DISABLED_LOAD_FAILED", "error": str(error), "records": 0}
    finally:
        adapter.unload()
    status = "ACTIVE" if records and not failures else "ACTIVE_WITH_DECODE_WARNINGS"
    return records, {"status": status, "records": len(records), "failures": failures}


def _apply_trake_graph_tail(
    candidates: list[dict[str, Any]], xclip_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Preserve visual Top1 and let complete XCLIP-supported chains order only ranks 2..5."""

    scores: dict[tuple[str, int], list[float]] = {}
    for row in xclip_rows:
        key = str(row["query_id"]), int(row["candidate_rank"])
        scores.setdefault(key, []).append(float(row["score"]))
    graph_rows = []
    query_ids = sorted({str(row["query_id"]) for row in candidates if row["task_type"] == "TRAKE"})
    for query_id in query_ids:
        selected = sorted(
            [
                row
                for row in candidates
                if row["task_type"] == "TRAKE" and str(row["query_id"]) == query_id
            ],
            key=lambda row: int(row["candidate_rank"]),
        )
        head, tail = selected[0], selected[1:]
        tail.sort(
            key=lambda row: (
                -(
                    sum(scores[(query_id, int(row["candidate_rank"]))])
                    / len(scores[(query_id, int(row["candidate_rank"]))])
                    if scores.get((query_id, int(row["candidate_rank"])))
                    else float("-inf")
                ),
                int(row["candidate_rank"]),
            )
        )
        reranked = [head, *tail]
        for rank, row in enumerate(reranked, 1):
            old_rank = int(row["candidate_rank"])
            row["candidate_rank"] = rank
            row["primary_candidate"] = rank == 1
            values = scores.get((query_id, old_rank), [])
            if values:
                row["modalities"] = sorted(set(row["modalities"]) | {"XCLIP"})
                row["reason_short"] += "; XCLIP query-local graph tail support"
            graph_rows.append(
                {
                    "query_id": query_id,
                    "candidate_rank": rank,
                    "video_id": row["video_id"],
                    "nodes": [
                        {"event_index": index, "frame_id": int(frame_id)}
                        for index, frame_id in enumerate(row["frame_ids"])
                    ],
                    "edges": [
                        {"from_event": index, "to_event": index + 1, "strictly_increasing": True}
                        for index in range(len(row["frame_ids"]) - 1)
                    ],
                    "xclip_mean_score": sum(values) / len(values) if values else None,
                    "head_protected": rank == 1,
                    "graph_used_for_tail_order": bool(xclip_rows) and rank > 1,
                    "ground_truth_used": False,
                }
            )
    return graph_rows


def _write_report(
    path: Path,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    by_id = {str(query["query_id"]): query for query in queries}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for query_id in by_id:
        grouped[query_id] = sorted(
            [row for row in candidates if str(row["query_id"]) == query_id],
            key=lambda row: int(row["candidate_rank"]),
        )
    lines = [
        "# MY PRELIM1 R5 SAFE TEAM RESULTS",
        "",
        "Blind inference only. No ground truth, leaderboard tuning, or automatic upload.",
        "",
    ]
    for query_id, query in by_id.items():
        lines.extend([f"## {query_id} ({query['task']})", "", str(query["normalized_text"]), ""])
        for row in grouped[query_id]:
            coordinate = row.get("frame_ids", row.get("frame_id"))
            answer = f" answer={row.get('answer')!r}" if row["task_type"] == "QA" else ""
            lines.append(
                f"- #{row['candidate_rank']} `{row['video_id']}` {coordinate}{answer}; "
                f"{row['reason_short']}"
            )
        if any(row.get("status") == "MANUAL_REVIEW_REQUIRED" for row in grouped[query_id]):
            lines.append("- WARNING: manual QA review required; no answer was invented.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
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
    asr_root = Path(args.asr_root).expanduser().resolve(strict=True) if args.asr_root else None
    e5_root = Path(args.e5_root).expanduser().resolve(strict=True) if args.e5_root else None
    external_root = (
        Path(args.external_evidence_root).expanduser().resolve(strict=True)
        if args.external_evidence_root
        else None
    )
    xclip_root = (
        Path(args.xclip_root).expanduser().resolve(strict=True) if args.xclip_root else None
    )
    manifest = (
        parse_prelim1_zip(
            query_zip,
            expected_sha256="" if args.allow_repacked_query_zip else None,
        )
        if args.allow_repacked_query_zip
        else parse_prelim1_zip(query_zip)
    )
    queries = list(manifest["queries"])
    write_json(output / "query_manifest.json", manifest)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_dir, text=True
    ).strip()
    work = output.parent / "prelim1_team_work"
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
    a0_runtime = _runtime(Path(args.repo_dir), head, "prelim1_a0", roots, work)
    s1_runtime = _runtime(Path(args.repo_dir), head, "prelim1_s1", roots, work)
    a0_pipeline = SafeCoveragePipeline(a0_runtime, dataset_root)
    s1_pipeline = Siglip2GroundingPipeline(
        s1_runtime,
        dataset_root,
        grounding_encoder=siglip_encoder,
        grounding_backend=Siglip2ExactBackend(siglip_index, stage1_root=stage1_root),
    )
    asset_status = {
        "A0_OPENAI_CLIP": {"status": "ACTIVE"},
        "S1_SIGLIP2": {"status": "ACTIVE"},
        "ASR_EXTERNAL_V3": {"status": "DISABLED_MISSING_ASSET"},
        "E5_QUERY": {"status": "DISABLED_MISSING_ASSET"},
        "OCR_EXTERNAL_V3": {"status": "DISABLED_MISSING_ASSET"},
        "OBJECT_EXTERNAL_V3": {"status": "DISABLED_MISSING_ASSET"},
        "XCLIP": {"status": "DISABLED_MISSING_ASSET"},
        "EVENT_GRAPH": {"status": "PENDING_TRAKE_CANDIDATES"},
        "WHISPER": {"status": "PROHIBITED_NOT_RUN"},
        "COMPLETION_V11_EVIDENCE": {"status": "NOT_REQUIRED_NONEXISTENT"},
    }
    try:
        translation = _translate_sources({"prelim1": queries}, a0_runtime)
        visual, view_rows, visual_provenance = _views_and_visual_retrieval(
            queries, translation, a0_pipeline, s1_pipeline
        )
        asr_search: dict[str, Any] = {
            str(query["query_id"]): {"lexical": {}, "e5": {}} for query in queries
        }
        asr_loader = None
        if asr_root is not None:
            asr_loader = load_asr_evidence(asr_root, ASR_EXTERNAL_V3_SOURCE_TYPE)
            asset_status["ASR_EXTERNAL_V3"] = {"status": "ACTIVE", "root": str(asr_root)}
            if e5_root is not None:
                e5_encoder = OnnxE5QueryEncoder(e5_root, exact_revision=E5_EXACT_REVISION)
                asr_search = _asr_search_all_views(queries, visual, asr_loader, e5_encoder)
                asset_status["E5_QUERY"] = {
                    "status": "ACTIVE",
                    "provenance": e5_encoder.provenance,
                }
            else:
                for query in queries:
                    query_id = str(query["query_id"])
                    for view in visual[query_id]["views"]:
                        if view["view"] in asr_search[query_id]["lexical"]:
                            continue
                        asr_search[query_id]["lexical"][view["view"]] = asr_loader.retrieve_spans(
                            str(view["text"]), max_spans=100
                        )
        ocr = {str(query["query_id"]): [] for query in queries}
        objects = {str(query["query_id"]): [] for query in queries}
        if external_root is not None:
            ocr_path = _find_one(external_root, "ocr_records_external_v3.parquet", optional=True)
            object_path = _find_one(
                external_root, "object_records_external_v3.parquet", optional=True
            )
            if ocr_path:
                ocr = build_external_parquet_evidence(queries, ocr_path, "ocr", limit=200)
                asset_status["OCR_EXTERNAL_V3"] = {
                    "status": "ACTIVE",
                    "path": str(ocr_path),
                    "sha256": _sha256(ocr_path),
                }
            if object_path:
                objects = build_external_parquet_evidence(queries, object_path, "object", limit=200)
                asset_status["OBJECT_EXTERNAL_V3"] = {
                    "status": "ACTIVE_WEAK_CORROBORATION_ONLY",
                    "path": str(object_path),
                    "sha256": _sha256(object_path),
                }
        resolver = CatalogResolver(stage1_root)
        candidates, evidence_rows, qa_audit = [], [], []
        for query in queries:
            query_id = str(query["query_id"])
            if str(query["task"]) == "TRAKE":
                rows = fuse_team_chains(query, a0=visual[query_id]["a0"], s1=visual[query_id]["s1"])
            else:
                provenance = [row for row in visual_provenance if str(row["query_id"]) == query_id]
                lexical = [
                    row for values in asr_search[query_id]["lexical"].values() for row in values
                ]
                e5 = [row for values in asr_search[query_id]["e5"].values() for row in values]
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
                )
                evidence_rows.extend(audit)
                if str(query["task"]) == "QA":
                    rows, current_qa = build_qa_review_rows(
                        query, contexts, asr_rows=[*lexical, *e5], ocr_rows=ocr[query_id]
                    )
                    qa_audit.extend(current_qa)
                else:
                    rows = contexts[:5]
            candidates.extend(rows)
        siglip_encoder.close()
        xclip_rows, xclip_status = _xclip_audit(queries, candidates, dataset_root, xclip_root)
        asset_status["XCLIP"] = xclip_status
        evidence_rows.extend(xclip_rows)
        graph_rows = _apply_trake_graph_tail(candidates, xclip_rows)
        evidence_rows.extend(graph_rows)
        asset_status["EVENT_GRAPH"] = {
            "status": (
                "ACTIVE_XCLIP_TAIL_ORDER" if xclip_rows else "ACTIVE_VISUAL_CHAINS_XCLIP_DISABLED"
            ),
            "chain_count": len(graph_rows),
            "top1_protected": True,
        }
        for query in queries:
            query_id, task = str(query["query_id"]), str(query["task"])
            selected = sorted(
                [row for row in candidates if str(row["query_id"]) == query_id],
                key=lambda row: int(row["candidate_rank"]),
            )
            if task in {"KIS", "QA"}:
                panels = _contact_panels(selected, resolver, dataset_root)
                write_contact_sheet(
                    output / "review" / task / f"{query_id}.jpg",
                    str(query["normalized_text"]),
                    panels,
                )
                if task == "KIS":
                    primary_row = resolver.nearest_row(
                        str(selected[0]["video_id"]), int(selected[0]["frame_id"])
                    )
                    nearby = resolver.nearby_rows(primary_row, (-2.0, 0.0, 2.0))
                    write_contact_sheet(
                        output / "review" / task / f"{query_id}_primary_nearby.jpg",
                        f"{query_id} primary t-2s / t / t+2s",
                        [
                            (
                                resolver.image_path(dataset_root, global_row),
                                (
                                    f"{offset:+.0f}s f="
                                    f"{resolver.catalog.map_row(global_row)['original_frame_idx']}"
                                ),
                            )
                            for offset, global_row in zip((-2.0, 0.0, 2.0), nearby, strict=True)
                        ],
                    )
            else:
                for row in selected:
                    panels = []
                    for event_index, frame_id in enumerate(row["frame_ids"]):
                        global_row = resolver.nearest_row(str(row["video_id"]), int(frame_id))
                        mapped = resolver.catalog.map_row(global_row)
                        panels.append(
                            (
                                resolver.image_path(dataset_root, global_row),
                                (
                                    f"E{event_index + 1} {row['video_id']} f={frame_id} "
                                    f"t={mapped['pts_time']:.1f}s"
                                ),
                            )
                        )
                    write_contact_sheet(
                        output / "review" / "TRAKE" / f"{query_id}_rank{row['candidate_rank']}.jpg",
                        str(query["normalized_text"]),
                        panels,
                    )
        embedding_summary = export_candidate_embeddings(
            candidates,
            resolver,
            siglip_index,
            output / "team_candidate_embeddings.npz",
            output / "team_candidate_embedding_index.csv",
        )
        _csv(output / "my_prelim1_top5.csv", candidates)
        write_json(output / "my_prelim1_top5.json", candidates)
        primary = []
        max_events = max(int(query.get("event_count", 0)) for query in queries)
        for row in candidates:
            if not row["primary_candidate"]:
                continue
            value = {
                "query_id": row["query_id"],
                "task_type": row["task_type"],
                "video_id": row["video_id"],
                "frame_id": row.get("frame_id", ""),
                "answer": row.get("answer", ""),
                "status": row.get("status", "READY"),
            }
            for event_index in range(max_events):
                frames = row.get("frame_ids", [])
                value[f"frame{event_index + 1}"] = (
                    frames[event_index] if event_index < len(frames) else ""
                )
            primary.append(value)
        _csv(output / "my_prelim1_primary.csv", primary)
        _jsonl(output / "query_view_provenance.jsonl", view_rows)
        _jsonl(output / "candidate_evidence.jsonl", evidence_rows)
        _csv(output / "qa_evidence_review.csv", qa_audit)
        write_json(
            output / "trake_top5.json",
            [row for row in candidates if row["task_type"] == "TRAKE"],
        )
        write_json(output / "asset_status.json", asset_status)
        _write_report(output / "MY_PRELIM1_RESULTS.md", queries, candidates)
        draft_ready = all(
            row.get("answer") and row.get("status") == "EVIDENCE_SUPPORTED"
            for row in primary
            if row["task_type"] == "QA"
        )
        draft_path = output / "prelim1_MY_PRELIM1_R5_SAFE_TEAM_DRAFT.zip"
        if draft_ready:
            predictions = []
            for row in primary:
                value = {"query_id": row["query_id"], "rank": 1, **row}
                if row["task_type"] == "TRAKE":
                    value["frame_ids"] = [row[f"frame{index + 1}"] for index in range(max_events)]
                predictions.append(value)
            create_submission_zip(queries, predictions, draft_path)
        provenance = {
            "HEAD": head,
            "query_package_sha256": manifest["official_package_sha256"],
            "resolved_archive_sha256": manifest["source_zip_sha256"],
            "query_content_sha256": manifest["content_sha256"],
            "elapsed_seconds": time.monotonic() - started,
            "source_system": "MY_PRELIM1_R5_SAFE_TEAM",
            "asset_provenance": [_asset_provenance(name, path) for name, path in roots.items()]
            + ([_asset_provenance("xclip", xclip_root)] if xclip_root is not None else []),
            "siglip_index_root": str(siglip_index),
            "ground_truth_opened": False,
            "leaderboard_used": False,
            "whisper_run": False,
            "submission_uploaded": False,
            "draft_oj_status": "DRAFT_OJ_READY" if draft_ready else "BLOCKED_MANUAL_QA",
            "embedding_summary": embedding_summary,
        }
        write_json(output / "run_provenance.json", provenance)
        validation = validate_team_packet(output, manifest, candidates)
        write_json(output / "packet_validation.json", validation)
        if validation["status"] != "PASS":
            raise RuntimeError(f"PRELIM1_TEAM_PACKET_VALIDATION_FAILED:{validation['issues']}")
        bundle = Path(args.output_zip).expanduser().resolve()
        bundle.unlink(missing_ok=True)
        with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
        summary = {
            "query_package_sha256": manifest["official_package_sha256"],
            "resolved_archive_sha256": manifest["source_zip_sha256"],
            "query_content_sha256": manifest["content_sha256"],
            "parsed_task_counts": manifest["task_counts"],
            "asset_branches": asset_status,
            "KIS_queries_completed": sum(query["task"] == "KIS" for query in queries),
            "QA_queries_completed": sum(query["task"] == "QA" for query in queries),
            "TRAKE_queries_completed": sum(query["task"] == "TRAKE" for query in queries),
            "contact_sheets_count": validation["contact_sheet_count"],
            "embeddings_exported_count": embedding_summary["embedding_array_count"],
            "status": "TEAM_PACKET_READY",
            "my_prelim1_primary.csv": str(output / "my_prelim1_primary.csv"),
            "my_prelim1_top5.csv": str(output / "my_prelim1_top5.csv"),
            "MY_PRELIM1_RESULTS.md": str(output / "MY_PRELIM1_RESULTS.md"),
            "team_candidate_embeddings.npz": str(output / "team_candidate_embeddings.npz"),
            "output_zip": str(bundle),
            "draft_oj_status": provenance["draft_oj_status"],
            "no_gt": True,
            "no_leaderboard": True,
            "no_upload": True,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
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
    parser.add_argument("--asr-root")
    parser.add_argument("--e5-root")
    parser.add_argument("--external-evidence-root")
    parser.add_argument("--xclip-root")
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="outputs/prelim1_team_candidates")
    parser.add_argument("--output-zip", default="outputs/prelim1_team_candidates_bundle.zip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--siglip-batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
