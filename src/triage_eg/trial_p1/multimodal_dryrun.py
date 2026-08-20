"""Fail-closed Trial P1 multimodal dry-run and review artifact helpers.

This module is diagnostic-only.  It never reads ground truth, changes the
production retrieval policy, or uploads a submission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aic2026_eval.validation import validate_predictions
from triage_eg.fs1.fusion import default_key
from triage_eg.fs1_v11.pipeline import build_completion_arm, grouped, semantic_content_hash

EXPECTED_TASK_COUNTS = {"KIS": 18, "QA": 3, "TRAKE": 3}
QA_IDS = ("query-p1-15-qa", "query-p1-19-qa", "query-p1-22-qa")
ASR_SANITY_IDS = (
    "query-p1-1-kis",
    "query-p1-8-kis",
    "query-p1-11-kis",
    "query-p1-17-kis",
    "query-p1-23-kis",
    "query-p1-25-kis",
)
GENERIC_QA = re.compile(
    r"^(?:người|ghế|bàn|sách|đồ vật|vật|nhà thơ|câu thơ|không rõ|"
    r"không đủ bằng chứng|unknown|n/?a)$",
    re.I,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_trial_plans(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert frozen compiler plans to the team query contract without GT."""

    queries = []
    for plan in plans:
        query = dict(plan.get("team_query") or {})
        query.setdefault("query_id", str(plan["query_id"]))
        query.setdefault("task", str(plan["task"]).upper())
        query.setdefault("language", "vi")
        query.setdefault("query", str(plan.get("raw_text", "")))
        if plan.get("answer_type"):
            query["answer_type"] = str(plan["answer_type"])
        if plan.get("answer_policy"):
            query["answer_policy"] = str(plan["answer_policy"])
        if query["task"] == "TRAKE":
            events = plan.get("events") or []
            query["event_count"] = len(events)
            query["event_descriptions"] = [str(event["description"]) for event in events]
            query["raw_event_labels"] = [str(event["raw_event_label"]) for event in events]
        queries.append(query)
    validate_trial_contract(queries)
    return queries


def validate_trial_contract(queries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("task", "")).upper() for row in queries)
    ids = [str(row.get("query_id")) for row in queries]
    issues: list[str] = []
    if len(queries) != 24 or len(set(ids)) != 24:
        issues.append("TRIAL_QUERY_COUNT_OR_ID_UNIQUENESS_FAILED")
    if dict(counts) != EXPECTED_TASK_COUNTS:
        issues.append(f"TRIAL_TASK_COUNTS_FAILED:{dict(counts)}")
    trake = [row for row in queries if str(row.get("task")).upper() == "TRAKE"]
    for row in trake:
        events = list(row.get("event_descriptions") or [])
        labels = list(row.get("raw_event_labels") or [])
        if int(row.get("event_count", 0)) != 4 or len(events) != 4:
            issues.append(f"TRAKE_NOT_EXACTLY_FOUR_EVENTS:{row.get('query_id')}")
        if row.get("query_id") == "query-p1-18-trake" and labels != ["E1", "E2", "E2", "E4"]:
            issues.append("P1_18_DUPLICATED_RAW_E2_NOT_PRESERVED")
    if set(QA_IDS) != {row["query_id"] for row in queries if row["task"] == "QA"}:
        issues.append("TRIAL_QA_ID_SET_MISMATCH")
    if issues:
        raise RuntimeError(";".join(issues))
    return {"status": "PASS", "query_count": 24, "task_counts": dict(counts)}


def group_evidence(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        output[str(row["query_id"])].append(row)
    for values in output.values():
        values.sort(key=lambda row: (int(row.get("event_index", -1)), int(row.get("rank", 999999))))
    return dict(output)


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return {token for token in re.findall(r"[^\W_]+", normalized, re.UNICODE) if len(token) > 1}


def build_asr_candidate_evidence(
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    loader: Any,
    *,
    e5_results: dict[str, list[dict[str, Any]]] | None = None,
    max_spans: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """Map second-based lexical/E5 evidence onto frozen BCF1 frame identities."""

    baseline = grouped(baseline_rows)
    output: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        text = str(query.get("query") or query.get("question") or "")
        lexical = loader.retrieve_spans(text, max_spans=max_spans)
        combined = [*lexical, *(e5_results or {}).get(query_id, [])]
        best_by_video: dict[str, dict[str, Any]] = {}
        for span in combined:
            best_by_video.setdefault(str(span["video_id"]), dict(span))
        rows = []
        for baseline_row in baseline[query_id]:
            video_id = str(baseline_row["video_id"])
            if video_id not in best_by_video:
                continue
            span = best_by_video[video_id]
            rows.append(
                {
                    **baseline_row,
                    "rank": len(rows) + 1,
                    "source": "asr_external_v3",
                    "asr_span": span,
                    "asr_branch": span.get("branch", "LEXICAL"),
                }
            )
        output[query_id] = rows
    return output


def build_external_parquet_evidence(
    queries: list[dict[str, Any]],
    parquet_path: str | Path,
    modality: str,
    *,
    limit: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """Bounded deterministic lexical retrieval over accepted OCR/object corpora."""

    if modality not in {"ocr", "object"}:
        raise ValueError("external parquet modality must be ocr or object")
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - Kaggle/local dependency gate
        raise RuntimeError("PYARROW_REQUIRED_FOR_EXTERNAL_EVIDENCE") from error
    columns = (
        ["video_id", "frame_idx", "corrected_text", "combined_text", "mean_confidence"]
        if modality == "ocr"
        else ["video_id", "actual_frame_id", "search_text", "object_scores"]
    )
    rows = pq.read_table(Path(parquet_path), columns=columns).to_pylist()
    output: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        query_tokens = _tokens(str(query.get("query") or query.get("question") or ""))
        scored = []
        for raw in rows:
            text = (
                str(raw.get("corrected_text") or raw.get("combined_text") or "")
                if modality == "ocr"
                else str(raw.get("search_text") or "")
            )
            overlap = query_tokens.intersection(_tokens(text))
            if not overlap:
                continue
            confidence = float(raw.get("mean_confidence") or 0.0) if modality == "ocr" else 0.0
            score = len(overlap) + confidence * 0.01
            frame_id = int(raw.get("frame_idx") if modality == "ocr" else raw["actual_frame_id"])
            scored.append(
                (
                    -score,
                    str(raw["video_id"]),
                    frame_id,
                    {
                        "query_id": query_id,
                        "video_id": str(raw["video_id"]),
                        "frame_id": frame_id,
                        "source": f"{modality}_external_v3",
                        "text": text,
                        "matched_tokens": sorted(overlap),
                        "source_score": score,
                        "evidence_only": True,
                    },
                )
            )
        scored.sort(key=lambda item: item[:3])
        seen, selected = set(), []
        for _, _, _, row in scored:
            identity = (row["video_id"], row["frame_id"])
            if identity in seen:
                continue
            seen.add(identity)
            selected.append({**row, "rank": len(selected) + 1})
            if len(selected) == limit:
                break
        output[query_id] = selected
    return output


def build_xclip_event_evidence(
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    score_window: Any,
    *,
    candidates_per_event: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Score frozen TRAKE candidate windows with the official XCLIP adapter boundary."""

    baseline = grouped(baseline_rows)
    output: dict[str, list[dict[str, Any]]] = {}
    for query in (row for row in queries if row["task"] == "TRAKE"):
        query_id = str(query["query_id"])
        events = list(query["event_descriptions"])
        rows = []
        for event_index, event_text in enumerate(events):
            scored = []
            for baseline_row in baseline[query_id][:candidates_per_event]:
                frames = list(baseline_row["frame_ids"])
                frame_id = int(frames[event_index])
                result = score_window(str(event_text), str(baseline_row["video_id"]), frame_id)
                if not result or not result.get("finite"):
                    continue
                scored.append(
                    (-float(result["score"]), str(baseline_row["video_id"]), frame_id, result)
                )
            scored.sort(key=lambda item: item[:3])
            for event_rank, (_, video_id, frame_id, result) in enumerate(scored, 1):
                rows.append(
                    {
                        "query_id": query_id,
                        "event_index": event_index,
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "rank": event_rank,
                        "source": "xclip_official",
                        "xclip": result,
                    }
                )
        output[query_id] = rows
    return output


def _rank_and_fill(
    rows: list[dict[str, Any]], baseline: list[dict[str, Any]], *, limit: int = 100
) -> list[dict[str, Any]]:
    output, seen = [], set()
    for raw in [*rows, *baseline]:
        row = dict(raw)
        identity = default_key(row)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(row)
        if len(output) == limit:
            break
    return [{**row, "rank": rank} for rank, row in enumerate(output, 1)]


def prioritize_qa_sufficient(
    rows: list[dict[str, Any]], qwen_rows: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Put audited sufficient QA evidence first; unsupported candidates stay behind it."""

    sufficient = []
    for raw in qwen_rows:
        answer = " ".join(str(raw.get("answer", "")).split())
        if not raw.get("evidence_sufficient") or not answer or GENERIC_QA.fullmatch(answer):
            continue
        sufficient.append(
            {
                **raw,
                "answer": answer,
                "evidence_sufficient": True,
                "source": str(raw.get("source", "qwen_bounded_evidence")),
            }
        )
    unsupported = []
    for raw in rows:
        row = dict(raw)
        if not row.get("evidence_sufficient"):
            row["answer"] = "không đủ bằng chứng"
            row["evidence_sufficient"] = False
        unsupported.append(row)
    return _rank_and_fill([*sufficient, *unsupported], baseline)


def _candidate_arm(
    arm: str,
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    revision_provider: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available = {name for name in ("asr", "ocr", "action", "object", "qwen") if evidence.get(name)}
    full, safe, diagnostics = build_completion_arm(
        arm,
        queries,
        baseline_rows,
        evidence,
        available,
        revision_provider=revision_provider,
    )
    baseline, full_group, safe_group = grouped(baseline_rows), grouped(full), grouped(safe)
    output, safe_output = [], []
    for query in queries:
        query_id, task = query["query_id"], query["task"]
        if task == "QA":
            ranked = prioritize_qa_sufficient(
                full_group[query_id], evidence.get("qwen", {}).get(query_id, []), baseline[query_id]
            )
            # SAFE has deliberately no BCF1 prefix protection for QA.
            safe_ranked = list(ranked)
        else:
            ranked = _rank_and_fill(full_group[query_id], baseline[query_id])
            safe_ranked = _rank_and_fill(safe_group[query_id], baseline[query_id])
        output.extend({**row, "query_id": query_id, "system_variant": arm} for row in ranked)
        safe_output.extend(
            {**row, "query_id": query_id, "system_variant": "TRIAGEEG_SAFE"} for row in safe_ranked
        )
    return output, safe_output, diagnostics


def build_trial_candidates(
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    revision_provider: Any,
) -> dict[str, Any]:
    validate_trial_contract(queries)
    required = {"asr", "ocr", "action", "object", "qwen"}
    missing = sorted(name for name in required if not evidence.get(name))
    if missing:
        raise RuntimeError("TRIAL_MULTIMODAL_EVIDENCE_MISSING:" + ",".join(missing))
    m0, _, m0_diagnostics = _candidate_arm("M0_v11", queries, baseline_rows, evidence)
    m1, safe, m1_diagnostics = _candidate_arm(
        "M1_v11", queries, baseline_rows, evidence, revision_provider=revision_provider
    )
    candidates = {"TRIAGEEG_M0_FULL": m0, "TRIAGEEG_M1_FULL": m1, "TRIAGEEG_SAFE": safe}
    validation = {}
    for name, rows in candidates.items():
        summary, issues = validate_predictions(queries, rows)
        exact = len(rows) == 2400 and all(len(values) == 100 for values in grouped(rows).values())
        validation[name] = {**summary, "exact_100_per_query": exact, "issues": issues}
        if summary["status"] != "PASS" or not exact:
            raise RuntimeError(f"TRIAL_CANDIDATE_VALIDATION_FAILED:{name}")
    return {
        "candidates": candidates,
        "m0_diagnostics": m0_diagnostics,
        "m1_diagnostics": m1_diagnostics,
        "validation": validation,
    }


def qa_evidence_summary(
    queries: list[dict[str, Any]], candidates: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    output: dict[str, Any] = {"queries": {}, "hard_gate": "PASS"}
    for query_id in QA_IDS:
        per_arm = {}
        for arm, rows in candidates.items():
            selected = grouped(rows).get(query_id, [])
            sufficient = [row for row in selected if row.get("evidence_sufficient") is True]
            violations = [
                row["rank"]
                for row in selected
                if row.get("evidence_sufficient") is not True
                and any(
                    later.get("evidence_sufficient") is True
                    for later in selected[int(row["rank"]) :]
                )
            ]
            per_arm[arm] = {
                "top100_evidence_sufficient_count": len(sufficient),
                "unsupported_above_sufficient_ranks": violations,
                "top_answer": selected[0].get("answer") if selected else None,
            }
            if not sufficient or violations:
                output["hard_gate"] = "QA_BLOCK_SUBMISSION_2"
        output["queries"][query_id] = per_arm
    return output


def trake_graph_summary(
    queries: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    diagnostics: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    causal_fixture_pass: bool,
) -> dict[str, Any]:
    by_query = {row["query_id"]: row for row in diagnostics if row.get("graph")}
    m0, m1 = grouped(candidates["TRIAGEEG_M0_FULL"]), grouped(candidates["TRIAGEEG_M1_FULL"])
    rows, issues, changed = {}, [], 0
    for query in (row for row in queries if row["task"] == "TRAKE"):
        query_id = query["query_id"]
        graph = (by_query.get(query_id) or {}).get("graph") or {}
        routed_action_events = {
            int(row.get("event_index"))
            for row in (by_query.get(query_id) or {}).get("routing", [])
            if "action" in row.get("modalities", ())
        }
        evidence_action_events = {
            int(row["event_index"])
            for row in evidence.get("action", {}).get(query_id, [])
            if row.get("event_index") is not None
        }
        final_rows = m1.get(query_id, [])
        increasing = all(
            all(
                left < right
                for left, right in zip(
                    item.get("frame_ids", []), item.get("frame_ids", [])[1:], strict=False
                )
            )
            for item in final_rows
        )
        content_changed = semantic_content_hash(m0[query_id]) != semantic_content_hash(m1[query_id])
        changed += int(content_changed)
        active = (
            graph.get("query_event_count") == 4
            and graph.get("revision_count") == 1
            and (graph.get("revision") or {}).get("evidence_added", 0) > 0
            and graph.get("chain_candidates_added", 0) > 0
            and routed_action_events == {0, 1, 2, 3}
            and evidence_action_events == {0, 1, 2, 3}
            and increasing
        )
        if not active:
            issues.append(f"TRAKE_GRAPH_OR_ACTION_GATE_FAILED:{query_id}")
        rows[query_id] = {
            "query_event_count": graph.get("query_event_count"),
            "routed_action_event_indices": sorted(routed_action_events),
            "xclip_evidence_event_indices": sorted(evidence_action_events),
            "revision_count": graph.get("revision_count"),
            "revision_evidence_added": (graph.get("revision") or {}).get("evidence_added", 0),
            "graph_chains_consumed": graph.get("chain_candidates_added", 0),
            "all_final_frame_ids_strictly_increasing": increasing,
            "m1_content_differs_from_m0": content_changed,
        }
    if changed == 0 and not causal_fixture_pass:
        issues.append("GRAPH_NOT_EXERCISED_AND_CAUSAL_FIXTURE_FAILED")
    return {
        "status": "PASS" if not issues else "FAIL",
        "queries": rows,
        "changed_trake_query_count": changed,
        "causal_fixture_pass": causal_fixture_pass,
        "issues": issues,
    }


def candidate_comparison(
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    groups = {"BCF1": grouped(baseline_rows)} | {
        name: grouped(rows) for name, rows in candidates.items()
    }
    table, top1_changed, top5_changed = [], 0, 0
    for query in queries:
        query_id = query["query_id"]
        b0, m0, m1, safe = (
            groups[name][query_id]
            for name in ("BCF1", "TRIAGEEG_M0_FULL", "TRIAGEEG_M1_FULL", "TRIAGEEG_SAFE")
        )
        changed_m0 = b0[0]["video_id"] != m0[0]["video_id"]
        top1_changed += int(changed_m0)
        changed5 = {row["video_id"] for row in b0[:5]} != {row["video_id"] for row in m0[:5]}
        top5_changed += int(changed5)
        provenance = m1[0].get("source") or m1[0].get("fs1_source_ranks") or "b0_visual"

        def concentration(values: list[dict[str, Any]]) -> float:
            counts = Counter(str(item["video_id"]) for item in values[:10])
            return max(counts.values(), default=0) / max(min(len(values), 10), 1)

        table.append(
            {
                "query_id": query_id,
                "task": query["task"],
                "bcf1_top1_video": b0[0]["video_id"],
                "m0_top1_video": m0[0]["video_id"],
                "m1_top1_video": m1[0]["video_id"],
                "safe_top1_video": safe[0]["video_id"],
                "bcf1_vs_m0_changed": changed_m0,
                "m0_vs_m1_changed": default_key(m0[0]) != default_key(m1[0]),
                "dominant_new_evidence": provenance,
                "sanity_flag": "REVIEW" if changed_m0 else "STABLE",
                "m0_top10_video_concentration": concentration(m0),
                "m1_top10_video_concentration": concentration(m1),
                "safe_top10_video_concentration": concentration(safe),
            }
        )
    return {
        "rows": table,
        "top1_changed_vs_bcf1": top1_changed,
        "top5_set_changed_vs_bcf1": top5_changed,
    }


def asr_regression_warnings(
    candidates: dict[str, list[dict[str, Any]]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    warnings = []
    for query_id in ASR_SANITY_IDS:
        strong = evidence.get("asr", {}).get(query_id, [])[:1]
        if not strong:
            warnings.append({"query_id": query_id, "code": "ASR_SANITY_EVIDENCE_MISSING"})
            continue
        video_id = str(strong[0]["video_id"])
        for arm, rows in candidates.items():
            rank = next(
                (row["rank"] for row in grouped(rows)[query_id] if row["video_id"] == video_id),
                None,
            )
            if rank is None or rank > 20:
                warnings.append(
                    {
                        "query_id": query_id,
                        "arm": arm,
                        "code": "FUSION_REGRESSION_WARNING",
                        "asr_video_id": video_id,
                        "final_rank": rank,
                    }
                )
    return warnings


def _candidate_zip(
    root: Path, name: str, rows: list[dict[str, Any]], validation: dict[str, Any]
) -> Path:
    source = root / "predictions" / f"{name}.jsonl"
    write_jsonl(source, rows)
    manifest = root / "predictions" / f"{name}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate": name,
                "prediction_sha256": sha256_file(source),
                "validation": validation,
                "gt_opened": False,
                "submission_uploaded": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    target = root / f"{name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, source.name)
        archive.write(manifest, manifest.name)
    return target


def write_blocked_artifacts(
    root: str | Path, blockers: list[str], provenance: dict[str, Any]
) -> Path:
    """Materialize the required report skeleton without fabricating candidates."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    empty = {"status": "NOT_RUN_FAIL_CLOSED", "blockers": blockers}
    for name in (
        "trial_p1_candidate_comparison.json",
        "qa_evidence_summary.json",
        "trake_graph_summary.json",
        "cross_l21_pre_gt_hashes.json",
    ):
        (output / name).write_text(json.dumps(empty, indent=2) + "\n", encoding="utf-8")
    (output / "trial_p1_candidate_comparison.csv").write_text(
        "status,blocker\nNOT_RUN_FAIL_CLOSED,MANDATORY_INPUT_MISSING\n", encoding="utf-8"
    )
    (output / "trial_p1_human_review_packet.md").write_text(
        "# Trial P1 Multimodal Dry Run\n\nStatus: NOT RUN (fail-closed).\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n",
        encoding="utf-8",
    )
    (output / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output / "SUBMISSION_2_DECISION.md").write_text(
        "# Submission #2 Decision\n\n`DO_NOT_SUBMIT_2_YET`\n\n"
        "The mandatory Trial dry-run was not executed because fail-closed preflight found:\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n\nNo GT was opened and no submission was uploaded.\n",
        encoding="utf-8",
    )
    return output


def write_dryrun_artifacts(
    root: str | Path,
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    result: dict[str, Any],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    causal_fixture_pass: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    qa = qa_evidence_summary(queries, candidates)
    trake = trake_graph_summary(
        queries,
        candidates,
        result["m1_diagnostics"],
        evidence,
        causal_fixture_pass=causal_fixture_pass,
    )
    comparison = candidate_comparison(queries, baseline_rows, candidates)
    comparison["strong_asr_or_ocr_evidence_query_count"] = sum(
        bool(evidence.get("asr", {}).get(query["query_id"]))
        or bool(evidence.get("ocr", {}).get(query["query_id"]))
        for query in queries
    )
    comparison["trake_graph_activation_count"] = sum(
        row.get("revision_count") == 1 and row.get("graph_chains_consumed", 0) > 0
        for row in trake["queries"].values()
    )
    comparison["qa_sufficient_counts"] = {
        query_id: {arm: value["top100_evidence_sufficient_count"] for arm, value in arms.items()}
        for query_id, arms in qa["queries"].items()
    }
    warnings = asr_regression_warnings(candidates, evidence)
    hard_pass = qa["hard_gate"] == "PASS" and trake["status"] == "PASS"
    recommendation = "DO_NOT_SUBMIT_2_YET"
    blockers = []
    if qa["hard_gate"] != "PASS":
        blockers.append(qa["hard_gate"])
    if trake["status"] != "PASS":
        blockers.extend(trake["issues"])
    if not hard_pass:
        recommendation = "DO_NOT_SUBMIT_2_YET"
    # Human review remains mandatory even when structural gates pass.
    if hard_pass:
        blockers.append("HUMAN_REVIEW_AND_CROSS_L21_EVALUATION_PENDING")
    (output / "qa_evidence_summary.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "trake_graph_summary.json").write_text(
        json.dumps(trake, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "trial_p1_candidate_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    with (output / "trial_p1_candidate_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison["rows"][0]))
        writer.writeheader()
        writer.writerows(comparison["rows"])
    (output / "cross_l21_pre_gt_hashes.json").write_text(
        json.dumps(
            {
                "status": "NOT_REACHED" if not hard_pass else "PENDING_SEPARATE_PRE_GT_RUN",
                "gt_opened": False,
                "hashes": {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    review = ["# Trial P1 Human Review Packet", "", "No ground truth was opened.", ""]
    evidence_snapshot = []
    for row in comparison["rows"]:
        query_id = row["query_id"]
        review.extend(
            [
                f"## {query_id} ({row['task']})",
                "",
                "Top1 videos: "
                f"BCF1={row['bcf1_top1_video']}, M0={row['m0_top1_video']}, "
                f"M1={row['m1_top1_video']}, SAFE={row['safe_top1_video']}.",
                "",
            ]
        )
        for arm in ("TRIAGEEG_M0_FULL", "TRIAGEEG_M1_FULL", "TRIAGEEG_SAFE"):
            top = grouped(candidates[arm])[query_id][:10]
            preview = "; ".join(
                f"#{item['rank']} {item['video_id']}:"
                f"{item.get('frame_id', item.get('frame_ids'))} "
                f"[{item.get('source', 'fused')}]"
                for item in top
            )
            review.append(f"- {arm}: {preview}")
        for modality in ("asr", "ocr", "action", "object", "qwen"):
            modality_rows = evidence.get(modality, {}).get(query_id, [])[:10]
            evidence_snapshot.append(
                {
                    "query_id": query_id,
                    "modality": modality,
                    "top_evidence": modality_rows,
                }
            )
            preview = "; ".join(
                f"{item.get('video_id')}:{item.get('frame_id', item.get('start_seconds'))}"
                for item in modality_rows[:3]
            )
            review.append(f"- evidence/{modality}: {preview or 'NONE'}")
        review.append("")
    review.extend(
        [
            "## ASR sanity warnings",
            "",
            *(
                f"- {json.dumps(row, ensure_ascii=False)}"
                for row in warnings or [{"status": "NONE"}]
            ),
        ]
    )
    (output / "trial_p1_human_review_packet.md").write_text(
        "\n".join(review) + "\n", encoding="utf-8"
    )
    write_jsonl(output / "trial_p1_evidence_top10.jsonl", evidence_snapshot)
    zips = {
        name: str(_candidate_zip(output, name, rows, result["validation"][name]))
        for name, rows in candidates.items()
    }
    decision = (
        "# Submission #2 Decision\n\n`DO_NOT_SUBMIT_2_YET`\n\n"
        "Structural Trial artifacts were generated, but upload is prohibited in this task.\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n\nNo GT was opened and no submission was uploaded.\n"
    )
    (output / "SUBMISSION_2_DECISION.md").write_text(decision, encoding="utf-8")
    (output / "run_provenance.json").write_text(
        json.dumps(
            provenance | {"recommendation": recommendation, "candidate_zips": zips},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "recommendation": recommendation,
        "hard_trial_gates_pass": hard_pass,
        "candidate_zips": zips,
        "qa": qa,
        "trake": trake,
        "comparison": comparison,
        "asr_warnings": warnings,
    }
