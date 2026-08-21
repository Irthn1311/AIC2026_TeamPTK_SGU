"""Trial P1 R4 surgical repair with no GT and no production-policy mutation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aic2026_eval.validation import validate_predictions
from triage_eg.fs1.fusion import default_key
from triage_eg.fs1_v11.pipeline import grouped, semantic_content_hash
from triage_eg.submission.aic26_prelim import create_submission_zip, validate_submission_zip

from .r2_policy import _complete_ranked_rows
from .r3_policy import (
    MEANINGFUL_CLASSES,
    augment_qa_context_r3,
    build_r3_candidates,
    evaluate_context_relevance,
    match_r3_anchors,
    write_jsonl,
)

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_QUOTE_SPLIT = re.compile(r"(?:\r?\n|\s*[;|]\s*)")
_R4_TITLE_WRAPPER = re.compile(
    r"^(?:món\s+ăn\s+có\s+tên(?:\s+gọi)?\s+là|tên\s+món(?:\s+ăn)?\s+là|"
    r"tên\s+công\s+thức\s+là|công\s+thức(?:\s+này)?\s+là|"
    r"công\s+thức\s+có\s+tên\s+là)\s*[:\-]?\s*",
    re.I,
)
_ADMIN_LEVELS = {
    "xa",
    "phuong",
    "thi tran",
    "huyen",
    "quan",
    "tinh",
    "thanh pho",
}


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char)).replace("đ", "d")


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(_fold(value)))


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[index : index + len(needle)] == needle for index in range(len(haystack)))


def _identity(row: dict[str, Any], task: str) -> tuple[Any, ...]:
    if task == "TRAKE":
        return str(row["video_id"]), tuple(map(int, row["frame_ids"]))
    return str(row["video_id"]), int(row["frame_id"])


def _support_decision(
    answer: str,
    spans: list[str],
    *,
    answer_type: str,
    requested_lines: int | None,
) -> dict[str, Any]:
    answer_tokens = _tokens(answer)
    span_tokens = [_tokens(value) for value in spans]
    exact_span_indexes = [
        index for index, value in enumerate(span_tokens) if _contains(value, answer_tokens)
    ]
    joined = tuple(token for value in span_tokens for token in value)
    faithful = bool(answer_tokens and (exact_span_indexes or _contains(joined, answer_tokens)))
    reason = "ANSWER_EXACT_NORMALIZED_SUBSPAN" if faithful else "ANSWER_TEXT_ABSENT_FROM_SUPPORT"
    line_decisions: list[dict[str, Any]] = []
    if answer_type == "QUOTE_OR_VISIBLE_TEXT" and requested_lines:
        answer_lines = [value.strip() for value in _QUOTE_SPLIT.split(answer) if value.strip()]
        for line in answer_lines:
            tokens = _tokens(line)
            line_decisions.append(
                {
                    "line": line,
                    "supported": any(_contains(value, tokens) for value in span_tokens),
                }
            )
        faithful = bool(
            len(answer_lines) >= int(requested_lines)
            and len(spans) >= int(requested_lines)
            and all(row["supported"] for row in line_decisions[: int(requested_lines)])
        )
        reason = (
            "ORDERED_QUOTE_LINES_EXACTLY_SUPPORTED"
            if faithful
            else "REQUESTED_QUOTE_LINES_NOT_EXACTLY_SUPPORTED"
        )
    return {
        "normalized_answer": " ".join(answer_tokens),
        "normalized_supporting_spans": [" ".join(value) for value in span_tokens],
        "answer_support_pass": faithful,
        "answer_support_reason": reason,
        "exact_supporting_span_indexes": exact_span_indexes,
        "quote_line_decisions": line_decisions,
    }


def _location_granularity_pass(
    answer: str, cited_text: list[str], requested: str | None
) -> tuple[bool, str]:
    if not requested:
        return True, "GENERIC_LOCATION_REQUEST"
    requested_tokens = _tokens(requested)
    answer_tokens = _tokens(answer)
    linked_answer_tokens = (
        answer_tokens[len(requested_tokens) :]
        if answer_tokens[: len(requested_tokens)] == requested_tokens
        else answer_tokens
    )
    admin_sequences = [tuple(level.split()) for level in _ADMIN_LEVELS]
    for text in cited_text:
        tokens = _tokens(text)
        for index in range(len(tokens)):
            if tokens[index : index + len(requested_tokens)] != requested_tokens:
                continue
            tail = tokens[index + len(requested_tokens) :]
            boundary = len(tail)
            for offset in range(1, len(tail)):
                if any(tail[offset : offset + len(level)] == level for level in admin_sequences):
                    boundary = offset
                    break
            linked_phrase = tail[:boundary]
            if _contains(
                linked_phrase[: max(len(linked_answer_tokens) + 3, 6)],
                linked_answer_tokens,
            ):
                return True, "REQUESTED_GRANULARITY_LINKED_TO_ANSWER"
    return False, "LOCATION_REQUESTED_GRANULARITY_NOT_LINKED_TO_ANSWER"


def verify_answer_r4(
    extraction: dict[str, Any] | None,
    answer_type: str,
    evidence_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    grounding_plausibility: float,
) -> dict[str, Any]:
    """Apply independent parse, claim, context, support, and type gates."""

    kind = str(answer_type).upper()
    if extraction is None:
        return {
            "answer": "",
            "unstripped_answer": "",
            "qwen_parse_pass": False,
            "qwen_claims_sufficient": False,
            "context_relevance_pass": False,
            "answer_support_pass": False,
            "answer_type_pass": False,
            "final_evidence_sufficient": False,
            "answer_support_reason": "QWEN_EXTRACTION_PARSE_FAILED",
            "answer_type_reasons": ["QWEN_EXTRACTION_PARSE_FAILED"],
            "grounding_plausibility": grounding_plausibility,
        }
    answer = str(extraction.get("answer", "")).strip()
    sources = extraction.get("supporting_source_ids")
    spans = extraction.get("supporting_spans")
    parse_pass = bool(
        answer
        and len(answer) <= 100
        and isinstance(sources, list)
        and all(isinstance(value, str) for value in sources)
        and len(sources) <= 3
        and isinstance(spans, list)
        and all(isinstance(value, str) and len(value) <= 160 for value in spans)
        and len(spans) <= 3
        and isinstance(extraction.get("evidence_sufficient"), bool)
    )
    source_map = {str(row.get("source_id")): row for row in evidence_rows if row.get("source_id")}
    cited_ids = [str(value) for value in sources] if isinstance(sources, list) else []
    cited_rows = [source_map[value] for value in cited_ids if value in source_map]
    cited_text = [str(row.get("text", "")) for row in cited_rows]
    supplied_spans = [str(value) for value in spans] if isinstance(spans, list) else []
    claims = bool(
        parse_pass
        and extraction.get("evidence_sufficient") is True
        and cited_ids
        and len(cited_rows) == len(cited_ids)
        and supplied_spans
        and all(
            any(_contains(_tokens(text), _tokens(span)) for text in cited_text)
            for span in supplied_spans
        )
    )
    context = evaluate_context_relevance(profile, evidence_rows)
    support = _support_decision(
        answer,
        supplied_spans,
        answer_type=kind,
        requested_lines=profile.get("requested_quote_line_count"),
    )
    type_pass, type_reasons = True, []
    canonical, rule = answer, None
    if kind == "LOCATION_NAME":
        type_pass, reason = _location_granularity_pass(
            answer, cited_text, profile.get("location_granularity")
        )
        if not type_pass:
            type_reasons.append(reason)
    elif kind == "QUOTE_OR_VISIBLE_TEXT":
        if not support["answer_support_pass"]:
            type_pass = False
            type_reasons.append("QUOTE_NOT_TEXT_PRESERVING")
    elif kind == "TITLE":
        compact = " ".join(answer.split())
        canonical = _R4_TITLE_WRAPPER.sub("", compact).strip(" :-–—") or compact
        changed = canonical != compact
        rule = "STRIP_GENERIC_TITLE_WRAPPER" if changed else "NO_WRAPPER_STRIPPED"
        available_meaningful = {
            tuple(anchor.get("folded_tokens", []))
            for anchor in profile.get("anchors", [])
            if anchor.get("anchor_class") in MEANINGFUL_CLASSES
        }
        bounded_meaningful = {
            tuple(match.get("folded_tokens", []))
            for row in evidence_rows
            for match in match_r3_anchors(str(row.get("text", "")), profile)
            if match.get("anchor_class") in MEANINGFUL_CLASSES
        }
        if len(available_meaningful) >= 2 and len(bounded_meaningful) < 2:
            type_pass = False
            type_reasons.append("TITLE_COMPOUND_CLUE_NOT_SUPPORTED")
    if not canonical or len(canonical) > 100:
        type_pass = False
        type_reasons.append("CANONICAL_ANSWER_INVALID")
    final = bool(
        parse_pass
        and claims
        and context["context_relevant"]
        and support["answer_support_pass"]
        and type_pass
    )
    return {
        **extraction,
        **context,
        **support,
        "video_id": extraction.get("video_id"),
        "frame_id": extraction.get("frame_id"),
        "grounding_rank": extraction.get("grounding_rank"),
        "grounding_plausibility": grounding_plausibility,
        "unstripped_answer": answer,
        "answer": canonical,
        "canonical_answer": canonical,
        "canonicalization_rule": rule,
        "verified_supporting_source_ids": cited_ids if len(cited_rows) == len(cited_ids) else [],
        "corroborating_modalities": sorted({str(row.get("modality")) for row in cited_rows}),
        "corroborating_source_count": len(cited_rows),
        "qwen_parse_pass": parse_pass,
        "qwen_claims_sufficient": claims,
        "context_relevance_pass": bool(context["context_relevant"]),
        "answer_type_pass": type_pass,
        "answer_type_reasons": type_reasons,
        "final_evidence_sufficient": final,
    }


def needs_local_ocr_r4(
    query: dict[str, Any], profile: dict[str, Any], evidence_rows: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Decide whether bounded local OCR is needed without using expected answers or GT."""

    kind = str(query.get("answer_type", "OTHER")).upper()
    text_rows = []
    for row in evidence_rows:
        text = str(row.get("text") or (row.get("asr_span") or {}).get("text", "")).strip()
        if text:
            text_rows.append(text)
    if kind == "LOCATION_NAME":
        granularity = profile.get("location_granularity")
        if granularity and any(
            _contains(_tokens(text), _tokens(str(granularity))) for text in text_rows
        ):
            return True, "TEXT_HEAVY_LOCATION_BOUNDED_LOCAL_CONFIRMATION"
        return True, "LOCATION_GRANULARITY_MISSING_EXTERNAL"
    if kind == "QUOTE_OR_VISIBLE_TEXT":
        required = int(profile.get("requested_quote_line_count") or 1)
        contextual = [text for text in text_rows if match_r3_anchors(text, profile)]
        return True, (
            "TEXT_HEAVY_QUOTE_BOUNDED_LOCAL_CONFIRMATION"
            if len(contextual) >= required
            else "QUOTE_LINES_MISSING_EXTERNAL"
        )
    if kind == "TITLE":
        strongest = max(
            (
                len(
                    {
                        tuple(match.get("folded_tokens", []))
                        for match in match_r3_anchors(text, profile)
                        if match.get("anchor_class") in MEANINGFUL_CLASSES
                    }
                )
                for text in text_rows
            ),
            default=0,
        )
        return True, (
            "TEXT_HEAVY_TITLE_BOUNDED_LOCAL_CONFIRMATION"
            if strongest >= 2
            else "TITLE_COMPOUND_CLUE_MISSING_EXTERNAL"
        )
    return False, "LOCAL_OCR_NOT_REQUIRED_FOR_ANSWER_TYPE"


def build_bounded_local_ocr_rescue(
    queries: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    rescue_diagnostics: list[dict[str, Any]],
    ocr_provider: Callable[[str, list[int]], list[dict[str, Any]]],
    *,
    max_frames_per_query: int = 12,
    neighbor_offsets: tuple[int, ...] = (-48, -24, 0, 24, 48),
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Run local OCR only on query-local shortlisted candidate and neighbor frames."""

    diagnostic_map = {str(row["query_id"]): row for row in rescue_diagnostics}
    output: dict[str, list[dict[str, Any]]] = {}
    audit = []
    for query in (row for row in queries if str(row["task"]).upper() == "QA"):
        query_id = str(query["query_id"])
        selected_videos = list(diagnostic_map.get(query_id, {}).get("selected_context_videos", []))
        bounded_rows = [
            row
            for modality in ("asr", "ocr")
            for row in evidence.get(modality, {}).get(query_id, [])
            if str(row.get("video_id")) in set(selected_videos)
        ]
        context_videos = []
        for video_id in selected_videos:
            rows = [
                {
                    "source_id": f"context:{index}",
                    "modality": "bounded_text",
                    "text": str(row.get("text") or (row.get("asr_span") or {}).get("text", "")),
                }
                for index, row in enumerate(bounded_rows)
                if str(row.get("video_id")) == video_id
            ]
            if evaluate_context_relevance(profiles[query_id], rows)["context_relevant"]:
                context_videos.append(video_id)
        bounded_rows = [
            row for row in bounded_rows if str(row.get("video_id")) in set(context_videos)
        ]
        needed, reason = needs_local_ocr_r4(query, profiles[query_id], bounded_rows)
        frames_by_video: dict[str, list[int]] = {}
        if needed:
            ordered = sorted(
                bounded_rows,
                key=lambda row: (
                    0
                    if match_r3_anchors(
                        str(row.get("text") or (row.get("asr_span") or {}).get("text", "")),
                        profiles[query_id],
                    )
                    else 1,
                    int(row.get("rank", 10**9)),
                    str(row.get("video_id")),
                    int(row.get("frame_id", 0)),
                ),
            )
            for row in ordered:
                video_id, center = str(row["video_id"]), int(row["frame_id"])
                values = frames_by_video.setdefault(video_id, [])
                for offset in neighbor_offsets:
                    frame = max(0, center + offset)
                    if frame not in values:
                        values.append(frame)
                    if sum(map(len, frames_by_video.values())) >= max_frames_per_query:
                        break
                if sum(map(len, frames_by_video.values())) >= max_frames_per_query:
                    break
        rows, failures = [], []
        for video_id, frame_ids in frames_by_video.items():
            try:
                for raw in ocr_provider(video_id, frame_ids):
                    text = str(raw.get("text", "")).strip()
                    if not text:
                        continue
                    rows.append(
                        {
                            "query_id": query_id,
                            "video_id": video_id,
                            "frame_id": int(raw["frame_id"]),
                            "rank": len(rows) + 1,
                            "source": "local_ocr_r4_bounded_candidate_neighbor",
                            "text": text,
                            "source_confidence": raw.get("confidence"),
                            "local_ocr_engine": raw.get("engine"),
                        }
                    )
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(f"{video_id}:{type(error).__name__}:{error}")
        output[query_id] = rows
        audit.append(
            {
                "query_id": query_id,
                "local_ocr_required": needed,
                "reason": reason,
                "shortlisted_videos": selected_videos,
                "context_relevant_videos": context_videos,
                "requested_frame_count": sum(map(len, frames_by_video.values())),
                "local_ocr_row_count": len(rows),
                "failures": failures,
                "bounded": True,
                "corpus_job_launched": False,
                "gt_used": False,
            }
        )
    return output, audit


def build_r4_candidates(
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    tiered_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    verified_qa: dict[str, list[dict[str, Any]]],
    revision_provider: Any,
    *,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Carry R3 forward and surgically enforce coordinate-exact SAFE protection."""

    r3 = build_r3_candidates(
        queries,
        baseline_rows,
        tiered_evidence,
        verified_qa,
        revision_provider,
        inventory=inventory,
    )
    baseline = grouped(baseline_rows)
    r3_groups = {name: grouped(rows) for name, rows in r3["candidates"].items()}
    candidates: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("M0_R4", "M1_R4", "SAFE_R4")
    }
    safe_audit = []
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"]).upper()
        for old, new in (("M0_R3", "M0_R4"), ("M1_R3", "M1_R4")):
            candidates[new].extend(
                {**row, "system_variant": new} for row in r3_groups[old][query_id]
            )
        if task in {"KIS", "TRAKE"}:
            frozen = baseline[query_id][:5]
            protected = {_identity(row, task) for row in frozen}
            tail = [
                row
                for row in r3_groups["SAFE_R3"][query_id]
                if _identity(row, task) not in protected
            ]
            safe = _complete_ranked_rows(
                [
                    *({**row, "system_variant": "SAFE_R4"} for row in frozen),
                    *({**row, "system_variant": "SAFE_R4"} for row in tail),
                ],
                ({**row, "system_variant": "SAFE_R4"} for row in baseline[query_id]),
                query_id=query_id,
                variant="SAFE_R4",
            )
            expected = [_identity(row, task) for row in frozen]
            actual = [_identity(row, task) for row in safe[:5]]
            passed = actual == expected
            safe_audit.append(
                {
                    "query_id": query_id,
                    "task": task,
                    "protection_policy": "NO_TOP5_OVERRIDE",
                    "expected_top5": expected,
                    "actual_top5": actual,
                    "exact_tuple_count": sum(
                        left == right for left, right in zip(expected, actual, strict=True)
                    ),
                    "pass": passed,
                }
            )
            if not passed:
                raise RuntimeError(f"SAFE_R4_TOP5_TUPLE_GATE_FAILED:{query_id}")
        else:
            safe = [{**row, "system_variant": "SAFE_R4"} for row in r3_groups["SAFE_R3"][query_id]]
        candidates["SAFE_R4"].extend(safe)
    validation = {}
    for name, rows in candidates.items():
        summary, issues = validate_predictions(queries, rows, inventory=inventory)
        counts = {key: len(values) for key, values in grouped(rows).items()}
        exact = len(queries) == 24 and len(rows) == 2400 and set(counts.values()) == {100}
        validation[name] = {
            **summary,
            "exact_100_per_query": exact,
            "per_query_counts": counts,
            "issues": issues,
        }
        if summary["status"] != "PASS" or not exact:
            raise RuntimeError(f"TRIAL_R4_CANDIDATE_VALIDATION_FAILED:{name}")
    candidate_groups = {name: grouped(rows) for name, rows in candidates.items()}
    trake_checks = {}
    for query in (row for row in queries if str(row["task"]).upper() == "TRAKE"):
        query_id = str(query["query_id"])
        arms = {}
        for name, values in candidate_groups.items():
            rows = values[query_id]
            arms[name] = {
                "event_count_correct": all(
                    len(row.get("frame_ids", [])) == int(query["event_count"]) for row in rows
                ),
                "strictly_increasing": all(
                    all(
                        left < right
                        for left, right in zip(row["frame_ids"], row["frame_ids"][1:], strict=False)
                    )
                    for row in rows
                ),
            }
        graph_pass = bool(r3["trake_checks"][query_id]["graph_causal_gate_pass"])
        if not graph_pass or not all(
            row["event_count_correct"] and row["strictly_increasing"] for row in arms.values()
        ):
            raise RuntimeError(f"TRIAL_R4_TRAKE_GATE_FAILED:{query_id}")
        trake_checks[query_id] = {
            "arms": arms,
            "graph_causal_gate_pass": graph_pass,
        }
    return {
        **r3,
        "candidates": candidates,
        "validation": validation,
        "safe_top5_audit": safe_audit,
        "strong_asr_audit": [
            {**row, "policy_version": "R4_CARRIED_FROM_R3"} for row in r3["strong_asr_audit"]
        ],
        "trake_checks": trake_checks,
    }


def _parse_summary(qa_extractions: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [row for row in qa_extractions if isinstance(row.get("audit"), dict)]
    passed = sum(bool(row["audit"].get("qwen_parse_pass")) for row in attempts)
    rate = passed / len(attempts) if attempts else 0.0
    return {
        "attempt_count": len(attempts),
        "parse_pass_count": passed,
        "parse_failure_count": len(attempts) - passed,
        "parse_success_rate": rate,
        "threshold": 0.8,
        "pass": bool(attempts and rate >= 0.8),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_r4_artifacts(
    root: str | Path,
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    result: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    qa_extractions: list[dict[str, Any]],
    qa_verifications: list[dict[str, Any]],
    local_ocr_audit: list[dict[str, Any]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    groups = {name: grouped(rows) for name, rows in candidates.items()}
    baseline = grouped(baseline_rows)
    parse_summary = _parse_summary(qa_extractions)
    qa_readiness = {}
    for query in (row for row in queries if str(row["task"]).upper() == "QA"):
        query_id = str(query["query_id"])
        attempts = [row for row in qa_verifications if row.get("query_id") == query_id]
        sufficient = [row for row in attempts if row.get("final_evidence_sufficient") is True]
        gates = {
            name: all(bool(row.get(name)) for row in sufficient) if sufficient else False
            for name in (
                "qwen_parse_pass",
                "qwen_claims_sufficient",
                "context_relevance_pass",
                "answer_support_pass",
                "answer_type_pass",
            )
        }
        qa_readiness[query_id] = {
            "attempt_count": len(attempts),
            "verified_sufficient_count": len(sufficient),
            "all_sufficient_rows_pass_independent_gates": all(gates.values()),
            "gates": gates,
            "status": "PASS" if sufficient and all(gates.values()) else "FAIL",
            "sufficient_answers": sufficient,
        }
    structural = bool(
        len(queries) == 24
        and all(
            row["status"] == "PASS" and row["exact_100_per_query"]
            for row in result["validation"].values()
        )
        and all(row["graph_causal_gate_pass"] for row in result["trake_checks"].values())
    )
    hard_gates = {
        "official_query_count_24": len(queries) == 24,
        "structural_validator_inventory_trake_graph": structural,
        "safe_exact_bcf1_top5_all_protected": all(row["pass"] for row in result["safe_top5_audit"]),
        "qualified_strong_asr_drop_count_zero": all(
            row.get("inclusion_status") != "DROPPED" for row in result["strong_asr_audit"]
        ),
        "all_three_qa_have_sufficient_answer": len(qa_readiness) == 3
        and all(row["verified_sufficient_count"] > 0 for row in qa_readiness.values()),
        "all_sufficient_qa_pass_independent_gates": bool(qa_readiness)
        and all(row["all_sufficient_rows_pass_independent_gates"] for row in qa_readiness.values()),
        "qwen_parse_success_at_least_80_percent": parse_summary["pass"],
        "malformed_qwen_never_sufficient": all(
            row.get("qwen_parse_pass") or not row.get("final_evidence_sufficient")
            for row in qa_verifications
        ),
        "no_systematic_runtime_failure": int(provenance.get("runtime_candidate_failure_count", 0))
        == 0,
        "gt_not_opened": provenance.get("gt_opened") is False,
        "submission_not_uploaded": provenance.get("submission_uploaded") is False,
        "whisper_not_run": provenance.get("whisper_run") is False,
    }
    hard_pass = all(hard_gates.values())
    recommendation = "SUBMIT_2_SAFE_R4" if hard_pass else "DO_NOT_SUBMIT_2_YET"
    write_jsonl(output / "safe_r4_top5_exact_audit.jsonl", result["safe_top5_audit"])
    write_jsonl(output / "strong_asr_r4_inclusion_audit.jsonl", result["strong_asr_audit"])
    write_jsonl(output / "qa_r4_extractions.jsonl", qa_extractions)
    write_jsonl(output / "qa_r4_context_relevance.jsonl", qa_verifications)
    write_jsonl(output / "qa_r4_answer_support.jsonl", qa_verifications)
    write_jsonl(output / "qa_r4_semantic_verifier.jsonl", qa_verifications)
    write_jsonl(output / "qa_r4_local_ocr_audit.jsonl", local_ocr_audit)
    (output / "qa_r4_readiness.json").write_text(
        json.dumps(qa_readiness, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output / "qwen_r4_parse_summary.json").write_text(
        json.dumps(parse_summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "trake_r4_graph_summary.json").write_text(
        json.dumps(result["trake_checks"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "asset_hashes.json").write_text(
        json.dumps(provenance.get("asset_hashes", {}), indent=2) + "\n", encoding="utf-8"
    )
    comparison = []
    for query in queries:
        query_id = str(query["query_id"])
        strong = next(
            (row for row in result["strong_asr_audit"] if row["query_id"] == query_id), {}
        )
        comparison.append(
            {
                "query_id": query_id,
                "task": query["task"],
                "bcf1_top1": default_key(baseline[query_id][0]),
                "m0_r4_top1": default_key(groups["M0_R4"][query_id][0]),
                "m1_r4_top1": default_key(groups["M1_R4"][query_id][0]),
                "safe_r4_top1": default_key(groups["SAFE_R4"][query_id][0]),
                "strong_asr_best_video": strong.get("best_strong_asr_video"),
                "strong_asr_best_rank": strong.get("final_best_rank"),
                "evidence_tier": groups["M0_R4"][query_id][0].get("evidence_tier"),
                "warnings": []
                if qa_readiness.get(query_id, {"status": "PASS"})["status"] == "PASS"
                else ["QA_R4_NOT_READY"],
                "m0_r4_top10": groups["M0_R4"][query_id][:10],
                "m1_r4_top10": groups["M1_R4"][query_id][:10],
                "safe_r4_top10": groups["SAFE_R4"][query_id][:10],
            }
        )
    (output / "trial_p1_r4_candidate_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (output / "trial_p1_r4_candidate_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    review = [
        "# Trial P1 R4 Human Review",
        "",
        "No GT was opened. No submission was uploaded.",
        "",
        "## All 24 queries",
        "",
        "| Query | Task | BCF1 | M0 | M1 | SAFE | Strong ASR | Tier | Warnings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison:
        top_summary = (
            f"| {row['query_id']} | {row['task']} | `{row['bcf1_top1']}` | "
            f"`{row['m0_r4_top1']}` | `{row['m1_r4_top1']}` | "
            f"`{row['safe_r4_top1']}` | "
            f"`{row['strong_asr_best_video']}:{row['strong_asr_best_rank']}` | "
            f"`{row['evidence_tier']}` | `{row['warnings'] or 'NONE'}` |"
        )
        review.append(top_summary)
    review.extend(
        [
            "",
            "## QA hypotheses",
            "",
            "| Query | Video/frame | Context | Parse | Claims | Raw | Canonical | "
            "Sources | Spans | Support | Type | Final |",
            "|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|",
        ]
    )

    def safe_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")[:240]

    for row in qa_verifications:
        review.append(
            f"| {row.get('query_id')} | {row.get('video_id')}:{row.get('frame_id')} | "
            f"{bool(row.get('context_relevance_pass'))} | "
            f"{bool(row.get('qwen_parse_pass'))} | "
            f"{bool(row.get('qwen_claims_sufficient'))} | "
            f"{safe_cell(row.get('unstripped_answer', ''))} | "
            f"{safe_cell(row.get('answer', ''))} | "
            f"{safe_cell(row.get('supporting_source_ids', []))} | "
            f"{safe_cell(row.get('supporting_spans', []))} | "
            f"{bool(row.get('answer_support_pass'))} | "
            f"{bool(row.get('answer_type_pass'))} | "
            f"{bool(row.get('final_evidence_sufficient'))} |"
        )
    review.extend(["", "## Final-sufficient QA answers", "", f"`{qa_readiness}`", ""])
    (output / "trial_p1_r4_human_review.md").write_text("\n".join(review), encoding="utf-8")
    for name, rows in candidates.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    oj = {}
    if hard_pass:
        for name, rows in candidates.items():
            path = create_submission_zip(queries, rows, output / f"trial_p1_{name}_submission.zip")
            oj[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "validation": validate_submission_zip(path, queries),
            }
    decision = {
        "recommendation": recommendation,
        "hard_automated_gates_pass": hard_pass,
        "hard_gates": hard_gates,
        "qa_readiness": qa_readiness,
        "qwen_parse_summary": parse_summary,
        "oj_ready_submissions": oj,
        "gt_opened": False,
        "submission_uploaded": False,
    }
    (output / "SUBMISSION_2_R4_DECISION.md").write_text(
        "# Submission #2 R4 Decision\n\n"
        f"`{recommendation}`\n\n"
        f"Hard gates: `{json.dumps(hard_gates, sort_keys=True)}`\n\n"
        "No upload was performed. STOP after Trial R4.\n",
        encoding="utf-8",
    )
    (output / "run_provenance.json").write_text(
        json.dumps(
            provenance
            | {
                "decision": decision,
                "candidate_hashes": {
                    name: semantic_content_hash(rows) for name, rows in candidates.items()
                },
                "local_ocr_audit": local_ocr_audit,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return decision


__all__ = [
    "augment_qa_context_r3",
    "build_bounded_local_ocr_rescue",
    "build_r4_candidates",
    "needs_local_ocr_r4",
    "verify_answer_r4",
    "write_r4_artifacts",
]
