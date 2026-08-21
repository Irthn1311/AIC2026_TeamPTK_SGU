"""Conservative rank-level fusion and evidence-only QA for blind Prelim-1."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from .packet import SOURCE_SYSTEM, CatalogResolver

RRF_K = 60
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_LOCATION = re.compile(
    r"\b(?:đèo|xã|phường|thị trấn|huyện|quận|tỉnh|thành phố)\s+"
    r"([A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*(?:\s+[A-ZÀ-ỸĐ][\wÀ-ỹĐđ-]*){0,5})",
    re.UNICODE,
)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value).casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[^\W_]+", _fold(value), re.UNICODE) if len(token) >= 2}


def _coordinate(row: dict[str, Any]) -> tuple[str, int]:
    frame = row.get("frame_id", row.get("original_frame_idx"))
    return str(row["video_id"]), int(frame)


def _chain(row: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(row["video_id"]), tuple(int(value) for value in row["frame_ids"])


def _canonical_evidence(
    row: dict[str, Any], resolver: CatalogResolver, *, source: str
) -> dict[str, Any] | None:
    video_id = str(row.get("video_id", ""))
    if not video_id:
        return None
    try:
        if row.get("frame_id") is not None:
            global_row = resolver.nearest_row(video_id, int(row["frame_id"]))
        else:
            seconds = row.get("start_seconds", row.get("pts_time"))
            if seconds is None:
                return None
            global_row = resolver.nearest_time_row(video_id, float(seconds))
    except KeyError:
        return None
    mapped = resolver.catalog.map_row(global_row)
    return {
        **row,
        "video_id": video_id,
        "frame_id": int(mapped["original_frame_idx"]),
        "video_time_sec": float(mapped["pts_time"]),
        "global_row": global_row,
        "source": source,
    }


def fuse_team_frames(
    query: dict[str, Any],
    *,
    a0: list[dict[str, Any]],
    s1: list[dict[str, Any]],
    a0_provenance: list[dict[str, Any]],
    s1_provenance: list[dict[str, Any]],
    asr_lexical: list[dict[str, Any]],
    asr_e5: list[dict[str, Any]],
    ocr: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    resolver: CatalogResolver,
    asr_specificity: list[dict[str, Any]] | None = None,
    limit: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fuse ranks only; visual agreement chooses the head and weak evidence cannot take Top1."""

    if limit < 5:
        raise ValueError("PRELIM1_TEAM_LIMIT_TOO_SMALL")
    query_id = str(query["query_id"])
    provenance = {}
    for branch, rows in (("A0", a0_provenance), ("S1", s1_provenance)):
        for row in rows:
            key = tuple(row["candidate_key"][:2])
            provenance[(branch, str(key[0]), int(key[1]))] = row
    specificity = asr_specificity or []
    branches: dict[str, tuple[list[dict[str, Any]], float]] = {
        "A0": (a0, 1.0),
        "S1": (s1, 1.0),
        "ASR_LEX": (
            [
                mapped
                for row in asr_lexical
                if (mapped := _canonical_evidence(row, resolver, source="ASR_LEX"))
            ],
            0.55,
        ),
        "ASR_E5": (
            [
                mapped
                for row in asr_e5
                if (mapped := _canonical_evidence(row, resolver, source="ASR_E5"))
            ],
            0.55,
        ),
        "OCR": (
            [mapped for row in ocr if (mapped := _canonical_evidence(row, resolver, source="OCR"))],
            0.25,
        ),
        "OBJECT": (
            [
                mapped
                for row in objects
                if (mapped := _canonical_evidence(row, resolver, source="OBJECT"))
            ],
            0.10,
        ),
        "ASR_HIGH": (
            [row for row in specificity if row.get("specificity_tier") == "HIGH"],
            0.85,
        ),
        "ASR_MEDIUM": (
            [row for row in specificity if row.get("specificity_tier") == "MEDIUM"],
            0.40,
        ),
        "ASR_LOW": (
            [row for row in specificity if row.get("specificity_tier") == "LOW"],
            0.12,
        ),
    }
    evidence: dict[tuple[str, int], dict[str, Any]] = {}
    for branch, (rows, weight) in branches.items():
        seen = set()
        for fallback_rank, raw in enumerate(rows[:100], 1):
            key = _coordinate(raw)
            if key in seen:
                continue
            seen.add(key)
            rank = int(raw.get("rank", fallback_rank))
            item = evidence.setdefault(
                key,
                {
                    "score": 0.0,
                    "source_ranks": {},
                    "representative": dict(raw),
                    "best_rank": rank,
                },
            )
            item["score"] += weight / (RRF_K + rank)
            item["source_ranks"][branch] = rank
            if rank < int(item["best_rank"]):
                item["best_rank"] = rank
                item["representative"] = dict(raw)
    visual_agreement = [
        (key, item) for key, item in evidence.items() if {"A0", "S1"}.issubset(item["source_ranks"])
    ]
    visual_agreement.sort(
        key=lambda value: (
            -float(value[1]["score"]),
            max(value[1]["source_ranks"]["A0"], value[1]["source_ranks"]["S1"]),
            value[0],
        )
    )
    a0_head = next(
        (
            (_coordinate(row), evidence[_coordinate(row)])
            for row in a0
            if _coordinate(row) in evidence
        ),
        None,
    )
    head = visual_agreement[0] if visual_agreement else a0_head
    if head is None:
        raise RuntimeError(f"PRELIM1_NO_VISUAL_HEAD:{query_id}")
    ordered = sorted(
        evidence.items(),
        key=lambda value: (
            -float(value[1]["score"]),
            -sum(name in value[1]["source_ranks"] for name in ("A0", "S1")),
            int(value[1]["best_rank"]),
            value[0],
        ),
    )
    ordered = [head, *(value for value in ordered if value[0] != head[0])]
    strong_asr = next(
        (
            value
            for value in ordered
            if "ASR_HIGH" in value[1]["source_ranks"]
            or {"ASR_LEX", "ASR_E5"}.issubset(value[1]["source_ranks"])
        ),
        None,
    )
    if strong_asr is not None and strong_asr not in ordered[:20]:
        ordered = [value for value in ordered if value[0] != strong_asr[0]]
        ordered.insert(19, strong_asr)
    rows, audit = [], []
    query_tokens = _tokens(str(query.get("query", query.get("question", ""))))
    for key, item in ordered[:limit]:
        ranks = item["source_ranks"]
        visual = [name for name in ("A0", "S1") if name in ranks]
        text = [
            name
            for name in ("ASR_HIGH", "ASR_MEDIUM", "ASR_LEX", "ASR_E5", "OCR")
            if name in ranks
        ]
        if len(visual) == 2:
            tier = "TIER_A_VISUAL_AGREEMENT"
        elif visual and text:
            tier = "TIER_B_VISUAL_TEXT_CORROBORATED"
        elif visual:
            tier = "TIER_B_SINGLE_VISUAL"
        elif "ASR_HIGH" in ranks or {"ASR_LEX", "ASR_E5"}.issubset(ranks):
            tier = "TIER_B_ASR_DUAL"
        else:
            tier = "TIER_C_WEAK_RECALL"
        view_ranks = {}
        for branch in ("A0", "S1"):
            row = provenance.get((branch, key[0], key[1]))
            if row:
                view_ranks.update(
                    {f"{branch}:{name}": rank for name, rank in row["view_ranks"].items()}
                )
        representative = item["representative"]
        evidence_text = str(
            representative.get("text") or representative.get("asr_span", {}).get("text", "") or ""
        )
        matched = sorted(query_tokens.intersection(_tokens(evidence_text)))
        row = {
            "query_id": query_id,
            "task_type": str(query["task"]),
            "candidate_rank": len(rows) + 1,
            "video_id": key[0],
            "frame_id": key[1],
            "video_time_sec": float(
                representative.get("video_time_sec", representative.get("pts_time", 0.0))
            ),
            "primary_candidate": len(rows) == 0,
            "A0_rank": ranks.get("A0"),
            "S1_rank": ranks.get("S1"),
            "ASR_lex_rank": ranks.get("ASR_LEX"),
            "ASR_E5_rank": ranks.get("ASR_E5"),
            "ASR_specificity": (
                "HIGH"
                if "ASR_HIGH" in ranks
                else "MEDIUM"
                if "ASR_MEDIUM" in ranks
                else "LOW"
                if "ASR_LOW" in ranks
                else None
            ),
            "evidence_tier": tier,
            "query_views_hit": sorted(view_ranks),
            "matched_high_anchors": matched[:10],
            "modalities": sorted(ranks),
            "reason_short": f"{tier}; sources={','.join(sorted(ranks))}",
            "source_system": SOURCE_SYSTEM,
        }
        rows.append(row)
        audit.append({**row, "rrf_score": item["score"], "source_ranks": ranks})
    if len(rows) < limit:
        raise RuntimeError(f"PRELIM1_TEAM_CANDIDATE_SHORTFALL:{query_id}:{len(rows)}")
    return rows, audit


def fuse_team_chains(
    query: dict[str, Any],
    *,
    a0: list[dict[str, Any]],
    s1: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    event_count = int(query["event_count"])
    evidence: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for branch, rows in (("A0", a0), ("S1", s1)):
        seen = set()
        for fallback_rank, row in enumerate(rows[:100], 1):
            key = _chain(row)
            if key in seen:
                continue
            seen.add(key)
            if len(key[1]) != event_count or any(
                left >= right for left, right in zip(key[1], key[1][1:], strict=False)
            ):
                continue
            rank = int(row.get("rank", fallback_rank))
            item = evidence.setdefault(key, {"score": 0.0, "ranks": {}, "row": dict(row)})
            item["score"] += 1.0 / (RRF_K + rank)
            item["ranks"][branch] = rank
    ordered = sorted(
        evidence.items(),
        key=lambda value: (
            -len(value[1]["ranks"]),
            -float(value[1]["score"]),
            max(value[1]["ranks"].values()),
            value[0],
        ),
    )
    if len(ordered) < limit:
        raise RuntimeError(f"PRELIM1_TRAKE_CHAIN_SHORTFALL:{len(ordered)}")
    output = []
    for rank, (key, item) in enumerate(ordered[:limit], 1):
        output.append(
            {
                "query_id": str(query["query_id"]),
                "task_type": "TRAKE",
                "candidate_rank": rank,
                "video_id": key[0],
                "frame_ids": list(key[1]),
                "primary_candidate": rank == 1,
                "A0_rank": item["ranks"].get("A0"),
                "S1_rank": item["ranks"].get("S1"),
                "ASR_lex_rank": None,
                "ASR_E5_rank": None,
                "evidence_tier": (
                    "TIER_A_VISUAL_AGREEMENT" if len(item["ranks"]) == 2 else "TIER_B_SINGLE_VISUAL"
                ),
                "query_views_hit": [],
                "matched_high_anchors": [],
                "modalities": sorted(item["ranks"]),
                "reason_short": "strict ordinal chain; rank-only A0/S1 consensus",
                "source_system": SOURCE_SYSTEM,
            }
        )
    return output


def _answer_candidates(text: str, answer_type: str) -> list[str]:
    if answer_type in {"NUMBER_OR_COUNT", "VISIBLE_NUMBER"}:
        return _NUMBER.findall(text)
    if answer_type == "VISUAL_COUNT":
        return []
    if answer_type == "LOCATION_OR_NAME":
        return [match.group(1).strip(" .,;:()[]") for match in _LOCATION.finditer(text)]
    if answer_type in {"TEXT_PRESERVING", "UNKNOWN_MANUAL"}:
        quoted = re.findall(r"[\"“”']([^\"“”']{2,100})[\"“”']", text)
        return [" ".join(value.split()) for value in quoted]
    return []


def build_qa_review_rows(
    query: dict[str, Any],
    contexts: list[dict[str, Any]],
    *,
    asr_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return five context frames and only evidence-preserving answer hypotheses."""

    answer_type = str(query.get("answer_type", "UNKNOWN_MANUAL"))
    by_video: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for source, rows in (("OCR", ocr_rows), ("ASR", asr_rows)):
        for row in rows:
            by_video[str(row["video_id"])].append((source, row))
    hypotheses, seen_answers = [], set()
    for context in contexts[:10]:
        video_id = str(context["video_id"])
        for source, evidence in by_video.get(video_id, []):
            if answer_type == "VISIBLE_NUMBER" and source != "OCR":
                continue
            text = str(
                evidence.get("text")
                or evidence.get("corrected_text")
                or evidence.get("combined_text")
                or evidence.get("asr_span", {}).get("text", "")
            )
            for answer in _answer_candidates(text, answer_type):
                folded = _fold(answer)
                if not folded or folded in seen_answers:
                    continue
                seen_answers.add(folded)
                hypotheses.append(
                    {
                        "answer": answer,
                        "video_id": video_id,
                        "frame_id": int(context["frame_id"]),
                        "support_span": text[:500],
                        "evidence_type": source,
                        "confidence_bucket": "HIGH" if source == "OCR" else "MEDIUM",
                    }
                )
    rows, audit = [], []
    for rank, context in enumerate(contexts[:5], 1):
        hypothesis = next(
            (value for value in hypotheses if value["video_id"] == str(context["video_id"])),
            None,
        )
        answer = str(hypothesis["answer"]) if hypothesis else ""
        status = "EVIDENCE_SUPPORTED" if answer else "MANUAL_REVIEW_REQUIRED"
        row = {
            **context,
            "query_id": str(query["query_id"]),
            "task_type": "QA",
            "candidate_rank": rank,
            "primary_candidate": rank == 1,
            "answer": answer,
            "status": status,
            "support_spans": [hypothesis["support_span"]] if hypothesis else [],
            "evidence_type": hypothesis["evidence_type"] if hypothesis else "NONE",
            "confidence_bucket": hypothesis["confidence_bucket"] if hypothesis else "MANUAL",
            "reason_short": (
                f"{hypothesis['evidence_type']} exact evidence"
                if hypothesis
                else "manual review; no supported answer"
            ),
            "source_system": SOURCE_SYSTEM,
        }
        rows.append(row)
        audit.append(
            {
                "query_id": str(query["query_id"]),
                "candidate_rank": rank,
                "answer_type": answer_type,
                "answer": answer,
                "status": status,
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
                "support_spans": row["support_spans"],
                "ground_truth_used": False,
            }
        )
    return rows, audit


__all__ = ["build_qa_review_rows", "fuse_team_chains", "fuse_team_frames"]
