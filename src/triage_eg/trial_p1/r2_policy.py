"""GT-free Trial P1 R2 evidence tiers, QA verification, and ranking policy."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aic2026_eval.validation import validate_predictions
from triage_eg.fs1.fusion import default_key, reciprocal_rank_fusion
from triage_eg.fs1_v11.pipeline import build_completion_arm, grouped, semantic_content_hash

TIER_A = "TIER_A_DIRECT"
TIER_B = "TIER_B_CORROBORATED"
TIER_C = "TIER_C_WEAK"
TIER_ORDER = {TIER_A: 0, TIER_B: 1, TIER_C: 2}

GENERIC_TERMS = frozenset(
    {
        "nghiên cứu",
        "nhà thơ",
        "món ăn",
        "chương trình",
        "sự kiện",
        "người",
        "video",
        "hình ảnh",
        "đang",
        "cảnh",
        "thực hiện",
        "giới thiệu",
    }
)
STOPWORDS = frozenset(
    {
        "và",
        "của",
        "có",
        "là",
        "được",
        "trong",
        "một",
        "những",
        "với",
        "cho",
        "đang",
        "sau",
        "trước",
        "này",
        "đó",
        "khi",
        "tại",
        "the",
        "and",
        "with",
        "from",
        "this",
        "that",
    }
)
LOCATION_ANCHORS = re.compile(
    r"\b(xã|phường|huyện|thị trấn|tỉnh|thành phố|quận|ấp|thôn|làng)\b", re.I
)
EXPLANATORY_TEXT = re.compile(
    r"^(câu thơ gợi nhắc|nội dung nói về|đoạn này cho thấy|đây là|the text says|"
    r"this shows|it refers to)\b",
    re.I,
)
TITLE_GENERIC = frozenset({"món ăn", "công thức", "chương trình", "tiêu đề", "recipe"})
LOCATION_REJECT = re.compile(
    r"\b(chương trình|tổ chức|công ty|quỹ|dự án|bệnh viện|trường đại học)\b", re.I
)


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", normalized, re.UNICODE))


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in _plain(value).split() if len(token) > 1)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _anchor_texts(values: Iterable[Any]) -> list[str]:
    return [
        str(value.get("text", "")) if isinstance(value, dict) else str(value)
        for value in values
    ]


def derive_anchor_profiles(
    queries: list[dict[str, Any]], plans: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Compile query anchors without GT, IDs, or expected result knowledge."""

    plan_by_id = {str(plan["query_id"]): plan for plan in plans}
    document_frequency: Counter[str] = Counter()
    query_tokens: dict[str, set[str]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        plan = plan_by_id[query_id]
        text = " ".join(
            [
                str(plan.get("raw_text", "")),
                str(plan.get("semantic_core", "")),
                *map(str, plan.get("text_entity_anchors", [])),
                *map(str, plan.get("action_anchors", [])),
                *_anchor_texts(plan.get("knowledge_expansions", [])),
            ]
        )
        values = set(_tokens(text))
        query_tokens[query_id] = values
        document_frequency.update(values)

    profiles = {}
    for query in queries:
        query_id = str(query["query_id"])
        plan = plan_by_id[query_id]
        raw = str(plan.get("raw_text", query.get("query", "")))
        proper_phrases = _unique(
            " ".join(match.group(0).split())
            for match in re.finditer(
                r"\b[A-ZÀ-Ỹ][\wÀ-ỹ-]+(?:\s+[A-ZÀ-Ỹ0-9][\wÀ-ỹ0-9-]+){0,4}", raw
            )
        )
        rare = sorted(
            token
            for token in query_tokens[query_id]
            if document_frequency[token] <= 2
            and token not in STOPWORDS
            and _plain(token) not in {_plain(value) for value in GENERIC_TERMS}
            and len(token) >= 3
        )
        constraints = {
            "count": re.findall(r"\b\d+\b|\b(?:hai|ba|bốn|năm|sáu|bảy|tám|chín)\b", raw, re.I),
            "color": re.findall(
                r"\b(?:đen|trắng|đỏ|cam|vàng|xanh|tím|hồng|nâu|xám)\b", raw, re.I
            ),
            "action": list(map(str, plan.get("action_anchors", []))),
        }
        phrases = _unique(
            [
                *proper_phrases,
                *map(str, plan.get("text_entity_anchors", [])),
                *_anchor_texts(plan.get("knowledge_expansions", [])),
            ]
        )
        profiles[query_id] = {
            "query_id": query_id,
            "named_entity_phrases": proper_phrases[:12],
            "distinctive_phrases": phrases[:12],
            "important_tokens": rare[:24],
            "generic_terms": sorted(
                term for term in GENERIC_TERMS if _plain(term) in _plain(raw)
            ),
            "constraints": constraints,
            "gt_used": False,
        }
    return profiles


def classify_asr_specificity(
    row: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    span = dict(row.get("asr_span") or row)
    text = str(span.get("text", ""))
    normalized = _plain(text)
    important = set(map(_plain, profile.get("important_tokens", [])))
    matched = sorted(token for token in important if token and token in set(_tokens(text)))
    phrase_matches = [
        phrase
        for phrase in profile.get("distinctive_phrases", [])
        if len(_tokens(phrase)) >= 2 and _plain(phrase) in normalized
    ]
    branches = {
        str(item.get("branch", "")).upper() for item in row.get("asr_source_ranks", [])
    }
    agreement = {"LEXICAL", "E5"}.issubset(branches)
    generic_matches = [
        term for term in profile.get("generic_terms", []) if _plain(term) in normalized
    ]
    generic_only = bool(generic_matches and not matched and not phrase_matches)
    reasons = []
    if phrase_matches:
        reasons.append("EXACT_DISTINCTIVE_PHRASE_COVERAGE")
    if matched:
        reasons.append(f"DISTINCT_IMPORTANT_ANCHORS_{len(matched)}")
    if agreement:
        reasons.append("LEXICAL_E5_VIDEO_AGREEMENT")
    if generic_only:
        reasons.append("GENERIC_TOKEN_ONLY_PENALTY")
    if phrase_matches or len(matched) >= 3 or (agreement and len(matched) >= 2):
        tier = TIER_A
    elif len(matched) >= 2 or (agreement and len(matched) >= 1):
        tier = TIER_B
    else:
        tier = TIER_C
    return {
        **row,
        "evidence_tier": tier,
        "evidence_tier_reasons": reasons or ["ASR_LOW_SPECIFICITY"],
        "matched_query_anchors": _unique([*phrase_matches, *matched]),
        "asr_specificity": {
            "exact_anchor_phrases": phrase_matches,
            "distinct_important_anchors": matched,
            "lexical_e5_agreement": agreement,
            "generic_matches": generic_matches,
            "generic_token_only": generic_only,
            "source_ranks": row.get("asr_source_ranks", []),
        },
    }


def _classify_non_asr(
    modality: str, row: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    text = str(row.get("text", ""))
    normalized = _plain(text)
    important = set(map(_plain, profile.get("important_tokens", [])))
    matched = sorted(token for token in important if token and token in set(_tokens(text)))
    phrase_matches = [
        phrase
        for phrase in profile.get("distinctive_phrases", [])
        if len(_tokens(phrase)) >= 2 and _plain(phrase) in normalized
    ]
    reasons: list[str]
    if modality == "ocr":
        confidence = float(row.get("source_confidence") or 0.0)
        normalized_confidence = confidence / 100.0 if confidence > 1.0 else confidence
        if phrase_matches or (len(matched) >= 2 and normalized_confidence >= 0.5):
            tier, reasons = TIER_A, ["OCR_DISTINCTIVE_ENTITY_CONTEXT_MATCH"]
        elif matched and normalized_confidence >= 0.5:
            tier, reasons = TIER_B, ["OCR_QUERY_ANCHOR_WITH_CONFIDENCE"]
        else:
            tier, reasons = TIER_C, ["OCR_ONLY_NO_DISTINCTIVE_CONTEXT"]
    elif modality == "action":
        rank = int(row.get("rank", 10**9))
        tier = TIER_A if rank <= 3 else TIER_B if rank <= 10 else TIER_C
        reasons = [f"XCLIP_EVENT_RANK_{rank}"]
    elif modality == "object":
        tier, reasons = TIER_C, ["OBJECT_ONLY_STANDALONE_WEAK"]
    else:
        tier, reasons = TIER_C, [f"{modality.upper()}_UNCLASSIFIED_WEAK"]
    return {
        **row,
        "evidence_tier": tier,
        "evidence_tier_reasons": reasons,
        "matched_query_anchors": _unique([*phrase_matches, *matched]),
    }


def tier_evidence(
    queries: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    profiles: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    output = {
        name: {key: [dict(row) for row in rows] for key, rows in values.items()}
        for name, values in evidence.items()
    }
    diagnostics = []
    for query in queries:
        query_id = str(query["query_id"])
        profile = profiles[query_id]
        for modality in ("asr", "ocr", "object", "action", "action_revision"):
            rows = []
            for raw in evidence.get(modality, {}).get(query_id, []):
                row = (
                    classify_asr_specificity(raw, profile)
                    if modality == "asr"
                    else _classify_non_asr(
                        "action" if modality == "action_revision" else modality,
                        raw,
                        profile,
                    )
                )
                rows.append(row)
                diagnostics.append(
                    {
                        "query_id": query_id,
                        "modality": modality,
                        "video_id": row.get("video_id"),
                        "frame_id": row.get("frame_id"),
                        "rank": row.get("rank"),
                        "evidence_tier": row["evidence_tier"],
                        "reasons": row["evidence_tier_reasons"],
                        "matched_query_anchors": row["matched_query_anchors"],
                        "asr_specificity": row.get("asr_specificity"),
                    }
                )
            rows.sort(
                key=lambda row: (
                    TIER_ORDER[row["evidence_tier"]],
                    int(row.get("rank", 10**9)),
                    str(row.get("video_id", "")),
                    int(row.get("frame_id", 0)),
                )
            )
            output.setdefault(modality, {})[query_id] = [
                {**row, "rank": rank} for rank, row in enumerate(rows, 1)
            ]
    return output, diagnostics


def _video_assessments(
    baseline: list[dict[str, Any]], modality_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "modalities": set(),
            "tiers": [],
            "reasons": [],
            "matched_query_anchors": [],
            "rows": [],
            "bcf1_rank": None,
        }
    )
    for row in baseline:
        item = values[str(row["video_id"])]
        rank = int(row["rank"])
        item["bcf1_rank"] = (
            rank if item["bcf1_rank"] is None else min(rank, int(item["bcf1_rank"]))
        )
    for modality, rows in modality_rows.items():
        for row in rows:
            item = values[str(row["video_id"])]
            item["modalities"].add(modality)
            item["tiers"].append(row.get("evidence_tier", TIER_C))
            item["reasons"].extend(row.get("evidence_tier_reasons", []))
            item["matched_query_anchors"].extend(row.get("matched_query_anchors", []))
            item["rows"].append((modality, row))
    for item in values.values():
        modalities = item["modalities"]
        tiers = item["tiers"]
        if TIER_A in tiers:
            tier = TIER_A
        elif TIER_B in tiers and item["bcf1_rank"] is not None:
            tier = TIER_B
            item["reasons"].append("SEMANTIC_EVIDENCE_WITH_VISUAL_SUPPORT")
        elif len(modalities) >= 2:
            tier = TIER_B
            item["reasons"].append("TWO_INDEPENDENT_WEAK_MODALITIES_AGREE")
        else:
            tier = TIER_C
        item["evidence_tier"] = tier
        item["modalities"] = sorted(modalities)
        item["reasons"] = _unique(item["reasons"])
        item["matched_query_anchors"] = _unique(item["matched_query_anchors"])
    return dict(values)


def _coordinate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if "frame_ids" in row:
        return str(row["video_id"]), tuple(map(int, row["frame_ids"]))
    return str(row["video_id"]), int(row["frame_id"])


def _complete_ranked_rows(
    primary: Iterable[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    query_id: str,
    variant: str,
    identity: Any = _coordinate_key,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fill to the submission limit without collapsing BCF1 frame coordinates."""

    selected, seen = [], set()
    for raw in [*primary, *fallback]:
        row = dict(raw)
        key = identity(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == limit:
            break
    # The shared validator permits repeated coordinates. Preserve frozen rows as
    # the final structural fallback if the frozen ranking itself contains repeats.
    if len(selected) < limit:
        for raw in fallback:
            selected.append(dict(raw))
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise RuntimeError(
            f"R2_STRUCTURAL_FILL_FAILED:{query_id}:{variant}:{len(selected)}:{limit}"
        )
    return [
        {**row, "query_id": query_id, "system_variant": variant, "rank": rank}
        for rank, row in enumerate(selected, 1)
    ]


def _representative(
    video_id: str, assessment: dict[str, Any], baseline: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = sorted(
        assessment["rows"],
        key=lambda item: (
            TIER_ORDER[item[1].get("evidence_tier", TIER_C)],
            int(item[1].get("rank", 10**9)),
            item[0],
        ),
    )
    baseline_row = next((row for row in baseline if str(row["video_id"]) == video_id), None)
    source = (
        rows[0][1]
        if rows and assessment["evidence_tier"] in {TIER_A, TIER_B}
        else baseline_row
    )
    if source is None and rows:
        source = rows[0][1]
    if source is None:
        raise RuntimeError(f"R2_CANDIDATE_WITHOUT_REPRESENTATIVE:{video_id}")
    return {
        **source,
        "video_id": video_id,
        "evidence_tier": assessment["evidence_tier"],
        "evidence_tier_reasons": assessment["reasons"],
        "dominant_modalities": assessment["modalities"],
        "matched_query_anchors": assessment["matched_query_anchors"],
        "corroborating_sources": assessment["modalities"],
        "bcf1_rank": assessment["bcf1_rank"],
    }


def repair_kis_ranking(
    query: dict[str, Any],
    baseline: list[dict[str, Any]],
    tiered_evidence: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rank KIS by discrete class, with RRF60 only inside comparable classes."""

    modalities = {name: tiered_evidence.get(name, []) for name in ("asr", "ocr", "object")}
    assessments = _video_assessments(baseline, modalities)
    by_tier = {
        tier: [video_id for video_id, item in assessments.items() if item["evidence_tier"] == tier]
        for tier in (TIER_A, TIER_B, TIER_C)
    }

    def tier_order(tier: str) -> list[str]:
        allowed = set(by_tier[tier])
        sources = [
            [row for row in rows if str(row["video_id"]) in allowed]
            for rows in modalities.values()
        ]
        sources = [rows for rows in sources if rows]
        if not sources:
            return sorted(allowed)
        return [
            str(row["video_id"])
            for row in reciprocal_rank_fusion(sources, key=lambda row: str(row["video_id"]))
        ]

    protected_rows = [dict(row) for row in baseline[:5]]
    protected_keys = {_coordinate_key(row) for row in protected_rows}
    weak_sources = [
        [row for row in rows if assessments[str(row["video_id"])]["evidence_tier"] == TIER_C]
        for rows in modalities.values()
    ]
    normal = reciprocal_rank_fusion(
        [baseline, *weak_sources], key=_coordinate_key
    )

    def annotate(raw: dict[str, Any]) -> dict[str, Any]:
        row = dict(raw)
        video_id = str(row["video_id"])
        item = assessments[video_id]
        exact_bcf1 = next(
            (
                candidate
                for candidate in baseline
                if _coordinate_key(candidate) == _coordinate_key(row)
            ),
            None,
        )
        return {
            **row,
            "evidence_tier": item["evidence_tier"],
            "evidence_tier_reasons": item["reasons"],
            "dominant_modalities": item["modalities"],
            "matched_query_anchors": item["matched_query_anchors"],
            "corroborating_sources": item["modalities"],
            "bcf1_rank": (
                int(exact_bcf1["rank"]) if exact_bcf1 is not None else item["bcf1_rank"]
            ),
        }

    tier_representatives = {
        tier: [
            _representative(video_id, assessments[video_id], baseline)
            for video_id in tier_order(tier)
        ]
        for tier in (TIER_A, TIER_B, TIER_C)
    }
    full_primary = [
        *tier_representatives[TIER_A],
        *tier_representatives[TIER_B],
        *map(annotate, protected_rows),
        *map(annotate, normal),
        *tier_representatives[TIER_C],
    ]
    annotated_baseline = [annotate(row) for row in baseline]
    full = _complete_ranked_rows(
        full_primary,
        annotated_baseline,
        query_id=str(query["query_id"]),
        variant="M0_R2",
    )
    direct = tier_representatives[TIER_A]
    safe_prefix = []
    prefix_seen = set()
    for row in [*direct, *map(annotate, protected_rows)]:
        key = _coordinate_key(row)
        if key in prefix_seen:
            continue
        prefix_seen.add(key)
        safe_prefix.append(row)
        if len(safe_prefix) == 5:
            break
    safe = _complete_ranked_rows(
        [*safe_prefix, *full],
        annotated_baseline,
        query_id=str(query["query_id"]),
        variant="SAFE_R2",
    )
    overrides = []
    for rank, row in enumerate(safe_prefix, 1):
        if _coordinate_key(row) not in protected_keys:
            video_id = str(row["video_id"])
            item = assessments[video_id]
            overrides.append(
                {
                    "new_rank": rank,
                    "video_id": video_id,
                    "frame_id": int(row["frame_id"]),
                    "original_bcf1_rank": item["bcf1_rank"],
                    "evidence_tier": item["evidence_tier"],
                    "reason": item["reasons"],
                    "corroborating_sources": item["modalities"],
                }
            )
    weak_override = [row for row in overrides if row["evidence_tier"] != TIER_A]
    if weak_override:
        raise RuntimeError(f"WEAK_MODALITY_OVERRIDE:{query['query_id']}:{weak_override}")
    strong_asr = {
        str(row["video_id"])
        for row in modalities["asr"]
        if row.get("evidence_tier") == TIER_A
    }
    strong_dropped = [
        video_id
        for video_id in sorted(strong_asr)
        if next((row["rank"] for row in full if row["video_id"] == video_id), 10**9) > 20
    ]
    diagnostics = {
        "query_id": query["query_id"],
        "task": "KIS",
        "tier_counts": dict(Counter(item["evidence_tier"] for item in assessments.values())),
        "safe_top5_overrides": overrides,
        "weak_modality_override": bool(weak_override),
        "strong_asr_dropped": strong_dropped,
        "candidate_assessments": {
            video_id: {key: value for key, value in item.items() if key != "rows"}
            for video_id, item in assessments.items()
        },
    }
    return full, safe, diagnostics


def _ordered_overlap(answer: str, evidence: str) -> float:
    answer_tokens, evidence_tokens = list(_tokens(answer)), list(_tokens(evidence))
    if not answer_tokens or not evidence_tokens:
        return 0.0
    cursor = 0
    for token in evidence_tokens:
        if cursor < len(answer_tokens) and token == answer_tokens[cursor]:
            cursor += 1
    subsequence = cursor / len(answer_tokens)
    token_recall = len(set(answer_tokens).intersection(evidence_tokens)) / len(set(answer_tokens))
    return max(subsequence, token_recall)


def verify_answer_extraction(
    extraction: dict[str, Any],
    answer_type: str,
    evidence_rows: list[dict[str, Any]],
    *,
    grounding_plausibility: float,
) -> dict[str, Any]:
    """Independent deterministic verifier; Qwen's sufficiency bit is never final."""

    kind = str(answer_type).upper()
    answer = " ".join(str(extraction.get("answer", "")).split())
    catalog = {str(row.get("source_id")): row for row in evidence_rows if row.get("source_id")}
    source_ids = extraction.get("supporting_source_ids")
    spans = extraction.get("supporting_spans")
    reasons = []
    if not isinstance(source_ids, list) or not all(isinstance(value, str) for value in source_ids):
        source_ids = []
        reasons.append("SUPPORT_SOURCE_IDS_INVALID")
    if not isinstance(spans, list) or not all(isinstance(value, str) for value in spans):
        spans = []
        reasons.append("SUPPORTING_SPANS_INVALID")
    unknown = sorted(set(source_ids) - set(catalog))
    if unknown:
        reasons.append("SUPPORT_SOURCE_ID_UNKNOWN")
    cited = [catalog[source_id] for source_id in source_ids if source_id in catalog]
    cited_text = [str(row.get("text", "")) for row in cited if str(row.get("text", "")).strip()]
    span_scores = [
        max((_ordered_overlap(span, text) for text in cited_text), default=0.0)
        for span in spans
    ]
    answer_score = max((_ordered_overlap(answer, text) for text in cited_text), default=0.0)
    if not answer or len(answer) > 100:
        reasons.append("ANSWER_EMPTY_OR_OVER_100_CHARS")
    if not cited:
        reasons.append("NO_VALID_CITED_EVIDENCE")
    if spans and min(span_scores, default=0.0) < 0.8:
        reasons.append("SUPPORTING_SPAN_NOT_NEAR_VERBATIM")
    compatible = True
    if kind == "LOCATION_NAME":
        location_context = any(
            LOCATION_ANCHORS.search(text) and _ordered_overlap(answer, text) >= 0.8
            for text in cited_text
        )
        compatible = bool(location_context and not LOCATION_REJECT.search(answer))
        if not compatible:
            reasons.append("LOCATION_NAME_SEMANTIC_SUPPORT_FAILED")
    elif kind == "QUOTE_OR_VISIBLE_TEXT":
        compatible = bool(
            len(_tokens(answer)) >= 3
            and answer_score >= 0.8
            and not EXPLANATORY_TEXT.search(answer)
        )
        if not compatible:
            reasons.append("QUOTE_FAITHFUL_EXTRACTION_FAILED")
    elif kind == "TITLE":
        compatible = bool(
            1 <= len(_tokens(answer)) <= 12
            and _plain(answer) not in {_plain(value) for value in TITLE_GENERIC}
            and answer_score >= 0.8
            and not EXPLANATORY_TEXT.search(answer)
        )
        if not compatible:
            reasons.append("TITLE_SEMANTIC_SUPPORT_FAILED")
    elif kind == "COUNT":
        compatible = bool(re.fullmatch(r"\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín", _plain(answer)))
        if not compatible:
            reasons.append("COUNT_TYPE_MISMATCH")
    elif kind == "COLOR":
        compatible = bool(
            re.fullmatch(r"(?:đen|trắng|đỏ|cam|vàng|xanh|tím|hồng|nâu|xám)", _plain(answer))
        )
        if not compatible:
            reasons.append("COLOR_TYPE_MISMATCH")
    elif kind in {"OBJECT", "PERSON", "ACTION", "YES_NO", "OTHER"}:
        compatible = bool(
            answer and len(_tokens(answer)) <= 12 and not EXPLANATORY_TEXT.search(answer)
        )
        if not compatible:
            reasons.append(f"{kind}_TYPE_MISMATCH")
    qwen_sufficient = extraction.get("evidence_sufficient") is True
    if not qwen_sufficient:
        reasons.append("QWEN_DECLARED_INSUFFICIENT")
    verifier_pass = bool(not reasons and compatible)
    modalities = sorted({str(row.get("modality", "unknown")) for row in cited})
    return {
        **extraction,
        "answer": answer,
        "answer_type": kind,
        "qwen_evidence_sufficient": qwen_sufficient,
        "deterministic_verifier_pass": verifier_pass,
        "final_evidence_sufficient": bool(qwen_sufficient and verifier_pass),
        "verifier_reasons": _unique(reasons),
        "evidence_type_compatible": compatible,
        "verified_supporting_source_ids": [
            source_id for source_id in source_ids if source_id in catalog
        ],
        "supporting_span_overlap": span_scores,
        "answer_support_overlap": answer_score,
        "corroborating_source_count": len(set(modalities)),
        "corroborating_modalities": modalities,
        "grounding_plausibility": float(grounding_plausibility),
    }


def rank_qa_r2(
    query: dict[str, Any], verified_rows: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fallback = [
        {
            **row,
            "answer": "không đủ bằng chứng",
            "qwen_evidence_sufficient": False,
            "deterministic_verifier_pass": False,
            "final_evidence_sufficient": False,
            "evidence_type_compatible": False,
            "corroborating_source_count": 0,
            "grounding_plausibility": 0.0,
            "verifier_reasons": ["STRUCTURAL_FALLBACK_ONLY"],
            "evidence_tier": TIER_C,
            "evidence_tier_reasons": ["QA_UNVERIFIED_STRUCTURAL_FALLBACK"],
            "dominant_modalities": [],
            "matched_query_anchors": [],
            "corroborating_sources": [],
        }
        for row in baseline
    ]
    verified_rows = [
        {
            **row,
            "evidence_tier": TIER_A if row.get("final_evidence_sufficient") else TIER_C,
            "evidence_tier_reasons": (
                ["QA_QWEN_EXTRACTION_AND_DETERMINISTIC_VERIFIER_PASS"]
                if row.get("final_evidence_sufficient")
                else ["QA_DETERMINISTIC_VERIFIER_NOT_PASS"]
            ),
            "dominant_modalities": row.get("corroborating_modalities", []),
            "matched_query_anchors": row.get("supporting_spans", []),
            "corroborating_sources": row.get("verified_supporting_source_ids", []),
        }
        for row in verified_rows
        if row.get("final_evidence_sufficient") is True
    ]
    combined = [*verified_rows, *fallback]
    combined.sort(
        key=lambda row: (
            0 if row.get("final_evidence_sufficient") is True else 1,
            0 if row.get("evidence_type_compatible") is True else 1,
            -int(row.get("corroborating_source_count", 0)),
            -float(row.get("grounding_plausibility", 0.0)),
            int(row.get("grounding_rank", row.get("rank", 10**9))),
            str(row.get("video_id", "")),
            int(row.get("frame_id", 0)),
            str(row.get("answer", "")),
        )
    )
    return _complete_ranked_rows(
        combined,
        fallback,
        query_id=str(query["query_id"]),
        variant="QA_R2",
        identity=default_key,
    )


def build_r2_candidates(
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    tiered_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    verified_qa: dict[str, list[dict[str, Any]]],
    revision_provider: Any,
    *,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = grouped(baseline_rows)
    available = {
        name for name in ("asr", "ocr", "action", "object", "qwen") if tiered_evidence.get(name)
    }
    m0_raw, _, m0_diagnostics = build_completion_arm(
        "M0_v11", queries, baseline_rows, tiered_evidence, available
    )
    m1_raw, trake_safe, m1_diagnostics = build_completion_arm(
        "M1_v11",
        queries,
        baseline_rows,
        tiered_evidence,
        available,
        revision_provider=revision_provider,
    )
    m0_group, m1_group, trake_safe_group = map(grouped, (m0_raw, m1_raw, trake_safe))
    m0, m1, safe, kis_diagnostics = [], [], [], []
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"]).upper()
        if task == "KIS":
            full_rows, safe_rows, diagnostic = repair_kis_ranking(
                query,
                baseline[query_id],
                {
                    name: tiered_evidence.get(name, {}).get(query_id, [])
                    for name in ("asr", "ocr", "object")
                },
            )
            m0.extend(full_rows)
            m1.extend({**row, "system_variant": "M1_R2"} for row in full_rows)
            safe.extend(safe_rows)
            kis_diagnostics.append(diagnostic)
        elif task == "QA":
            qa_rows = rank_qa_r2(query, verified_qa.get(query_id, []), baseline[query_id])
            m0.extend({**row, "system_variant": "M0_R2"} for row in qa_rows)
            m1.extend({**row, "system_variant": "M1_R2"} for row in qa_rows)
            safe.extend({**row, "system_variant": "SAFE_R2"} for row in qa_rows)
        else:
            action_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in tiered_evidence.get("action", {}).get(query_id, []):
                action_by_video[str(row["video_id"])].append(row)

            def annotate_trake(
                row: dict[str, Any],
                variant: str,
                action_support: dict[str, list[dict[str, Any]]] = action_by_video,
            ) -> dict[str, Any]:
                support = action_support.get(str(row["video_id"]), [])
                tier = min(
                    (item.get("evidence_tier", TIER_C) for item in support),
                    key=lambda value: TIER_ORDER[value],
                    default=TIER_C,
                )
                return {
                    **row,
                    "system_variant": variant,
                    "evidence_tier": tier,
                    "evidence_tier_reasons": _unique(
                        reason
                        for item in support
                        for reason in item.get("evidence_tier_reasons", [])
                    )
                    or ["B0_TEMPORAL_CHAIN_WITHOUT_XCLIP_VIDEO_SUPPORT"],
                    "dominant_modalities": ["b0_visual", *("xclip" for _ in support[:1])],
                    "matched_query_anchors": _unique(
                        anchor
                        for item in support
                        for anchor in item.get("matched_query_anchors", [])
                    ),
                    "corroborating_sources": _unique(
                        f"xclip:{item.get('video_id')}:{item.get('event_index')}"
                        for item in support
                    ),
                }

            m0.extend(annotate_trake(row, "M0_R2") for row in m0_group[query_id])
            m1.extend(annotate_trake(row, "M1_R2") for row in m1_group[query_id])
            safe.extend(annotate_trake(row, "SAFE_R2") for row in trake_safe_group[query_id])
    candidates = {"M0_R2": m0, "M1_R2": m1, "SAFE_R2": safe}
    validation = {}
    for name, rows in candidates.items():
        summary, issues = validate_predictions(queries, rows, inventory=inventory)
        exact = len(rows) == 2400 and all(len(values) == 100 for values in grouped(rows).values())
        validation[name] = {**summary, "exact_100_per_query": exact, "issues": issues}
        if summary["status"] != "PASS" or not exact:
            counts = {key: len(values) for key, values in grouped(rows).items()}
            payload = {
                "variant": name,
                "summary": validation[name],
                "per_query_counts": counts,
            }
            raise RuntimeError(
                "TRIAL_R2_CANDIDATE_VALIDATION_FAILED:"
                + json.dumps(payload, ensure_ascii=False, default=str)
            )
    trake_checks = {}
    diagnostics_by_id = {
        str(row["query_id"]): row for row in m1_diagnostics if row.get("task") == "TRAKE"
    }
    for query in (row for row in queries if row["task"] == "TRAKE"):
        query_id = str(query["query_id"])
        per_arm = {}
        for name, rows in candidates.items():
            selected = grouped(rows)[query_id]
            per_arm[name] = {
                "event_count_correct": all(
                    len(row.get("frame_ids", [])) == int(query["event_count"])
                    for row in selected
                ),
                "strictly_increasing": all(
                    all(left < right for left, right in zip(frames, frames[1:], strict=False))
                    for frames in (row.get("frame_ids", []) for row in selected)
                ),
            }
        graph = (diagnostics_by_id.get(query_id) or {}).get("graph") or {}
        graph_pass = bool(
            graph.get("query_event_count") == int(query["event_count"])
            and graph.get("revision_count") == 1
            and (graph.get("revision") or {}).get("evidence_added", 0) > 0
            and graph.get("chain_candidates_added", 0) > 0
        )
        trake_checks[query_id] = {"arms": per_arm, "graph_causal_gate_pass": graph_pass}
        if not graph_pass or not all(
            value["event_count_correct"] and value["strictly_increasing"]
            for value in per_arm.values()
        ):
            raise RuntimeError(f"TRIAL_R2_TRAKE_GATE_FAILED:{query_id}")
    return {
        "candidates": candidates,
        "validation": validation,
        "kis_diagnostics": kis_diagnostics,
        "m0_diagnostics": m0_diagnostics,
        "m1_diagnostics": m1_diagnostics,
        "trake_checks": trake_checks,
    }


def choose_r2_recommendation(
    queries: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any]:
    candidates = result["candidates"]
    qa_ids = [str(query["query_id"]) for query in queries if query["task"] == "QA"]
    readiness = {}
    for query_id in qa_ids:
        rows = grouped(candidates["M0_R2"])[query_id]
        sufficient = [row for row in rows if row.get("final_evidence_sufficient") is True]
        readiness[query_id] = {
            "verified_sufficient_count": len(sufficient),
            "target_at_least_five_met": len(sufficient) >= 5,
            "status": (
                "READY_FOR_MANUAL_REVIEW"
                if sufficient
                else "QA_NOT_READY_FOR_SERIOUS_SUBMISSION"
            ),
        }
    weak_override = any(row["weak_modality_override"] for row in result["kis_diagnostics"])
    strong_dropped = any(row["strong_asr_dropped"] for row in result["kis_diagnostics"])
    structural = all(value["status"] == "PASS" for value in result["validation"].values())
    hard_pass = bool(
        structural
        and all(value["verified_sufficient_count"] > 0 for value in readiness.values())
        and not weak_override
        and not strong_dropped
        and all(
            row["graph_causal_gate_pass"]
            and all(
                value["event_count_correct"] and value["strictly_increasing"]
                for value in row["arms"].values()
            )
            for row in result["trake_checks"].values()
        )
    )
    # Manual Top10 semantic review is intentionally not automated.
    return {
        "recommendation": "DO_NOT_SUBMIT_2_YET",
        "hard_structural_gates_pass": hard_pass,
        "manual_review_required": True,
        "qa_readiness": readiness,
        "weak_modality_override_present": weak_override,
        "strong_asr_dropped_present": strong_dropped,
        "gt_opened": False,
        "submission_uploaded": False,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_r2_artifacts(
    root: str | Path,
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    previous_candidates: dict[str, list[dict[str, Any]]],
    result: dict[str, Any],
    tier_diagnostics: list[dict[str, Any]],
    qa_extractions: list[dict[str, Any]],
    qa_verifications: list[dict[str, Any]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    decision = choose_r2_recommendation(queries, result)
    write_jsonl(output / "kis_evidence_tier_diagnostics.jsonl", tier_diagnostics)
    write_jsonl(
        output / "asr_specificity_diagnostics.jsonl",
        (row for row in tier_diagnostics if row["modality"] == "asr"),
    )
    write_jsonl(output / "qa_extractions.jsonl", qa_extractions)
    write_jsonl(output / "qa_semantic_verifier.jsonl", qa_verifications)
    (output / "qa_r2_readiness.json").write_text(
        json.dumps(decision["qa_readiness"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    baseline = grouped(baseline_rows)
    groups = {name: grouped(rows) for name, rows in previous_candidates.items()} | {
        name: grouped(rows) for name, rows in candidates.items()
    }
    comparison = []
    for query in queries:
        query_id = str(query["query_id"])
        rows = {name: values[query_id] for name, values in groups.items()}
        b0 = baseline[query_id]
        kis_diagnostic = next(
            (row for row in result["kis_diagnostics"] if row["query_id"] == query_id), None
        )
        warnings = []
        if kis_diagnostic and kis_diagnostic["weak_modality_override"]:
            warnings.append("WEAK_MODALITY_OVERRIDE")
        if kis_diagnostic and kis_diagnostic["strong_asr_dropped"]:
            warnings.append("STRONG_ASR_DROPPED")
        if query["task"] == "QA":
            if not rows["M0_R2"][0].get("evidence_type_compatible"):
                warnings.append("QA_TYPE_MISMATCH")
            if not rows["M0_R2"][0].get("final_evidence_sufficient"):
                warnings.append("QA_UNSUPPORTED")
        item = {
            "query_id": query_id,
            "task": query["task"],
            "bcf1_top1": default_key(b0[0]),
            "previous_m0_top1": default_key(rows["TRIAGEEG_M0_FULL"][0]),
            "m0_r2_top1": default_key(rows["M0_R2"][0]),
            "previous_safe_top1": default_key(rows["TRIAGEEG_SAFE"][0]),
            "safe_r2_top1": default_key(rows["SAFE_R2"][0]),
            "m1_r2_top1": default_key(rows["M1_R2"][0]),
            "m0_r2_top5_changed_vs_bcf1": {
                default_key(row) for row in rows["M0_R2"][:5]
            }
            != {default_key(row) for row in b0[:5]},
            "m0_r2_top5_changed_vs_previous_m0": {
                default_key(row) for row in rows["M0_R2"][:5]
            }
            != {default_key(row) for row in rows["TRIAGEEG_M0_FULL"][:5]},
            "safe_r2_top5_changed_vs_previous_safe": {
                default_key(row) for row in rows["SAFE_R2"][:5]
            }
            != {default_key(row) for row in rows["TRIAGEEG_SAFE"][:5]},
            "promotion_tier": rows["M0_R2"][0].get("evidence_tier"),
            "promotion_reason": rows["M0_R2"][0].get("evidence_tier_reasons"),
            "strongest_evidence": rows["M0_R2"][0].get("dominant_modalities"),
            "warnings": warnings,
            "m0_r2_top10": [
                {
                    "rank": row["rank"],
                    "identity": default_key(row),
                    "tier": row.get("evidence_tier"),
                    "modalities": row.get("dominant_modalities"),
                    "bcf1_rank": row.get("bcf1_rank"),
                    "matched_query_anchors": row.get("matched_query_anchors", []),
                    "corroborating_sources": row.get("corroborating_sources", []),
                    "answer": row.get("answer"),
                    "qwen_evidence_sufficient": row.get("qwen_evidence_sufficient"),
                    "deterministic_verifier_pass": row.get(
                        "deterministic_verifier_pass"
                    ),
                    "final_evidence_sufficient": row.get("final_evidence_sufficient"),
                }
                for row in rows["M0_R2"][:10]
            ],
            "m1_r2_top10": [
                {
                    "rank": row["rank"],
                    "identity": default_key(row),
                    "tier": row.get("evidence_tier"),
                    "modalities": row.get("dominant_modalities"),
                    "bcf1_rank": row.get("bcf1_rank"),
                    "matched_query_anchors": row.get("matched_query_anchors", []),
                    "corroborating_sources": row.get("corroborating_sources", []),
                    "answer": row.get("answer"),
                    "final_evidence_sufficient": row.get("final_evidence_sufficient"),
                }
                for row in rows["M1_R2"][:10]
            ],
            "safe_r2_top10": [
                {
                    "rank": row["rank"],
                    "identity": default_key(row),
                    "tier": row.get("evidence_tier"),
                    "modalities": row.get("dominant_modalities"),
                    "bcf1_rank": row.get("bcf1_rank"),
                    "matched_query_anchors": row.get("matched_query_anchors", []),
                    "corroborating_sources": row.get("corroborating_sources", []),
                    "answer": row.get("answer"),
                    "final_evidence_sufficient": row.get("final_evidence_sufficient"),
                }
                for row in rows["SAFE_R2"][:10]
            ],
        }
        comparison.append(item)
    comparison_json = {"rows": comparison, "decision": decision}
    (output / "trial_p1_r2_candidate_comparison.json").write_text(
        json.dumps(comparison_json, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    with (output / "trial_p1_r2_candidate_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    trake_summary = {
        row["query_id"]: {
            "graph": row["graph"],
            "structural_checks": result["trake_checks"][row["query_id"]],
        }
        for row in result["m1_diagnostics"]
        if row.get("task") == "TRAKE"
    }
    (output / "trake_r2_graph_summary.json").write_text(
        json.dumps(trake_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review = [
        "# Trial P1 R2 Human Review",
        "",
        "No GT was opened. No submission was uploaded.",
        "",
    ]
    for item in comparison:
        review.extend(
            [
                f"## {item['query_id']} ({item['task']})",
                "",
                f"- BCF1 Top1: `{item['bcf1_top1']}`",
                f"- Previous M0 Top1: `{item['previous_m0_top1']}`",
                f"- M0_R2 Top1: `{item['m0_r2_top1']}`",
                f"- SAFE_R2 Top1: `{item['safe_r2_top1']}`",
                f"- M1_R2 Top1: `{item['m1_r2_top1']}`",
                f"- Tier/reason: `{item['promotion_tier']}` / `{item['promotion_reason']}`",
                f"- Top5 changed vs BCF1: `{item['m0_r2_top5_changed_vs_bcf1']}`",
                f"- Top5 changed vs previous M0: `{item['m0_r2_top5_changed_vs_previous_m0']}`",
                "- SAFE_R2 Top5 changed vs previous SAFE: "
                f"`{item['safe_r2_top5_changed_vs_previous_safe']}`",
                f"- Warnings: `{item['warnings'] or 'NONE'}`",
                f"- M0_R2 Top10: `{item['m0_r2_top10']}`",
                f"- M1_R2 Top10: `{item['m1_r2_top10']}`",
                f"- SAFE_R2 Top10: `{item['safe_r2_top10']}`",
                "",
            ]
        )
    (output / "trial_p1_r2_human_review.md").write_text("\n".join(review), encoding="utf-8")
    candidate_zips = {}
    for name, rows in candidates.items():
        prediction = output / f"{name}.jsonl"
        write_jsonl(prediction, rows)
        target = output / f"{name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(prediction, prediction.name)
        candidate_zips[name] = str(target)
    (output / "SUBMISSION_2_R2_DECISION.md").write_text(
        "# Submission #2 R2 Decision\n\n`DO_NOT_SUBMIT_2_YET`\n\n"
        "Manual Top10 evidence review remains mandatory. No upload was performed.\n",
        encoding="utf-8",
    )
    (output / "run_provenance.json").write_text(
        json.dumps(
            provenance
            | {
                "decision": decision,
                "candidate_zips": candidate_zips,
                "m0_r2_sha256": semantic_content_hash(candidates["M0_R2"]),
                "m1_r2_sha256": semantic_content_hash(candidates["M1_R2"]),
                "safe_r2_sha256": semantic_content_hash(candidates["SAFE_R2"]),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return decision | {"candidate_zips": candidate_zips}


__all__ = [
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "build_r2_candidates",
    "choose_r2_recommendation",
    "classify_asr_specificity",
    "derive_anchor_profiles",
    "rank_qa_r2",
    "repair_kis_ranking",
    "tier_evidence",
    "verify_answer_extraction",
    "write_r2_artifacts",
]
