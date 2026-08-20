"""GT-free Trial P1 R3 precision policy built on the frozen R2 runtime."""

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

from .r2_policy import (
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_ORDER,
    _complete_ranked_rows,
    _coordinate_key,
    _ordered_overlap,
    verify_answer_extraction,
)

ENTITY_HIGH = "ENTITY_PHRASE_HIGH"
DISTINCTIVE_HIGH = "DISTINCTIVE_PHRASE_HIGH"
OBJECT_MEDIUM = "SPECIFIC_OBJECT_MEDIUM"
ACTION_MEDIUM = "ACTION_MEDIUM"
COUNT_COLOR_MEDIUM = "COUNT_COLOR_MEDIUM"
GENERIC_LOW = "GENERIC_LOW"
HIGH_CLASSES = {ENTITY_HIGH, DISTINCTIVE_HIGH}
MEDIUM_CLASSES = {OBJECT_MEDIUM, ACTION_MEDIUM, COUNT_COLOR_MEDIUM}
MEANINGFUL_CLASSES = HIGH_CLASSES | MEDIUM_CLASSES

GENERIC_WORDS = frozenset(
    {
        "anh",
        "bang",
        "ben",
        "biet",
        "buoi",
        "cac",
        "can",
        "canh",
        "cau",
        "chinh",
        "cho",
        "chuong",
        "co",
        "con",
        "cua",
        "cung",
        "day",
        "dang",
        "de",
        "deu",
        "do",
        "doan",
        "duoi",
        "duoc",
        "gioi",
        "giua",
        "hay",
        "hien",
        "hinh",
        "hoi",
        "khi",
        "la",
        "lai",
        "len",
        "ma",
        "mon",
        "mot",
        "nha",
        "nghien",
        "nghiem",
        "nhieu",
        "nhu",
        "nhiem",
        "nguoi",
        "nhung",
        "noi",
        "phan",
        "phia",
        "qua",
        "ra",
        "roi",
        "sang",
        "sau",
        "su",
        "tai",
        "the",
        "thay",
        "ten",
        "theo",
        "thieu",
        "thuc",
        "tim",
        "tren",
        "truoc",
        "trong",
        "tu",
        "ve",
        "vi",
        "cuu",
        "vao",
        "video",
        "viec",
        "vu",
        "voi",
        "xuong",
        "xac",
    }
)
GENERIC_PHRASES = frozenset(
    {
        "chuong trinh",
        "doan clip",
        "doan video",
        "mon an",
        "nha tho",
        "nghien cuu",
    }
)
COLORS = frozenset(
    {"den", "trang", "do", "cam", "vang", "xanh", "tim", "hong", "nau", "xam"}
)
LOCATION_PATTERN = re.compile(
    r"\b(xã|phường|thị trấn|huyện|tỉnh|thành phố|quận|ấp|thôn|làng)\b", re.I
)
TITLE_WRAPPER = re.compile(
    r"^(?:món\s+ăn\s+có\s+tên\s+gọi\s+là|tên\s+món(?:\s+ăn)?\s+là|"
    r"công\s+thức(?:\s+này)?\s+là)\s*[:\-]?\s*",
    re.I,
)
_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char)).replace(
        "đ", "d"
    )


def _tokens(value: str, *, folded: bool = False) -> tuple[str, ...]:
    text = _fold(value) if folded else unicodedata.normalize("NFKC", str(value).casefold())
    return tuple(_TOKEN.findall(text))


def _unique(values: Iterable[Any], key: Any) -> list[Any]:
    output, seen = [], set()
    for value in values:
        identity = key(value)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(value)
    return output


def _contains_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[index : index + len(needle)] == needle for index in range(len(haystack)))


def _meaningful(tokens: Iterable[str]) -> list[str]:
    return [
        token
        for token in tokens
        if len(_fold(token)) >= 3 and _fold(token) not in GENERIC_WORDS
    ]


def _anchor(
    text: str,
    anchor_class: str,
    source: str,
) -> dict[str, Any] | None:
    normalized = _tokens(text)
    folded = _tokens(text, folded=True)
    if not normalized:
        return None
    if anchor_class in HIGH_CLASSES and len(normalized) == 1 and len(folded[0]) < 4:
        return None
    return {
        "text": " ".join(str(text).split()),
        "normalized_tokens": list(normalized),
        "folded_tokens": list(folded),
        "anchor_class": anchor_class,
        "source": source,
        "is_phrase": len(normalized) >= 2,
    }


def _proper_phrases(text: str, *, allow_single_title: bool = False) -> list[str]:
    raw_text = str(text)
    matches = list(_TOKEN.finditer(raw_text))
    values: list[str] = []
    run: list[tuple[int, str, int, int]] = []

    def flush() -> None:
        if not run:
            return
        selected = run[:5]
        if re.fullmatch(r"e\d+", _fold(selected[0][1])):
            selected = selected[1:]
        if not selected:
            run.clear()
            return
        phrase = " ".join(token for _, token, _, _ in selected)
        folded = " ".join(_tokens(phrase, folded=True))
        single = len(selected) == 1
        token = selected[0][1] if single else ""
        mixed_case_brand = single and any(char.isupper() for char in token[1:])
        if (
            folded not in GENERIC_PHRASES
            and folded not in GENERIC_WORDS
            and _meaningful(_tokens(phrase))
            and (
                not single
                or token.isupper()
                or mixed_case_brand
                or allow_single_title
            )
        ):
            values.append(phrase)
        run.clear()

    for index, match in enumerate(matches):
        token = match.group(0)
        named = bool(token and token[0].isupper())
        if named:
            if run and not raw_text[run[-1][3] : match.start()].isspace():
                flush()
            run.append((index, token, match.start(), match.end()))
        else:
            flush()
    flush()
    return list(dict.fromkeys(values))


def derive_r3_anchor_profiles(
    queries: list[dict[str, Any]], plans: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Classify query anchors without substring/stem matching or GT."""

    plan_by_id = {str(row["query_id"]): row for row in plans}
    document_frequency: Counter[str] = Counter()
    token_rows: dict[str, tuple[str, ...]] = {}
    raw_token_rows: dict[str, tuple[str, ...]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        raw = str(plan_by_id[query_id].get("raw_text", query.get("query", "")))
        values = tuple(_meaningful(_tokens(raw)))
        token_rows[query_id] = values
        raw_token_rows[query_id] = _tokens(raw)
        document_frequency.update(set(_fold(token) for token in values))

    output = {}
    for query in queries:
        query_id = str(query["query_id"])
        plan = plan_by_id[query_id]
        raw = str(plan.get("raw_text", query.get("query", "")))
        anchors: list[dict[str, Any]] = []
        for phrase in _proper_phrases(raw):
            value = _anchor(phrase, ENTITY_HIGH, "PROPER_OR_NAMED_PHRASE")
            if value:
                anchors.append(value)
        for quoted in re.findall(r"[“\"]([^”\"]{3,160})[”\"]", raw):
            value = _anchor(quoted, DISTINCTIVE_HIGH, "QUOTED_QUERY_PHRASE")
            if value and len(_meaningful(value["normalized_tokens"])) >= 2:
                anchors.append(value)
        expansions = plan.get("knowledge_expansions", [])
        for expansion in expansions:
            text = str(expansion.get("text", "")) if isinstance(expansion, dict) else str(expansion)
            value = _anchor(text, DISTINCTIVE_HIGH, "HIGH_CONFIDENCE_KNOWLEDGE_EXPANSION")
            if value:
                anchors.append(value)
            for phrase in _proper_phrases(text, allow_single_title=True):
                entity = _anchor(phrase, ENTITY_HIGH, "KNOWLEDGE_EXPANSION_ENTITY")
                if entity:
                    anchors.append(entity)

        tokens = token_rows[query_id]
        ngrams = []
        raw_tokens = raw_token_rows[query_id]
        for width in (4, 3, 2):
            for index in range(len(raw_tokens) - width + 1):
                gram = raw_tokens[index : index + width]
                meaningful = _meaningful(gram)
                rare = sum(document_frequency[_fold(token)] <= 2 for token in meaningful)
                if (
                    rare >= 2
                    and len(meaningful) / width >= 0.67
                    and " ".join(map(_fold, gram)) not in GENERIC_PHRASES
                ):
                    ngrams.append((rare, width, " ".join(gram)))
        for _, _, text in sorted(ngrams, key=lambda row: (-row[0], -row[1], row[2]))[:16]:
            value = _anchor(text, DISTINCTIVE_HIGH, "RARE_MULTIWORD_QUERY_PHRASE")
            if value:
                anchors.append(value)

        action_text = " ".join(map(str, plan.get("action_anchors", [])))
        action_tokens = set(_tokens(action_text, folded=True))
        for token in sorted(set(tokens), key=lambda item: (_fold(item), item)):
            folded = _fold(token)
            if folded in GENERIC_WORDS:
                continue
            anchor_class = ACTION_MEDIUM if folded in action_tokens else OBJECT_MEDIUM
            value = _anchor(token, anchor_class, "WHOLE_QUERY_TOKEN")
            if value:
                anchors.append(value)
        for token in _tokens(raw):
            folded = _fold(token)
            if folded.isdigit() or folded in COLORS:
                value = _anchor(token, COUNT_COLOR_MEDIUM, "COUNT_OR_COLOR_CONSTRAINT")
                if value:
                    anchors.append(value)
        for phrase in sorted(GENERIC_PHRASES):
            if _contains_sequence(_tokens(raw, folded=True), tuple(phrase.split())):
                value = _anchor(phrase, GENERIC_LOW, "GENERIC_SEMANTIC_PHRASE")
                if value:
                    anchors.append(value)

        priority = {
            ENTITY_HIGH: 0,
            DISTINCTIVE_HIGH: 1,
            OBJECT_MEDIUM: 2,
            ACTION_MEDIUM: 3,
            COUNT_COLOR_MEDIUM: 4,
            GENERIC_LOW: 5,
        }
        anchors.sort(
            key=lambda row: (
                priority[row["anchor_class"]],
                -len(row["normalized_tokens"]),
                row["text"],
            )
        )
        anchors = _unique(anchors, key=lambda row: tuple(row["folded_tokens"]))
        output[query_id] = {
            "query_id": query_id,
            "raw_query": raw,
            "anchors": anchors,
            "high_anchors": [row for row in anchors if row["anchor_class"] in HIGH_CLASSES],
            "meaningful_anchors": [
                row for row in anchors if row["anchor_class"] in MEANINGFUL_CLASSES
            ],
            "gt_used": False,
            "location_granularity": next(
                (
                    value
                    for value in (
                        "xã",
                        "phường",
                        "thị trấn",
                        "huyện",
                        "tỉnh",
                        "thành phố",
                        "quận",
                        "ấp",
                        "thôn",
                        "làng",
                    )
                    if _contains_sequence(
                        _tokens(raw, folded=True), _tokens(value, folded=True)
                    )
                ),
                None,
            )
            if str(query.get("answer_type", "")).upper() == "LOCATION_NAME"
            else None,
            "requested_quote_line_count": (
                2
                if re.search(r"\b(?:hai|2)\s+(?:câu|dòng)\b", raw, re.I)
                else None
            ),
        }
    return output


def match_r3_anchors(text: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    normalized, folded = _tokens(text), _tokens(text, folded=True)
    matches = []
    for anchor in profile.get("anchors", []):
        anchor_tokens = tuple(anchor["normalized_tokens"])
        folded_tokens = tuple(anchor["folded_tokens"])
        if _contains_sequence(normalized, anchor_tokens):
            reason = "EXACT_NORMALIZED_TOKEN_SEQUENCE"
        elif _contains_sequence(folded, folded_tokens):
            reason = "DIACRITIC_FOLDED_EXACT_TOKEN_SEQUENCE"
        else:
            continue
        matches.append({**anchor, "match_reason": reason})
    return matches


def _branch_ranks(row: dict[str, Any]) -> tuple[int | None, int | None]:
    values = {
        str(item.get("branch", "")).upper(): int(item["rank"])
        for item in row.get("asr_source_ranks", [])
        if item.get("rank") is not None
    }
    return values.get("LEXICAL"), values.get("E5")


def classify_asr_r3(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    visual_corroborated: bool = False,
) -> dict[str, Any]:
    span = dict(row.get("asr_span") or row)
    matches = match_r3_anchors(str(span.get("text", "")), profile)
    high = [match for match in matches if match["anchor_class"] in HIGH_CLASSES]
    meaningful = [match for match in matches if match["anchor_class"] in MEANINGFUL_CLASSES]
    lexical_rank, e5_rank = _branch_ranks(row)
    agreement = lexical_rank is not None and e5_rank is not None
    exact_high_phrase = any(match["is_phrase"] for match in high)
    exact_high_entity = any(match["anchor_class"] == ENTITY_HIGH for match in high)
    if exact_high_phrase or exact_high_entity:
        tier, reason = TIER_A, "EXACT_HIGH_ENTITY_OR_DISTINCTIVE_PHRASE"
    elif len({tuple(match["folded_tokens"]) for match in meaningful}) >= 2 and agreement:
        tier, reason = TIER_A, "TWO_MEANINGFUL_ANCHORS_WITH_LEXICAL_E5_AGREEMENT"
    elif meaningful and agreement:
        tier, reason = TIER_B, "ONE_MEANINGFUL_ANCHOR_WITH_LEXICAL_E5_AGREEMENT"
    elif meaningful and e5_rank is not None and visual_corroborated:
        tier, reason = TIER_B, "MEANINGFUL_E5_WITH_BCF1_VISUAL_CORROBORATION"
    else:
        tier, reason = TIER_C, "GENERIC_OR_LOW_SPECIFICITY_ASR_OVERLAP"
    return {
        **row,
        "evidence_tier": tier,
        "evidence_tier_reasons": [reason],
        "matched_query_anchors": [match["text"] for match in matches],
        "matched_phrase_anchors": [match["text"] for match in matches if match["is_phrase"]],
        "matched_token_anchors": [match["text"] for match in matches if not match["is_phrase"]],
        "matched_anchor_classes": [match["anchor_class"] for match in matches],
        "anchor_matches": matches,
        "asr_specificity": {
            "lexical_rank": lexical_rank,
            "e5_rank": e5_rank,
            "lexical_e5_agreement": agreement,
            "bcf1_visual_corroboration": visual_corroborated,
            "meaningful_anchor_count": len(
                {tuple(match["folded_tokens"]) for match in meaningful}
            ),
            "exact_high_phrase": exact_high_phrase,
            "tier_reason": reason,
        },
    }


def _classify_other_r3(
    modality: str, row: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    matches = match_r3_anchors(str(row.get("text", "")), profile)
    high_phrases = [
        match
        for match in matches
        if match["anchor_class"] in HIGH_CLASSES and match["is_phrase"]
    ]
    meaningful = [match for match in matches if match["anchor_class"] in MEANINGFUL_CLASSES]
    if modality == "ocr":
        confidence = float(row.get("source_confidence") or 0.0)
        normalized_confidence = confidence / 100.0 if confidence > 1.0 else confidence
        if high_phrases and normalized_confidence >= 0.5:
            tier, reason = TIER_A, "OCR_EXACT_HIGH_MULTIWORD_PHRASE_WITH_CONFIDENCE"
        elif meaningful and normalized_confidence >= 0.5:
            tier, reason = TIER_B, "OCR_MEANINGFUL_ANCHOR_WITH_CONFIDENCE"
        else:
            tier, reason = TIER_C, "OCR_GENERIC_OR_ISOLATED_TOKEN_ONLY"
    elif modality == "object":
        tier, reason = TIER_C, "OBJECT_ONLY_ALWAYS_WEAK"
    elif modality == "action":
        rank = int(row.get("rank", 10**9))
        tier = TIER_A if rank <= 3 else TIER_B if rank <= 10 else TIER_C
        reason = f"XCLIP_RELEVANT_EVENT_RANK_{rank}"
    else:
        tier, reason = TIER_C, f"{modality.upper()}_UNCLASSIFIED_WEAK"
    return {
        **row,
        "evidence_tier": tier,
        "evidence_tier_reasons": [reason],
        "matched_query_anchors": [match["text"] for match in matches],
        "matched_phrase_anchors": [match["text"] for match in matches if match["is_phrase"]],
        "matched_token_anchors": [match["text"] for match in matches if not match["is_phrase"]],
        "matched_anchor_classes": [match["anchor_class"] for match in matches],
        "anchor_matches": matches,
    }


def tier_evidence_r3(
    queries: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
    profiles: dict[str, dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    output = {
        name: {key: [dict(row) for row in rows] for key, rows in values.items()}
        for name, values in evidence.items()
    }
    diagnostics = []
    baseline_videos = {
        query_id: {str(row["video_id"]) for row in rows}
        for query_id, rows in grouped(baseline_rows or []).items()
    }
    for query in queries:
        query_id = str(query["query_id"])
        for modality in ("asr", "ocr", "object", "action", "action_revision"):
            rows = []
            for raw in evidence.get(modality, {}).get(query_id, []):
                row = (
                    classify_asr_r3(
                        raw,
                        profiles[query_id],
                        visual_corroborated=str(raw.get("video_id"))
                        in baseline_videos.get(query_id, set()),
                    )
                    if modality == "asr"
                    else _classify_other_r3(
                        "action" if modality == "action_revision" else modality,
                        raw,
                        profiles[query_id],
                    )
                )
                rows.append(row)
                diagnostics.append(
                    {
                        "query_id": query_id,
                        "modality": modality,
                        "video_id": row.get("video_id"),
                        "frame_id": row.get("frame_id"),
                        "source_rank": raw.get("rank"),
                        "evidence_tier": row["evidence_tier"],
                        "tier_reasons": row["evidence_tier_reasons"],
                        "anchor_matches": row.get("anchor_matches", []),
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


def _video_assessments_r3(
    baseline: list[dict[str, Any]], modality_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "modalities": set(),
            "tiers": [],
            "reasons": [],
            "anchors": [],
            "high_anchors": [],
            "rows": [],
            "bcf1_rank": None,
        }
    )
    for row in baseline:
        item = values[str(row["video_id"])]
        rank = int(row["rank"])
        item["bcf1_rank"] = rank if item["bcf1_rank"] is None else min(item["bcf1_rank"], rank)
    for modality, rows in modality_rows.items():
        for row in rows:
            item = values[str(row["video_id"])]
            item["modalities"].add(modality)
            item["tiers"].append(row.get("evidence_tier", TIER_C))
            item["reasons"].extend(row.get("evidence_tier_reasons", []))
            item["anchors"].extend(row.get("matched_query_anchors", []))
            item["high_anchors"].extend(
                match["text"]
                for match in row.get("anchor_matches", [])
                if match.get("anchor_class") in HIGH_CLASSES
            )
            item["rows"].append((modality, row))
    for item in values.values():
        modalities = set(item["modalities"])
        direct = [
            (modality, row)
            for modality, row in item["rows"]
            if row.get("evidence_tier") == TIER_A and modality != "object"
        ]
        coordinates: dict[tuple[Any, ...], set[str]] = defaultdict(set)
        for modality, row in item["rows"]:
            coordinates[_coordinate_key(row)].add(modality)
        same_coordinate_corroboration = any(len(names) >= 2 for names in coordinates.values())
        if direct:
            tier = TIER_A
        elif TIER_B in item["tiers"] or same_coordinate_corroboration:
            tier = TIER_B
        else:
            tier = TIER_C
        trusted = {name for name in modalities if name in {"asr", "ocr", "action"}}
        if item["bcf1_rank"] is not None:
            trusted.add("b0_visual")
        item["evidence_tier"] = tier
        item["strong_corroboration"] = bool(tier == TIER_A and len(trusted) >= 2)
        item["modalities"] = sorted(modalities)
        item["trusted_modalities"] = sorted(trusted)
        item["reasons"] = list(dict.fromkeys(item["reasons"]))
        item["anchors"] = list(dict.fromkeys(item["anchors"]))
        item["high_anchors"] = list(dict.fromkeys(item["high_anchors"]))
    return dict(values)


def _representative_r3(
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
    source = rows[0][1] if rows else baseline_row
    if source is None:
        raise RuntimeError(f"R3_CANDIDATE_WITHOUT_REPRESENTATIVE:{video_id}")
    return {
        **source,
        "evidence_tier": assessment["evidence_tier"],
        "evidence_tier_reasons": assessment["reasons"],
        "dominant_modalities": assessment["modalities"],
        "corroborating_sources": assessment["trusted_modalities"],
        "matched_query_anchors": assessment["anchors"],
        "matched_high_query_anchors": assessment["high_anchors"],
        "strong_corroboration": assessment["strong_corroboration"],
        "bcf1_rank": assessment["bcf1_rank"],
    }


def _qualified_strong_asr(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("evidence_tier") == TIER_A
        and (
            (row.get("asr_specificity") or {}).get("exact_high_phrase")
            or (
                (row.get("asr_specificity") or {}).get("lexical_e5_agreement")
                and int((row.get("asr_specificity") or {}).get("meaningful_anchor_count", 0)) >= 2
            )
        )
    ]


def repair_kis_r3(
    query: dict[str, Any],
    baseline: list[dict[str, Any]],
    tiered_evidence: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    modalities = {name: tiered_evidence.get(name, []) for name in ("asr", "ocr", "object")}
    assessments = _video_assessments_r3(baseline, modalities)
    representatives = {
        video_id: _representative_r3(video_id, item, baseline)
        for video_id, item in assessments.items()
    }

    def ordered(predicate: Any) -> list[dict[str, Any]]:
        allowed = {video_id for video_id, item in assessments.items() if predicate(item)}
        sources = [
            [row for row in rows if str(row["video_id"]) in allowed]
            for rows in modalities.values()
        ]
        sources = [rows for rows in sources if rows]
        if not sources:
            return [representatives[video_id] for video_id in sorted(allowed)]
        fused = reciprocal_rank_fusion(sources, key=lambda row: str(row["video_id"]))
        return [representatives[str(row["video_id"])] for row in fused]

    strong_a = ordered(
        lambda item: item["evidence_tier"] == TIER_A and item["strong_corroboration"]
    )
    other_a = ordered(
        lambda item: item["evidence_tier"] == TIER_A and not item["strong_corroboration"]
    )
    tier_b = ordered(lambda item: item["evidence_tier"] == TIER_B)
    tier_c = ordered(lambda item: item["evidence_tier"] == TIER_C)
    annotated_baseline = [
        {
            **row,
            **{
                key: value
                for key, value in representatives[str(row["video_id"])].items()
                if key
                in {
                    "evidence_tier",
                    "evidence_tier_reasons",
                    "dominant_modalities",
                    "corroborating_sources",
                    "matched_query_anchors",
                    "matched_high_query_anchors",
                    "strong_corroboration",
                    "bcf1_rank",
                }
            },
        }
        for row in baseline
    ]
    weak_sources = [
        [row for row in rows if assessments[str(row["video_id"])]["evidence_tier"] == TIER_C]
        for rows in modalities.values()
    ]
    normal = reciprocal_rank_fusion([baseline, *weak_sources], key=_coordinate_key)
    full = _complete_ranked_rows(
        [*strong_a, *annotated_baseline[:5], *other_a, *tier_b, *normal, *tier_c],
        annotated_baseline,
        query_id=str(query["query_id"]),
        variant="M0_R3",
    )
    qualified = _qualified_strong_asr(modalities["asr"])
    best = min(qualified, key=lambda row: int(row.get("rank", 10**9)), default=None)
    inclusion = {
        "query_id": query["query_id"],
        "qualified_direct_count": len(qualified),
        "best_strong_asr_video": str(best["video_id"]) if best else None,
        "best_strong_asr_source_rank": int(best["rank"]) if best else None,
        "final_best_rank": None,
        "inclusion_status": "NO_QUALIFIED_DIRECT_ASR",
        "displacement_reason": None,
    }
    if best:
        video_id = str(best["video_id"])
        final_rank = next(
            (row["rank"] for row in full[:20] if str(row["video_id"]) == video_id), None
        )
        if final_rank is None:
            candidate = representatives[video_id]
            remainder = [row for row in full if _coordinate_key(row) != _coordinate_key(candidate)]
            full = _complete_ranked_rows(
                [*remainder[:19], candidate, *remainder[19:]],
                annotated_baseline,
                query_id=str(query["query_id"]),
                variant="M0_R3",
            )
            final_rank = next(
                row["rank"] for row in full[:20] if str(row["video_id"]) == video_id
            )
            inclusion["inclusion_status"] = "GUARANTEED_INSERTION_AT_OR_BEFORE_TOP20"
        else:
            inclusion["inclusion_status"] = "ALREADY_INCLUDED_BY_VIDEO_HYPOTHESIS"
        inclusion["final_best_rank"] = final_rank
    protected_keys = {_coordinate_key(row) for row in baseline[:5]}
    safe_tail = [row for row in full if _coordinate_key(row) not in protected_keys]
    safe = _complete_ranked_rows(
        [*annotated_baseline[:5], *safe_tail],
        annotated_baseline,
        query_id=str(query["query_id"]),
        variant="SAFE_R3",
    )
    if [_coordinate_key(row) for row in safe[:5]] != [
        _coordinate_key(row) for row in baseline[:5]
    ]:
        raise RuntimeError(f"SAFE_R3_BCF1_TOP5_NOT_EXACT:{query['query_id']}")
    object_only_a = [
        video_id
        for video_id, item in assessments.items()
        if item["evidence_tier"] == TIER_A and set(item["modalities"]) == {"object"}
    ]
    ocr_object_a = [
        video_id
        for video_id, item in assessments.items()
        if item["evidence_tier"] == TIER_A
        and set(item["modalities"]).issubset({"ocr", "object"})
        and not any(
            row.get("evidence_tier") == TIER_A
            and row.get("matched_phrase_anchors")
            for modality, row in item["rows"]
            if modality == "ocr"
        )
    ]
    if object_only_a or ocr_object_a:
        raise RuntimeError(
            f"R3_FORBIDDEN_WEAK_TIER_A:{query['query_id']}:{object_only_a}:{ocr_object_a}"
        )
    return full, safe, {
        "query_id": query["query_id"],
        "strong_asr_inclusion": inclusion,
        "safe_top5_exact": True,
        "object_only_tier_a_count": len(object_only_a),
        "ocr_object_without_direct_phrase_tier_a_count": len(ocr_object_a),
        "candidate_assessments": {
            video_id: {key: value for key, value in item.items() if key != "rows"}
            for video_id, item in assessments.items()
        },
    }


def evaluate_context_relevance(
    profile: dict[str, Any], evidence_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    matches = []
    for row in evidence_rows:
        text = str(row.get("text", ""))
        for match in match_r3_anchors(text, profile):
            matches.append(
                {
                    **match,
                    "source_id": row.get("source_id"),
                    "modality": row.get("modality"),
                }
            )
    high = [match for match in matches if match["anchor_class"] in HIGH_CLASSES]
    meaningful = [match for match in matches if match["anchor_class"] in MEANINGFUL_CLASSES]
    unique_meaningful = {tuple(match["folded_tokens"]) for match in meaningful}
    if high:
        status, reason = True, "EXACT_HIGH_QUERY_CONTEXT_ANCHOR"
    elif len(unique_meaningful) >= 2:
        status, reason = True, "MULTIPLE_MEANINGFUL_QUERY_CONTEXT_ANCHORS"
    else:
        status, reason = False, "QUERY_CONTEXT_NOT_ESTABLISHED"
    return {
        "context_relevant": status,
        "context_reason": reason,
        "context_anchor_matches": matches,
        "high_context_anchor_count": len(high),
        "meaningful_context_anchor_count": len(unique_meaningful),
    }


def canonicalize_title(answer: str) -> tuple[str, bool]:
    compact = " ".join(str(answer).split())
    canonical = TITLE_WRAPPER.sub("", compact).strip(" :-–—")
    return canonical or compact, canonical != compact


def verify_answer_r3(
    extraction: dict[str, Any],
    answer_type: str,
    evidence_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    grounding_plausibility: float,
) -> dict[str, Any]:
    base = verify_answer_extraction(
        extraction,
        answer_type,
        evidence_rows,
        grounding_plausibility=grounding_plausibility,
    )
    context = evaluate_context_relevance(profile, evidence_rows)
    non_qwen_reasons = [
        reason for reason in base["verifier_reasons"] if reason != "QWEN_DECLARED_INSUFFICIENT"
    ]
    type_pass = bool(base["evidence_type_compatible"] and not non_qwen_reasons)
    original = str(base["answer"])
    canonical, changed = (
        canonicalize_title(original)
        if str(answer_type).upper() == "TITLE" and type_pass
        else (original, False)
    )
    if not canonical or len(canonical) > 100:
        type_pass = False
        non_qwen_reasons.append("R3_CANONICAL_ANSWER_INVALID")
    kind = str(answer_type).upper()
    cited_ids = set(base.get("verified_supporting_source_ids", []))
    cited_text = [
        str(row.get("text", ""))
        for row in evidence_rows
        if str(row.get("source_id")) in cited_ids
    ]
    if kind == "LOCATION_NAME" and profile.get("location_granularity"):
        granularity = str(profile["location_granularity"])
        supported_granularity = any(
            _contains_sequence(_tokens(text, folded=True), _tokens(granularity, folded=True))
            and _ordered_overlap(canonical, text) >= 0.8
            for text in cited_text
        )
        if not supported_granularity:
            type_pass = False
            non_qwen_reasons.append("LOCATION_GRANULARITY_NOT_SUPPORTED")
    requested_lines = profile.get("requested_quote_line_count")
    if kind == "QUOTE_OR_VISIBLE_TEXT" and requested_lines:
        spans = [str(value).strip() for value in extraction.get("supporting_spans", [])]
        if len({value for value in spans if value}) < int(requested_lines):
            type_pass = False
            non_qwen_reasons.append("QUOTE_REQUESTED_LINE_COUNT_NOT_SUPPORTED")
    qwen_supports = extraction.get("evidence_sufficient") is True
    final = bool(qwen_supports and context["context_relevant"] and type_pass)
    return {
        **base,
        **context,
        "unstripped_answer": original,
        "answer": canonical,
        "title_canonicalization_applied": changed,
        "qwen_supports": qwen_supports,
        "answer_type_verifier_pass": type_pass,
        "answer_type_verifier_reasons": non_qwen_reasons,
        "final_evidence_sufficient": final,
    }


def rank_qa_r3(
    query: dict[str, Any], verified_rows: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sufficient = [
        {
            **row,
            "evidence_tier": TIER_A,
            "evidence_tier_reasons": ["QWEN_CONTEXT_AND_TYPE_VERIFIER_PASS"],
            "dominant_modalities": row.get("corroborating_modalities", []),
            "matched_query_anchors": [
                match["text"] for match in row.get("context_anchor_matches", [])
            ],
            "matched_high_query_anchors": [
                match["text"]
                for match in row.get("context_anchor_matches", [])
                if match.get("anchor_class") in HIGH_CLASSES
            ],
            "corroborating_sources": row.get("verified_supporting_source_ids", []),
        }
        for row in verified_rows
        if row.get("final_evidence_sufficient") is True
    ]
    sufficient.sort(
        key=lambda row: (
            -int(row.get("high_context_anchor_count", 0)),
            -int(row.get("corroborating_source_count", 0)),
            -float(row.get("grounding_plausibility", 0.0)),
            int(row.get("grounding_rank", row.get("rank", 10**9))),
            default_key(row),
        )
    )
    fallback = [
        {
            **row,
            "answer": "không đủ bằng chứng",
            "qwen_supports": False,
            "context_relevant": False,
            "answer_type_verifier_pass": False,
            "final_evidence_sufficient": False,
            "evidence_tier": TIER_C,
            "evidence_tier_reasons": ["QA_R3_STRUCTURAL_FALLBACK"],
            "dominant_modalities": [],
                "matched_query_anchors": [],
                "matched_high_query_anchors": [],
            "corroborating_sources": [],
        }
        for row in baseline
    ]
    return _complete_ranked_rows(
        [*sufficient, *fallback],
        fallback,
        query_id=str(query["query_id"]),
        variant="QA_R3",
        identity=default_key,
    )


def augment_qa_context_r3(
    profile: dict[str, Any],
    candidate_video: str,
    local_rows: list[dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Keep local answer evidence and add bounded same-video HIGH context rows."""

    if limit < 3:
        raise ValueError("R3_QA_CONTEXT_LIMIT_TOO_SMALL")
    output = [dict(row) for row in local_rows[:limit]]
    seen = {str(row.get("source_id")) for row in output if row.get("source_id")}
    candidates = []
    for modality in ("asr", "ocr"):
        for raw in evidence.get(modality, []):
            if str(raw.get("video_id")) != str(candidate_video):
                continue
            span = dict(raw.get("asr_span") or {})
            text = str(raw.get("text") or span.get("text") or "").strip()
            matches = [
                match
                for match in match_r3_anchors(text, profile)
                if match["anchor_class"] in HIGH_CLASSES
            ]
            if not matches:
                continue
            frame_id = int(raw.get("frame_id", 0))
            source_id = (
                f"asr:{span.get('chunk_id')}"
                if modality == "asr" and span.get("chunk_id")
                else f"{modality}:{candidate_video}:{frame_id}:{int(raw.get('rank', 999999))}"
            )
            candidates.append(
                (
                    -len(matches),
                    int(raw.get("rank", 999999)),
                    source_id,
                    {
                        "source_id": source_id,
                        "modality": modality,
                        "video_id": str(candidate_video),
                        "frame_id": frame_id,
                        "distance_frames": None,
                        "time_distance_seconds": None,
                        "text": text,
                        "rank": int(raw.get("rank", 999999)),
                        "confidence": raw.get("source_confidence"),
                        "asr_span": span or None,
                        "source": raw.get("source"),
                        "r3_context_anchor_matches": [match["text"] for match in matches],
                    },
                )
            )
    for _, _, source_id, row in sorted(candidates, key=lambda item: item[:3]):
        if source_id in seen:
            continue
        seen.add(source_id)
        output.append(row)
        if len(output) == limit:
            break
    return output


def build_bounded_qa_rescue_evidence(
    queries: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    loader: Any,
    ocr_parquet: str | Path,
    canonical_mapper: Any,
    initial_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    baseline_rows: list[dict[str, Any]],
    *,
    max_videos: int = 5,
    max_rows: int = 40,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Read-only QA rescue inside already shortlisted context videos."""

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - Kaggle dependency gate
        raise RuntimeError("PYARROW_REQUIRED_FOR_R3_QA_RESCUE") from error
    baseline = grouped(baseline_rows)
    output: dict[str, dict[str, list[dict[str, Any]]]] = {"asr": {}, "ocr": {}}
    diagnostics = []
    for query in (row for row in queries if row["task"] == "QA"):
        query_id = str(query["query_id"])
        profile = profiles[query_id]
        video_scores: dict[str, tuple[int, int]] = {}
        for modality in ("asr", "ocr"):
            for row in initial_evidence.get(modality, {}).get(query_id, []):
                matches = match_r3_anchors(
                    str(row.get("text") or row.get("asr_span", {}).get("text", "")),
                    profile,
                )
                high = sum(match["anchor_class"] in HIGH_CLASSES for match in matches)
                meaningful = sum(match["anchor_class"] in MEANINGFUL_CLASSES for match in matches)
                video_id = str(row["video_id"])
                score = (high * 10 + meaningful * 2, -int(row.get("rank", 10**9)))
                video_scores[video_id] = max(video_scores.get(video_id, (-1, -10**9)), score)
        ordered_videos = [
            video_id
            for video_id, score in sorted(
                video_scores.items(), key=lambda item: (-item[1][0], -item[1][1], item[0])
            )
            if score[0] > 0
        ]
        if not ordered_videos:
            ordered_videos = [
                str(row["video_id"])
                for row in initial_evidence.get("asr", {}).get(query_id, [])[:max_videos]
            ]
        if len(ordered_videos) < max_videos:
            ordered_videos.extend(
                str(row["video_id"])
                for row in baseline[query_id]
                if str(row["video_id"]) not in ordered_videos
            )
        selected_videos = list(dict.fromkeys(ordered_videos))[:max_videos]
        selected_set = set(selected_videos)
        ocr_rows = pq.read_table(
            Path(ocr_parquet),
            columns=[
                "video_id",
                "frame_idx",
                "corrected_text",
                "combined_text",
                "mean_confidence",
            ],
            filters=[("video_id", "in", selected_videos)],
        ).to_pylist()
        context_times: dict[str, list[float]] = defaultdict(list)
        for row in loader.transcripts:
            if str(row["video_id"]) not in selected_set:
                continue
            if match_r3_anchors(str(row.get("text", "")), profile):
                context_times[str(row["video_id"])].append(
                    (float(row["start_seconds"]) + float(row["end_seconds"])) / 2
                )
        asr_candidates = []
        for row in loader.transcripts:
            video_id = str(row["video_id"])
            if video_id not in selected_set or not str(row.get("text", "")).strip():
                continue
            text = str(row["text"])
            midpoint = (float(row["start_seconds"]) + float(row["end_seconds"])) / 2
            matches = match_r3_anchors(text, profile)
            local = min(
                (abs(midpoint - value) for value in context_times.get(video_id, [])),
                default=float("inf"),
            )
            kind = str(query.get("answer_type", "OTHER")).upper()
            answer_bearing = (
                bool(LOCATION_PATTERN.search(text))
                if kind == "LOCATION_NAME"
                else local <= 180
            )
            if not answer_bearing and not matches:
                continue
            mapped = loader.map_span_to_frame(row, canonical_mapper)
            asr_candidates.append(
                {
                    "query_id": query_id,
                    "video_id": video_id,
                    "frame_id": int(mapped["frame_id"]),
                    "source": "asr_external_v3_r3_bounded_rescue",
                    "text": text,
                    "asr_span": mapped,
                    "source_confidence": None,
                    "rescue_context_distance_seconds": local if local != float("inf") else None,
                    "rescue_anchor_matches": [match["text"] for match in matches],
                }
            )
        asr_candidates.sort(
            key=lambda row: (
                0 if LOCATION_PATTERN.search(row["text"]) else 1,
                -len(row["rescue_anchor_matches"]),
                row.get("rescue_context_distance_seconds") or 10**9,
                row["video_id"],
                row["frame_id"],
            )
        )
        ocr_candidates = []
        for raw in ocr_rows:
            video_id = str(raw["video_id"])
            if video_id not in selected_set:
                continue
            text = str(raw.get("corrected_text") or raw.get("combined_text") or "").strip()
            if not text:
                continue
            matches = match_r3_anchors(text, profile)
            kind = str(query.get("answer_type", "OTHER")).upper()
            if kind == "LOCATION_NAME" and not LOCATION_PATTERN.search(text) and not matches:
                continue
            if kind == "QUOTE_OR_VISIBLE_TEXT" and not matches:
                continue
            ocr_candidates.append(
                {
                    "query_id": query_id,
                    "video_id": video_id,
                    "frame_id": int(raw["frame_idx"]),
                    "source": "ocr_external_v3_r3_bounded_rescue",
                    "text": text,
                    "source_confidence": float(raw.get("mean_confidence") or 0.0),
                    "rescue_anchor_matches": [match["text"] for match in matches],
                }
            )
        ocr_candidates.sort(
            key=lambda row: (
                0 if LOCATION_PATTERN.search(row["text"]) else 1,
                -len(row["rescue_anchor_matches"]),
                -float(row["source_confidence"]),
                row["video_id"],
                row["frame_id"],
            )
        )
        for modality, rows in (("asr", asr_candidates), ("ocr", ocr_candidates)):
            selected = _unique(rows, key=lambda row: (_coordinate_key(row), row["text"]))[:max_rows]
            output[modality][query_id] = [
                {**row, "rank": rank} for rank, row in enumerate(selected, 1)
            ]
        diagnostics.append(
            {
                "query_id": query_id,
                "selected_context_videos": selected_videos,
                "asr_rescue_count": len(output["asr"][query_id]),
                "ocr_rescue_count": len(output["ocr"][query_id]),
                "corpus_job_launched": False,
                "gt_used": False,
            }
        )
    return output, diagnostics


def build_r3_candidates(
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
    m0_group, m1_group, safe_group = map(grouped, (m0_raw, m1_raw, trake_safe))
    candidates = {"M0_R3": [], "M1_R3": [], "SAFE_R3": []}
    kis_diagnostics = []
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"]).upper()
        if task == "KIS":
            full, safe, diagnostic = repair_kis_r3(
                query,
                baseline[query_id],
                {
                    name: tiered_evidence.get(name, {}).get(query_id, [])
                    for name in ("asr", "ocr", "object")
                },
            )
            candidates["M0_R3"].extend(full)
            candidates["M1_R3"].extend({**row, "system_variant": "M1_R3"} for row in full)
            candidates["SAFE_R3"].extend(safe)
            kis_diagnostics.append(diagnostic)
        elif task == "QA":
            rows = rank_qa_r3(query, verified_qa.get(query_id, []), baseline[query_id])
            for name in candidates:
                candidates[name].extend({**row, "system_variant": name} for row in rows)
        else:
            support: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in tiered_evidence.get("action", {}).get(query_id, []):
                support[str(row["video_id"])].append(row)

            def annotate(
                row: dict[str, Any],
                variant: str,
                support_rows: dict[str, list[dict[str, Any]]] = support,
            ) -> dict[str, Any]:
                values = support_rows.get(str(row["video_id"]), [])
                tier = min(
                    (item.get("evidence_tier", TIER_C) for item in values),
                    key=lambda value: TIER_ORDER[value],
                    default=TIER_C,
                )
                return {
                    **row,
                    "system_variant": variant,
                    "evidence_tier": tier,
                    "evidence_tier_reasons": list(
                        dict.fromkeys(
                            reason
                            for item in values
                            for reason in item.get("evidence_tier_reasons", [])
                        )
                    )
                    or ["B0_TEMPORAL_CHAIN_WITHOUT_XCLIP_SUPPORT"],
                    "dominant_modalities": ["b0_visual", *("xclip" for _ in values[:1])],
                    "matched_query_anchors": list(
                        dict.fromkeys(
                            anchor
                            for item in values
                            for anchor in item.get("matched_query_anchors", [])
                        )
                    ),
                    "matched_high_query_anchors": list(
                        dict.fromkeys(
                            match["text"]
                            for item in values
                            for match in item.get("anchor_matches", [])
                            if match.get("anchor_class") in HIGH_CLASSES
                        )
                    ),
                    "corroborating_sources": [
                        f"xclip:{item.get('video_id')}:{item.get('event_index')}"
                        for item in values
                    ],
                }

            fallback = [annotate(row, "B0_FALLBACK") for row in baseline[query_id]]
            for name, source in (
                ("M0_R3", m0_group[query_id]),
                ("M1_R3", m1_group[query_id]),
                ("SAFE_R3", safe_group[query_id]),
            ):
                candidates[name].extend(
                    _complete_ranked_rows(
                        (annotate(row, name) for row in source),
                        fallback,
                        query_id=query_id,
                        variant=name,
                    )
                )
    validation = {}
    for name, rows in candidates.items():
        summary, issues = validate_predictions(queries, rows, inventory=inventory)
        counts = {key: len(values) for key, values in grouped(rows).items()}
        exact = len(rows) == 2400 and set(counts.values()) == {100}
        validation[name] = {
            **summary,
            "exact_100_per_query": exact,
            "per_query_counts": counts,
            "issues": issues,
        }
        if summary["status"] != "PASS" or not exact:
            raise RuntimeError(
                "TRIAL_R3_CANDIDATE_VALIDATION_FAILED:"
                + json.dumps({"variant": name, "validation": validation[name]}, default=str)
            )
    diagnostics_by_id = {
        str(row["query_id"]): row for row in m1_diagnostics if row.get("task") == "TRAKE"
    }
    trake_checks = {}
    for query in (row for row in queries if row["task"] == "TRAKE"):
        query_id = str(query["query_id"])
        arms = {}
        for name, rows in candidates.items():
            selected = grouped(rows)[query_id]
            arms[name] = {
                "event_count_correct": all(
                    len(row.get("frame_ids", [])) == int(query["event_count"])
                    for row in selected
                ),
                "strictly_increasing": all(
                    all(left < right for left, right in zip(frames, frames[1:], strict=False))
                    for frames in (row["frame_ids"] for row in selected)
                ),
            }
        graph = (diagnostics_by_id.get(query_id) or {}).get("graph") or {}
        graph_pass = bool(
            graph.get("query_event_count") == int(query["event_count"])
            and graph.get("revision_count") == 1
            and (graph.get("revision") or {}).get("evidence_added", 0) > 0
            and graph.get("chain_candidates_added", 0) > 0
        )
        trake_checks[query_id] = {"arms": arms, "graph_causal_gate_pass": graph_pass}
        if not graph_pass or not all(
            value["event_count_correct"] and value["strictly_increasing"]
            for value in arms.values()
        ):
            raise RuntimeError(f"TRIAL_R3_TRAKE_GATE_FAILED:{query_id}")
    return {
        "candidates": candidates,
        "validation": validation,
        "kis_diagnostics": kis_diagnostics,
        "strong_asr_audit": [row["strong_asr_inclusion"] for row in kis_diagnostics],
        "m0_diagnostics": m0_diagnostics,
        "m1_diagnostics": m1_diagnostics,
        "trake_checks": trake_checks,
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_r3_artifacts(
    root: str | Path,
    queries: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    r2_candidates: dict[str, list[dict[str, Any]]],
    result: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    tier_diagnostics: list[dict[str, Any]],
    qa_extractions: list[dict[str, Any]],
    qa_verifications: list[dict[str, Any]],
    rescue_diagnostics: list[dict[str, Any]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    candidates = result["candidates"]
    groups = {name: grouped(rows) for name, rows in r2_candidates.items()} | {
        name: grouped(rows) for name, rows in candidates.items()
    }
    baseline = grouped(baseline_rows)
    qa_readiness = {}
    for query in (row for row in queries if row["task"] == "QA"):
        query_id = str(query["query_id"])
        sufficient = [
            row
            for row in groups["M0_R3"][query_id]
            if row.get("final_evidence_sufficient") is True
        ]
        qa_readiness[query_id] = {
            "verified_sufficient_count": len(sufficient),
            "status": (
                "READY_FOR_MANUAL_REVIEW" if sufficient else "QA_NOT_READY_FOR_SERIOUS_SUBMISSION"
            ),
            "sufficient_answers": sufficient,
        }
    hard = bool(
        all(value["status"] == "PASS" for value in result["validation"].values())
        and all(value["verified_sufficient_count"] > 0 for value in qa_readiness.values())
        and all(row["inclusion_status"] != "DROPPED" for row in result["strong_asr_audit"])
        and all(row["safe_top5_exact"] for row in result["kis_diagnostics"])
        and all(
            row.get("object_only_tier_a_count", 0) == 0
            and row.get("ocr_object_without_direct_phrase_tier_a_count", 0) == 0
            for row in result["kis_diagnostics"]
        )
        and int(provenance.get("runtime_candidate_failure_count", 0)) == 0
        and all(
            row["graph_causal_gate_pass"]
            and all(
                arm["event_count_correct"] and arm["strictly_increasing"]
                for arm in row["arms"].values()
            )
            for row in result["trake_checks"].values()
        )
    )
    decision = {
        "recommendation": "DO_NOT_SUBMIT_2_YET",
        "hard_automated_gates_pass": hard,
        "manual_review_required": True,
        "qa_readiness": qa_readiness,
        "gt_opened": False,
        "submission_uploaded": False,
    }
    write_jsonl(
        output / "query_anchor_diagnostics.jsonl",
        ({"query_id": key, **value} for key, value in profiles.items()),
    )
    write_jsonl(
        output / "asr_r3_specificity_diagnostics.jsonl",
        (row for row in tier_diagnostics if row["modality"] == "asr"),
    )
    write_jsonl(output / "strong_asr_inclusion_audit.jsonl", result["strong_asr_audit"])
    write_jsonl(output / "qa_r3_extractions.jsonl", qa_extractions)
    write_jsonl(output / "qa_context_relevance.jsonl", qa_verifications)
    write_jsonl(output / "qa_r3_semantic_verifier.jsonl", qa_verifications)
    (output / "qa_r3_readiness.json").write_text(
        json.dumps(qa_readiness, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "asset_hashes.json").write_text(
        json.dumps(provenance.get("asset_hashes", {}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = []
    for query in queries:
        query_id = str(query["query_id"])
        strong = next(
            (row for row in result["strong_asr_audit"] if row["query_id"] == query_id), None
        )
        r3_top = groups["M0_R3"][query_id][0]
        comparison.append(
            {
                "query_id": query_id,
                "task": query["task"],
                "bcf1_top1": default_key(baseline[query_id][0]),
                "r2_top1": default_key(groups["M0_R2"][query_id][0]),
                "m0_r3_top1": default_key(r3_top),
                "safe_r3_top1": default_key(groups["SAFE_R3"][query_id][0]),
                "m1_r3_top1": default_key(groups["M1_R3"][query_id][0]),
                "best_strong_asr_video": (strong or {}).get("best_strong_asr_video"),
                "best_strong_asr_rank": (strong or {}).get("final_best_rank"),
                "evidence_tier": r3_top.get("evidence_tier"),
                "tier_reason": r3_top.get("evidence_tier_reasons"),
                "exact_high_phrase_matches": r3_top.get(
                    "matched_high_query_anchors", []
                ),
                "warnings": (
                    ["QA_ZERO_SUFFICIENT"]
                    if query["task"] == "QA"
                    and qa_readiness[query_id]["verified_sufficient_count"] == 0
                    else []
                ),
                "m0_r3_top10": groups["M0_R3"][query_id][:10],
                "safe_r3_top10": groups["SAFE_R3"][query_id][:10],
                "m1_r3_top10": groups["M1_R3"][query_id][:10],
            }
        )
    (output / "trial_p1_r3_candidate_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    with (output / "trial_p1_r3_candidate_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    trake_summary = {
        query_id: checks for query_id, checks in result["trake_checks"].items()
    }
    (output / "trake_r3_graph_summary.json").write_text(
        json.dumps(trake_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review = [
        "# Trial P1 R3 Human Review",
        "",
        "No GT was opened. No submission was uploaded.",
        "",
    ]
    for row in comparison:
        review.extend(
            [
                f"## {row['query_id']} ({row['task']})",
                "",
                f"- BCF1 / R2 / M0_R3 Top1: `{row['bcf1_top1']}` / "
                f"`{row['r2_top1']}` / `{row['m0_r3_top1']}`",
                f"- SAFE_R3 / M1_R3 Top1: `{row['safe_r3_top1']}` / `{row['m1_r3_top1']}`",
                f"- Strong ASR video/rank: `{row['best_strong_asr_video']}` / "
                f"`{row['best_strong_asr_rank']}`",
                f"- Tier/reason: `{row['evidence_tier']}` / `{row['tier_reason']}`",
                f"- High phrase matches: `{row['exact_high_phrase_matches']}`",
                f"- Warnings: `{row['warnings'] or 'NONE'}`",
                "",
            ]
        )
    review.extend(
        [
            "## QA candidate verification",
            "",
            "| Query | Video/frame | Context | Answer | Supporting span | Qwen | Type | Final |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in qa_verifications:
        spans = row.get("supporting_spans", [])
        supporting = str(spans[0] if spans else "").replace("|", "\\|")[:160]
        answer = str(row.get("answer", "")).replace("|", "\\|")
        review.append(
            f"| {row.get('query_id')} | {row.get('video_id')}:{row.get('frame_id')} | "
            f"{bool(row.get('context_relevant'))} | {answer} | {supporting} | "
            f"{bool(row.get('qwen_supports'))} | "
            f"{bool(row.get('answer_type_verifier_pass'))} | "
            f"{bool(row.get('final_evidence_sufficient'))} |"
        )
    review.extend(["", "## All sufficient QA answers", "", f"`{qa_readiness}`", ""])
    (output / "trial_p1_r3_human_review.md").write_text("\n".join(review), encoding="utf-8")
    candidate_zips = {}
    for name, rows in candidates.items():
        prediction = output / f"{name}.jsonl"
        write_jsonl(prediction, rows)
        target = output / f"{name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(prediction, prediction.name)
        candidate_zips[name] = str(target)
    (output / "SUBMISSION_2_R3_DECISION.md").write_text(
        "# Submission #2 R3 Decision\n\n`DO_NOT_SUBMIT_2_YET`\n\n"
        "Manual Top10 review remains mandatory. No upload was performed.\n",
        encoding="utf-8",
    )
    (output / "run_provenance.json").write_text(
        json.dumps(
            provenance
            | {
                "decision": decision,
                "candidate_zips": candidate_zips,
                "rescue_diagnostics": rescue_diagnostics,
                "candidate_hashes": {
                    name: semantic_content_hash(rows) for name, rows in candidates.items()
                },
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
    "ACTION_MEDIUM",
    "COUNT_COLOR_MEDIUM",
    "DISTINCTIVE_HIGH",
    "ENTITY_HIGH",
    "GENERIC_LOW",
    "OBJECT_MEDIUM",
    "build_bounded_qa_rescue_evidence",
    "build_r3_candidates",
    "augment_qa_context_r3",
    "canonicalize_title",
    "classify_asr_r3",
    "derive_r3_anchor_profiles",
    "evaluate_context_relevance",
    "match_r3_anchors",
    "rank_qa_r3",
    "repair_kis_r3",
    "tier_evidence_r3",
    "verify_answer_r3",
    "write_r3_artifacts",
]
