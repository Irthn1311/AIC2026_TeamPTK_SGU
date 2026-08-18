# ==============================================================================================================
# Canonical Generic OCR Span Candidate Generator & Scorer (Sprint R3-S2E1)
# ==============================================================================================================

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from system_tai.quality.l21_150_answers import normalize_answer


@dataclass(frozen=True, slots=True)
class OCRSpanCandidate:
    normalized_span: str
    raw_span: str
    n_gram: int
    line_idx: int
    mean_confidence: float
    score: float
    score_components: dict[str, float]

    @property
    def sort_key(self) -> tuple[float, float, int, int, str]:
        """
        Deterministic multi-tier sort key:
        1. Final Score (descending)
        2. Mean OCR Word Confidence (descending)
        3. Span Length (ascending: concise n-grams preferred)
        4. Line Index (ascending: earlier lines first)
        5. Normalized Span string (alphabetical ascending: strict tie-break)
        """
        return (
            -round(self.score, 6),
            -round(self.mean_confidence, 2),
            self.n_gram,
            self.line_idx,
            self.normalized_span,
        )


def is_junk_token(tok: str) -> bool:
    """Check if token is punctuation-only, symbol junk, or non-word artifact."""
    cleaned = re.sub(r"[^\w\s]", "", tok, flags=re.UNICODE).strip()
    if not cleaned or len(cleaned) == 0:
        return True
    alphanumeric_count = sum(c.isalnum() for c in tok)
    if alphanumeric_count / max(len(tok), 1) < 0.65:
        return True
    # If token has symbols like / # ! @ $ ~ ^ ` | \
    if any(c in "/#!@$~^`|\\¿¡†‡" for c in tok):
        return True
    return False


def score_span_candidate(
    *,
    tokens: list[str],
    confidences: list[float],
    line_idx: int,
    line_conf: float,
) -> tuple[float, dict[str, float]]:
    """
    Evaluates candidates using only runtime-safe, GT-blind generic features:
    1. Average word-level OCR confidence (0.0 .. 1.0)
    2. Alphabetic cleanliness density (ratio of letters in span)
    3. Junk and broken-symbol penalty
    4. Bounded concise length prior (1..3 tokens preferred)
    5. Line prominence and position factor
    6. Capitalization / entity bonus (uppercase words often denote brand names / entities)
    """
    raw_span_text = " ".join(tokens)
    norm_span = normalize_answer(raw_span_text)

    if not norm_span:
        return 0.0, {"valid": 0.0}

    # 1. Average Word Confidence
    mean_conf = sum(confidences) / max(len(confidences), 1)
    conf_factor = mean_conf / 100.0

    # 2. Cleanliness Ratio (Alphanumeric Density)
    total_chars = len(raw_span_text)
    alpha_chars = sum(c.isalpha() for c in raw_span_text)
    cleanliness = alpha_chars / max(total_chars, 1)

    # 3. Junk Penalty
    junk_count = sum(is_junk_token(t) for t in tokens)
    junk_factor = 1.0 / (1.0 + 2.0 * junk_count)
    if any(is_junk_token(t) for t in tokens):
        junk_factor *= 0.5

    # 4. Length Prior
    num_words = len(tokens)
    if num_words == 1:
        len_prior = 1.0
    elif num_words == 2:
        len_prior = 0.95
    elif num_words == 3:
        len_prior = 0.85
    elif num_words == 4:
        len_prior = 0.75
    else:
        len_prior = 0.5

    # 5. Line Prominence Factor
    line_factor = (line_conf / 100.0) * (1.0 / (1.0 + 0.1 * line_idx))

    # 6. Entity / Capitalization Bonus
    cap_bonus = 1.0
    if any(t.isupper() and len(t) >= 2 for t in tokens):
        cap_bonus = 1.15

    final_score = conf_factor * cleanliness * junk_factor * len_prior * line_factor * cap_bonus

    components = {
        "mean_conf": mean_conf,
        "conf_factor": conf_factor,
        "cleanliness": cleanliness,
        "junk_factor": junk_factor,
        "len_prior": len_prior,
        "line_factor": line_factor,
        "cap_bonus": cap_bonus,
        "final_score": final_score,
    }
    return final_score, components


def extract_and_rank_canonical_ocr_spans(
    tsv_payload: bytes,
    max_n: int = 4,
    max_candidates: int | None = None,
) -> list[OCRSpanCandidate]:
    """
    Canonical helper to extract, score, and deterministically rank all contiguous 1..max_n spans.
    Guarantees 100% platform-independent, deterministic ordering.
    """
    text = tsv_payload.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)

    # Group words by line: (page_num, block_num, par_num, line_num)
    lines_dict: dict[tuple[int, int, int, int], list[tuple[int, str, float]]] = defaultdict(list)
    for row in reader:
        try:
            raw_text = str(row.get("text", "")).strip()
            conf = float(row.get("conf", -1))
            if not raw_text or conf < 0:
                continue
            key = (int(row["page_num"]), int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
            word_num = int(row["word_num"])
            lines_dict[key].append((word_num, raw_text, conf))
        except Exception:
            continue

    candidates_by_norm: dict[str, OCRSpanCandidate] = {}

    for line_idx, (line_key, words) in enumerate(sorted(lines_dict.items())):
        sorted_words = sorted(words, key=lambda x: x[0])
        raw_tokens = [w[1] for w in sorted_words]
        confs = [w[2] for w in sorted_words]
        line_mean_conf = sum(confs) / max(len(confs), 1)

        num_words = len(sorted_words)
        for n in range(1, max_n + 1):
            for i in range(num_words - n + 1):
                span_tokens = raw_tokens[i : i + n]
                span_confs = confs[i : i + n]
                span_text = " ".join(span_tokens)
                norm_span = normalize_answer(span_text)

                if not norm_span:
                    continue

                mean_span_conf = sum(span_confs) / max(len(span_confs), 1)
                score, components = score_span_candidate(
                    tokens=span_tokens,
                    confidences=span_confs,
                    line_idx=line_idx,
                    line_conf=line_mean_conf,
                )

                candidate = OCRSpanCandidate(
                    normalized_span=norm_span,
                    raw_span=span_text,
                    n_gram=n,
                    line_idx=line_idx,
                    mean_confidence=mean_span_conf,
                    score=score,
                    score_components=components,
                )

                # Deduplicate by keeping the highest scoring candidate instance per normalized span
                if norm_span not in candidates_by_norm or candidate.sort_key < candidates_by_norm[norm_span].sort_key:
                    candidates_by_norm[norm_span] = candidate

    # Strictly deterministic sorting
    ranked = sorted(candidates_by_norm.values(), key=lambda c: c.sort_key)
    if max_candidates is not None and max_candidates > 0:
        return ranked[:max_candidates]
    return ranked
