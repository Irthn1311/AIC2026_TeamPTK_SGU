"""
Negation Extractor — Detects and Scopes Negation in Vi/En Queries.

Handles:
  - Vietnamese negations: không, chẳng, chả, không phải, không có,
    ngoại trừ, trừ, mà không, thay vì
  - English negations: not, without, no, except, neither, nor, rather than
  - Scope resolution: maps each negation trigger to its governed attribute
  - Returns positively-stated constraints (must_have) alongside negated ones

Output:
  NegationResult(
      negated_attributes: ["áo đỏ", "phòng học"]
      must_have:          ["người phụ nữ", "phòng"]
      negation_scopes:    [("không mặc", "áo đỏ"), ...]
      has_negation:       True
  )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================
# Negation Trigger Patterns
# ============================================================

# Vietnamese negation triggers — ordered longest-first
_VI_NEGATION_TRIGGERS: List[str] = [
    "không phải là",
    "không phải",
    "không có",
    "không được",
    "không mặc",
    "không đội",
    "không đeo",
    "không cầm",
    "không đứng",
    "không ngồi",
    "ngoại trừ",
    "thay vì",
    "mà không",
    "chứ không",
    "chẳng phải",
    "chẳng có",
    "chẳng",
    "chả",
    "không",
    "trừ",
]

# English negation triggers — ordered longest-first
_EN_NEGATION_TRIGGERS: List[str] = [
    "rather than",
    "instead of",
    "other than",
    "not wearing",
    "not holding",
    "not standing",
    "not sitting",
    "not a",
    "neither",
    "without",
    "except",
    "not",
    "no",
    "nor",
]

# Maximum number of words to capture after a negation trigger as its scope
_SCOPE_WINDOW = 5

# Words that stop scope capture (conjunctions & sentence boundaries)
_SCOPE_STOP_WORDS = {
    # Vietnamese
    "và", "hoặc", "nhưng", "mà", "thì", "với", "trong", "trên",
    "đang", "đang ở", "có", "tại", "là",
    # English
    "and", "or", "but", "while", "who", "that", "which", "with",
    "in", "at", "on", "is", "are", "was", "were",
}


@dataclass
class NegationResult:
    """Result of negation extraction from a query."""
    negated_attributes: List[str] = field(default_factory=list)
    must_have: List[str] = field(default_factory=list)
    negation_scopes: List[Tuple[str, str]] = field(default_factory=list)
    has_negation: bool = False

    def to_vlm_constraint_text(self) -> str:
        """
        Generate a constraint string for VLM verification prompts.

        Example output:
            "The image MUST contain: người phụ nữ, phòng.
             The image must NOT contain or show: áo đỏ, phòng học."
        """
        parts = []
        if self.must_have:
            parts.append("The image MUST contain or show: " + ", ".join(self.must_have))
        if self.negated_attributes:
            parts.append(
                "The image must NOT contain or show: "
                + ", ".join(self.negated_attributes)
            )
        return " ".join(parts) if parts else ""


class NegationExtractor:
    """
    Extracts negation structure from Vietnamese/English query text.

    Usage:
        extractor = NegationExtractor()
        result = extractor.extract("Người phụ nữ không mặc áo đỏ trong phòng không phải phòng học")
        # result.negated_attributes = ["áo đỏ", "phòng học"]
        # result.must_have = ["người phụ nữ", "phòng"]
    """

    def __init__(self):
        # Pre-compile sorted trigger list (longest first, case-insensitive)
        all_triggers = sorted(
            _VI_NEGATION_TRIGGERS + _EN_NEGATION_TRIGGERS,
            key=len,
            reverse=True,
        )
        self._trigger_patterns = [
            (trigger, re.compile(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", re.IGNORECASE))
            for trigger in all_triggers
        ]

    # ----------------------------------------------------------
    # Main Entry
    # ----------------------------------------------------------

    def extract(self, text: str) -> NegationResult:
        """
        Extract negation structure from a query string.

        Algorithm:
          1. Find all negation trigger positions in text
          2. For each trigger, capture the N words following it as scope
          3. Stop scope at sentence boundaries or stop-words
          4. Collect positive phrases (text BEFORE first negation, or between negations)
          5. Deduplicate and return structured result
        """
        if not text or not text.strip():
            return NegationResult()

        text_lower = text.lower()

        # Step 1: Find all negation trigger positions
        trigger_matches: List[Tuple[int, int, str]] = []  # (start, end, trigger)
        for trigger, pattern in self._trigger_patterns:
            for m in pattern.finditer(text_lower):
                # Don't double-count (e.g. "không phải" should not also match "không")
                overlap = any(
                    existing_start <= m.start() < existing_end
                    for existing_start, existing_end, _ in trigger_matches
                )
                if not overlap:
                    trigger_matches.append((m.start(), m.end(), trigger))

        if not trigger_matches:
            # No negation found — all content is must_have
            must_have = self._extract_must_have_phrases(text)
            return NegationResult(must_have=must_have, has_negation=False)

        # Sort by position
        trigger_matches.sort(key=lambda x: x[0])

        # Step 2: Extract negated scopes
        negated_attrs: List[str] = []
        scopes: List[Tuple[str, str]] = []
        used_spans: List[Tuple[int, int]] = []  # spans consumed by negation

        for trig_start, trig_end, trigger in trigger_matches:
            after_trigger = text[trig_end:].strip()
            scope = self._capture_scope(after_trigger)
            if scope:
                negated_attrs.append(scope)
                scopes.append((trigger, scope))
                # Mark the scope span as used
                scope_end = trig_end + text[trig_end:].find(scope) + len(scope)
                used_spans.append((trig_start, scope_end))

        # Step 3: Extract must_have from non-negated regions
        must_have = self._extract_positive_phrases(text, trigger_matches, used_spans)

        return NegationResult(
            negated_attributes=list(dict.fromkeys(negated_attrs)),
            must_have=list(dict.fromkeys(must_have)),
            negation_scopes=scopes,
            has_negation=True,
        )

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _capture_scope(self, text_after_trigger: str) -> str:
        """
        Capture the negated attribute phrase following a trigger.

        Takes up to _SCOPE_WINDOW words, stopping at:
        - Stop-words (conjunctions, prepositions)
        - Punctuation (,  . ; !)
        - Another negation trigger
        - End of string
        """
        # Split into tokens (words + punctuation)
        tokens = re.split(r"(\s+|[,\.!;])", text_after_trigger.strip())
        scope_tokens: List[str] = []

        for tok in tokens:
            tok_stripped = tok.strip()
            if not tok_stripped:
                continue
            # Stop at punctuation
            if tok_stripped in (",", ".", "!", ";", ":", "?"):
                break
            # Stop at stop-words (only if we have at least 1 token collected)
            if scope_tokens and tok_stripped.lower() in _SCOPE_STOP_WORDS:
                break
            # Stop at negation triggers
            if any(
                tok_stripped.lower().startswith(trig.split()[0])
                for trig, _ in self._trigger_patterns
            ):
                if scope_tokens:
                    break
            scope_tokens.append(tok_stripped)
            if len(scope_tokens) >= _SCOPE_WINDOW:
                break

        return " ".join(scope_tokens).strip()

    def _extract_positive_phrases(
        self,
        text: str,
        trigger_matches: List[Tuple[int, int, str]],
        used_spans: List[Tuple[int, int]],
    ) -> List[str]:
        """
        Extract noun phrases from the positive (non-negated) regions.
        Positive regions = text BEFORE the first trigger and BETWEEN negated spans.
        """
        positive_text_parts: List[str] = []

        # Text before the first trigger
        first_trig_start = trigger_matches[0][0]
        pre_text = text[:first_trig_start].strip()
        if pre_text:
            positive_text_parts.append(pre_text)

        # Text between consecutive triggers (excluding used spans)
        for i in range(len(trigger_matches) - 1):
            _, span1_end, _ = trigger_matches[i]
            span2_start, _, _ = trigger_matches[i + 1]

            between = text[span1_end:span2_start].strip()
            between = re.sub(r"^[,\s]+", "", between)
            between = re.sub(r"[,\s]+$", "", between)
            if between and len(between) > 2:
                positive_text_parts.append(between)

        # Extract meaningful phrases from positive regions
        must_have = []
        for part in positive_text_parts:
            phrases = self._extract_must_have_phrases(part)
            must_have.extend(phrases)

        return must_have

    def _extract_must_have_phrases(self, text: str) -> List[str]:
        """
        Naively extract meaningful noun-like phrases from a text segment.
        Filters out short filler words.
        """
        # Split by comma / "và" / "hoặc"
        segments = re.split(r"[,;]|\bvà\b|\bhoặc\b|\band\b|\bor\b", text, flags=re.IGNORECASE)
        phrases = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) >= 3:
                # Remove leading prepositions
                seg = re.sub(r"^(trong|ở|tại|trên|dưới|cạnh|gần|at|in|on|near)\s+", "", seg, flags=re.IGNORECASE)
                if seg:
                    phrases.append(seg)
        return phrases
