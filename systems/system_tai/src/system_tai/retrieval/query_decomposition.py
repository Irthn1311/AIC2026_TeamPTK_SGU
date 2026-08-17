# ==============================================================================================================
# Deterministic Runtime-Safe Query Decomposition for Multi-Variant Retrieval
# ==============================================================================================================

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Common linguistic markers for deterministic decomposition (English & Vietnamese)
STOPWORDS_EN = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "what", "which", "who", "whom", "whose", "why", "how",
    "many", "much", "there", "this", "that", "these", "those", "can", "could",
    "would", "should", "visible", "frame", "image", "photo", "picture", "scene",
    "video", "clip", "show", "shown", "see", "seen", "tell", "describe",
}

ACTION_MARKERS_EN = [
    r"\b(walking|walks|walked)\b",
    r"\b(running|runs|ran)\b",
    r"\b(standing|stands|stood)\b",
    r"\b(sitting|sits|sat)\b",
    r"\b(speaking|speaks|spoke|talking|talks)\b",
    r"\b(getting out|gets out|got out|exiting|exits)\b",
    r"\b(entering|enters|entered|getting in)\b",
    r"\b(driving|drives|drove|riding|rides)\b",
    r"\b(holding|holds|held|carrying|carries)\b",
    r"\b(presenting|presents|hosting|hosts)\b",
    r"\b(dancing|dances|singing|sings)\b",
    r"\b(cooking|cooks|eating|eats)\b",
    r"\b(playing|plays|working|works)\b",
    r"\b(burning|burns|burned|flaming)\b",
]

SCENE_MARKERS_EN = [
    r"\b(studio|television studio|newsroom)\b",
    r"\b(street|road|highway|intersection|alley)\b",
    r"\b(room|office|hall|stage|podium)\b",
    r"\b(park|garden|field|forest|mountain|hill)\b",
    r"\b(outdoor|outdoors|indoor|indoors)\b",
    r"\b(daytime|day|night|nighttime|evening|morning)\b",
    r"\b(water|river|lake|sea|beach|pool)\b",
]

TEMPORAL_RELATION_MARKERS_EN = [
    r"\b(after|before|during|while|then|next|following|first|second)\b",
    r"\b(when|as soon as|start|end|beginning)\b",
]


@dataclass(frozen=True, slots=True)
class QueryVariants:
    """Deterministic structured query variants for multi-channel retrieval."""

    literal: str
    entity_focused: str | None = None
    action_focused: str | None = None
    scene_context: str | None = None
    temporal_relation: str | None = None
    compact_keywords: str | None = None

    def as_list(self) -> list[tuple[str, str]]:
        """Returns non-empty (variant_name, text) pairs in deterministic order."""
        res: list[tuple[str, str]] = [("literal", self.literal)]
        if self.entity_focused and self.entity_focused != self.literal:
            res.append(("entity_focused", self.entity_focused))
        if self.action_focused and self.action_focused != self.literal:
            res.append(("action_focused", self.action_focused))
        if self.scene_context and self.scene_context != self.literal:
            res.append(("scene_context", self.scene_context))
        if self.temporal_relation and self.temporal_relation != self.literal:
            res.append(("temporal_relation", self.temporal_relation))
        if self.compact_keywords and self.compact_keywords != self.literal:
            res.append(("compact_keywords", self.compact_keywords))
        return res


def decompose_query(
    query_text_vi: str,
    query_text_en: str | None = None,
) -> QueryVariants:
    """
    Decomposes a query into multiple deterministic, semantic-focused search variants.

    Fail-closed policy:
    - Only derives variants from actual input query tokens.
    - If no distinct action/entity/scene marker exists, sets the variant to None.
    - Never hallucinates non-existent entities or constants.
    """
    raw_en = query_text_en.strip() if query_text_en else ""
    raw_vi = query_text_vi.strip() if query_text_vi else ""
    base_text = raw_en if raw_en else raw_vi
    norm_base = _normalize_text(base_text)

    if not norm_base:
        return QueryVariants(literal="")

    # 1. Literal normalized variant
    literal_variant = norm_base

    # 2. Compact keywords (filter out high-frequency generic stopwords)
    words = [w for w in norm_base.split() if w not in STOPWORDS_EN and len(w) > 1]
    compact_keywords = " ".join(words) if len(words) >= 2 else None

    # 3. Entity-focused variant
    # Extract noun/entity tokens by removing common question words, auxiliaries, and action verbs
    entity_words = [
        w for w in words
        if not any(re.search(pattern, w) for pattern in ACTION_MARKERS_EN)
        and not any(re.search(pattern, w) for pattern in TEMPORAL_RELATION_MARKERS_EN)
    ]
    entity_variant = " ".join(entity_words) if len(entity_words) >= 2 and " ".join(entity_words) != compact_keywords else None

    # 4. Action-focused variant
    found_actions: list[str] = []
    for pattern in ACTION_MARKERS_EN:
        match = re.search(pattern, norm_base)
        if match:
            found_actions.append(match.group(0))
    action_variant = None
    if found_actions and entity_words:
        action_variant = f"{' '.join(entity_words[:3])} {' '.join(found_actions)}"

    # 5. Scene/context variant
    found_scenes: list[str] = []
    for pattern in SCENE_MARKERS_EN:
        match = re.search(pattern, norm_base)
        if match:
            found_scenes.append(match.group(0))
    scene_variant = None
    if found_scenes and entity_words:
        scene_variant = f"{' '.join(found_scenes)} with {' '.join(entity_words[:3])}"

    # 6. Temporal relation variant
    found_relations: list[str] = []
    for pattern in TEMPORAL_RELATION_MARKERS_EN:
        match = re.search(pattern, norm_base)
        if match:
            found_relations.append(match.group(0))
    relation_variant = None
    if found_relations and (found_actions or entity_words):
        core_tokens = found_actions + entity_words[:2]
        relation_variant = f"{' '.join(found_relations)} {' '.join(core_tokens)}"

    return QueryVariants(
        literal=literal_variant,
        entity_focused=entity_variant,
        action_focused=action_variant,
        scene_context=scene_variant,
        temporal_relation=relation_variant,
        compact_keywords=compact_keywords,
    )
