"""Deterministic Vietnamese semantic units translated by the canonical VinAI provider."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)


class SemanticUnitRole(StrEnum):
    FULL_QUERY = "full_query"
    TEMPORAL_SCENE = "temporal_scene"
    PRIMARY_SCENE = "primary_scene"
    SUPPORTING_ATTRIBUTE = "supporting_attribute"


@dataclass(frozen=True, slots=True)
class SemanticQueryConfig:
    full_query_weight: float = 1.0
    primary_scene_weight: float = 1.0
    supporting_attribute_weight: float = 0.35
    enable_visual_content_filter: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("full_query_weight", self.full_query_weight),
            ("primary_scene_weight", self.primary_scene_weight),
            ("supporting_attribute_weight", self.supporting_attribute_weight),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class VietnameseSemanticUnit:
    unit_id: str
    text: str
    role: SemanticUnitRole
    weight: float
    temporal_index: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledSemanticVariant:
    query_variant: QueryVariant
    semantic_unit_id: str
    semantic_role: SemanticUnitRole
    source_vietnamese: str
    raw_english: str
    segment_index: int
    segment_count: int
    clip_token_count: int
    temporal_index: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledSemanticQuery:
    query_id: str
    source_vietnamese: str
    units: tuple[VietnameseSemanticUnit, ...]
    variants: tuple[CompiledSemanticVariant, ...]
    provider_name: str

    @property
    def query_variants(self) -> tuple[QueryVariant, ...]:
        return tuple(item.query_variant for item in self.variants)

    @property
    def primary_variant_ids(self) -> frozenset[str]:
        return frozenset(
            item.query_variant.variant_id
            for item in self.variants
            if item.semantic_role in (SemanticUnitRole.PRIMARY_SCENE, SemanticUnitRole.TEMPORAL_SCENE)
        )

    @property
    def temporal_scene_variants(self) -> tuple[CompiledSemanticVariant, ...]:
        return tuple(
            item for item in self.variants
            if item.semantic_role is SemanticUnitRole.TEMPORAL_SCENE
        )

    @property
    def is_temporal_compound(self) -> bool:
        temporal_indices = {
            item.temporal_index
            for item in self.variants
            if item.semantic_role is SemanticUnitRole.TEMPORAL_SCENE and item.temporal_index is not None
        }
        return len(temporal_indices) >= 2

    @property
    def supporting_variant_ids(self) -> frozenset[str]:
        return frozenset(
            item.query_variant.variant_id
            for item in self.variants
            if item.semantic_role is SemanticUnitRole.SUPPORTING_ATTRIBUTE
        )

    @property
    def has_supporting_attributes(self) -> bool:
        return any(
            item.semantic_role is SemanticUnitRole.SUPPORTING_ATTRIBUTE
            for item in self.variants
        )

    @property
    def full_query_variant_id(self) -> str:
        return self.variants[0].query_variant.variant_id

    def to_metadata(self) -> dict[str, object]:
        return {
            "dynamic_translation_enabled": True,
            "semantic_clause_compilation_enabled": True,
            "provider": self.provider_name,
            "source_vietnamese": self.source_vietnamese,
            "is_temporal_compound": self.is_temporal_compound,
            "unit_count": len(self.units),
            "segment_count": len(self.variants),
            "lossless_segmentation": True,
            "was_truncated": False,
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "role": unit.role.value,
                    "temporal_index": unit.temporal_index,
                    "weight": unit.weight,
                    "source_vietnamese": unit.text,
                    "raw_english": next(
                        variant.raw_english
                        for variant in self.variants
                        if variant.semantic_unit_id == unit.unit_id
                    ),
                    "segments": [
                        {
                            "variant_id": variant.query_variant.variant_id,
                            "text": variant.query_variant.text,
                            "weight": variant.query_variant.weight,
                            "clip_token_count": variant.clip_token_count,
                            "segment_index": variant.segment_index,
                            "temporal_index": variant.temporal_index,
                        }
                        for variant in self.variants
                        if variant.semantic_unit_id == unit.unit_id
                    ],
                }
                for unit in self.units
            ],
        }


class BatchTranslationProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    def translate_many(self, texts: Sequence[str]) -> tuple[str, ...]: ...


class ClipTokenBudgetGuard(Protocol):
    def split_for_clip(self, text: str) -> tuple[str, ...]: ...

    def count_tokens(self, text: str) -> int: ...


_TEMPORAL_BOUNDARY = re.compile(
    r"(?i)(?=\b(?:sau\s+đó|sau\s+vài\s+giây|tiếp\s+theo|trước\s+đó|"
    r"cuối\s+cùng|đoạn\s+clip\s+kết\s+thúc|cảnh\s+quay\s+kết\s+thúc)\b)"
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?;]+")
_WHITESPACE = re.compile(r"\s+")

_PRIMARY_MARKERS = re.compile(
    r"(?i)\b(?:đang|bắt\s+đầu|kết\s+thúc|thực\s+hiện|xếp|đi|đứng|ngồi|"
    r"chạy|nhảy|nấu|đặt|đưa|lấy|cắt|kéo|quay|lia|chạm|cầm|đổ|rót|"
    r"trình\s+bày|xuất\s+hiện|di\s+chuyển|tiến\s+đến|bước|đào|múc|"
    r"nhìn|ghi|cân|thả|chiên|luộc|nướng|xào|giảng|nói|trang\s+trí)\b"
)
_SUPPORTING_MARKERS = re.compile(
    r"(?i)\b(?:chỉ\s+có|trong\s+nhóm|có\s+màu|đeo\s+kính|đội\s+nón|"
    r"mặc|số\s+lượng|bao\s+nhiêu|một\s+người|hai\s+người|ba\s+người|"
    r"bốn\s+người|năm\s+người|màu\s+đỏ|màu\s+xanh|màu\s+đen|"
    r"bên\s+trái|bên\s+phải|phía\s+sau|phía\s+trước)\b"
)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip(" ,:\t\r\n")


def _split_semantic_clauses(query_vi: str) -> tuple[str, ...]:
    # Check if temporal boundary markers exist to preserve cohesive multi-sentence scene phases
    raw_blocks = [b.strip() for b in _TEMPORAL_BOUNDARY.split(query_vi) if b.strip()]
    if len(raw_blocks) >= 2:
        clauses = []
        for block in raw_blocks:
            c = _clean(block)
            if c:
                clauses.append(c)
        return tuple(clauses)

    clauses: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(query_vi):
        sentence = _clean(sentence)
        if not sentence:
            continue
        for fragment in _TEMPORAL_BOUNDARY.split(sentence):
            fragment = _clean(fragment)
            if fragment:
                clauses.append(fragment)
    return tuple(clauses)


_BOILERPLATE_PREFIX = re.compile(
    r"(?i)^(?:cảnh\s+quay|đoạn\s+phim|đoạn\s+clip|mẩu\s+tin|video|hình\s+ảnh)\s+"
    r"(?:bắt\s+đầu\s+với\s+hình\s+ảnh|bắt\s+đầu\s+với|bắt\s+đầu\s+bằng\s+việc|bắt\s+đầu\s+bằng|"
    r"cho\s+thấy|ghi\s+lại|mô\s+tả|về|chiếu\s+cảnh)?\s*",
)

_PURE_CONNECTIVE_PATTERN = re.compile(
    r"(?i)^(?:\s*next\s*,?|\s*then\s*,?|\s*after\s+that\s*,?|\s*sau\s+đó\s*,?|\s*tiếp\s+theo\s*,?|\s*sau\s+vài\s+giây\s*,?)$"
)


def has_visual_content(text: str) -> bool:
    cleaned = _clean(text)
    if not cleaned:
        return False
    if _PURE_CONNECTIVE_PATTERN.match(cleaned):
        return False
    return True


def strip_boilerplate(text: str) -> str:
    cleaned = _clean(text)
    stripped = _BOILERPLATE_PREFIX.sub("", cleaned).strip(" ,:\t\r\n")
    return stripped if len(stripped) >= 10 else cleaned


def _classify_clause(text: str) -> SemanticUnitRole:
    has_primary = _PRIMARY_MARKERS.search(text) is not None
    has_supporting = _SUPPORTING_MARKERS.search(text) is not None
    if has_supporting and not has_primary:
        return SemanticUnitRole.SUPPORTING_ATTRIBUTE
    return SemanticUnitRole.PRIMARY_SCENE


def decompose_vietnamese_semantic_units(
    *,
    query_id: str,
    query_vi: str,
    config: SemanticQueryConfig = SemanticQueryConfig(),
) -> tuple[VietnameseSemanticUnit, ...]:
    query_id = query_id.strip()
    source = _clean(query_vi)
    if not query_id:
        raise ValueError("query_id must not be empty")
    if not source:
        raise ValueError("query_vi must not be empty")

    units = [
        VietnameseSemanticUnit(
            unit_id=f"{query_id}::full",
            text=source,
            role=SemanticUnitRole.FULL_QUERY,
            weight=config.full_query_weight,
            temporal_index=None,
        )
    ]
    clauses = _split_semantic_clauses(source)
    if len(clauses) == 1 and clauses[0] == source:
        return tuple(units)

    # Pre-classify clauses and clean boilerplate
    classified = [(_classify_clause(clause), strip_boilerplate(clause)) for clause in clauses]
    scene_clause_count = sum(1 for role, _ in classified if role is not SemanticUnitRole.SUPPORTING_ATTRIBUTE)
    is_temporal = scene_clause_count >= 2

    current_temporal_idx = 1
    for index, (initial_role, clause) in enumerate(classified, start=1):
        if initial_role is SemanticUnitRole.SUPPORTING_ATTRIBUTE:
            role = SemanticUnitRole.SUPPORTING_ATTRIBUTE
            weight = config.supporting_attribute_weight
            temp_idx = None
        elif is_temporal:
            role = SemanticUnitRole.TEMPORAL_SCENE
            weight = config.primary_scene_weight
            temp_idx = current_temporal_idx
            current_temporal_idx += 1
        else:
            role = SemanticUnitRole.PRIMARY_SCENE
            weight = config.primary_scene_weight
            temp_idx = None

        units.append(
            VietnameseSemanticUnit(
                unit_id=f"{query_id}::clause_{index:02d}",
                text=clause,
                role=role,
                weight=weight,
                temporal_index=temp_idx,
            )
        )
    return tuple(units)


def compile_vietnamese_semantic_query(
    *,
    query_id: str,
    query_vi: str,
    provider: BatchTranslationProvider,
    token_budget_guard: ClipTokenBudgetGuard,
    config: SemanticQueryConfig = SemanticQueryConfig(),
) -> CompiledSemanticQuery:
    """Translate a full Vietnamese query and its semantic clauses without truncation."""

    units = decompose_vietnamese_semantic_units(
        query_id=query_id,
        query_vi=query_vi,
        config=config,
    )
    translations = tuple(provider.translate_many(tuple(unit.text for unit in units)))
    if len(translations) != len(units):
        raise ValueError(
            "translation provider returned an unexpected row count: "
            f"{len(translations)} != {len(units)}"
        )

    compiled: list[CompiledSemanticVariant] = []
    for unit_index, (unit, raw_english) in enumerate(
        zip(units, translations, strict=True),
        start=1,
    ):
        raw_english = _clean(raw_english)
        if not raw_english:
            raise ValueError(f"translation for {unit.unit_id} is empty")
        segments = tuple(token_budget_guard.split_for_clip(raw_english))
        if not segments:
            raise ValueError(f"translation for {unit.unit_id} produced no CLIP segments")
        segment_weight = unit.weight / len(segments)
        for segment_index, segment in enumerate(segments, start=1):
            if config.enable_visual_content_filter and not has_visual_content(segment):
                continue
            compiled.append(
                CompiledSemanticVariant(
                    query_variant=QueryVariant(
                        variant_id=(
                            f"{query_id}::semantic_{unit_index:02d}_s{segment_index:02d}"
                        ),
                        text=segment,
                        language=QueryLanguage.ENGLISH,
                        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                        weight=segment_weight,
                    ),
                    semantic_unit_id=unit.unit_id,
                    semantic_role=unit.role,
                    source_vietnamese=unit.text,
                    raw_english=raw_english,
                    segment_index=segment_index,
                    segment_count=len(segments),
                    clip_token_count=token_budget_guard.count_tokens(segment),
                    temporal_index=unit.temporal_index,
                )
            )
    return CompiledSemanticQuery(
        query_id=query_id,
        source_vietnamese=_clean(query_vi),
        units=units,
        variants=tuple(compiled),
        provider_name=provider.provider_name,
    )


@dataclass(frozen=True, slots=True)
class CompiledParaphraseGroup:
    group_id: str
    source_text: str
    compiled_query: CompiledSemanticQuery
    group_weight_mass: float


@dataclass(frozen=True, slots=True)
class CompiledParaphraseEnsemble:
    query_id: str
    source_vietnamese: str
    groups: tuple[CompiledParaphraseGroup, ...]
    normalized_weights: Mapping[str, float]
    hierarchical_quotas_c1: Mapping[str, int]
    hierarchical_quotas_c2: Mapping[str, int]
    provider_name: str
    baseline_weight_mass: float
    total_semantic_budget_c1: int
    total_semantic_budget_c2: int

    @property
    def all_variants(self) -> tuple[QueryVariant, ...]:
        result = []
        for g in self.groups:
            for v in g.compiled_query.variants:
                norm_w = self.normalized_weights[v.query_variant.variant_id]
                result.append(
                    QueryVariant(
                        variant_id=v.query_variant.variant_id,
                        text=v.query_variant.text,
                        language=v.query_variant.language,
                        variant_type=v.query_variant.variant_type,
                        weight=norm_w,
                    )
                )
        return tuple(result)

    def to_metadata(self) -> dict[str, object]:
        return {
            "dynamic_translation_enabled": True,
            "paraphrase_ensemble_enabled": True,
            "provider": self.provider_name,
            "source_vietnamese": self.source_vietnamese,
            "group_count": len(self.groups),
            "baseline_weight_mass": self.baseline_weight_mass,
            "total_semantic_budget_c1": self.total_semantic_budget_c1,
            "total_semantic_budget_c2": self.total_semantic_budget_c2,
            "groups": [
                {
                    "group_id": g.group_id,
                    "source_text": g.source_text,
                    "group_weight_mass": g.group_weight_mass,
                    "unit_count": len(g.compiled_query.units),
                    "segment_count": len(g.compiled_query.variants),
                    "is_temporal_compound": g.compiled_query.is_temporal_compound,
                    "units": g.compiled_query.to_metadata().get("units", []),
                }
                for g in self.groups
            ],
            "normalized_weights": dict(self.normalized_weights),
            "quotas_c1": dict(self.hierarchical_quotas_c1),
            "quotas_c2": dict(self.hierarchical_quotas_c2),
        }


def allocate_hierarchical_quotas(
    compiled_groups: Sequence[CompiledSemanticQuery],
    *,
    total_semantic_budget: int,
) -> dict[str, int]:
    """Hierarchical two-level deterministic divmod quota allocation across paraphrase groups and compiled variants."""
    n_groups = len(compiled_groups)
    if n_groups == 0:
        raise ValueError("compiled_groups must not be empty")
    if type(total_semantic_budget) is not int or total_semantic_budget <= 0:
        raise ValueError("total_semantic_budget must be a positive integer")

    total_variants = sum(len(g.variants) for g in compiled_groups)
    if total_semantic_budget < total_variants:
        raise ValueError(
            f"Insufficient semantic budget: total_budget={total_semantic_budget} < total_variants={total_variants}"
        )

    # Level 1: divmod across groups
    q_grp, r_grp = divmod(total_semantic_budget, n_groups)
    group_budgets: list[int] = [
        q_grp + 1 if i < r_grp else q_grp
        for i in range(n_groups)
    ]
    assert sum(group_budgets) == total_semantic_budget

    # Level 2: divmod across variants within each group
    variant_quotas: dict[str, int] = {}
    for g_idx, (group, b_i) in enumerate(zip(compiled_groups, group_budgets, strict=True)):
        n_vars = len(group.variants)
        assert n_vars > 0
        q_i, r_i = divmod(b_i, n_vars)
        allocated_for_group = 0
        for v_idx, variant in enumerate(group.variants):
            k_ij = q_i + 1 if v_idx < r_i else q_i
            variant_quotas[variant.query_variant.variant_id] = k_ij
            allocated_for_group += k_ij
        assert allocated_for_group == b_i, f"Group {g_idx} allocation mismatch: {allocated_for_group} != {b_i}"

    assert sum(variant_quotas.values()) == total_semantic_budget
    return variant_quotas


def compute_normalized_ensemble_weights(
    compiled_groups: Sequence[CompiledSemanticQuery],
    *,
    baseline_weight_mass: float,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Compute strictly normalized RRF weights preserving total weight mass W_0 and intra-group relative importance."""
    n_groups = len(compiled_groups)
    if n_groups == 0:
        raise ValueError("compiled_groups must not be empty")
    if not math.isfinite(baseline_weight_mass) or baseline_weight_mass <= 0:
        raise ValueError("baseline_weight_mass must be a positive finite float")

    target_group_mass = baseline_weight_mass / n_groups
    normalized_weights: dict[str, float] = {}

    for g_idx, group in enumerate(compiled_groups):
        raw_weights = [v.query_variant.weight for v in group.variants]
        sum_raw = sum(raw_weights)
        if not math.isfinite(sum_raw) or sum_raw <= 0:
            raise ValueError(f"Group {g_idx} has non-positive or non-finite weight sum: {sum_raw}")

        group_mass_sum = 0.0
        for variant in group.variants:
            raw_w = variant.query_variant.weight
            norm_w = float(target_group_mass * (raw_w / sum_raw))
            if not math.isfinite(norm_w) or norm_w <= 0:
                raise ValueError(f"Variant {variant.query_variant.variant_id} computed invalid normalized weight: {norm_w}")
            normalized_weights[variant.query_variant.variant_id] = norm_w
            group_mass_sum += norm_w

        if abs(group_mass_sum - target_group_mass) > tolerance:
            raise ValueError(
                f"Group {g_idx} normalized mass mismatch: computed={group_mass_sum}, target={target_group_mass}"
            )

    total_mass = sum(normalized_weights.values())
    if abs(total_mass - baseline_weight_mass) > tolerance:
        raise ValueError(
            f"Total normalized ensemble mass mismatch: computed={total_mass}, target={baseline_weight_mass}"
        )

    return normalized_weights
