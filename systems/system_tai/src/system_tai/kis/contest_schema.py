"""Typed immutable input schema for contest Textual KIS queries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
)


def _positive_weight(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be a finite positive number")
    return float(value)


@dataclass(frozen=True, slots=True)
class ContestQuery:
    query_id: str
    query_vi: str
    query_en: str | None = None
    query_en_expansion: str | None = None
    weight_vi: float = 1.0
    weight_en: float = 1.0
    weight_en_expansion: float = 1.0
    output_top_k: int = 100
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must not be empty")
        if not isinstance(self.query_vi, str) or not self.query_vi.strip():
            raise ValueError("query_vi must not be empty")
        for field, value in (
            ("query_en", self.query_en),
            ("query_en_expansion", self.query_en_expansion),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be a non-empty string when provided")
        for field, value in (
            ("weight_vi", self.weight_vi),
            ("weight_en", self.weight_en),
            ("weight_en_expansion", self.weight_en_expansion),
        ):
            _positive_weight(value, field)
        if type(self.output_top_k) is not int or not 1 <= self.output_top_k <= 100:
            raise ValueError("output_top_k must be an integer between 1 and 100")
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise ValueError("metadata must be an object when provided")
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def variants(self) -> tuple[QueryVariant, ...]:
        variants = [
            QueryVariant(
                variant_id=f"{self.query_id}:vi",
                text=self.query_vi,
                language=QueryLanguage.VIETNAMESE,
                variant_type=QueryVariantType.VIETNAMESE_DIRECT,
                weight=self.weight_vi,
            )
        ]
        if self.query_en is not None:
            variants.append(
                QueryVariant(
                    variant_id=f"{self.query_id}:en",
                    text=self.query_en,
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_TRANSLATION,
                    weight=self.weight_en,
                )
            )
        if self.query_en_expansion is not None:
            variants.append(
                QueryVariant(
                    variant_id=f"{self.query_id}:en_expansion",
                    text=self.query_en_expansion,
                    language=QueryLanguage.ENGLISH,
                    variant_type=QueryVariantType.ENGLISH_EXPANSION,
                    weight=self.weight_en_expansion,
                )
            )
        return tuple(variants)


@dataclass(frozen=True, slots=True)
class ContestQueryBatch:
    schema_version: int
    queries: tuple[ContestQuery, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported contest query schema_version")
        if not self.queries:
            raise ValueError("contest query batch must not be empty")
        query_ids = [query.query_id for query in self.queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("contest query_id values must be unique")


def _parse_query(item: Any, *, index: int) -> ContestQuery:
    if not isinstance(item, dict):
        raise ValueError(f"queries[{index}] must be an object")
    weights = item.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError(f"queries[{index}].weights must be an object")
    return ContestQuery(
        query_id=item.get("query_id"),
        query_vi=item.get("query_vi"),
        query_en=item.get("query_en"),
        query_en_expansion=item.get("query_en_expansion"),
        weight_vi=weights.get("vietnamese_direct", 1.0),
        weight_en=weights.get("english_translation", 1.0),
        weight_en_expansion=weights.get("english_expansion", 1.0),
        output_top_k=item.get("output_top_k", 100),
        metadata=item.get("metadata"),
    )


def parse_contest_queries(payload: Any) -> ContestQueryBatch:
    if not isinstance(payload, dict):
        raise ValueError("contest query file root must be an object")
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("contest query file must contain a queries list")
    queries = tuple(_parse_query(item, index=index) for index, item in enumerate(raw_queries))
    return ContestQueryBatch(
        schema_version=payload.get("schema_version", 1),
        queries=queries,
    )


def load_contest_queries(path: Path) -> ContestQueryBatch:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"contest query file not found: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"contest query file is not valid UTF-8: {exc}") from exc
    try:
        payload = (
            json.loads(text)
            if source.suffix.casefold() == ".json"
            else yaml.safe_load(text)
        )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse contest query file: {exc}") from exc
    return parse_contest_queries(payload)
