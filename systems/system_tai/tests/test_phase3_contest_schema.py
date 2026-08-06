from __future__ import annotations

from pathlib import Path

import pytest

from system_tai.kis.contest_schema import (
    ContestQuery,
    load_contest_queries,
    parse_contest_queries,
)
from system_tai.retrieval.multi_query import QueryVariantType


def test_single_query_builds_only_explicit_variants() -> None:
    query = ContestQuery(
        query_id="Q1",
        query_vi="người đi bộ",
        query_en="a pedestrian",
        weight_vi=2.0,
        output_top_k=50,
        metadata={"source": "human"},
    )
    variants = query.variants()
    assert [variant.variant_type for variant in variants] == [
        QueryVariantType.VIETNAMESE_DIRECT,
        QueryVariantType.ENGLISH_TRANSLATION,
    ]
    assert [variant.weight for variant in variants] == [2.0, 1.0]
    assert query.output_top_k == 50
    with pytest.raises(TypeError):
        assert query.metadata is not None
        query.metadata["changed"] = True  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_id": "", "query_vi": "text"},
        {"query_id": "Q", "query_vi": ""},
        {"query_id": "Q", "query_vi": "text", "weight_vi": 0},
        {"query_id": "Q", "query_vi": "text", "output_top_k": 101},
    ],
)
def test_invalid_query_fields_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ContestQuery(**kwargs)  # type: ignore[arg-type]


def test_batch_safe_utf8_yaml_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "queries.yaml"
    path.write_text(
        """schema_version: 1
queries:
  - query_id: Q1
    query_vi: nhiều người đi bộ
    query_en: many pedestrians
    weights:
      vietnamese_direct: 1.5
    output_top_k: 100
""",
        encoding="utf-8",
    )
    batch = load_contest_queries(path)
    assert batch.queries[0].query_vi == "nhiều người đi bộ"
    duplicate_payload = {
        "schema_version": 1,
        "queries": [
            {"query_id": "Q1", "query_vi": "a"},
            {"query_id": "Q1", "query_vi": "b"},
        ],
    }
    with pytest.raises(ValueError, match="unique"):
        parse_contest_queries(duplicate_payload)
