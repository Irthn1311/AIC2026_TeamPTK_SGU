from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from tests.phase2_helpers import make_store


def _candidate(
    video_id: str,
    frame_id: int,
    rank: int,
    *,
    score: float = 0.5,
    clip_row: int | None = None,
) -> CandidateFrame:
    return CandidateFrame(
        video_id=video_id,
        frame_id=frame_id,
        clip_row=rank - 1 if clip_row is None else clip_row,
        keyframe_order=rank,
        score=score,
        rank=rank,
        source="clip_exact",
    )


class FakeExactRetriever:
    text_encoder = type(
        "Encoder",
        (),
        {
            "identifiers": MappingProxyType(
                {"model": "fake", "device": "cpu", "library": "test"}
            )
        },
    )()

    def __init__(self, results_by_text: dict[str, tuple[CandidateFrame, ...]]) -> None:
        self.results_by_text = results_by_text
        self.calls: list[KISQuery] = []

    def retrieve(self, query: KISQuery) -> KISResult:
        self.calls.append(query)
        candidates = self.results_by_text[query.text][: query.top_k]
        reranked = tuple(
            CandidateFrame(
                video_id=item.video_id,
                frame_id=item.frame_id,
                clip_row=item.clip_row,
                keyframe_order=item.keyframe_order,
                score=item.score,
                rank=rank,
                source=item.source,
            )
            for rank, item in enumerate(candidates, start=1)
        )
        return KISResult(query_id=query.query_id, ranked_candidates=reranked)


def _variant(
    variant_id: str,
    text: str,
    variant_type: QueryVariantType,
    *,
    weight: float = 1.0,
) -> QueryVariant:
    language = (
        QueryLanguage.VIETNAMESE
        if variant_type is QueryVariantType.VIETNAMESE_DIRECT
        else QueryLanguage.ENGLISH
    )
    return QueryVariant(variant_id, text, language, variant_type, weight)


def test_weighted_rrf_formula_identity_dedup_and_provenance() -> None:
    exact = FakeExactRetriever(
        {
            "vi": (
                _candidate("L21_V001", 100, 1, score=0.01, clip_row=7),
                _candidate("L21_V002", 200, 2),
            ),
            "translation": (
                _candidate("L21_V001", 100, 1, score=99.0, clip_row=70),
                _candidate("L22_V001", 300, 2),
            ),
            "expansion": (_candidate("L21_V001", 100, 1, clip_row=700),),
        }
    )
    variants = (
        _variant("vi", "vi", QueryVariantType.VIETNAMESE_DIRECT, weight=1.0),
        _variant(
            "translation",
            "translation",
            QueryVariantType.ENGLISH_TRANSLATION,
            weight=2.0,
        ),
        _variant(
            "expansion",
            "expansion",
            QueryVariantType.ENGLISH_EXPANSION,
            weight=0.5,
        ),
    )
    result = WeightedRRFRetriever(exact).retrieve(
        query_id="group",
        variants=variants,
        top_k_per_variant=10,
        output_top_k=10,
        rrf_constant=60.0,
    )
    first = result.ranked_candidates[0]
    assert (first.video_id, first.frame_id) == ("L21_V001", 100)
    assert first.clip_row == 7
    assert first.score == pytest.approx((1.0 + 2.0 + 0.5) / 61.0)
    by_pair = {
        (candidate.video_id, candidate.frame_id): candidate
        for candidate in result.ranked_candidates
    }
    assert by_pair[("L21_V002", 200)].score == pytest.approx(1.0 / 62.0)
    metadata = first.diagnostic_metadata
    assert metadata is not None
    assert metadata["variant_hit_count"] == 3
    assert metadata["best_individual_rank"] == 1
    assert [entry["variant_id"] for entry in metadata["per_variant"]] == [
        "expansion",
        "translation",
        "vi",
    ]
    assert len(
        {
            (candidate.video_id, candidate.frame_id)
            for candidate in result.ranked_candidates
        }
    ) == len(result.ranked_candidates)
    assert [candidate.rank for candidate in result.ranked_candidates] == list(
        range(1, len(result.ranked_candidates) + 1)
    )


def test_candidate_appearing_in_one_variant_uses_only_that_branch() -> None:
    exact = FakeExactRetriever(
        {
            "vi": (_candidate("A", 1, 1),),
            "en": (_candidate("B", 2, 1),),
        }
    )
    result = WeightedRRFRetriever(exact).retrieve(
        query_id="q",
        variants=(
            _variant("vi", "vi", QueryVariantType.VIETNAMESE_DIRECT),
            _variant("en", "en", QueryVariantType.ENGLISH_TRANSLATION, weight=2),
        ),
        rrf_constant=10,
    )
    by_pair = {
        (candidate.video_id, candidate.frame_id): candidate
        for candidate in result.ranked_candidates
    }
    assert by_pair[("A", 1)].score == pytest.approx(1 / 11)
    assert by_pair[("B", 2)].score == pytest.approx(2 / 11)


@pytest.mark.parametrize("weight", [0.0, -1.0, math.inf, math.nan])
def test_query_variant_rejects_invalid_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        _variant("v", "text", QueryVariantType.VIETNAMESE_DIRECT, weight=weight)


@pytest.mark.parametrize(
    ("variant_id", "text", "message"),
    [("", "text", "variant_id"), ("v", "", "variant text")],
)
def test_query_variant_rejects_empty_fields(
    variant_id: str, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _variant(variant_id, text, QueryVariantType.VIETNAMESE_DIRECT)


def test_retriever_rejects_empty_or_duplicate_variants() -> None:
    retriever = WeightedRRFRetriever(FakeExactRetriever({"a": ()}))
    with pytest.raises(ValueError, match="at least one"):
        retriever.retrieve(query_id="q", variants=())
    variant = _variant("same", "a", QueryVariantType.VIETNAMESE_DIRECT)
    with pytest.raises(ValueError, match="unique"):
        retriever.retrieve(query_id="q", variants=(variant, variant))


def test_deterministic_ties_top_100_and_raw_score_scale_irrelevance() -> None:
    ordered = tuple(
        _candidate(
            "B" if index % 2 else "A",
            1000 - index,
            index + 1,
            score=float(index) * 1_000_000,
            clip_row=index,
        )
        for index in range(120)
    )
    exact = FakeExactRetriever({"query": ordered})
    variant = _variant("v", "query", QueryVariantType.VIETNAMESE_DIRECT)
    result = WeightedRRFRetriever(exact).retrieve(
        query_id="q",
        variants=(variant,),
        top_k_per_variant=120,
        output_top_k=100,
    )
    assert len(result.ranked_candidates) == 100
    assert [candidate.rank for candidate in result.ranked_candidates] == list(
        range(1, 101)
    )
    # One-variant RRF preserves the input rank, irrespective of cosine magnitude.
    assert [candidate.frame_id for candidate in result.ranked_candidates] == [
        candidate.frame_id for candidate in ordered[:100]
    ]

    tied = FakeExactRetriever(
        {
            "one": (_candidate("B", 20, 1, clip_row=2),),
            "two": (_candidate("A", 30, 1, clip_row=3),),
        }
    )
    tied_result = WeightedRRFRetriever(tied).retrieve(
        query_id="tie",
        variants=(
            _variant("one", "one", QueryVariantType.VIETNAMESE_DIRECT),
            _variant("two", "two", QueryVariantType.ENGLISH_TRANSLATION),
        ),
    )
    assert [(item.video_id, item.frame_id) for item in tied_result.ranked_candidates] == [
        ("A", 30),
        ("B", 20),
    ]


def test_core_exporter_does_not_leak_fusion_provenance(tmp_path: Path) -> None:
    result = WeightedRRFRetriever(
        FakeExactRetriever({"vi": (_candidate("L21_V001", 12345, 1),)})
    ).retrieve(
        query_id="q",
        variants=(
            _variant("vi", "vi", QueryVariantType.VIETNAMESE_DIRECT),
        ),
    )
    destination = tmp_path / "result.jsonl"
    CheckpointExporter().export(result, destination)
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record == {
        "query_id": "q",
        "rank": 1,
        "video_id": "L21_V001",
        "frame_id": 12345,
    }


class FakeEncoder:
    dimension = 2
    identifiers = MappingProxyType({"model": "fake", "device": "cpu"})

    def encode(self, _text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_existing_exact_retriever_tie_breaking_is_unchanged() -> None:
    registry = FeatureStoreRegistry(
        [
            make_store("B", np.asarray([[1.0, 0.0]], dtype=np.float32), [5]),
            make_store("A", np.asarray([[1.0, 0.0]], dtype=np.float32), [6]),
        ]
    )
    result = ExactNumpyRetriever(registry, FakeEncoder()).retrieve(
        KISQuery("single", "unchanged", 2)
    )
    assert [(item.video_id, item.frame_id) for item in result.ranked_candidates] == [
        ("A", 6),
        ("B", 5),
    ]
