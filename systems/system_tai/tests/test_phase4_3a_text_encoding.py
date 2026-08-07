from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.kis.session_engine import KISResult, OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest
from system_tai.refinement.models import (
    Phase3Candidate,
    RefinementConfig,
    RefinementQuery,
)
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType
from tests.test_phase4_2_session import setup_runtime


class CountingFakeEncoder:
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.identifiers = {"model": "ViT-B/32", "device": "cpu"}
        self.encode_calls = 0
        self.encode_texts_calls = 0
        self.encode_images_calls = 0

    def encode(self, text: str) -> np.ndarray:
        self.encode_calls += 1
        return np.ones(self.dimension, dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encode_texts_calls += 1
        return np.ones((len(texts), self.dimension), dtype=np.float32)

    def encode_images(self, images: list[Any], batch_size: int = 1) -> np.ndarray:
        self.encode_images_calls += 1
        return np.ones((len(images), self.dimension), dtype=np.float32)


def test_retrieval_batch_encoding_equivalence(tmp_path: Path) -> None:
    """BATCH VS OLD PER-TEXT RETRIEVAL EQUIVALENCE"""
    runtime_old, _ = setup_runtime(tmp_path / "old")
    runtime_new, _ = setup_runtime(tmp_path / "new")

    old_fake = CountingFakeEncoder()
    runtime_old.shared_encoder = old_fake
    new_fake = CountingFakeEncoder()
    runtime_new.shared_encoder = new_fake

    def old_handle_query(self: OperationalKISRuntime, request: QueryRequest) -> dict[str, Any]:
        variants = request.variants()
        rankings: dict[str, KISResult] = {}
        for variant in variants:
            vector = self.shared_encoder.encode(variant.text)
            rankings[variant.variant_id] = self.exact_retriever.search_vector(
                query_id=f"{request.query_id}::{variant.variant_id}",
                query_vector=vector,
                top_k=request.top_k_per_variant,
            )
        fused_result = self.weighted_rrf.fuse_rankings(
            query_id=request.query_id,
            variants=variants,
            rankings=rankings,
            output_top_k=request.output_top_k,
            rrf_constant=self.config.rrf_constant,
        )
        out_path = self.output_root / "queries" / request.query_id / "top100.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.exporter.export(fused_result, out_path)
        return {
            "artifacts": {
                "top100_jsonl": str(out_path.relative_to(self.output_root)).replace("\\", "/")
            },
            "fused_result": fused_result,
        }

    import types

    runtime_old.handle_query = types.MethodType(old_handle_query, runtime_old)

    req = QueryRequest(
        request_id="req1",
        query_id="q1",
        query_vi="nhiều người đi bộ trên phố",
        query_en="many people walking on a city street",
        query_en_expansion="a crowd of pedestrians moving along a busy urban street",
        refine_top_n=0,
    )
    variants = req.variants()
    assert len(variants) == 3

    res_old = runtime_old.handle_query(req)
    res_new = runtime_new.handle_query(req)

    old_path = runtime_old.output_root / res_old["artifacts"]["top100_jsonl"]
    new_path = runtime_new.output_root / res_new["artifacts"]["top100_jsonl"]

    assert old_path.read_bytes() == new_path.read_bytes()
    assert old_fake.encode_calls == 3
    assert old_fake.encode_texts_calls == 0
    assert new_fake.encode_calls == 0
    assert new_fake.encode_texts_calls == 1


def test_one_text_forward_per_session_query(tmp_path: Path) -> None:
    """ONE TEXT FORWARD PER SESSION QUERY"""
    runtime, _ = setup_runtime(tmp_path)
    fake_encoder = CountingFakeEncoder()
    runtime.shared_encoder = fake_encoder
    runtime.refiner.encoder = fake_encoder  # same encoder

    # Retrieval only
    req_retrieval = QueryRequest(
        request_id="req-ret",
        query_id="q-ret",
        query_vi="vi query",
        query_en="en query",
        query_en_expansion="en expansion",
        refine_top_n=0,
    )
    assert len(req_retrieval.variants()) == 3

    before = fake_encoder.encode_texts_calls
    runtime.handle_query(req_retrieval)
    after = fake_encoder.encode_texts_calls

    assert after - before == 1
    assert fake_encoder.encode_calls == 0

    # Refinement
    req_refine = QueryRequest(
        request_id="req-ref",
        query_id="q-ref",
        query_vi="vi query refine",
        query_en="en query refine",
        query_en_expansion="en expansion refine",
        refine_top_n=3,
    )
    assert len(req_refine.variants()) == 3

    before = fake_encoder.encode_texts_calls
    runtime.handle_query(req_refine)
    after = fake_encoder.encode_texts_calls

    assert after - before == 1
    assert fake_encoder.encode_calls == 0


def test_refiner_backward_compatibility(tmp_path: Path) -> None:
    """REFINER BACKWARD COMPATIBILITY"""
    runtime, _ = setup_runtime(tmp_path)
    fake_encoder = CountingFakeEncoder()
    runtime.refiner.encoder = fake_encoder

    query = RefinementQuery(
        query_id="q1",
        variants=(
            QueryVariant(
                "vi", "text", QueryLanguage.VIETNAMESE, QueryVariantType.VIETNAMESE_DIRECT, 1.0
            ),
        ),
        candidates=(Phase3Candidate("q1", 1, "L21_V001", 100, 1.0, {}),),
    )
    config = RefinementConfig()

    runtime.refiner.refine_query(query, config)

    # Text encoded internally because precomputed was not supplied
    assert fake_encoder.encode_texts_calls == 1


def test_precomputed_refinement_equivalence(tmp_path: Path) -> None:
    """PRECOMPUTED REFINEMENT EQUIVALENCE"""
    runtime, _ = setup_runtime(tmp_path)

    # Create fake query and candidate with 3 variants
    query = RefinementQuery(
        query_id="q1",
        variants=(
            QueryVariant(
                "vi",
                "nhiều người",
                QueryLanguage.VIETNAMESE,
                QueryVariantType.VIETNAMESE_DIRECT,
                1.0,
            ),
            QueryVariant(
                "en",
                "many people",
                QueryLanguage.ENGLISH,
                QueryVariantType.ENGLISH_TRANSLATION,
                1.0,
            ),
            QueryVariant(
                "en-exp",
                "crowd of pedestrians",
                QueryLanguage.ENGLISH,
                QueryVariantType.ENGLISH_EXPANSION,
                1.0,
            ),
        ),
        candidates=(Phase3Candidate("q1", 1, "L21_V001", 100, 1.0, {}),),
    )
    config = RefinementConfig(top_candidates_to_refine=1)

    # Path A: Normal
    outcome_a = runtime.refiner.refine_query(query, config)

    # Path B: Precomputed
    text_embeddings = runtime.refiner.encoder.encode_texts([v.text for v in query.variants])
    outcome_b = runtime.refiner.refine_query(
        query, config, precomputed_text_embeddings=text_embeddings
    )

    assert len(outcome_a.candidates) == len(outcome_b.candidates)
    assert (
        outcome_a.candidates[0].original_candidate_rank
        == outcome_b.candidates[0].original_candidate_rank
    )
    assert outcome_a.candidates[0].video_id == outcome_b.candidates[0].video_id
    assert outcome_a.candidates[0].candidate_frame_id == outcome_b.candidates[0].candidate_frame_id
    assert outcome_a.candidates[0].refined_frame_id == outcome_b.candidates[0].refined_frame_id
    assert outcome_a.candidates[0].status == outcome_b.candidates[0].status

    # Timing fields might differ, but output KISResult must match
    assert len(outcome_a.result.ranked_candidates) == len(outcome_b.result.ranked_candidates)
    if outcome_a.result.ranked_candidates:
        assert (
            outcome_a.result.ranked_candidates[0].rank == outcome_b.result.ranked_candidates[0].rank
        )
        assert (
            outcome_a.result.ranked_candidates[0].video_id
            == outcome_b.result.ranked_candidates[0].video_id
        )
        assert (
            outcome_a.result.ranked_candidates[0].frame_id
            == outcome_b.result.ranked_candidates[0].frame_id
        )
        assert (
            outcome_a.result.ranked_candidates[0].source
            == outcome_b.result.ranked_candidates[0].source
        )
        assert (
            outcome_a.result.ranked_candidates[0].score
            == outcome_b.result.ranked_candidates[0].score
        )


def test_validation_tests(tmp_path: Path) -> None:
    """VALIDATION TESTS"""
    runtime, _ = setup_runtime(tmp_path)
    fake_encoder = CountingFakeEncoder()
    runtime.refiner.encoder = fake_encoder

    query = RefinementQuery(
        query_id="q1",
        variants=(
            QueryVariant(
                "vi", "text", QueryLanguage.VIETNAMESE, QueryVariantType.VIETNAMESE_DIRECT, 1.0
            ),
        ),
        candidates=(Phase3Candidate("q1", 1, "L21_V001", 100, 1.0, {}),),
    )
    config = RefinementConfig()

    valid_embeds = np.ones((1, 2), dtype=np.float32)

    # wrong ndim
    with pytest.raises(ValueError, match="2-dimensional"):
        runtime.refiner.refine_query(
            query, config, precomputed_text_embeddings=np.ones(2, dtype=np.float32)
        )
    assert fake_encoder.encode_texts_calls == 0

    # wrong row count
    with pytest.raises(ValueError, match="match number of query variants"):
        runtime.refiner.refine_query(
            query, config, precomputed_text_embeddings=np.ones((2, 2), dtype=np.float32)
        )
    assert fake_encoder.encode_texts_calls == 0

    # NaN
    invalid_nan = valid_embeds.copy()
    invalid_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        runtime.refiner.refine_query(query, config, precomputed_text_embeddings=invalid_nan)
    assert fake_encoder.encode_texts_calls == 0

    # Infinity
    invalid_inf = valid_embeds.copy()
    invalid_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        runtime.refiner.refine_query(query, config, precomputed_text_embeddings=invalid_inf)
    assert fake_encoder.encode_texts_calls == 0

    # zero-norm row
    invalid_zero = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="zero-norm row"):
        runtime.refiner.refine_query(query, config, precomputed_text_embeddings=invalid_zero)
    assert fake_encoder.encode_texts_calls == 0
