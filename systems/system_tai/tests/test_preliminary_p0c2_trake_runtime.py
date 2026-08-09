import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.kis.session import session_config_from_args
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    HealthRequest,
    InvalidRequestError,
    TRAKEQueryRequest,
    parse_session_request,
)
from system_tai.refinement.models import (
    RefinedCandidate,
    RefinementConfig,
    RefinementStatus,
)
from system_tai.trake.engine import TRAKEEngine
from system_tai.trake.runtime import TRAKERuntimePipeline


class FakeEncoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.identifiers = {"device": "cpu", "model": "fake"}
        self.encode_texts_calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self.dim

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encode_texts_calls.append(list(texts))
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            if "event1" in t or "event_0" in t or "xe" in t:
                v[0] = 1.0
            elif "event2" in t or "event_1" in t or "người" in t:
                v[1] = 1.0
            elif "event3" in t or "event_2" in t:
                v[2] = 1.0
            else:
                v[3] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


class FakeRetriever:
    def __init__(self, cands_per_event: list[list[dict]] | None = None) -> None:
        self.cands_per_event = cands_per_event
        self.search_vector_calls: list[dict] = []
        self.search_vectors_calls: list[dict] = []

    def search_vectors(
        self,
        query_ids: Sequence[str],
        query_vectors: Sequence[np.ndarray],
        top_k: int,
    ) -> dict[str, KISResult]:
        self.search_vectors_calls.append({"query_ids": list(query_ids), "top_k": top_k})
        results: dict[str, KISResult] = {}
        for qid, vec in zip(query_ids, query_vectors):
            results[qid] = self._single_search(query_id=qid, query_vector=vec, top_k=top_k)
        return results

    def search_vector(self, query_id: str, query_vector: np.ndarray, top_k: int) -> KISResult:
        self.search_vector_calls.append({"query_id": query_id, "top_k": top_k})
        return self._single_search(query_id=query_id, query_vector=query_vector, top_k=top_k)

    def _single_search(
        self, query_id: str, query_vector: np.ndarray, top_k: int
    ) -> KISResult:
        if self.cands_per_event and "::e" in query_id:
            e_part = query_id.split("::e")[1].split("::")[0]
            e_idx = int(e_part)
            raw_cands = self.cands_per_event[e_idx] if e_idx < len(self.cands_per_event) else []
            cands = [
                CandidateFrame(
                    rank=c["rank"],
                    video_id=c["video_id"],
                    frame_id=c["frame_id"],
                    score=c.get("score", 0.9),
                    clip_row=c.get("clip_row", 1),
                    keyframe_order=c.get("keyframe_order", 1),
                    source="fake_exact",
                    diagnostic_metadata=c.get(
                        "diagnostic_metadata",
                        {"variant_hit_count": 1, "best_individual_rank": 1},
                    ),
                )
                for c in raw_cands
            ]
            return KISResult(query_id=query_id, ranked_candidates=tuple(cands))

        cands = [
            CandidateFrame(
                rank=1,
                video_id="V001",
                frame_id=100,
                score=0.9,
                clip_row=1,
                keyframe_order=1,
                source="fake_exact",
                diagnostic_metadata={"variant_hit_count": 1, "best_individual_rank": 1},
            ),
            CandidateFrame(
                rank=2,
                video_id="V001",
                frame_id=200,
                score=0.8,
                clip_row=2,
                keyframe_order=2,
                source="fake_exact",
                diagnostic_metadata={"variant_hit_count": 1, "best_individual_rank": 1},
            ),
        ]
        return KISResult(query_id=query_id, ranked_candidates=tuple(cands))


class FakeWeightedRRF:
    def fuse_rankings(
        self,
        *,
        query_id: str,
        variants: tuple,
        rankings: dict,
        output_top_k: int,
        rrf_constant: float = 60.0,
    ) -> KISResult:
        empty_res = KISResult(query_id=query_id, ranked_candidates=())
        first_ranking = list(rankings.values())[0] if rankings else empty_res
        if isinstance(first_ranking, KISResult):
            ranked_cands = first_ranking.ranked_candidates
        elif isinstance(first_ranking, dict):
            ranked_cands = first_ranking.get("ranked_candidates", [])
        else:
            ranked_cands = ()

        cands = []
        for rc in list(ranked_cands)[:output_top_k]:
            if isinstance(rc, CandidateFrame):
                cands.append(rc)
            else:
                cands.append(
                    CandidateFrame(
                        rank=rc["rank"],
                        video_id=rc["video_id"],
                        frame_id=rc["frame_id"],
                        score=rc.get("score", 0.8),
                        clip_row=rc.get("clip_row", 0),
                        keyframe_order=rc.get("keyframe_order", 0),
                        source="rrf",
                        diagnostic_metadata=rc.get("diagnostic_metadata"),
                    )
                )
        return KISResult(query_id=query_id, ranked_candidates=tuple(cands))


class FakeRefiner:
    def __init__(self, proposal_map: dict[tuple[int, str, int], int] | None = None) -> None:
        self.proposal_map = proposal_map or {}
        self.refine_query_calls: list[dict] = []

    def refine_query(
        self,
        query: Any,
        config: RefinementConfig,
        *,
        precomputed_text_embeddings: np.ndarray | None = None,
        frame_embedding_cache: Any | None = None,
    ):
        from system_tai.refinement.models import RefinementQuery

        assert isinstance(query, RefinementQuery), "query must be RefinementQuery"
        ranks = [c.rank for c in query.candidates]
        assert ranks == list(range(1, len(query.candidates) + 1)), "candidate ranks must be 1..M"
        assert all(c.query_id == query.query_id for c in query.candidates), "query_id mismatch"
        assert precomputed_text_embeddings is not None, "embeddings required"
        assert isinstance(precomputed_text_embeddings, np.ndarray), "must be ndarray"
        assert precomputed_text_embeddings.ndim == 2, "must be 2D"
        assert precomputed_text_embeddings.shape[0] == len(query.variants), "rows mismatch"

        self.refine_query_calls.append({
            "query_id": query.query_id,
            "candidate_count": len(query.candidates),
            "precomputed_text_embeddings": precomputed_text_embeddings,
            "frame_embedding_cache": frame_embedding_cache,
        })
        refined = []
        e_idx = 0
        if "::trake_refine_e" in query.query_id:
            e_idx = int(query.query_id.split("::trake_refine_e")[1])

        for c in query.candidates:
            key = (e_idx, c.video_id, c.frame_id)
            prop = self.proposal_map.get(key, c.frame_id)
            is_refined = prop != c.frame_id
            status = RefinementStatus.REFINED if is_refined else RefinementStatus.KEEP_ORIGINAL
            rc = RefinedCandidate(
                query_id=query.query_id,
                original_candidate_rank=c.rank,
                video_id=c.video_id,
                candidate_frame_id=c.frame_id,
                refined_frame_id=prop,
                candidate_timestamp_seconds=1.0,
                refined_timestamp_seconds=1.2 if is_refined else 1.0,
                fps=25.0,
                total_frame_count=1000,
                window_start_frame=None,
                window_end_frame=None,
                coarse_frame_ids=(),
                fine_frame_ids=(),
                coarse_sample_count=0,
                fine_sample_count=0,
                decoded_frame_count=0,
                encoded_image_count=0,
                refinement_fusion_score=0.95 if is_refined else None,
                variant_hit_count=1,
                best_individual_rank=1,
                per_variant_provenance=(),
                decoder_backend="opencv",
                raw_video_path=Path(f"{c.video_id}.mp4"),
                status=status,
                warnings=(),
                failure_reason=None,
                original_retrieval_provenance={},
                timings={},
            )
            refined.append(rc)

        outcome = type("Outcome", (), {
            "query_id": query.query_id,
            "candidates": tuple(refined),
            "result": KISResult(query_id=query.query_id, ranked_candidates=()),
            "warnings": (),
            "timings": {},
        })()
        return outcome


def make_test_pipeline(
    cands_per_event: list[list[dict]] | None = None,
    proposal_map: dict[tuple[int, str, int], int] | None = None,
):
    encoder = FakeEncoder()
    retriever = FakeRetriever(cands_per_event=cands_per_event)
    rrf = FakeWeightedRRF()
    refiner = FakeRefiner(proposal_map=proposal_map)
    engine = TRAKEEngine()

    pipeline = TRAKERuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=rrf,
        refiner=refiner,
        shared_encoder=encoder,
        trake_engine=engine,
    )
    return pipeline, encoder, retriever, rrf, refiner, engine


class FakeVideoStore:
    def contains_frame(self, frame_id: int) -> bool:
        return True


class FakeRegistry:
    embedding_dimension = 4
    total_rows = 100
    stores = ()

    def get(self, video_id: str):
        return FakeVideoStore()


# ======================================================================
# TEST CASES A - AR
# ======================================================================


def test_a_schema_validation_trake_query_parsing():
    line = json.dumps({
        "type": "trake_query",
        "request_id": "req-001",
        "query_id": "T001",
        "events": [{"description": "event 1"}, {"description": "event 2"}],
    })
    req = parse_session_request(line)
    assert isinstance(req, TRAKEQueryRequest)
    assert req.request_id == "req-001"
    assert req.query_id == "T001"
    assert len(req.events) == 2


def test_b_schema_validation_strict_integer_rejection():
    # bool rejected
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            json.dumps({
                "type": "trake_query",
                "request_id": "req-1",
                "query_id": "T1",
                "events": [{"description": "e1"}],
                "top_k_per_variant": True,
            })
        )
    # float rejected
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            json.dumps({
                "type": "trake_query",
                "request_id": "req-1",
                "query_id": "T1",
                "events": [{"description": "e1"}],
                "beam_width": 100.0,
            })
        )
    # str rejected
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            json.dumps({
                "type": "trake_query",
                "request_id": "req-1",
                "query_id": "T1",
                "events": [{"description": "e1"}],
                "output_top_k": "10",
            })
        )


def test_c_schema_validation_empty_fields():
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            json.dumps({
                "type": "trake_query",
                "request_id": "",
                "query_id": "T1",
                "events": [{"description": "e1"}],
            })
        )
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            json.dumps({
                "type": "trake_query",
                "request_id": "req-1",
                "query_id": "T1",
                "events": [],
            })
        )


def test_d_schema_validation_malformed_event_objects():
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            json.dumps({
                "type": "trake_query",
                "request_id": "req-1",
                "query_id": "T1",
                "events": [{"description": ""}],
            })
        )


def test_f_single_batched_text_encode_call():
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
    ]
    pipeline, encoder, _, _, _, _ = make_test_pipeline(cands_per_event=cands)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-f",
            "query_id": "TF",
            "events": [
                {"description": "xe chạy", "description_en": "car running"},
                {"description": "người ngã"},
            ],
        })
    )
    ref_config = RefinementConfig()
    pipeline.process_trake_query(req, refinement_config=ref_config)

    assert len(encoder.encode_texts_calls) == 1
    # 3 total variants (e0_vi, e0_en, e1_vi)
    assert len(encoder.encode_texts_calls[0]) == 3
    assert encoder.encode_texts_calls[0] == ["xe chạy", "car running", "người ngã"]


def test_g_vietnamese_only_event_variants():
    pipeline, _, _, _, _, _ = make_test_pipeline()
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-g",
            "query_id": "TG",
            "events": [{"description": "xe chạy"}],
        })
    )
    ref_config = RefinementConfig()
    res, _, extra = pipeline.process_trake_query(req, refinement_config=ref_config)
    assert extra["flattened_variants"][0]["variant_id"] == "TG::e0::v1_vi"
    assert len(extra["flattened_variants"]) == 1


def test_h_bilingual_event_variants():
    pipeline, _, _, _, _, _ = make_test_pipeline()
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-h",
            "query_id": "TH",
            "events": [{"description": "xe chạy", "description_en": "car running"}],
        })
    )
    ref_config = RefinementConfig()
    _, _, extra = pipeline.process_trake_query(req, refinement_config=ref_config)
    assert len(extra["flattened_variants"]) == 2
    assert extra["flattened_variants"][0]["variant_id"] == "TH::e0::v1_vi"
    assert extra["flattened_variants"][1]["variant_id"] == "TH::e0::v2_en"


def test_k_refine_top_n_zero_bypasses_refiner():
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
    ]
    pipeline, _, _, _, refiner, _ = make_test_pipeline(cands_per_event=cands)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-k",
            "query_id": "TK",
            "events": [{"description": "e0"}, {"description": "e1"}],
            "refine_top_n": 0,
        })
    )
    ref_config = RefinementConfig()
    res, _, _ = pipeline.process_trake_query(req, refinement_config=ref_config)
    assert len(refiner.refine_query_calls) == 0
    assert len(res.predictions) == 1
    assert res.predictions[0].frame_ids == (10, 20)


def test_l_m_n_o_refine_top_n_unique_node_batching_and_ranks():
    cands = [
        [
            {"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9},
            {"rank": 2, "video_id": "V1", "frame_id": 15, "score": 0.8},
        ],
        [
            {"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9},
            {"rank": 2, "video_id": "V1", "frame_id": 25, "score": 0.8},
        ],
    ]
    # Proposal map: (0, 'V1', 10) -> 12
    prop_map = {(0, "V1", 10): 12}
    pipeline, _, _, _, refiner, _ = make_test_pipeline(
        cands_per_event=cands, proposal_map=prop_map
    )
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-l",
            "query_id": "TL",
            "events": [{"description": "e0"}, {"description": "e1"}],
            "refine_top_n": 2,
        })
    )
    ref_config = RefinementConfig()
    res, _, _ = pipeline.process_trake_query(req, refinement_config=ref_config)

    assert len(refiner.refine_query_calls) == 2  # 1 call per event
    # Check precomputed text embeddings passed
    for call in refiner.refine_query_calls:
        assert call["precomputed_text_embeddings"] is not None
    # Check prediction has refined frame_id 12 for event 0
    assert res.predictions[0].frame_ids == (12, 20)


def test_p_temporal_safety_valid_refinement_accepted():
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
    ]
    # Refined: (10 -> 12), (20 -> 22). 12 <= 22 is valid
    prop_map = {(0, "V1", 10): 12, (1, "V1", 20): 22}
    pipeline, _, _, _, _, _ = make_test_pipeline(cands_per_event=cands, proposal_map=prop_map)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-p",
            "query_id": "TP",
            "events": [{"description": "e0"}, {"description": "e1"}],
            "refine_top_n": 1,
        })
    )
    res, _, extra = pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    assert res.predictions[0].frame_ids == (12, 22)
    assert extra["path_diagnostics"][0]["temporal_fallback_reason"] is None


def test_q_r_temporal_safety_violation_triggers_whole_path_fallback():
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
    ]
    # Refined: (10 -> 25), (20 -> 15). Proposed (25, 15) breaks temporal order 25 > 15!
    prop_map = {(0, "V1", 10): 25, (1, "V1", 20): 15}
    pipeline, _, _, _, _, _ = make_test_pipeline(cands_per_event=cands, proposal_map=prop_map)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-q",
            "query_id": "TQ",
            "events": [{"description": "e0"}, {"description": "e1"}],
            "refine_top_n": 1,
        })
    )
    res, _, extra = pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    # ENTIRE path falls back to original C1 frames (10, 20)
    assert res.predictions[0].frame_ids == (10, 20)
    assert (
        extra["path_diagnostics"][0]["temporal_fallback_reason"]
        == "refinement_temporal_order_violation"
    )
    assert res.diagnostics["temporal_fallback_path_count"] == 1


def test_s_t_u_v_duplicate_resolution_and_rank_preservation():
    # C1 produces 2 paths:
    # Path 1: (V1, 10, 20) -> proposed (V1, 12, 22)
    # Path 2: (V1, 12, 22) -> proposed (V1, 12, 22) -> Option A collides with Path 1 Option A!
    # Path 2 Option B is original (V1, 12, 22) which also collides!
    cands = [
        [
            {"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9},
            {"rank": 2, "video_id": "V1", "frame_id": 12, "score": 0.8},
        ],
        [
            {"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9},
            {"rank": 2, "video_id": "V1", "frame_id": 22, "score": 0.8},
        ],
    ]
    prop_map = {(0, "V1", 10): 12, (1, "V1", 20): 22}
    pipeline, _, _, _, _, _ = make_test_pipeline(cands_per_event=cands, proposal_map=prop_map)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-s",
            "query_id": "TS",
            "events": [{"description": "e0"}, {"description": "e1"}],
            "refine_top_n": 2,
        })
    )
    res, _, _ = pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    # 3 predictions emitted (ranks 1, 2, 3)
    assert len(res.predictions) == 3
    assert res.predictions[0].rank == 1
    assert res.predictions[0].frame_ids == (12, 22)
    assert res.predictions[1].rank == 2
    assert res.predictions[1].frame_ids == (10, 22)
    assert res.predictions[2].rank == 3
    assert res.predictions[2].frame_ids == (12, 20)


def test_w_x_y_output_validation_gate():
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
    ]
    pipeline, _, _, _, _, engine = make_test_pipeline(cands_per_event=cands)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-w",
            "query_id": "TW",
            "events": [{"description": "e0"}, {"description": "e1"}],
        })
    )
    res, _, _ = pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    assert len(res.predictions) == 1
    assert res.predictions[0].frame_ids == (10, 20)


def test_z_aa_ab_ac_artifacts_and_response_payload(tmp_path: Path):
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.refinement.video import OpenCVVideoDecoder, RawVideoRegistry

    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
    ]
    encoder = FakeEncoder()
    retriever = FakeRetriever(cands_per_event=cands)
    rrf = FakeWeightedRRF()
    refiner = FakeRefiner()

    conf_args = type(
        "Args",
        (),
        {
            "input_root": tmp_path,
            "reuse_manifest": None,
            "manifest_cache": None,
            "output_root": tmp_path / "session_out",
            "device": "cpu",
            "allow_model_download": True,
            "clip_cache_dir": None,
            "rrf_constant": 60.0,
            "chunk_size": 4096,
            "default_top_k_per_variant": 100,
            "default_output_top_k": 100,
            "default_refine_top_n": 3,
            "max_requests": None,
            "continue_on_request_error": True,
            "fail_fast_protocol": False,
            "session_id": "test-sess",
            "window_before_seconds": 5.0,
            "window_after_seconds": 5.0,
            "coarse_stride_frames": 15,
            "coarse_top_n": 3,
            "fine_radius_frames": 30,
            "fine_stride_frames": 1,
            "image_batch_size": 32,
            "max_decoded_frames_per_candidate": 500,
            "missing_raw_video_policy": "keep-original",
            "candidate_failure_policy": "keep-original",
            "coarse_decode_strategy": "sequential",
        },
    )()
    sess_config = session_config_from_args(conf_args)

    manifest_stub = type(
        "CorpusManifestStub", (), {"fingerprint": "fp123", "schema_version": 1, "videos": {}}
    )()
    raw_video_reg = RawVideoRegistry(records={})
    decoder = OpenCVVideoDecoder()
    reg = FakeRegistry()

    runtime = OperationalKISRuntime(
        config=sess_config,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest_stub,
        registry=reg,
        raw_video_registry=raw_video_reg,
        shared_encoder=encoder,
        decoder=decoder,
    )
    # Monkeypatch inner retriever & rrf
    runtime.exact_retriever = retriever
    runtime.weighted_rrf = rrf
    runtime.refiner = refiner
    runtime.trake_pipeline.exact_retriever = retriever
    runtime.trake_pipeline.weighted_rrf = rrf
    runtime.trake_pipeline.refiner = refiner

    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-z",
            "query_id": "TZ",
            "events": [{"description": "e0"}, {"description": "e1"}],
        })
    )

    resp = runtime.handle_trake_query(req)
    assert resp["type"] == "trake_result"
    assert resp["status"] == "SUCCESS"
    assert resp["prediction_count"] == 1

    artifacts = resp["artifacts"]
    assert "trake_predictions_jsonl" in artifacts
    assert "trake_event_candidates_json" in artifacts
    assert "trake_refinement_json" in artifacts
    assert "trake_request_manifest" in artifacts
    assert "trake_timings" in artifacts

    req_dir = tmp_path / "session_out" / Path(artifacts["trake_predictions_jsonl"]).parent
    assert (req_dir / "trake_predictions.jsonl").exists()
    assert (req_dir / "trake_event_candidates.json").exists()
    assert (req_dir / "trake_refinement.json").exists()
    assert (req_dir / "trake_request_manifest.json").exists()
    assert (req_dir / "trake_timings.json").exists()


def test_e_duplicate_request_id_across_types(tmp_path: Path):
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.refinement.video import OpenCVVideoDecoder, RawVideoRegistry

    sess_config = session_config_from_args(
        type(
            "Args",
            (),
            {
                "input_root": tmp_path,
                "reuse_manifest": None,
                "manifest_cache": None,
                "output_root": tmp_path / "out",
                "device": "cpu",
                "allow_model_download": True,
                "clip_cache_dir": None,
                "rrf_constant": 60.0,
                "chunk_size": 4096,
                "default_top_k_per_variant": 100,
                "default_output_top_k": 100,
                "default_refine_top_n": 3,
                "max_requests": None,
                "continue_on_request_error": True,
                "fail_fast_protocol": False,
                "session_id": "test-sess-e",
                "window_before_seconds": 5.0,
                "window_after_seconds": 5.0,
                "coarse_stride_frames": 15,
                "coarse_top_n": 3,
                "fine_radius_frames": 30,
                "fine_stride_frames": 1,
                "image_batch_size": 32,
                "max_decoded_frames_per_candidate": 500,
                "missing_raw_video_policy": "keep-original",
                "candidate_failure_policy": "keep-original",
                "coarse_decode_strategy": "sequential",
            },
        )()
    )
    manifest_stub = type(
        "CorpusManifestStub", (), {"fingerprint": "fp123", "schema_version": 1, "videos": {}}
    )()
    runtime = OperationalKISRuntime(
        config=sess_config,
        manifest_path=tmp_path / "m.json",
        manifest=manifest_stub,
        registry=FakeRegistry(),
        raw_video_registry=RawVideoRegistry({}),
        shared_encoder=FakeEncoder(),
        decoder=OpenCVVideoDecoder(),
    )

    # First handle health request with ID "dupe-id"
    runtime.handle_health(HealthRequest(request_id="dupe-id"))

    # Now attempt trake request with same "dupe-id"
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "dupe-id",
            "query_id": "TE",
            "events": [{"description": "e0"}],
        })
    )
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_trake_query(req)


def test_af_zero_valid_paths_return_empty():
    # Event 0 has V1 at frame 100, Event 1 has V1 at frame 50
    # (decreasing order, impossible to form path)
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 100, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 50, "score": 0.9}],
    ]
    pipeline, _, _, _, _, _ = make_test_pipeline(cands_per_event=cands)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-af",
            "query_id": "TAF",
            "events": [{"description": "e0"}, {"description": "e1"}],
        })
    )
    res, _, _ = pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    assert len(res.predictions) == 0
    assert res.diagnostics["zero_output_reason"] == "no_temporal_valid_path"


def test_aq_multi_event_3_events_chain():
    cands = [
        [{"rank": 1, "video_id": "V1", "frame_id": 10, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 20, "score": 0.9}],
        [{"rank": 1, "video_id": "V1", "frame_id": 30, "score": 0.9}],
    ]
    pipeline, _, _, _, _, _ = make_test_pipeline(cands_per_event=cands)
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-aq",
            "query_id": "TAQ",
            "events": [{"description": "e0"}, {"description": "e1"}, {"description": "e2"}],
        })
    )
    res, _, _ = pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    assert len(res.predictions) == 1
    assert res.predictions[0].frame_ids == (10, 20, 30)


# ======================================================================
# PRODUCTION COMPATIBILITY AUDIT TESTS
# ======================================================================

def test_queryvariant_production_enums_and_regression():
    from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType

    # Valid instantiation with enums
    v_vi = QueryVariant(
        variant_id="Q1::e0::v1_vi",
        text="mô tả",
        language=QueryLanguage.VIETNAMESE,
        variant_type=QueryVariantType.VIETNAMESE_DIRECT,
    )
    assert v_vi.language == QueryLanguage.VIETNAMESE
    assert v_vi.variant_type == QueryVariantType.VIETNAMESE_DIRECT

    # Raw string language raises ValueError
    with pytest.raises(ValueError, match="language must be a supported QueryLanguage"):
        QueryVariant(
            variant_id="Q1::e0::v1_vi",
            text="mô tả",
            language="vi",  # type: ignore
            variant_type=QueryVariantType.VIETNAMESE_DIRECT,
        )

    # Raw string variant_type raises ValueError
    with pytest.raises(ValueError, match="variant_type must be a supported QueryVariantType"):
        QueryVariant(
            variant_id="Q1::e0::v1_vi",
            text="mô tả",
            language=QueryLanguage.VIETNAMESE,
            variant_type="vietnamese_direct",  # type: ignore
        )


def test_dynamic_variant_ids_across_queries():
    pipeline, _, _, _, _, _ = make_test_pipeline()

    req_alpha = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-alpha",
            "query_id": "Q_ALPHA",
            "events": [{"description": "xe chạy", "description_en": "car running"}],
        })
    )
    req_beta = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-beta",
            "query_id": "Q_BETA",
            "events": [{"description": "xe chạy", "description_en": "car running"}],
        })
    )

    _, _, diag_alpha = pipeline.process_trake_query(req_alpha, refinement_config=RefinementConfig())
    _, _, diag_beta = pipeline.process_trake_query(req_beta, refinement_config=RefinementConfig())

    var_ids_alpha = [v["variant_id"] for v in diag_alpha["flattened_variants"]]
    var_ids_beta = [v["variant_id"] for v in diag_beta["flattened_variants"]]

    assert var_ids_alpha == ["Q_ALPHA::e0::v1_vi", "Q_ALPHA::e0::v2_en"]
    assert var_ids_beta == ["Q_BETA::e0::v1_vi", "Q_BETA::e0::v2_en"]
    assert var_ids_alpha != var_ids_beta


def test_real_contract_integration_flow():
    """Integration test verifying real domain/schema objects across TRAKE runtime flow."""
    pipeline, encoder, retriever, rrf, refiner, engine = make_test_pipeline()

    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-int-1",
            "query_id": "Q_INT",
            "events": [
                {"description": "xe dừng lại", "description_en": "car stopping"},
                {"description": "người bước xuống", "description_en": "person stepping out"},
            ],
            "refine_top_n": 1,
        })
    )

    ref_cfg = RefinementConfig()
    res, timings, extra_diag = pipeline.process_trake_query(req, refinement_config=ref_cfg)

    # 1. Verify single batch encode call
    assert len(encoder.encode_texts_calls) == 1
    assert encoder.encode_texts_calls[0] == [
        "xe dừng lại",
        "car stopping",
        "người bước xuống",
        "person stepping out",
    ]

    # 2. Verify search_vectors was called with variant IDs
    assert len(retriever.search_vectors_calls) == 1
    called_ids = retriever.search_vectors_calls[0]["query_ids"]
    assert called_ids == [
        "Q_INT::e0::v1_vi",
        "Q_INT::e0::v2_en",
        "Q_INT::e1::v1_vi",
        "Q_INT::e1::v2_en",
    ]

    # 3. Verify refiner received precomputed 2D numpy text embeddings
    assert len(refiner.refine_query_calls) == 2
    for call in refiner.refine_query_calls:
        embeddings = call["precomputed_text_embeddings"]
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.dtype == np.float32

    # 4. Verify predictions result
    assert res.query_id == "Q_INT"
    assert len(res.predictions) > 0
    pred = res.predictions[0]
    assert pred.query_id == "Q_INT"
    assert pred.video_id == "V001"
    assert len(pred.frame_ids) == 2


def test_single_batched_text_encode_proof():
    pipeline, encoder, _, _, _, _ = make_test_pipeline()
    req = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "req-encode-proof",
            "query_id": "Q_ENCODE",
            "events": [
                {"description": "sự kiện 1", "description_en": "event 1"},
                {"description": "sự kiện 2", "description_en": "event 2"},
            ],
            "refine_top_n": 2,
        })
    )
    pipeline.process_trake_query(req, refinement_config=RefinementConfig())
    assert len(encoder.encode_texts_calls) == 1
    assert encoder.encode_texts_calls[0] == [
        "sự kiện 1",
        "event 1",
        "sự kiện 2",
        "event 2",
    ]


def test_cross_task_request_id_uniqueness_exact(tmp_path):
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.kis.session_schema import QAQueryRequest, QueryRequest
    from system_tai.refinement.video import OpenCVVideoDecoder, RawVideoRegistry

    sess_config = session_config_from_args(
        type(
            "Args",
            (),
            {
                "input_root": tmp_path,
                "reuse_manifest": None,
                "manifest_cache": None,
                "output_root": tmp_path / "session_out",
                "device": "cpu",
                "allow_model_download": False,
                "clip_cache_dir": None,
                "rrf_constant": 60.0,
                "default_top_k_per_variant": 100,
                "default_output_top_k": 100,
                "default_refine_top_n": 3,
                "max_requests": None,
                "chunk_size": 4096,
                "continue_on_request_error": True,
                "fail_fast_protocol": False,
                "session_id": "test-sess-cross",
                "window_before_seconds": 5.0,
                "window_after_seconds": 5.0,
                "coarse_stride_frames": 15,
                "coarse_top_n": 3,
                "fine_radius_frames": 30,
                "fine_stride_frames": 1,
                "image_batch_size": 32,
                "max_decoded_frames_per_candidate": 500,
                "missing_raw_video_policy": "keep-original",
                "candidate_failure_policy": "keep-original",
                "coarse_decode_strategy": "sequential",
            },
        )()
    )
    manifest_stub = type(
        "ManifestStub", (), {"fingerprint": "fp", "schema_version": 1, "videos": {}}
    )()

    runtime = OperationalKISRuntime(
        config=sess_config,
        manifest_path=tmp_path / "m.json",
        manifest=manifest_stub,
        registry=FakeRegistry(),
        raw_video_registry=RawVideoRegistry({}),
        shared_encoder=FakeEncoder(),
        decoder=OpenCVVideoDecoder(),
    )

    # A. KIS request_id = X -> TRAKE request_id = X => DuplicateRequestIdError
    try:
        q_req_x = QueryRequest(request_id="X", query_id="QKIS", query_vi="test", refine_top_n=0)
        runtime.handle_query(q_req_x)
    except Exception:
        pass
    trake_req_x = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "X",
            "query_id": "QT",
            "events": [{"description": "e0"}],
        })
    )
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_trake_query(trake_req_x)

    # B. QA request_id = Y -> TRAKE request_id = Y => DuplicateRequestIdError
    try:
        qa_req_y = QAQueryRequest(
            request_id="Y",
            query_id="QQA",
            event_description="desc",
            question="quest?",
            refine_top_n=1,
        )
        runtime.handle_qa_query(qa_req_y)
    except Exception:
        pass
    trake_req_y = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "Y",
            "query_id": "QT2",
            "events": [{"description": "e0"}],
        })
    )
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_trake_query(trake_req_y)

    # C. TRAKE request_id = Z -> Health request_id = Z => DuplicateRequestIdError
    trake_req_z = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "Z",
            "query_id": "QT3",
            "events": [{"description": "e0"}],
        })
    )
    runtime.handle_trake_query(trake_req_z)
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_health(HealthRequest(request_id="Z"))

    # C2. TRAKE request_id = W -> KIS request_id = W => DuplicateRequestIdError
    trake_req_w = parse_session_request(
        json.dumps({
            "type": "trake_query",
            "request_id": "W",
            "query_id": "QT4",
            "events": [{"description": "e0"}],
        })
    )
    runtime.handle_trake_query(trake_req_w)
    with pytest.raises(DuplicateRequestIdError):
        q_req_w = QueryRequest(request_id="W", query_id="QKIS2", query_vi="test", refine_top_n=0)
        runtime.handle_query(q_req_w)
