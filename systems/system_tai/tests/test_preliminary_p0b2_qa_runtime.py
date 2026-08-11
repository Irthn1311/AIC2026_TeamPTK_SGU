import json
from pathlib import Path

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    HealthRequest,
    InvalidRequestError,
    QAQueryRequest,
    QueryRequest,
    parse_session_request,
)
from system_tai.qa.models import QuestionType
from system_tai.qa.runtime import QARuntimePipeline
from system_tai.refinement.engine import QueryRefinementOutcome
from system_tai.refinement.models import (
    RefinedCandidate,
    RefinementConfig,
    RefinementStatus,
)
from system_tai.refinement.video import (
    DecodedFrame,
    DecodeRequest,
    DecodeResult,
    RawVideoRecord,
    RawVideoRegistry,
    VideoProbe,
)


class FakeEncoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.dimension = dim
        self.identifiers = {"device": "cpu", "model": "fake"}
        self.encode_texts_calls: list[list[str]] = []
        self.encode_images_calls: list[int] = []

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.encode_texts_calls.append(list(texts))
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            if "đỏ" in t or "red" in t:
                v[0] = 1.0
            elif "xanh" in t or "blue" in t:
                v[1] = 1.0
            else:
                v[2] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        self.encode_images_calls.append(len(images))
        vecs = []
        for _img in images:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


class FakeRetriever:
    def __init__(self, candidate_frame_id: int = 100) -> None:
        self.candidate_frame_id = candidate_frame_id
        self.search_vector_calls: list[dict] = []

    def search_vector(self, query_id: str, query_vector: np.ndarray, top_k: int) -> dict:
        self.search_vector_calls.append({"query_id": query_id, "top_k": top_k})
        return {
            "query_id": query_id,
            "ranked_candidates": [
                {
                    "rank": 1,
                    "video_id": "V001",
                    "frame_id": self.candidate_frame_id,
                    "score": 0.95,
                    "clip_row": 10,
                    "keyframe_order": 5,
                    "diagnostic_metadata": {"variant_hit_count": 1, "best_individual_rank": 1},
                }
            ],
        }


class FakeWeightedRRF:
    def fuse_rankings(
        self,
        query_id: str,
        variants: tuple,
        rankings: dict,
        output_top_k: int,
        rrf_constant: float = 60.0,
    ):
        return KISResult(
            query_id=query_id,
            ranked_candidates=(
                CandidateFrame(
                    rank=1,
                    video_id="V001",
                    frame_id=100,
                    score=0.95,
                    clip_row=10,
                    keyframe_order=5,
                    source="rrf",
                    diagnostic_metadata={"variant_hit_count": 1, "best_individual_rank": 1},
                ),
            ),
        )


class FakeRefiner:
    def __init__(
        self,
        refined_frame_id: int = 150,
        status: RefinementStatus = RefinementStatus.REFINED,
    ) -> None:
        self.refined_frame_id = refined_frame_id
        self.status = status

    def refine_query(self, query, config, precomputed_text_embeddings=None):
        res = KISResult(
            query_id=query.query_id,
            ranked_candidates=(
                CandidateFrame(
                    rank=1,
                    video_id="V001",
                    frame_id=self.refined_frame_id,
                    score=0.98,
                    clip_row=10,
                    keyframe_order=5,
                    source="refinement",
                ),
            ),
        )
        is_refined = self.status == RefinementStatus.REFINED
        cand = RefinedCandidate(
            query_id=query.query_id,
            original_candidate_rank=1,
            video_id="V001",
            candidate_frame_id=100,
            refined_frame_id=self.refined_frame_id if is_refined else None,
            candidate_timestamp_seconds=3.33,
            refined_timestamp_seconds=5.0 if is_refined else None,
            fps=30.0,
            total_frame_count=1000,
            window_start_frame=0,
            window_end_frame=300,
            coarse_frame_ids=(100, 120, 150),
            fine_frame_ids=(145, 150, 155),
            coarse_sample_count=3,
            fine_sample_count=3,
            decoded_frame_count=6,
            encoded_image_count=6,
            refinement_fusion_score=0.98 if is_refined else None,
            variant_hit_count=1,
            best_individual_rank=1,
            per_variant_provenance=(),
            decoder_backend="opencv",
            raw_video_path=Path("V001.mp4"),
            status=self.status,
            warnings=(),
            failure_reason=None if is_refined else "Refinement failed",
            original_retrieval_provenance={"fusion_score": 0.95},
            timings={},
        )
        timings = {"candidate_total_seconds": 0.1}
        return QueryRefinementOutcome(query.query_id, res, (cand,), warnings=(), timings=timings)


class FakeRawVideoRegistry:
    def __init__(self, video_ids=("V001",), tmp_dir: Path | None = None) -> None:
        self.records = tuple(
            RawVideoRecord(vid, (tmp_dir / f"{vid}.mp4") if tmp_dir else Path(f"{vid}.mp4"))
            for vid in video_ids
        )
        self.video_ids = set(video_ids)

    def get(self, video_id: str) -> RawVideoRecord:
        for r in self.records:
            if r.video_id == video_id:
                return r
        raise KeyError(f"Video {video_id} not found")


class FakeDecoder:
    backend_identifier = "fake"

    def __init__(self, frame_id_to_return: int = 150, fail_all: bool = False) -> None:
        self.frame_id_to_return = frame_id_to_return
        self.fail_all = fail_all
        self.decode_calls: list[tuple[str, tuple[int, ...]]] = []

    def probe(self, record: RawVideoRecord) -> VideoProbe:
        if self.fail_all:
            raise RuntimeError("Probe failed synthetic test")
        return VideoProbe(
            video_id=record.video_id,
            raw_video_path=record.raw_video_path or Path(f"{record.video_id}.mp4"),
            decoder_backend="fake",
            fps=30.0,
            total_frame_count=1000,
            width=1920,
            height=1080,
            duration_seconds=33.33,
        )

    def decode(self, request: DecodeRequest) -> DecodeResult:
        if self.fail_all:
            raise RuntimeError("Decode failed synthetic test")
        self.decode_calls.append((request.probe.video_id, request.frame_ids))
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = DecodedFrame(
            absolute_frame_id=self.frame_id_to_return,
            timestamp_seconds=5.0,
            image=img,
        )
        return DecodeResult(
            frames=(frame,),
            decoded_frame_count=1,
            video_open_seconds=0.01,
            decode_seconds=0.01,
            decoder_backend="fake",
            warnings=(),
        )


class FakeVideoStore:
    def contains_frame(self, frame_id: int) -> bool:
        return True


class FakeRegistry:
    embedding_dimension = 4
    total_rows = 100

    def get(self, video_id: str):
        return FakeVideoStore()


# Test A & D: Parser behavior
def test_qa_query_parser_valid_and_existing_query_unchanged():
    line_qa = (
        '{"type": "qa_query", "request_id": "r1", "query_id": "q1", '
        '"event_description": "Xe chạy", "question": "Chiếc xe màu gì?"}'
    )
    req_qa = parse_session_request(line_qa)
    assert isinstance(req_qa, QAQueryRequest)
    assert req_qa.request_id == "r1"
    assert req_qa.query_id == "q1"
    assert req_qa.event_description == "Xe chạy"
    assert req_qa.question == "Chiếc xe màu gì?"

    line_kis = '{"type": "query", "request_id": "r2", "query_id": "q2", "query_vi": "Xe chạy"}'
    req_kis = parse_session_request(line_kis)
    assert isinstance(req_kis, QueryRequest)
    assert req_kis.request_id == "r2"


# Test B: Optional EN fields validation
def test_qa_query_optional_fields_validation():
    for val in ["", "   "]:
        payload = {
            "type": "qa_query",
            "request_id": "r1",
            "query_id": "q1",
            "event_description": "Xe chạy",
            "question": "Màu gì?",
            "event_description_en": val,
        }
        with pytest.raises(InvalidRequestError, match="event_description_en"):
            parse_session_request(json.dumps(payload))


# Test C & Requirement 3: Strict integer validation from raw JSON
def test_qa_query_raw_json_strict_integer_validation():
    base = {
        "type": "qa_query",
        "request_id": "r1",
        "query_id": "q1",
        "event_description": "Xe",
        "question": "Màu gì?",
    }
    invalid_params = [
        {"top_k_per_variant": True},
        {"output_top_k": True},
        {"refine_top_n": True},
        {"top_k_per_variant": 1.5},
        {"output_top_k": 2.5},
        {"refine_top_n": 1.9},
        {"top_k_per_variant": "100"},
        {"output_top_k": "10"},
        {"refine_top_n": "3"},
    ]
    for param in invalid_params:
        payload = {**base, **param}
        with pytest.raises(InvalidRequestError, match="strict integers"):
            parse_session_request(json.dumps(payload))

    # Prove valid integer JSON parses correctly
    valid_payload = {
        **base,
        "top_k_per_variant": 50,
        "output_top_k": 20,
        "refine_top_n": 5,
    }
    req = parse_session_request(json.dumps(valid_payload))
    assert isinstance(req, QAQueryRequest)
    assert req.top_k_per_variant == 50
    assert req.output_top_k == 20
    assert req.refine_top_n == 5


# Test E: Unsupported QA returns zero predictions without calling dependencies
def test_unsupported_qa_short_circuits():
    encoder = FakeEncoder()
    retriever = FakeRetriever()
    refiner = FakeRefiner()
    decoder = FakeDecoder()
    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=FakeWeightedRRF(),
        refiner=refiner,
        raw_video_registry=FakeRawVideoRegistry(),
        decoder=decoder,
        shared_encoder=encoder,
    )
    req = QAQueryRequest("r1", "q1", "Mô tả", "Sau đó người phụ nữ làm gì?")
    res, timings, diags = pipeline.process_qa_query(req, RefinementConfig())

    assert res.question_type == QuestionType.UNSUPPORTED
    assert res.predictions == []
    assert len(encoder.encode_texts_calls) == 0
    assert len(encoder.encode_images_calls) == 0
    assert len(retriever.search_vector_calls) == 0
    assert len(decoder.decode_calls) == 0


# Test F & Requirement 6: Question excluded from retrieval vector encoding
def test_question_text_excluded_from_retrieval_text_encode(tmp_path: Path):
    encoder = FakeEncoder()
    retriever = FakeRetriever()
    refiner = FakeRefiner()
    dummy_file = tmp_path / "V001.mp4"
    dummy_file.touch()
    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=FakeWeightedRRF(),
        refiner=refiner,
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=FakeDecoder(),
        shared_encoder=encoder,
    )
    req = QAQueryRequest(
        "r1",
        "q1",
        event_description="Chiếc xe dừng cạnh đường.",
        question="Chiếc xe có màu đỏ không?",
    )
    pipeline.process_qa_query(req, RefinementConfig())

    assert len(encoder.encode_texts_calls) >= 2
    # Call 0 is event retrieval text encode
    event_encode_call = encoder.encode_texts_calls[0]
    assert event_encode_call == ["Chiếc xe dừng cạnh đường."]
    assert "đỏ" not in event_encode_call[0]
    assert "Chiếc xe có màu đỏ không?" not in event_encode_call

    # Call 1 is prompt text encode (separate log)
    prompt_encode_call = encoder.encode_texts_calls[1]
    assert prompt_encode_call != event_encode_call


# Test G: Optional EN event description produces second event variant
def test_optional_en_event_description_variants():
    req = QAQueryRequest(
        "r1",
        "q1",
        event_description="Chiếc xe dừng.",
        question="Chiếc xe màu gì?",
        event_description_en="The car stopped.",
    )
    vars = req.variants()
    assert len(vars) == 2
    assert vars[0].text == "Chiếc xe dừng."
    assert vars[1].text == "The car stopped."


# Test H, I, J, N, O & Requirement 5: Single shared encoder instance
def test_synthetic_color_vertical_slice_and_shared_encoder(tmp_path: Path):
    dummy_file = tmp_path / "V001.mp4"
    dummy_file.touch()
    encoder = FakeEncoder()
    retriever = FakeRetriever(candidate_frame_id=100)
    refiner = FakeRefiner(refined_frame_id=150)
    decoder = FakeDecoder(frame_id_to_return=150)

    pipeline = QARuntimePipeline(
        exact_retriever=retriever,
        weighted_rrf=FakeWeightedRRF(),
        refiner=refiner,
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=decoder,
        shared_encoder=encoder,
    )
    assert pipeline.shared_encoder is encoder

    req = QAQueryRequest("r1", "q1", "Chiếc xe chạy trên đường", "Chiếc xe có màu gì?")
    res, timings, diags = pipeline.process_qa_query(req, RefinementConfig())

    assert len(res.predictions) == 1
    pred = res.predictions[0]
    assert pred.answer == "đỏ"

    assert pred.query_id == "q1"
    assert pred.rank == 1
    assert pred.video_id == "V001"
    assert pred.frame_id == 150  # Refined frame ID

    assert len(decoder.decode_calls) == 1
    assert decoder.decode_calls[0] == ("V001", (150,))

    assert len(encoder.encode_texts_calls) == 2
    assert len(encoder.encode_images_calls) == 1


# Test K: Decoded absolute frame mismatch skips evidence
def test_decoded_frame_mismatch_skips_evidence(tmp_path: Path):
    dummy_file = tmp_path / "V001.mp4"
    dummy_file.touch()
    decoder_mismatched = FakeDecoder(frame_id_to_return=999)
    pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(refined_frame_id=150),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=decoder_mismatched,
        shared_encoder=FakeEncoder(),
    )
    req = QAQueryRequest("r1", "q1", "Xe chạy", "Chiếc xe có màu gì?")
    res, timings, diags = pipeline.process_qa_query(req, RefinementConfig())

    assert res.predictions == []
    assert any("Frame ID mismatch" in w for w in diags["warnings"])


# Test M: Prompt embedding cache prevents re-encoding
def test_prompt_embedding_cache():
    encoder = FakeEncoder()
    pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(),
        raw_video_registry=FakeRawVideoRegistry(),
        decoder=FakeDecoder(),
        shared_encoder=encoder,
    )
    prompts = ["red car", "blue car"]
    _, t1 = pipeline.get_prompt_embeddings(prompts)
    assert len(encoder.encode_texts_calls) == 1

    _, t2 = pipeline.get_prompt_embeddings(prompts)
    assert len(encoder.encode_texts_calls) == 1


# Test P: Failed refinement candidate is skipped
def test_failed_refinement_candidate_skipped():
    refiner_failed = FakeRefiner(status=RefinementStatus.FAILED)
    pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=refiner_failed,
        raw_video_registry=FakeRawVideoRegistry(),
        decoder=FakeDecoder(),
        shared_encoder=FakeEncoder(),
    )
    req = QAQueryRequest("r1", "q1", "Xe chạy", "Chiếc xe có màu gì?")
    res, timings, diags = pipeline.process_qa_query(req, RefinementConfig())

    assert res.predictions == []
    assert any("refinement failed" in w for w in diags["warnings"])


# Requirement 9: All-evidence-failure zero-output test
def test_all_evidence_failure_zero_output(tmp_path: Path):
    dummy_file = tmp_path / "V001.mp4"
    dummy_file.touch()
    decoder_failing = FakeDecoder(fail_all=True)
    pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=decoder_failing,
        shared_encoder=FakeEncoder(),
    )
    req = QAQueryRequest("r1", "q1", "Xe chạy", "Chiếc xe có màu gì?")
    res, timings, diags = pipeline.process_qa_query(req, RefinementConfig())

    assert res.predictions == []
    assert any("Decode exception" in w for w in diags["warnings"])


# Requirement 1: One global session request-id namespace across all request types
def test_one_global_session_request_id_namespace_cross_type(tmp_path: Path):
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.kis.session_schema import SessionConfig

    config = SessionConfig(output_root=tmp_path / "out")
    manifest_stub = type(
        "CorpusManifestStub", (), {"fingerprint": "fp123", "schema_version": 1, "videos": {}}
    )()
    runtime = OperationalKISRuntime(
        config=config,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest_stub,
        registry=FakeRegistry(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        shared_encoder=FakeEncoder(),
        decoder=FakeDecoder(),
    )
    runtime.exact_retriever = FakeRetriever()
    runtime.weighted_rrf = FakeWeightedRRF()
    runtime.refiner = FakeRefiner()
    runtime.qa_pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=FakeDecoder(),
        shared_encoder=FakeEncoder(),
    )

    # 1. KIS r1 then QA r1 -> DuplicateRequestIdError
    runtime.handle_query(QueryRequest("r1", "q1", "Xe chạy", refine_top_n=0))
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_qa_query(QAQueryRequest("r1", "q1", "Xe", "Màu gì?"))

    # 2. QA r2 then KIS r2 -> DuplicateRequestIdError
    runtime.handle_qa_query(QAQueryRequest("r2", "q2", "Xe", "Màu gì?"))
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_query(QueryRequest("r2", "q2", "Xe chạy", refine_top_n=0))

    # 3. QA r3 then QA r3 -> DuplicateRequestIdError
    runtime.handle_qa_query(QAQueryRequest("r3", "q3", "Xe", "Màu gì?"))
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_qa_query(QAQueryRequest("r3", "q3", "Xe", "Màu gì?"))

    # 4. Health r4 then QA r4 -> DuplicateRequestIdError
    runtime.handle_health(HealthRequest("r4"))
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_qa_query(QAQueryRequest("r4", "q4", "Xe", "Màu gì?"))


# Requirement 2 & 8: Directory keyed by request_id and complete request artifacts
def test_qa_request_directory_and_artifacts(tmp_path: Path):
    from system_tai.kis.session_engine import (
        OperationalKISRuntime,
        safe_request_directory_name,
    )
    from system_tai.kis.session_schema import SessionConfig

    config = SessionConfig(output_root=tmp_path / "out")
    manifest_stub = type(
        "CorpusManifestStub", (), {"fingerprint": "fp123", "schema_version": 1, "videos": {}}
    )()
    shared_enc = FakeEncoder()
    runtime = OperationalKISRuntime(
        config=config,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest_stub,
        registry=FakeRegistry(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        shared_encoder=shared_enc,
        decoder=FakeDecoder(),
    )
    runtime.qa_pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=FakeDecoder(),
        shared_encoder=shared_enc,
    )

    # 3. Assert single encoder object identity across runtime and qa_pipeline
    assert runtime.qa_pipeline.shared_encoder is runtime.shared_encoder

    dummy_file = tmp_path / "V001.mp4"
    dummy_file.touch()

    req1 = QAQueryRequest("qa-run-001", "q42", "Chiếc xe chạy", "Chiếc xe màu gì?")
    req2 = QAQueryRequest("qa-run-002", "q42", "Chiếc xe chạy", "Chiếc xe màu gì?")

    resp1 = runtime.handle_qa_query(req1)
    resp2 = runtime.handle_qa_query(req2)

    assert resp1["status"] == "SUCCESS"
    assert resp2["status"] == "SUCCESS"

    dir1 = tmp_path / "out" / "requests" / safe_request_directory_name("qa-run-001")
    dir2 = tmp_path / "out" / "requests" / safe_request_directory_name("qa-run-002")
    assert dir1.is_dir()
    assert dir2.is_dir()

    assert (dir1 / "qa_predictions.jsonl").is_file()
    assert (dir1 / "qa_evidence.json").is_file()
    assert (dir1 / "qa_request_manifest.json").is_file()
    assert (dir1 / "qa_timings.json").is_file()

    pred_lines = (dir1 / "qa_predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(pred_lines) == 1
    pred_obj = json.loads(pred_lines[0])
    assert set(pred_obj.keys()) == {"query_id", "rank", "video_id", "frame_id", "answer"}

    # Manifest verification
    manifest_obj = json.loads((dir1 / "qa_request_manifest.json").read_text(encoding="utf-8"))
    assert "event_variants" in manifest_obj
    assert len(manifest_obj["event_variants"]) > 0
    for v in manifest_obj["event_variants"]:
        assert "Chiếc xe màu gì?" not in v["text"]
        assert "đỏ" not in v["text"]
        assert "Chiếc xe chạy" in v["text"]

    assert "artifacts" in manifest_obj
    artifacts = manifest_obj["artifacts"]
    assert set(artifacts.keys()) == {
        "qa_predictions_jsonl",
        "qa_evidence_json",
        "qa_request_manifest",
        "qa_timings",
    }
    for rel_p in artifacts.values():
        assert (tmp_path / "out" / rel_p).is_file()

    # Evidence audit verification
    ev_obj = json.loads((dir1 / "qa_evidence.json").read_text(encoding="utf-8"))
    assert ev_obj["question_supported"] is True
    assert ev_obj["fused_retrieval_candidates"] == [
        {"rank": 1, "video_id": "V001", "frame_id": 100}
    ]
    assert ev_obj["refined_candidates"] == [
        {
            "original_rank": 1,
            "video_id": "V001",
            "candidate_frame_id": 100,
            "refined_frame_id": 150,
            "status": "REFINED",
        }
    ]
    assert ev_obj["usable_evidence_candidates"] == [
        {"rank": 1, "video_id": "V001", "frame_id": 150}
    ]
    assert "evidence" in ev_obj
    records = ev_obj["evidence"]
    assert len(records) == 1
    rec = records[0]
    assert rec["candidate_frame_id"] == 100
    assert rec["refined_frame_id"] == 150
    assert rec["output_frame_id"] == 150
    assert rec["candidate_frame_id"] != rec["refined_frame_id"]
    assert rec["output_frame_id"] == rec["refined_frame_id"]
    assert rec["answer"] == "đỏ"
    assert isinstance(rec["answer_score"], float)
    assert rec["skip_reason"] is None
    assert ev_obj["final_predictions"] == [
        {
            "rank": 1,
            "video_id": "V001",
            "frame_id": 150,
            "answer": "đỏ",
        }
    ]

    timings_obj = json.loads((dir1 / "qa_timings.json").read_text(encoding="utf-8"))
    expected_timing_keys = {
        "total_seconds",
        "text_encode_seconds",
        "retrieval_seconds",
        "fusion_seconds",
        "refinement_seconds",
        "evidence_decode_seconds",
        "evidence_encode_seconds",
        "answer_scoring_seconds",
        "prompt_encode_seconds",
    }
    assert expected_timing_keys.issubset(set(timings_obj.keys()))


def test_failed_refinement_evidence_skip_reason(tmp_path: Path):
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.kis.session_schema import SessionConfig

    config = SessionConfig(output_root=tmp_path / "out")
    manifest_stub = type(
        "CorpusManifestStub", (), {"fingerprint": "fp123", "schema_version": 1, "videos": {}}
    )()
    shared_enc = FakeEncoder()
    runtime = OperationalKISRuntime(
        config=config,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest_stub,
        registry=FakeRegistry(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        shared_encoder=shared_enc,
        decoder=FakeDecoder(),
    )
    runtime.qa_pipeline = QARuntimePipeline(
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(status=RefinementStatus.FAILED),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        decoder=FakeDecoder(),
        shared_encoder=shared_enc,
    )

    req = QAQueryRequest("qa-fail-001", "q99", "Chiếc xe chạy", "Chiếc xe màu gì?")
    resp = runtime.handle_qa_query(req)
    assert resp["status"] == "SUCCESS"
    assert resp["prediction_count"] == 0

    ev_path = tmp_path / "out" / resp["artifacts"]["qa_evidence_json"]
    ev_obj = json.loads(ev_path.read_text(encoding="utf-8"))
    assert "evidence" in ev_obj
    assert len(ev_obj["evidence"]) == 1
    failed_rec = ev_obj["evidence"][0]
    assert failed_rec["rank"] == 1
    assert failed_rec["refined_frame_id"] is None
    assert failed_rec["output_frame_id"] is None
    assert failed_rec["answer"] is None
    assert failed_rec["skip_reason"] == "refinement_failed"


def test_production_runtime_shared_encoder_identity(tmp_path: Path):
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.kis.session_schema import SessionConfig

    config = SessionConfig(output_root=tmp_path / "out")
    manifest_stub = type(
        "CorpusManifestStub", (), {"fingerprint": "fp123", "schema_version": 1, "videos": {}}
    )()
    shared_enc = FakeEncoder()
    runtime = OperationalKISRuntime(
        config=config,
        manifest_path=tmp_path / "manifest.json",
        manifest=manifest_stub,
        registry=FakeRegistry(),
        raw_video_registry=FakeRawVideoRegistry(tmp_dir=tmp_path),
        shared_encoder=shared_enc,
        decoder=FakeDecoder(),
    )

    # Assert default construction shares encoder
    assert runtime.qa_pipeline.shared_encoder is runtime.shared_encoder

    # Plug mocks except shared_encoder
    runtime.qa_pipeline.exact_retriever = FakeRetriever()
    runtime.qa_pipeline.weighted_rrf = FakeWeightedRRF()
    runtime.qa_pipeline.refiner = FakeRefiner()
    runtime.qa_pipeline.raw_video_registry = FakeRawVideoRegistry(tmp_dir=tmp_path)
    runtime.qa_pipeline.decoder = FakeDecoder()

    dummy_file = tmp_path / "V001.mp4"
    dummy_file.touch()

    req = QAQueryRequest("qa-shared-enc", "q100", "Chiếc xe chạy", "Chiếc xe màu gì?")
    runtime.handle_qa_query(req)

    # Prove SAME encoder handles text event encoding, prompt encoding, and image encoding
    assert len(shared_enc.encode_texts_calls) == 2
    assert shared_enc.encode_texts_calls[0] == ["Chiếc xe chạy"]
    assert "red" in shared_enc.encode_texts_calls[1]
    assert len(shared_enc.encode_images_calls) == 1


def test_real_raw_video_registry_compatibility(tmp_path: Path) -> None:
    """Verify QARuntimePipeline is fully compatible with real RawVideoRegistry class."""
    video_file = tmp_path / "V001.mp4"
    video_file.touch()

    real_registry = RawVideoRegistry([RawVideoRecord("V001", video_file)])
    pipeline = QARuntimePipeline(
        raw_video_registry=real_registry,
        shared_encoder=FakeEncoder(),
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(refined_frame_id=150),
        decoder=FakeDecoder(frame_id_to_return=150),
    )

    req = QAQueryRequest("req-real-reg", "q-real-1", "Chiếc xe chạy", "Chiếc xe màu gì?")
    res, timings, diags = pipeline.process_qa_query(req)

    assert res.question_type == QuestionType.COLOR
    assert len(res.predictions) == 1
    assert res.predictions[0].answer == "đỏ"
    assert res.predictions[0].frame_id == 150
    assert res.predictions[0].video_id == "V001"
    assert diags["evidence_candidate_count"] == 1
    assert diags["decoded_frame_count"] == 1


def test_question_en_fallback_classification_and_retrieval_exclusion(tmp_path: Path) -> None:
    """Verify question_en propagates for classification and is excluded from retrieval."""
    video_file = tmp_path / "V001.mp4"
    video_file.touch()

    shared_enc = FakeEncoder()
    pipeline = QARuntimePipeline(
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V001", video_file)]),
        shared_encoder=shared_enc,
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(refined_frame_id=150),
        decoder=FakeDecoder(frame_id_to_return=150),
    )

    # Question in VI alone is unsupported ("Nói gì đó"); question_en is "What color is the car?"
    req = QAQueryRequest(
        request_id="req-q-en-fallback",
        query_id="q-fallback-1",
        event_description="Chiếc xe chạy trên đường",
        question="Nói gì đó",
        event_description_en="A car driving on the road",
        question_en="What color is the car?",
    )

    res, timings, diags = pipeline.process_qa_query(req)

    assert res.question_type == QuestionType.COLOR
    assert len(res.predictions) == 1
    assert res.predictions[0].answer == "đỏ"

    # Verify English QUESTION was NOT passed to event retrieval encoder calls
    assert len(shared_enc.encode_texts_calls) >= 1
    retrieval_encoded_texts = shared_enc.encode_texts_calls[0]
    assert "What color is the car?" not in retrieval_encoded_texts
    assert "Chiếc xe chạy trên đường" in retrieval_encoded_texts


def test_fail_closed_p0a_validation_error(tmp_path: Path) -> None:
    """Verify QARuntimePipeline raises ValueError if QA engine outputs invalid predictions."""
    class BadQAEngine:
        def answer(
            self, query, evidence_candidates, image_embeddings=None, prompt_embeddings=None
        ):
            from system_tai.preliminary.schemas import QAPrediction
            from system_tai.qa.models import QAResult
            # Return prediction with query_id mismatch to trigger P0-A validation error
            bad_pred = QAPrediction(
                query_id="WRONG_QUERY_ID",
                rank=1,
                video_id="V001",
                frame_id=150,
                answer="đỏ",
            )
            return QAResult(
                query_id=query.query_id,
                question_type=QuestionType.COLOR,
                predictions=[bad_pred],
                warnings=[],
            )

    video_file = tmp_path / "V001.mp4"
    video_file.touch()

    pipeline = QARuntimePipeline(
        raw_video_registry=RawVideoRegistry([RawVideoRecord("V001", video_file)]),
        shared_encoder=FakeEncoder(),
        exact_retriever=FakeRetriever(),
        weighted_rrf=FakeWeightedRRF(),
        refiner=FakeRefiner(refined_frame_id=150),
        decoder=FakeDecoder(frame_id_to_return=150),
        qa_engine=BadQAEngine(),
    )

    req = QAQueryRequest("req-bad-engine", "q-bad", "Chiếc xe chạy", "Chiếc xe màu gì?")
    with pytest.raises(ValueError, match="QA prediction validation failed"):
        pipeline.process_qa_query(req)
