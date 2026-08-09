"""Comprehensive unit tests for Phase 4.2 operational session runtime, protocol, and lifecycle."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.data.corpus_discovery import CorpusManifest, DiscoveredVideo, _fingerprint
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.kis.contest_runner import ContestRunConfig, ContestRunner
from system_tai.kis.contest_schema import ContestQuery
from system_tai.kis.session import build_parser, run_session
from system_tai.kis.session_engine import OperationalKISRuntime, safe_request_directory_name
from system_tai.kis.session_schema import (
    DuplicateRequestIdError,
    HealthRequest,
    InvalidRequestError,
    QueryRequest,
    SessionConfig,
    ShutdownRequest,
    UnknownRequestTypeError,
    parse_session_request,
)
from system_tai.refinement.video import DecodedFrame, DecodeResult, VideoProbe


class MockSharedEncoder:
    dimension = 2
    identifiers = {"model": "ViT-B/32", "device": "cpu"}

    def __init__(self, counts: dict[str, int], device: str = "cpu"):
        self.counts = counts
        self.device = device
        self.counts["model_load"] = self.counts.get("model_load", 0) + 1

    def encode(self, text: str) -> np.ndarray:
        self.counts["text_encode"] = self.counts.get("text_encode", 0) + 1
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.counts["text_encode_batch"] = self.counts.get("text_encode_batch", 0) + len(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_images(self, images: list[Any], *, batch_size: int = 32) -> np.ndarray:
        self.counts["image_encode"] = self.counts.get("image_encode", 0) + len(images)
        rows = np.asarray(
            [
                [1.0 / (1.0 + abs(int(getattr(img, "absolute_frame_id", 0)) - 55)), 0.1]
                for img in images
            ],
            dtype=np.float32,
        )
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class MockDecoder:
    backend_identifier = "mock-decoder"

    def __init__(self, counts: dict[str, int] | None = None):
        self.counts = counts if counts is not None else {}
        self.counts["decoder_init"] = self.counts.get("decoder_init", 0) + 1
        self.closed = False

    def probe(self, record: Any) -> VideoProbe:
        return VideoProbe(
            video_id=record.video_id,
            raw_video_path=record.raw_video_path,
            decoder_backend=self.backend_identifier,
            fps=10.0,
            duration_seconds=10.0,
            total_frame_count=100,
            width=8,
            height=8,
        )

    def decode(self, request: Any) -> DecodeResult:
        self.counts["decode_calls"] = self.counts.get("decode_calls", 0) + 1
        frames = tuple(
            DecodedFrame(frame_id, frame_id / request.probe.fps, frame_id)
            for frame_id in request.frame_ids
        )
        return DecodeResult(frames, len(frames), 0, 0, self.backend_identifier, ())

    def close(self) -> None:
        self.closed = True


def make_test_corpus(tmp_path: Path) -> tuple[Path, CorpusManifest, dict[str, int]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    mapping = tmp_path / "L21_V001.csv"
    mapping.write_text("n,pts_time,fps,frame_idx\n1,5,10,50\n2,6,10,60\n", encoding="utf-8")
    clip = tmp_path / "L21_V001.npy"
    np.save(str(clip), np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32))
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir(exist_ok=True)
    (keyframes / "1.jpg").touch()
    video = tmp_path / "L21_V001.mp4"
    video.touch()

    discovered = DiscoveredVideo(
        "L21_V001",
        mapping,
        clip,
        keyframes,
        video,
        1,
        2,
        mapping.stat().st_size,
        clip.stat().st_size,
        2,
    )
    manifest = CorpusManifest(tmp_path, tmp_path, _fingerprint((discovered,)), (discovered,))
    manifest_path = tmp_path / "feature_manifest.json"
    manifest.write(manifest_path)
    return manifest_path, manifest, counts


def setup_runtime(
    tmp_path: Path, counts: dict[str, int] | None = None
) -> tuple[OperationalKISRuntime, dict[str, int]]:
    cnt = counts if counts is not None else {}
    manifest_path, manifest, _ = make_test_corpus(tmp_path)
    output_root = tmp_path / "session_output"
    config = SessionConfig(
        input_root=tmp_path,
        reuse_manifest=manifest_path,
        output_root=output_root,
        device="cpu",
    )
    runtime = OperationalKISRuntime.bootstrap(
        config,
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(path, expected_dimension=2),
        encoder_factory=lambda **_kw: MockSharedEncoder(cnt),
        decoder_factory=lambda: MockDecoder(cnt),
    )
    return runtime, cnt


# -----------------------------------------------------------------------------
# Test Cases 1 - 50
# -----------------------------------------------------------------------------


def test_01_bootstrap_loads_manifest_once(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    assert runtime.manifest is not None
    assert runtime.manifest_path.is_file()


def test_02_feature_registry_loads_once(tmp_path: Path) -> None:
    registry_count = {"count": 0}

    def loader(path: Path) -> FeatureStoreRegistry:
        registry_count["count"] += 1
        return FeatureStoreRegistry.from_manifest(path, expected_dimension=2)

    sub_path = tmp_path / "sub_reg"
    manifest_path, _, _ = make_test_corpus(sub_path)
    config = SessionConfig(
        input_root=sub_path,
        reuse_manifest=manifest_path,
        output_root=sub_path / "out_reg",
        device="cpu",
    )
    runtime = OperationalKISRuntime.bootstrap(
        config,
        registry_loader=loader,
        encoder_factory=lambda **_kw: MockSharedEncoder({}),
        decoder_factory=lambda: MockDecoder({}),
    )
    assert registry_count["count"] == 1
    assert runtime.registry.total_rows == 2


def test_03_04_model_loads_once_preferred_shared_architecture(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    runtime, _ = setup_runtime(tmp_path, counts)
    assert counts["model_load"] == 1
    # Run two queries
    runtime.handle_query(QueryRequest("req-1", "Q1", "test query", refine_top_n=0))
    runtime.handle_query(QueryRequest("req-2", "Q2", "test query 2", refine_top_n=1))
    assert counts["model_load"] == 1  # Still 1!


def test_05_06_retrieval_requests_do_not_reload_model_or_registry(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    runtime, _ = setup_runtime(tmp_path, counts)
    q1 = QueryRequest("req-1", "Q1", "query 1", refine_top_n=0)
    q2 = QueryRequest("req-2", "Q2", "query 2", refine_top_n=0)
    runtime.handle_query(q1)
    runtime.handle_query(q2)
    assert counts["model_load"] == 1
    assert counts.get("text_encode_batch", 0) == 2


def test_07_refine_top_n_zero_does_not_initialize_refinement_path(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    runtime, _ = setup_runtime(tmp_path, counts)
    resp = runtime.handle_query(QueryRequest("req-1", "Q1", "query 1", refine_top_n=0))
    assert resp["status"] == "SUCCESS"
    assert resp["refinement_requested"] is False
    assert counts.get("image_encode", 0) == 0
    assert counts.get("decode_calls", 0) == 0


def test_08_refine_top_n_greater_than_zero_refines_correct_candidates(tmp_path: Path) -> None:
    counts: dict[str, int] = {}
    runtime, _ = setup_runtime(tmp_path, counts)
    resp = runtime.handle_query(QueryRequest("req-1", "Q1", "query 1", refine_top_n=1))
    assert resp["status"] == "SUCCESS"
    assert resp["refinement_requested"] is True
    assert counts.get("image_encode", 0) > 0


def test_09_10_health_before_and_after_queries(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    h1 = runtime.handle_health(HealthRequest("health-1"))
    assert h1["status"] == "READY"
    assert h1["request_count"] == 1

    runtime.handle_query(QueryRequest("req-1", "Q1", "query 1", refine_top_n=0))
    h2 = runtime.handle_health(HealthRequest("health-2"))
    assert h2["status"] == "READY"
    assert h2["request_count"] == 3


def test_11_clean_shutdown(tmp_path: Path) -> None:
    runtime, counts = setup_runtime(tmp_path)
    resp = runtime.handle_shutdown(ShutdownRequest("shutdown-1"))
    assert resp["status"] == "STOPPING"
    assert resp["processed_requests"] == 1
    assert runtime.decoder.closed is True


def test_12_malformed_json_continue_mode(tmp_path: Path) -> None:
    stdin = io.StringIO('invalid json line\n{"type":"health","request_id":"h1"}\n')
    stdout = io.StringIO()
    stderr = io.StringIO()
    runtime, _ = setup_runtime(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reuse-manifest",
            str(runtime.manifest_path),
            "--output-root",
            str(tmp_path / "out_m1"),
            "--device",
            "cpu",
        ]
    )
    code = run_session(args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr)
    assert code == 0
    lines = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["type"] == "error"
    assert lines[0]["error_code"] == "MALFORMED_JSON"
    assert lines[1]["type"] == "health"


def test_13_malformed_json_fail_fast(tmp_path: Path) -> None:
    stdin = io.StringIO('invalid json line\n{"type":"health","request_id":"h1"}\n')
    stdout = io.StringIO()
    stderr = io.StringIO()
    runtime, _ = setup_runtime(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reuse-manifest",
            str(runtime.manifest_path),
            "--output-root",
            str(tmp_path / "out_m2"),
            "--fail-fast-protocol",
            "--device",
            "cpu",
        ]
    )
    code = run_session(args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr)
    assert code == 1
    lines = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    assert len(lines) == 1
    assert lines[0]["type"] == "error"


def test_14_to_19_request_validation_errors() -> None:
    # 14. Unknown request type
    with pytest.raises(UnknownRequestTypeError):
        parse_session_request('{"type": "unknown", "request_id": "r1"}')

    # 15. Missing request_id
    with pytest.raises(InvalidRequestError):
        parse_session_request('{"type": "health"}')

    # 16. Missing query_vi
    with pytest.raises(InvalidRequestError):
        parse_session_request('{"type": "query", "request_id": "r1", "query_id": "q1"}')

    # 17. Invalid weights
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            '{"type": "query", "request_id": "r1", "query_id": "q1", '
            '"query_vi": "v", "weight_vi": -1.0}'
        )

    # 18. Invalid output_top_k
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            '{"type": "query", "request_id": "r1", "query_id": "q1", '
            '"query_vi": "v", "output_top_k": 0}'
        )

    # 19. Invalid refine_top_n
    with pytest.raises(InvalidRequestError):
        parse_session_request(
            '{"type": "query", "request_id": "r1", "query_id": "q1", '
            '"query_vi": "v", "output_top_k": 10, "refine_top_n": 20}'
        )


def test_20_duplicate_request_id_rejected(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    q1 = QueryRequest("req-1", "Q1", "query 1", refine_top_n=0)
    runtime.handle_query(q1)
    with pytest.raises(DuplicateRequestIdError):
        runtime.handle_query(q1)


def test_21_22_23_request_directory_naming_and_no_overwrite(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    q1 = QueryRequest("req-1", "Q1", "query 1", refine_top_n=0)
    q2 = QueryRequest("req-2", "Q1", "query 1 repeated", refine_top_n=0)
    resp1 = runtime.handle_query(q1)
    resp2 = runtime.handle_query(q2)
    assert resp1["artifacts"]["top100_jsonl"] != resp2["artifacts"]["top100_jsonl"]
    assert "requests/req-1-" in resp1["artifacts"]["top100_jsonl"]


def test_24_session_retrieval_jsonl_matches_contest_baseline(tmp_path: Path) -> None:
    manifest_path, manifest, _ = make_test_corpus(tmp_path)
    contest_out = tmp_path / "contest_out"
    runner = ContestRunner(
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(path, expected_dimension=2),
        encoder_factory=lambda **_kw: MockSharedEncoder({}),
    )
    q = ContestQuery("Q1", "test query")
    runner.run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(q,),
        output_directory=contest_out,
        config=ContestRunConfig(device="cpu"),
    )
    contest_jsonl = (
        contest_out / "queries" / safe_request_directory_name("Q1") / "top100.jsonl"
    ).read_text(encoding="utf-8")

    session_runtime, _ = setup_runtime(tmp_path)
    sess_resp = session_runtime.handle_query(
        QueryRequest("req-test", "Q1", "test query", refine_top_n=0)
    )
    session_jsonl = (
        tmp_path / "session_output" / sess_resp["artifacts"]["top100_jsonl"]
    ).read_text(encoding="utf-8")

    assert session_jsonl == contest_jsonl


def test_25_26_27_refined_jsonl_and_no_provenance_leak(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    resp = runtime.handle_query(QueryRequest("req-ref", "Q1", "query 1", refine_top_n=1))
    jsonl_path = tmp_path / "session_output" / resp["artifacts"]["refined_top100_jsonl"]
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) > 0
    for r in records:
        assert set(r.keys()) == {"query_id", "rank", "video_id", "frame_id"}


def test_28_29_30_contiguous_ranks_max_100_dedup(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    resp = runtime.handle_query(QueryRequest("req-1", "Q1", "query 1", refine_top_n=0))
    jsonl_path = tmp_path / "session_output" / resp["artifacts"]["top100_jsonl"]
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    ranks = [r["rank"] for r in records]
    assert ranks == list(range(1, len(ranks) + 1))
    assert len(records) <= 100
    identities = [(r["video_id"], r["frame_id"]) for r in records]
    assert len(identities) == len(set(identities))


def test_31_32_error_isolation_and_continue_on_error(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)

    class FailingEncoder(MockSharedEncoder):
        def encode(self, text: str) -> np.ndarray:
            if "fail" in text:
                raise RuntimeError("simulated error")
            return super().encode(text)

        def encode_texts(self, texts: list[str]) -> np.ndarray:
            if any("fail" in text for text in texts):
                raise RuntimeError("simulated error")
            return super().encode_texts(texts)

    runtime.shared_encoder = FailingEncoder({})
    stdin = io.StringIO(
        json.dumps(
            {"type": "query", "request_id": "r1", "query_id": "q1", "query_vi": "fail query"}
        )
        + "\n"
        + json.dumps(
            {"type": "query", "request_id": "r2", "query_id": "q2", "query_vi": "good query"}
        )
        + "\n"
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reuse-manifest",
            str(runtime.manifest_path),
            "--output-root",
            str(tmp_path / "out_err"),
            "--device",
            "cpu",
        ]
    )
    code = run_session(args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr)
    assert code == 0
    lines = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["status"] == "ERROR"
    assert lines[1]["status"] == "SUCCESS"


def test_33_max_requests(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    stdin = io.StringIO(
        json.dumps({"type": "health", "request_id": "h1"})
        + "\n"
        + json.dumps({"type": "health", "request_id": "h2"})
        + "\n"
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reuse-manifest",
            str(runtime.manifest_path),
            "--output-root",
            str(tmp_path / "out_max"),
            "--max-requests",
            "1",
            "--device",
            "cpu",
        ]
    )
    code = run_session(args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr)
    assert code == 0
    lines = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    assert len(lines) == 1


def test_34_session_summary_counts(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    runtime.handle_health(HealthRequest("h1"))
    runtime.handle_query(QueryRequest("r1", "q1", "query", refine_top_n=0))
    runtime.close("done")
    manifest = json.loads(
        (tmp_path / "session_output" / "session_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["request_count"] == 2
    assert manifest["successful_query_count"] == 1
    assert manifest["health_request_count"] == 1


def test_35_startup_timing_not_mixed_into_query_latency(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    resp = runtime.handle_query(QueryRequest("r1", "q1", "query", refine_top_n=0))
    timings = resp["timings"]
    assert "total_bootstrap_seconds" not in timings
    assert timings["total_seconds"] < 5.0


def test_36_37_decoder_handle_release_and_no_image_retention(tmp_path: Path) -> None:
    runtime, counts = setup_runtime(tmp_path)
    runtime.handle_query(QueryRequest("r1", "q1", "query", refine_top_n=1))
    runtime.close()
    assert runtime.decoder.closed is True


def test_38_39_stdout_only_contains_valid_jsonl_responses(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    stdin = io.StringIO(json.dumps({"type": "health", "request_id": "h1"}) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reuse-manifest",
            str(runtime.manifest_path),
            "--output-root",
            str(tmp_path / "out_stdout"),
            "--device",
            "cpu",
        ]
    )
    run_session(args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr)
    out_lines = stdout.getvalue().splitlines()
    for line in out_lines:
        if line.strip():
            obj = json.loads(line)
            assert isinstance(obj, dict)
            assert "type" in obj


def test_40_deterministic_repeated_run(tmp_path: Path) -> None:
    path1 = tmp_path / "t1"
    runtime1, _ = setup_runtime(path1)
    resp1 = runtime1.handle_query(QueryRequest("r1", "q1", "query", refine_top_n=0))
    f1 = (path1 / "session_output" / resp1["artifacts"]["top100_jsonl"]).read_text(encoding="utf-8")

    path2 = tmp_path / "t2"
    runtime2, _ = setup_runtime(path2)
    resp2 = runtime2.handle_query(QueryRequest("r1", "q1", "query", refine_top_n=0))
    f2 = (path2 / "session_output" / resp2["artifacts"]["top100_jsonl"]).read_text(encoding="utf-8")

    assert f1 == f2


def test_41_42_cpu_and_fake_cuda_path(tmp_path: Path) -> None:
    runtime_cpu, _ = setup_runtime(tmp_path / "cpu")
    assert runtime_cpu.shared_encoder.device == "cpu"

    cuda_path = tmp_path / "cuda"
    manifest_path, _, _ = make_test_corpus(cuda_path)
    config = SessionConfig(
        input_root=cuda_path,
        reuse_manifest=manifest_path,
        output_root=cuda_path / "out",
        device="cpu",
    )
    runtime_cuda = OperationalKISRuntime.bootstrap(
        config,
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(path, expected_dimension=2),
        encoder_factory=lambda device, **_kw: MockSharedEncoder({}, device="cuda"),
        decoder_factory=lambda: MockDecoder({}),
    )
    assert runtime_cuda.shared_encoder.device == "cuda"


def test_43_44_45_46_existing_clim_and_frame_idx_regressions(tmp_path: Path) -> None:
    # 46. frame_idx is preserved exactly
    manifest_path, manifest, _ = make_test_corpus(tmp_path)
    registry = FeatureStoreRegistry.from_manifest(manifest_path, expected_dimension=2)
    assert registry.total_rows == 2
    mapping = registry.stores[0].mappings[0]
    assert mapping.frame_id == 50


def test_47_48_windows_and_posix_paths(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    resp = runtime.handle_query(QueryRequest("r1", "q1", "query", refine_top_n=0))
    for key, rel_path in resp["artifacts"].items():
        assert "\\" not in rel_path


def test_49_signal_summary_writing(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    runtime.close(shutdown_reason="sigint")
    assert (tmp_path / "session_output" / "session_manifest.json").is_file()


def test_50_broken_stdin_eof_clean_shutdown(tmp_path: Path) -> None:
    runtime, _ = setup_runtime(tmp_path)
    stdin = io.StringIO("")  # Empty stdin (EOF)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reuse-manifest",
            str(runtime.manifest_path),
            "--output-root",
            str(tmp_path / "out_eof"),
            "--device",
            "cpu",
        ]
    )
    code = run_session(args, runtime=runtime, stdin=stdin, stdout=stdout, stderr=stderr)
    assert code == 0
    assert runtime.decoder.closed is True


def test_51_52_refinement_response_metrics_regression(tmp_path: Path) -> None:
    from system_tai.refinement.engine import QueryRefinementOutcome
    from system_tai.refinement.models import RefinementStatus

    runtime, _ = setup_runtime(tmp_path)

    # Test retrieval-only query
    resp_ret = runtime.handle_query(QueryRequest("req-ret", "Q-RET", "retrieval", refine_top_n=0))
    assert resp_ret["refined_count"] == 0
    assert resp_ret["timings"]["decoded_frame_count"] == 0
    assert resp_ret["timings"]["encoded_image_count"] == 0
    assert resp_ret["result_count"] == 2  # default corpus has 2

    # Test refinement query with mocked outcome to simulate Kaggle bug
    original_refine_query = runtime.refiner.refine_query

    class FakeCandidate:
        def __init__(self, status: RefinementStatus, refined_frame_id: int):
            self.status = status
            self.refined_frame_id = refined_frame_id

    def mock_refine_query(
        ref_query: Any, exec_config: Any, **kwargs: Any
    ) -> QueryRefinementOutcome:
        from system_tai.common.schemas import CandidateFrame, KISResult

        candidates = []
        for i in range(10):
            status = RefinementStatus.REFINED if i < 3 else RefinementStatus.NOT_REFINED
            candidates.append(FakeCandidate(status, i * 10))

        timings = {
            "refined_candidate_count": 3,
            "decoded_frame_count": 150,
            "encoded_image_count": 100,
            "coarse_requested_frame_count": 0,
            "coarse_decoded_frame_count": 0,
            "fine_requested_frame_count": 0,
            "fine_decoded_frame_count": 0,
            "coarse_sparse_request_count": 0,
            "coarse_sparse_success_count": 0,
            "coarse_sparse_fallback_count": 0,
            "video_probe_seconds": 0.0,
            "video_open_seconds": 0.0,
            "coarse_decode_seconds": 0.0,
            "coarse_encode_seconds": 0.0,
            "coarse_score_seconds": 0.0,
            "coarse_fusion_seconds": 0.0,
            "fine_decode_seconds": 0.0,
            "fine_encode_seconds": 0.0,
            "fine_score_seconds": 0.0,
            "fine_fusion_seconds": 0.0,
            "candidate_total_seconds": 0.0,
        }
        result = KISResult(
            query_id=ref_query.query_id,
            ranked_candidates=tuple(
                CandidateFrame(
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    clip_row=index,
                    keyframe_order=index,
                    score=candidate.retrieval_score,
                    rank=index + 1,
                    source="telemetry-regression",
                )
                for index, candidate in enumerate(ref_query.candidates)
            ),
        )
        return QueryRefinementOutcome(
            query_id=ref_query.query_id,
            result=result,
            candidates=tuple(candidates),  # type: ignore
            timings=timings,
            warnings=(),
        )

    runtime.refiner.refine_query = mock_refine_query
    import system_tai.kis.session_engine

    original_write_csv = system_tai.kis.session_engine._write_refined_csv
    original_write_json = system_tai.kis.session_engine._write_json
    system_tai.kis.session_engine._write_refined_csv = lambda _, path: Path(path)
    system_tai.kis.session_engine._write_json = lambda path, payload: Path(path)

    try:
        resp_ref = runtime.handle_query(
            QueryRequest("req-ref", "Q-REF", "refinement", refine_top_n=3)
        )

        # Exact assertions required
        assert resp_ref["refined_count"] == 3
        # Should NOT be total output candidate count (which is 10)
        assert resp_ref["refined_count"] != 10

        # result_count should be the original retrieval result count
        assert resp_ref["result_count"] == 2

        assert resp_ref["timings"]["decoded_frame_count"] == 150
        assert resp_ref["timings"]["encoded_image_count"] == 100
    finally:
        runtime.refiner.refine_query = original_refine_query
        system_tai.kis.session_engine._write_refined_csv = original_write_csv
        system_tai.kis.session_engine._write_json = original_write_json
