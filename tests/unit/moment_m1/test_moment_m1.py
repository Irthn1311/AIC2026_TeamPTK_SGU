from __future__ import annotations

import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.moment_m1 import (
    DecodedFrame,
    M1RunnerConfig,
    M1Settings,
    VideoInfo,
    aggregate_refinement_metrics,
    blinded_mapping,
    clipped_local_window,
    coarse_frame_indices,
    create_m1_bundle,
    dense_frame_indices,
    failure_diagnostics,
    order_only_source_chain,
    reference_is_reachable,
    render_blinded_event_sheet,
    run_moment_m1,
    select_best_frame,
)
from triage_eg.experiments.moment_m1 import runner as m1_runner
from triage_eg.experiments.reference_rt2 import (
    BENCHMARK_TYPE,
    RT2BenchmarkQuery,
    RT2ReferenceEvent,
)
from triage_eg.retrieval.stage2 import EncodedQueryBatch, Stage2RuntimeConfig


class Catalog:
    def __init__(self) -> None:
        self.video_index = np.zeros(4, dtype=np.int32)
        self.n = np.arange(1, 5, dtype=np.int32)
        self.original_idx = np.asarray([0, 10, 20, 30], dtype=np.int64)
        self.video_table = [{"video_id": "L01_V001", "keyframe_prefix": "L01_V001"}]

    def map_row(self, row: int) -> dict[str, object]:
        return {
            "global_row": row,
            "video_id": "L01_V001",
            "n": int(self.n[row]),
            "original_frame_idx": int(self.original_idx[row]),
            "keyframe_relative_path": f"L01_V001/{self.n[row]:03d}.jpg",
        }


class Backend:
    def __init__(self) -> None:
        self.vectors = np.zeros((4, 512), dtype=np.float32)
        self.vectors[0, :2] = (1.0, 0.0)
        self.vectors[1, :2] = (0.7, 0.3)
        self.vectors[2, :2] = (0.0, 1.0)
        self.vectors[3, :2] = (0.3, 0.7)

    def vectors_at(self, rows: np.ndarray) -> np.ndarray:
        return self.vectors[rows]


def benchmark_query() -> RT2BenchmarkQuery:
    return RT2BenchmarkQuery(
        "rt2_test",
        BENCHMARK_TYPE,
        "L01_V001",
        "en",
        (
            RT2ReferenceEvent("E1", "first visible event", "S02", 1, 1, 2, 10),
            RT2ReferenceEvent("E2", "second visible event", "S04", 3, 3, 4, 30),
        ),
        ("MULTI_EVENT",),
        "GPT-5.6 Sol",
        False,
    )


def text_embeddings() -> np.ndarray:
    values = np.zeros((2, 512), dtype=np.float32)
    values[0, 0] = 1.0
    values[1, 1] = 1.0
    return values


def test_m1_settings_freeze_order_only_lambda_zero() -> None:
    settings = M1Settings()
    assert settings.distance_lambda == 0.0
    assert settings.local_window_seconds == 6.0
    try:
        M1Settings(distance_lambda=0.0001)
    except ValueError as error:
        assert "lambda=0" in str(error)
    else:
        raise AssertionError("positive M1 lambda was accepted")


def test_source_video_chain_reuses_lambda_zero_and_is_strict(monkeypatch) -> None:
    calls = []
    real = m1_runner.dante_monotonic_dp

    def recording_dp(scores, distance_lambda):
        calls.append(distance_lambda)
        return real(scores, distance_lambda)

    monkeypatch.setattr(m1_runner, "dante_monotonic_dp", recording_dp)
    chain = order_only_source_chain(
        backend=Backend(),
        catalog=Catalog(),
        source_rows=np.arange(4),
        event_ids=["E1", "E2"],
        text_embeddings=text_embeddings(),
    )
    assert calls == [0.0]
    positions = [item["catalog_position"] for item in chain]
    assert positions == sorted(positions) and positions[0] < positions[1]


def test_raw_window_clips_at_video_start_and_end() -> None:
    assert clipped_local_window(2, fps=2.0, total_frames=100) == (0, 14)
    assert clipped_local_window(98, fps=2.0, total_frames=100) == (86, 99)


def test_coarse_samples_are_raw_coordinates_and_include_anchor_boundaries() -> None:
    values = coarse_frame_indices(3, 41, 17, stride=12)
    assert values == [3, 15, 17, 27, 39, 41]
    assert {3, 17, 41} <= set(values)


def test_dense_refinement_stays_within_peak_radius_and_window() -> None:
    assert dense_frame_indices(10, 50, 12, radius=15) == list(range(10, 28))
    assert dense_frame_indices(10, 50, 48, radius=15) == list(range(33, 51))


def test_score_tie_selects_lower_actual_frame_index() -> None:
    frame_idx, score = select_best_frame([24, 12, 36], np.asarray([0.8, 0.8, 0.7]))
    assert frame_idx == 12 and np.isclose(score, 0.8)


def test_reference_reachability_and_failure_taxonomy() -> None:
    assert reference_is_reachable(10, 10, 20)
    assert not reference_is_reachable(9, 10, 20)
    assert failure_diagnostics(
        reference_reachable=False, coarse_error_frames=40, refined_error_frames=10
    ) == ["COARSE_REFERENCE_OUTSIDE_WINDOW", "LOCAL_REFINEMENT_IMPROVED"]
    assert failure_diagnostics(
        reference_reachable=True, coarse_error_frames=5, refined_error_frames=5
    ) == ["LOCAL_REFINEMENT_TIED"]
    assert failure_diagnostics(
        reference_reachable=True, coarse_error_frames=2, refined_error_frames=6
    ) == ["LOCAL_REFINEMENT_REGRESSED"]


def test_refinement_metrics_compute_wins_ties_regressions_and_hits() -> None:
    records = [
        {"coarse_error_frames": 20, "refined_error_frames": 5, "error_delta": 15},
        {"coarse_error_frames": 5, "refined_error_frames": 5, "error_delta": 0},
        {"coarse_error_frames": 1, "refined_error_frames": 10, "error_delta": -9},
    ]
    metrics = aggregate_refinement_metrics(records)
    assert metrics["REFINEMENT_WIN_COUNT"] == 1
    assert metrics["REFINEMENT_TIE_COUNT"] == 1
    assert metrics["REFINEMENT_REGRESSION_COUNT"] == 1
    assert metrics["M0_BTC_TECHNICAL_KEYFRAME_ANCHOR"]["MEDIAN_ABSOLUTE_ERROR_FRAMES"] == 5
    assert metrics["M1_LOCAL_RAW_CLIP_COARSE_TO_FINE"]["HIT_WITHIN_5_FRAMES"] == 2 / 3


class FakeRuntime:
    def __init__(self) -> None:
        self.catalog = Catalog()
        self.backend = Backend()
        self.preflight = {"stage1_index_fingerprint": "frozen"}
        self.encode_calls = 0
        self.translator_calls = 0

    def load(self):
        return self

    def encode_requests(self, requests):
        self.encode_calls += 1
        assert all(request.language == "en" for request in requests)
        resolutions = tuple(
            type("Resolution", (), {"as_dict": lambda self: {"resolved_language": "en"}})()
            for _ in requests
        )
        encodings = tuple(
            {"clip_input_text": request.text, "translation_applied": False} for request in requests
        )
        return EncodedQueryBatch(
            text_embeddings(), resolutions, encodings, tuple({} for _ in requests), 1.0
        )

    def runtime_manifest(self):
        return {"translator": {"loaded": False}, "network_required": False}


class FakeDecoder:
    info = VideoInfo(fps=2.0, total_frames=60)

    def __init__(self, video_id: str, video_path: Path) -> None:
        self.video_id = video_id
        self.video_path = video_path

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        frames = []
        for index in sorted(set(frame_indices)):
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            image[0, 0, 0] = index
            frames.append(DecodedFrame(self.video_id, index, image))
        return frames

    def close(self) -> None:
        return None


class FakeImageEncoder:
    def encode(self, frames: list[DecodedFrame]) -> np.ndarray:
        result = np.zeros((len(frames), 512), dtype=np.float32)
        for row, frame in enumerate(frames):
            index = frame.actual_frame_idx
            result[row, 0] = 1.0 / (1.0 + abs(index - 10))
            result[row, 1] = 1.0 / (1.0 + abs(index - 30))
        return result


def runner_config(tmp_path: Path) -> M1RunnerConfig:
    stage2 = Stage2RuntimeConfig(
        *(tmp_path / name for name in ("s1", "s1b", "s1e", "clip", "opus", "runtime", "s1d"))
    )
    return M1RunnerConfig(
        stage2,
        tmp_path / "dataset",
        tmp_path / "benchmark.jsonl",
        tmp_path / "output",
        M1Settings(),
    )


def test_runner_encodes_english_events_once_and_reuses_query_local_frames(tmp_path: Path) -> None:
    config = runner_config(tmp_path)
    raw_video = config.dataset_root / "Videos_L01/video/L01_V001.mp4"
    raw_video.parent.mkdir(parents=True)
    raw_video.write_bytes(b"fake")
    config.benchmark_path.write_text("{}\n", encoding="utf-8")
    runtime = FakeRuntime()
    summary = run_moment_m1(
        config,
        [benchmark_query()],
        runtime=runtime,  # type: ignore[arg-type]
        decoder_factory=FakeDecoder,
        local_image_encoder=FakeImageEncoder(),
        render_visuals=False,
    )
    assert runtime.encode_calls == 1 and runtime.translator_calls == 0
    assert summary["benchmark_event_count"] == 2
    results = [
        json.loads(line)
        for line in (config.output_root / "event_results.jsonl").read_text().splitlines()
    ]
    assert [item["coarse_anchor_frame_idx"] for item in results] == [0, 20]
    assert [item["refined_frame_idx"] for item in results] == [10, 30]
    assert all(item["diagnostics"] == ["LOCAL_REFINEMENT_IMPROVED"] for item in results)
    manifest = json.loads((config.output_root / "run_manifest.json").read_text())
    assert manifest["text_embedding_batches"] == 1
    assert manifest["raw_frames_added_to_global_index"] is False


def test_blinded_mapping_is_deterministic_and_visual_has_no_reference_input() -> None:
    assert blinded_mapping("q", "E1", 2026) == blinded_mapping("q", "E1", 2026)
    parameters = inspect.signature(render_blinded_event_sheet).parameters
    assert "reference_frame_idx" not in parameters
    assert "score" not in parameters and "error" not in parameters


def test_m1_bundle_excludes_models_vectors_raw_video_and_cache(tmp_path: Path) -> None:
    root = tmp_path / "m1"
    for name in ("m1_summary.json", "m1_metrics.json", "run_manifest.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "event_results.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "issues.jsonl").write_text("", encoding="utf-8")
    (root / "visuals").mkdir()
    (root / "visuals/review_key.json").write_text("{}\n", encoding="utf-8")
    (root / "visuals/q_E1_ab.jpg").write_bytes(b"jpg")
    (root / "_stage2_control").mkdir()
    np.save(root / "_stage2_control/vectors.npy", np.ones(2))
    (root / "raw.mp4").write_bytes(b"raw")
    (root / "weights.bin").write_bytes(b"weights")
    archive = create_m1_bundle(root, tmp_path / "m1.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "m1_summary.json" in names and "visuals/q_E1_ab.jpg" in names
    assert not any(name.endswith((".npy", ".bin", ".mp4")) for name in names)
    assert not any(name.startswith("_stage2_control/") for name in names)


def test_m1_source_has_no_forbidden_model_or_network_downloads() -> None:
    source = Path(m1_runner.__file__).read_text(encoding="utf-8")
    for forbidden in ("requests.get", "urlopen(", "hf_hub_download", "snapshot_download"):
        assert forbidden not in source
    assert "SigLIP" not in source and "optical_flow" not in source
