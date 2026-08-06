from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.data.corpus_discovery import CorpusManifest, DiscoveredVideo, _fingerprint
from system_tai.refinement.engine import ExactFrameRefiner
from system_tai.refinement.models import RefinementConfig
from system_tai.refinement.runner import RefinementRunner, load_phase3_run
from system_tai.refinement.video import DecodedFrame, DecodeResult, VideoProbe


class RunnerEncoder:
    dimension = 2
    identifiers = {"model": "runner-fake", "device": "cpu"}

    def __init__(self, counts):
        counts["model"] = counts.get("model", 0) + 1
        self.counts = counts

    def encode_texts(self, texts):
        self.counts["text"] = self.counts.get("text", 0) + 1
        return np.asarray([[1, 0] for _ in texts], dtype=np.float32)

    def encode_images(self, images, *, batch_size):
        self.counts["images"] = self.counts.get("images", 0) + len(images)
        rows = np.asarray(
            [[1 / (1 + abs(int(image) - 55)), 0.1] for image in images],
            dtype=np.float32,
        )
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class RunnerDecoder:
    backend_identifier = "runner-fake"

    def probe(self, record):
        return VideoProbe(
            record.video_id, record.raw_video_path, self.backend_identifier, 10, 100, 8, 8, 10
        )

    def decode(self, request):
        frames = tuple(
            DecodedFrame(frame_id, frame_id / request.probe.fps, frame_id)
            for frame_id in request.frame_ids
        )
        return DecodeResult(frames, len(frames), 0, 0, self.backend_identifier, ())


def phase3_run(tmp_path: Path, *, query_ids=("Q1",)) -> Path:
    run = tmp_path / "phase3"
    run.mkdir()
    mapping = tmp_path / "L21_V001.csv"
    mapping.write_text("n,pts_time,fps,frame_idx\n1,5,10,50\n", encoding="utf-8")
    clip = tmp_path / "L21_V001.npy"
    np.save(clip, np.asarray([[1.0, 0.0]], dtype=np.float32))
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
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
        1,
    )
    manifest = CorpusManifest(tmp_path, tmp_path, _fingerprint((discovered,)), (discovered,))
    manifest.write(run / "feature_manifest.json")
    core = []
    inspection = []
    queries = []
    for query_id in query_ids:
        core.append({"query_id": query_id, "rank": 1, "video_id": "L21_V001", "frame_id": 50})
        inspection.append(
            {
                "query_id": query_id,
                "rank": 1,
                "video_id": "L21_V001",
                "frame_id": 50,
                "fusion_score": 0.2,
                "variant_hit_count": 1,
                "best_individual_rank": 1,
                "per_variant": [],
                "clip_row_diagnostic": 0,
                "keyframe_order_diagnostic": 1,
            }
        )
        queries.append(
            {
                "query_id": query_id,
                "status": "SUCCESS",
                "variants": [
                    {
                        "variant_id": "vi",
                        "text": "target",
                        "language": "vi",
                        "variant_type": "vietnamese_direct",
                        "weight": 1.0,
                    }
                ],
            }
        )
    (run / "top100.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in core), encoding="utf-8"
    )
    (run / "candidates.json").write_text(json.dumps({"records": inspection}), encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({"queries": queries}), encoding="utf-8")
    return run


def test_phase3_artifacts_require_explicit_variants_and_provenance(tmp_path: Path) -> None:
    loaded = load_phase3_run(phase3_run(tmp_path))
    assert loaded.queries[0].variants[0].text == "target"
    assert loaded.queries[0].candidates[0].frame_id == 50


def test_runner_artifact_contract_model_once_and_core_safety(tmp_path: Path) -> None:
    counts = {}
    output = tmp_path / "phase4"
    runner = RefinementRunner(
        decoder_factory=RunnerDecoder,
        encoder_factory=lambda **_kwargs: RunnerEncoder(counts),
    )
    outcome = runner.run(
        run_directory=phase3_run(tmp_path, query_ids=("Q2", "Q1")),
        output_directory=output,
        config=RefinementConfig(
            top_candidates_to_refine=1,
            window_before_seconds=1,
            window_after_seconds=1,
            coarse_stride_frames=5,
            coarse_top_n=1,
            fine_radius_frames=2,
            output_top_k=1,
            max_decoded_frames_per_candidate=30,
        ),
    )
    assert outcome.exit_code == 0 and outcome.validation.valid
    assert counts["model"] == 1
    records = [
        json.loads(line)
        for line in (output / "refined_top100.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(set(item) == {"query_id", "rank", "video_id", "frame_id"} for item in records)
    assert [item["query_id"] for item in records] == ["Q1", "Q2"]
    assert all(item["frame_id"] == 55 for item in records)
    assert all(item["rank"] == 1 for item in records)
    expected = {
        "refined_top100.jsonl",
        "refined_top100.csv",
        "refinement_candidates.json",
        "refinement_trace.json",
        "refinement_timings.json",
        "refinement_validation_report.json",
        "refinement_run_manifest.json",
        "refinement_summary.md",
    }
    assert expected.issubset({path.name for path in outcome.output_files})
    trace = (output / "refinement_trace.json").read_text(encoding="utf-8")
    assert "embedding" not in trace and '"image":' not in trace
    timings = json.loads((output / "refinement_timings.json").read_text(encoding="utf-8"))
    for field in (
        "coarse_decode_seconds",
        "fine_encode_seconds",
        "decoded_frame_count",
        "refined_candidate_count",
        "total_run_seconds",
    ):
        assert field in timings
    manifest = json.loads((output / "refinement_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ranking_policy"].startswith("replace frame")
    assert manifest["contact_sheet_status"] == "not-created"


def test_repeated_engine_run_is_deterministic(tmp_path: Path) -> None:
    artifacts = load_phase3_run(phase3_run(tmp_path))
    config = RefinementConfig(
        top_candidates_to_refine=1,
        window_before_seconds=1,
        window_after_seconds=1,
        coarse_stride_frames=5,
        coarse_top_n=1,
        fine_radius_frames=2,
        output_top_k=1,
        max_decoded_frames_per_candidate=30,
    )
    raw = __import__(
        "system_tai.refinement.video", fromlist=["RawVideoRegistry"]
    ).RawVideoRegistry.from_manifest(artifacts.manifest)
    first = ExactFrameRefiner(
        raw_videos=raw, decoder=RunnerDecoder(), encoder=RunnerEncoder({})
    ).refine_query(artifacts.queries[0], config)
    second = ExactFrameRefiner(
        raw_videos=raw, decoder=RunnerDecoder(), encoder=RunnerEncoder({})
    ).refine_query(artifacts.queries[0], config)
    assert [item.frame_id for item in first.result.ranked_candidates] == [
        item.frame_id for item in second.result.ranked_candidates
    ]


class FailSecondTextEncoder(RunnerEncoder):
    def encode_texts(self, texts):
        count = self.counts.get("query_text", 0) + 1
        self.counts["query_text"] = count
        if count == 2:
            raise RuntimeError("controlled query failure")
        return super().encode_texts(texts)


def test_continue_on_query_error_and_fail_fast(tmp_path: Path) -> None:
    source = phase3_run(tmp_path, query_ids=("Q1", "Q2", "Q3"))
    continued = RefinementRunner(
        decoder_factory=RunnerDecoder,
        encoder_factory=lambda **_kwargs: FailSecondTextEncoder({}),
    ).run(
        run_directory=source,
        output_directory=tmp_path / "continued",
        config=RefinementConfig(top_candidates_to_refine=1, output_top_k=1),
        continue_on_query_error=True,
    )
    assert continued.exit_code == 2
    assert continued.successful_query_ids == ("Q1", "Q3")
    assert continued.failed_queries[0][0] == "Q2"

    stopped = RefinementRunner(
        decoder_factory=RunnerDecoder,
        encoder_factory=lambda **_kwargs: FailSecondTextEncoder({}),
    ).run(
        run_directory=source,
        output_directory=tmp_path / "stopped",
        config=RefinementConfig(top_candidates_to_refine=1, output_top_k=1),
        continue_on_query_error=False,
    )
    assert stopped.successful_query_ids == ("Q1",)
    assert stopped.failed_queries[0][0] == "Q2"


class InvalidCombinedExporter(CheckpointExporter):
    def export(self, results, destination, **kwargs):
        summary = super().export(results, destination, **kwargs)
        if (
            Path(destination).name == "refined_top100.jsonl"
            and "queries" not in Path(destination).parts
        ):
            Path(destination).write_text(
                '{"query_id":"Q1","rank":2,"video_id":"L21_V001","frame_id":55}\n',
                encoding="utf-8",
            )
        return summary


def test_invalid_final_checkpoint_returns_nonzero(tmp_path: Path) -> None:
    outcome = RefinementRunner(
        decoder_factory=RunnerDecoder,
        encoder_factory=lambda **_kwargs: RunnerEncoder({}),
        exporter=InvalidCombinedExporter(),
    ).run(
        run_directory=phase3_run(tmp_path),
        output_directory=tmp_path / "invalid",
        config=RefinementConfig(top_candidates_to_refine=1, output_top_k=1),
    )
    assert outcome.exit_code == 2
    assert not outcome.validation.valid
