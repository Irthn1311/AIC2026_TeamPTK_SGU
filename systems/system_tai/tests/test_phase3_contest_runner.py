from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.data.corpus_discovery import discover_corpus
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.inspection.candidate_report import resolve_keyframe_path
from system_tai.kis.contest import build_parser
from system_tai.kis.contest import run as contest_cli_run
from system_tai.kis.contest_runner import (
    ContestRunConfig,
    ContestRunner,
    safe_query_directory_name,
)
from system_tai.kis.contest_schema import ContestQuery
from tests.phase3_helpers import create_corpus, feature_matrix


class FakeContestEncoder:
    dimension = 512
    identifiers = MappingProxyType(
        {
            "library": "deterministic-test-fake",
            "model": "fixture-text-encoder",
            "device": "cpu",
        }
    )

    def encode(self, text: str) -> np.ndarray:
        if text == "FAIL":
            raise RuntimeError("synthetic query failure")
        vector = np.zeros(512, dtype=np.float32)
        vector[1 if "axis1" in text else 0] = 1.0
        return vector


def _setup(tmp_path: Path) -> tuple[Path, Any, Path]:
    input_root = tmp_path / "input"
    create_corpus(
        input_root,
        {
            "L21_V001": (
                [100, 101],
                feature_matrix([(0, 1.0), (1, 1.0)]),
            ),
            "L21_V002": ([200], feature_matrix([(0, 1.0)])),
        },
    )
    manifest = discover_corpus(input_root)
    manifest_path = manifest.write(tmp_path / "feature_manifest.json")
    return input_root, manifest, manifest_path


def _counted_runner(counts: dict[str, int]) -> ContestRunner:
    def registry_loader(path: Path) -> FeatureStoreRegistry:
        counts["registry"] = counts.get("registry", 0) + 1
        return FeatureStoreRegistry.from_manifest(path)

    def encoder_factory(**_kwargs: object) -> FakeContestEncoder:
        counts["encoder"] = counts.get("encoder", 0) + 1
        return FakeContestEncoder()

    return ContestRunner(
        registry_loader=registry_loader,
        encoder_factory=encoder_factory,
    )


def _query(query_id: str = "Q1") -> ContestQuery:
    return ContestQuery(
        query_id=query_id,
        query_vi="axis0 vi",
        query_en="axis0 en",
        query_en_expansion="axis0 expansion",
        output_top_k=3,
    )


def test_single_run_artifacts_rrf_core_and_frame_semantics(tmp_path: Path) -> None:
    _input, manifest, manifest_path = _setup(tmp_path)
    counts: dict[str, int] = {}
    output = tmp_path / "run"
    outcome = _counted_runner(counts).run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(_query(),),
        output_directory=output,
        config=ContestRunConfig(device="cpu", output_top_k_override=3),
        bootstrap_timings={
            "discovery_seconds": 0.1,
            "manifest_load_or_build_seconds": 0.2,
            "pre_runner_total_seconds": 0.3,
        },
    )
    assert outcome.exit_code == 0
    assert outcome.validation.valid
    assert counts == {"registry": 1, "encoder": 1}
    records = [
        json.loads(line)
        for line in (output / "top100.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["rank"] for record in records] == [1, 2, 3]
    assert records[0]["frame_id"] == 100
    assert all(set(record) == {"query_id", "rank", "video_id", "frame_id"} for record in records)
    candidates = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
    assert candidates["records"][0]["variant_hit_count"] == 3
    assert candidates["records"][0]["frame_id"] == 100
    expected = {
        "top100.jsonl",
        "top100.csv",
        "candidates.json",
        "candidate_inspection.md",
        "validation_report.json",
        "run_manifest.json",
        "timings.json",
        "run_summary.md",
    }
    assert expected.issubset({path.name for path in outcome.output_files})
    validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert validation == {"errors": [], "valid": True, "warnings": []}


def test_batch_loads_registry_and_encoder_once_and_is_deterministic(tmp_path: Path) -> None:
    _input, manifest, manifest_path = _setup(tmp_path)
    queries = (_query("Q2"), ContestQuery("Q1", "axis1 vi", output_top_k=3))
    counts: dict[str, int] = {}
    first_output = tmp_path / "first"
    first = _counted_runner(counts).run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=queries,
        output_directory=first_output,
        config=ContestRunConfig(device="cpu", output_top_k_override=3),
    )
    assert first.exit_code == 0
    assert counts == {"registry": 1, "encoder": 1}
    second_output = tmp_path / "second"
    second = _counted_runner({}).run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=queries,
        output_directory=second_output,
        config=ContestRunConfig(device="cpu", output_top_k_override=3),
    )
    assert second.exit_code == 0
    for filename in ("top100.jsonl", "top100.csv", "candidates.json", "candidate_inspection.md"):
        assert (first_output / filename).read_bytes() == (second_output / filename).read_bytes()


def test_continue_on_query_error_isolates_failure_without_fake_metrics(tmp_path: Path) -> None:
    _input, manifest, manifest_path = _setup(tmp_path)
    queries = (
        ContestQuery("BAD", "FAIL"),
        ContestQuery("GOOD", "axis0 vi", output_top_k=2),
    )
    output = tmp_path / "run"
    outcome = _counted_runner({}).run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=queries,
        output_directory=output,
        config=ContestRunConfig(
            device="cpu",
            output_top_k_override=2,
            continue_on_query_error=True,
        ),
    )
    assert outcome.exit_code == 2
    assert outcome.successful_query_ids == ("GOOD",)
    assert outcome.failed_queries[0][0] == "BAD"
    records = [
        json.loads(line)
        for line in (output / "top100.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {record["query_id"] for record in records} == {"GOOD"}
    manifest_payload = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    failed = next(item for item in manifest_payload["queries"] if item["query_id"] == "BAD")
    assert failed["status"] == "FAILED"
    assert failed["failure_reason"]


class InvalidExporter(CheckpointExporter):
    def export(self, results: Any, destination: Path, **_kwargs: object) -> Any:
        Path(destination).write_text(
            '{"query_id":"Q1","rank":2,"video_id":"L21_V001","frame_id":999}\n',
            encoding="utf-8",
        )
        return None


def test_invalid_checkpoint_returns_nonzero(tmp_path: Path) -> None:
    _input, manifest, manifest_path = _setup(tmp_path)
    runner = ContestRunner(
        encoder_factory=lambda **_kwargs: FakeContestEncoder(),
        exporter=InvalidExporter(),
    )
    outcome = runner.run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(_query(),),
        output_directory=tmp_path / "invalid",
        config=ContestRunConfig(device="cpu", output_top_k_override=3),
    )
    assert outcome.exit_code == 2
    assert not outcome.validation.valid


def test_timings_run_manifest_and_safe_paths(tmp_path: Path) -> None:
    _input, manifest, manifest_path = _setup(tmp_path)
    output = tmp_path / "run"
    outcome = _counted_runner({}).run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(_query("folder/unsafe:Q"),),
        output_directory=output,
        config=ContestRunConfig(device="cpu", output_top_k_override=2),
    )
    assert outcome.exit_code == 0
    timings = json.loads((output / "timings.json").read_text(encoding="utf-8"))
    required = {
        "discovery_seconds",
        "manifest_load_or_build_seconds",
        "registry_load_seconds",
        "model_load_seconds",
        "queries",
        "export_seconds",
        "validation_seconds",
        "total_batch_seconds",
        "corpus_video_count",
        "corpus_feature_row_count",
    }
    assert required.issubset(timings)
    assert {"encode_seconds", "retrieval_seconds"}.issubset(
        timings["queries"][0]["variants"][0]
    )
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    for field in (
        "timestamp_utc",
        "git_commit_hash",
        "model_identifier",
        "device",
        "manifest_fingerprint",
        "video_count",
        "feature_row_count",
        "queries",
        "rrf_constant",
        "top_k_per_variant",
        "exporter_mode",
        "validation_result",
        "output_filenames",
        "failures",
    ):
        assert field in run_manifest
    safe_name = safe_query_directory_name("folder/unsafe:Q")
    assert "/" not in safe_name and "\\" not in safe_name and ":" not in safe_name
    assert (output / "queries" / safe_name).is_dir()


def test_contact_sheet_path_resolution_and_cli_single_smoke(tmp_path: Path) -> None:
    input_root, manifest, _manifest_path = _setup(tmp_path)
    video = manifest.videos[0]
    assert resolve_keyframe_path(video.keyframe_directory, 1) == (
        video.keyframe_directory / "001.jpg"
    ).resolve()
    assert resolve_keyframe_path(video.keyframe_directory, 999) is None

    counts: dict[str, int] = {}
    output = tmp_path / "cli"
    args = build_parser().parse_args(
        [
            "--input-root",
            str(input_root),
            "--query-id",
            "CLI_Q",
            "--query-vi",
            "axis0 vi",
            "--query-en",
            "axis0 en",
            "--output-directory",
            str(output),
            "--device",
            "cpu",
            "--output-top-k",
            "2",
        ]
    )
    assert contest_cli_run(args, runner=_counted_runner(counts)) == 0
    assert counts == {"registry": 1, "encoder": 1}
    assert (output / "top100.jsonl").is_file()


def test_contest_output_is_capped_at_contiguous_top_100(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    frame_ids = list(range(1000, 1120))
    matrix = np.zeros((120, 512), dtype=np.float32)
    matrix[:, 0] = 1.0
    create_corpus(input_root, {"L21_V001": (frame_ids, matrix)})
    manifest = discover_corpus(input_root)
    manifest_path = manifest.write(tmp_path / "manifest.json")
    output = tmp_path / "top100"
    outcome = _counted_runner({}).run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(ContestQuery("Q100", "axis0 vi"),),
        output_directory=output,
        config=ContestRunConfig(
            device="cpu",
            top_k_per_variant=120,
            output_top_k_override=100,
        ),
    )
    assert outcome.exit_code == 0
    records = [
        json.loads(line)
        for line in (output / "top100.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 100
    assert [record["rank"] for record in records] == list(range(1, 101))
