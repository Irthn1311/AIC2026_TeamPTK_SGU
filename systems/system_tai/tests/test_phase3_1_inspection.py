from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from system_tai.common.schemas import CandidateFrame, KISResult
from system_tai.data.corpus_discovery import discover_corpus
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.inspection.candidate_report import (
    InspectionMode,
    ThumbnailResolver,
    combine_prepared_inspections,
    prepare_candidate_inspection,
    write_candidate_inspection,
)
from system_tai.kis.contest import build_parser
from system_tai.kis.contest import run as contest_cli_run
from system_tai.kis.contest_runner import ContestRunConfig, ContestRunner
from system_tai.kis.contest_schema import ContestQuery
from tests.phase3_helpers import create_corpus, feature_matrix
from tests.test_phase3_contest_runner import FakeContestEncoder


def _inspection_fixture(
    tmp_path: Path, *, candidate_count: int = 100
) -> tuple[KISResult, FeatureStoreRegistry, object]:
    input_root = tmp_path / "input"
    frame_ids = list(range(1000, 1000 + candidate_count))
    matrix = np.zeros((candidate_count, 512), dtype=np.float32)
    matrix[:, 0] = 1.0
    create_corpus(input_root, {"L21_V001": (frame_ids, matrix)})
    manifest = discover_corpus(input_root)
    manifest_path = manifest.write(tmp_path / "manifest.json")
    registry = FeatureStoreRegistry.from_manifest(manifest_path)
    candidates = tuple(
        CandidateFrame(
            video_id="L21_V001",
            frame_id=frame_id,
            clip_row=index,
            keyframe_order=index + 1,
            score=1.0 - index / 1000,
            rank=index + 1,
            source="weighted_rrf",
            diagnostic_metadata={"variant_hit_count": 1, "per_variant": []},
        )
        for index, frame_id in enumerate(frame_ids)
    )
    return KISResult("Q", candidates), registry, manifest


@pytest.mark.parametrize(
    ("mode", "expected_resolves"),
    [
        (InspectionMode.NONE, 0),
        (InspectionMode.TOP_N, 10),
        (InspectionMode.ALL, 100),
    ],
)
def test_inspection_modes_bound_thumbnail_resolution(
    tmp_path: Path,
    mode: InspectionMode,
    expected_resolves: int,
) -> None:
    result, registry, manifest = _inspection_fixture(tmp_path)
    resolver = ThumbnailResolver()
    prepared = prepare_candidate_inspection(
        (result,),
        registry,
        manifest,
        mode=mode,
        top_n=10,
        thumbnail_resolver=resolver,
    )

    assert len(prepared.records) == 100
    assert resolver.stats.resolve_count == expected_resolves
    assert resolver.stats.directory_scan_count == (0 if mode is InspectionMode.NONE else 1)
    assert sum(record["thumbnail_path"] is not None for record in prepared.records) == (
        expected_resolves
    )


def test_thumbnail_cache_is_reused_for_two_queries_and_combined_output(
    tmp_path: Path,
) -> None:
    result, registry, manifest = _inspection_fixture(tmp_path, candidate_count=20)
    second = KISResult(
        "Q2",
        tuple(
            CandidateFrame(
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                clip_row=candidate.clip_row,
                keyframe_order=candidate.keyframe_order,
                score=candidate.score,
                rank=candidate.rank,
                source=candidate.source,
                diagnostic_metadata=candidate.diagnostic_metadata,
            )
            for candidate in result.ranked_candidates
        ),
    )
    resolver = ThumbnailResolver()
    first_prepared = prepare_candidate_inspection(
        (result,),
        registry,
        manifest,
        mode=InspectionMode.TOP_N,
        top_n=10,
        thumbnail_resolver=resolver,
    )
    second_prepared = prepare_candidate_inspection(
        (second,),
        registry,
        manifest,
        mode=InspectionMode.TOP_N,
        top_n=10,
        thumbnail_resolver=resolver,
    )
    before_combined = resolver.stats
    combined = combine_prepared_inspections((first_prepared, second_prepared))

    assert resolver.stats.directory_scan_count == 1
    assert resolver.stats.resolve_count == 20
    assert resolver.stats == before_combined
    assert len(combined.records) == 40


def test_numeric_resolution_duplicate_and_missing_warnings(tmp_path: Path) -> None:
    directory = tmp_path / "keyframes"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "001.jpg").write_bytes(b"one")
    (nested / "2.PNG").write_bytes(b"two")
    (directory / "003.jpg").write_bytes(b"duplicate-a")
    (nested / "3.webp").write_bytes(b"duplicate-b")
    (directory / "not-numeric.jpeg").write_bytes(b"ignored")
    resolver = ThumbnailResolver()

    assert resolver.resolve(directory, 1).path == (directory / "001.jpg").resolve()
    assert resolver.resolve(directory, 2).path == (nested / "2.PNG").resolve()
    duplicate = resolver.resolve(directory, 3)
    missing = resolver.resolve(directory, 999)

    assert duplicate.path is None
    assert any("ambiguous thumbnail keyframe_order=3" in warning for warning in duplicate.warnings)
    assert missing.path is None
    assert any("thumbnail missing" in warning for warning in missing.warnings)
    assert resolver.stats.directory_scan_count == 1


def test_none_mode_writes_full_lightweight_records_and_disabled_summary(
    tmp_path: Path,
) -> None:
    result, registry, manifest = _inspection_fixture(tmp_path, candidate_count=12)
    resolver = ThumbnailResolver()
    prepared = prepare_candidate_inspection(
        (result,),
        registry,
        manifest,
        mode=InspectionMode.NONE,
        top_n=5,
        thumbnail_resolver=resolver,
    )
    artifact = write_candidate_inspection(prepared, tmp_path / "output")
    payload = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    markdown = artifact.markdown_path.read_text(encoding="utf-8")

    assert payload["inspection_mode"] == "none"
    assert len(payload["records"]) == 12
    assert all(record["thumbnail_path"] is None for record in payload["records"])
    assert "Thumbnail inspection disabled" in markdown
    assert resolver.stats.resolve_count == 0


def test_contact_sheet_and_fast_mode_conflicts_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contact sheet requires"):
        ContestRunConfig(
            inspection_mode=InspectionMode.NONE,
            create_contact_sheet=True,
        )

    parser = build_parser()
    common = [
        "--query-id",
        "Q",
        "--query-vi",
        "query",
        "--output-directory",
        str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="conflicts with --contact-sheet"):
        contest_cli_run(parser.parse_args([*common, "--fast-contest-mode", "--contact-sheet"]))
    with pytest.raises(ValueError, match="conflicts with --inspection-mode"):
        contest_cli_run(
            parser.parse_args(
                [*common, "--fast-contest-mode", "--inspection-mode", "all"]
            )
        )


def test_fast_mode_preserves_jsonl_validator_and_artifact_contract(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(
        input_root,
        {
            "L21_V001": (
                [0, 0, 10, 20],
                feature_matrix([(0, 0.8), (0, 1.0), (0, 0.9), (0, 0.7)]),
            )
        },
    )
    manifest = discover_corpus(input_root)
    manifest_path = manifest.write(tmp_path / "manifest.json")
    runner = ContestRunner(encoder_factory=lambda **_kwargs: FakeContestEncoder())
    query = ContestQuery("Q", "axis0 vi", query_en="axis0 en", output_top_k=3)
    default_output = tmp_path / "default"
    fast_output = tmp_path / "fast"

    default = runner.run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(query,),
        output_directory=default_output,
        config=ContestRunConfig(
            device="cpu",
            output_top_k_override=3,
            inspection_mode=InspectionMode.TOP_N,
            inspection_top_n=2,
        ),
    )
    fast = runner.run(
        manifest_path=manifest_path,
        manifest=manifest,
        queries=(query,),
        output_directory=fast_output,
        config=ContestRunConfig(
            device="cpu",
            output_top_k_override=3,
            inspection_mode=InspectionMode.NONE,
            fast_contest_mode=True,
        ),
    )

    assert default.validation.valid and fast.validation.valid
    assert (default_output / "top100.jsonl").read_bytes() == (
        fast_output / "top100.jsonl"
    ).read_bytes()
    core_records = [
        json.loads(line)
        for line in (fast_output / "top100.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        set(record) == {"query_id", "rank", "video_id", "frame_id"}
        for record in core_records
    )
    candidates = json.loads((fast_output / "candidates.json").read_text(encoding="utf-8"))
    assert len(candidates["records"]) == 3
    assert all(record["thumbnail_path"] is None for record in candidates["records"])
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
    assert expected <= {path.name for path in fast.output_files}

    run_manifest = json.loads((fast_output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["successful_query_ids"] == ["Q"]
    assert run_manifest["failed_query_ids"] == []
    assert run_manifest["successful_query_count"] == 1
    assert run_manifest["failed_query_count"] == 0
    assert run_manifest["queries"] and run_manifest["failures"] == []
    assert run_manifest["inspection_mode"] == "none"
    assert run_manifest["fast_contest_mode"] is True

    timings = json.loads((fast_output / "timings.json").read_text(encoding="utf-8"))
    required_timings = {
        "core_jsonl_export_seconds",
        "internal_csv_export_seconds",
        "candidate_json_seconds",
        "thumbnail_index_seconds",
        "thumbnail_resolve_seconds",
        "markdown_seconds",
        "contact_sheet_seconds",
        "combined_export_seconds",
        "total_export_seconds",
        "export_seconds",
    }
    assert required_timings <= timings.keys()
    assert timings["thumbnail_index_seconds"] == 0.0
    assert timings["thumbnail_resolve_seconds"] == 0.0
