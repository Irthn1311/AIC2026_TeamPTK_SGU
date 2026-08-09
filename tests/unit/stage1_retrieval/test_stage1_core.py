from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.numpy_index import NumPyMemmapExactIndex
from triage_eg.retrieval.stage1.builder import Stage1BuildConfig, build_index
from triage_eg.retrieval.stage1.catalog import load_catalog_rows
from triage_eg.retrieval.stage1.contracts import EncoderContract, SearchConfig
from triage_eg.retrieval.stage1.encoder import compatibility_gate, validate_encoder_output
from triage_eg.retrieval.stage1.runner import load_query_vector, search_vector
from triage_eg.retrieval.stage1.search import deduplicate_kis, group_videos
from triage_eg.retrieval.stage1.stage0_loader import load_stage0_bundle
from triage_eg.retrieval.stage1.writers import (
    INDEX_MEMBERS,
    REPORT_MEMBERS,
    STAGE1B_INPUT_MEMBERS,
    create_index_bundle,
    create_report_bundle,
    create_stage1b_input_bundle,
)


def make_fixture(
    root: Path,
    *,
    videos: int = 2,
    rows: int = 3,
    dimension: int = 512,
    complete: bool = True,
    gate: str = "PASS_WITH_WARNINGS",
) -> tuple[Path, Path]:
    stage0, data = root / "stage0", root / "data"
    stage0.mkdir(parents=True)
    data.mkdir(parents=True)
    video_ids = [f"L01_V{index:03d}" for index in range(1, videos + 1)]
    total = videos * rows
    summary = {
        "audit_version": "0.1.0",
        "mode": "full",
        "videos_discovered": videos,
        "videos_completed": videos,
        "mapping_rows": total,
        "clip_rows": total,
        "config_fingerprint": "stage0-fingerprint",
        "git_commit": "abc",
        "gates": {"btc_baseline": gate},
        "unknown_contracts": ["CLIP model compatibility"],
    }
    (stage0 / "audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (stage0 / "run_manifest.json").write_text(
        json.dumps({"status": "COMPLETE" if complete else "RUNNING"}), encoding="utf-8"
    )
    (stage0 / "contract_notes.json").write_text(
        json.dumps(
            {
                "original_frame_policy": (
                    "CSV frame_idx is authoritative; never reconstruct from pts_time*fps"
                )
            }
        ),
        encoding="utf-8",
    )
    frame_lines = []
    clip_lines = []
    for video_index, video_id in enumerate(video_ids):
        matrix = np.zeros((rows, dimension), dtype=np.float16)
        for row in range(rows):
            matrix[row, (video_index * rows + row) % dimension] = 1
        relative = f"clip-features/{video_id}.npy"
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        clip_lines.append(
            json.dumps(
                {
                    "video_id": video_id,
                    "relative_path": relative,
                    "shape": [rows, dimension],
                    "row_count": rows,
                    "dimension": dimension,
                    "dtype": "float16",
                }
            )
        )
        for n in range(1, rows + 1):
            frame_idx = 7 if n in {1, 2} and video_index == 0 else n + video_index * 10
            frame_lines.append(
                json.dumps(
                    {
                        "video_id": video_id,
                        "n": n,
                        "clip_row_index": n - 1,
                        "pts_time": n / 25,
                        "mapping_fps": 25.0,
                        "original_frame_idx": frame_idx,
                        "keyframe_relative_path": (
                            f"keyframes/Keyframes_L01/keyframes/{video_id}/{n:03d}.jpg"
                        ),
                        "duplicate_frame_idx_group_size": 2 if frame_idx == 7 else 1,
                    }
                )
            )
    (stage0 / "btc_frame_manifest.jsonl").write_text(
        "\n".join(reversed(frame_lines)) + "\n", encoding="utf-8"
    )
    (stage0 / "clip_manifest.jsonl").write_text("\n".join(clip_lines) + "\n", encoding="utf-8")
    return stage0, data


def config(
    root: Path, stage0: Path, data: Path, *, videos=2, rows=3, **kwargs
) -> Stage1BuildConfig:
    defaults = {
        "stage0_root": stage0,
        "dataset_root": data,
        "output_root": root / "output",
        "dimension": 512,
        "expected_rows": videos * rows,
        "expected_videos": videos,
        "self_queries": min(3, videos * rows),
    }
    defaults.update(kwargs)
    return Stage1BuildConfig(**defaults)


def test_valid_stage0_bundle(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path)
    bundle = load_stage0_bundle(stage0, require_full=False)
    assert len(bundle.clip_records) == 2


def test_incomplete_stage0_rejected(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match="COMPLETE"):
        load_stage0_bundle(stage0, require_full=False)


def test_btc_fail_gate_rejected(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path, gate="FAIL")
    with pytest.raises(ValueError, match="BTC"):
        load_stage0_bundle(stage0, require_full=False)


@pytest.mark.parametrize(
    "mutation",
    [lambda value: value.update(clip_rows=5), lambda value: value.update(mapping_rows=5)],
)
def test_stage0_count_mismatch_rejected(tmp_path: Path, mutation) -> None:
    stage0, _ = make_fixture(tmp_path)
    path = stage0 / "audit_summary.json"
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_stage0_bundle(stage0, require_full=False)


def test_original_frame_contract_checked(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path)
    (stage0 / "contract_notes.json").write_text(json.dumps({"original_frame_policy": "wrong"}))
    with pytest.raises(ValueError, match="original-frame"):
        load_stage0_bundle(stage0, require_full=False)


def test_catalog_deterministic_global_rows_and_duplicates(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path)
    catalog = load_catalog_rows(load_stage0_bundle(stage0, require_full=False))
    assert [item["global_row"] for item in catalog.rows] == list(range(6))
    assert [(item["video_id"], item["n"]) for item in catalog.rows] == sorted(
        (item["video_id"], item["n"]) for item in catalog.rows
    )
    assert catalog.rows[0]["original_frame_idx"] == catalog.rows[1]["original_frame_idx"] == 7
    assert len(catalog.rows) == 6


def test_catalog_rejects_bad_clip_row(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path)
    path = stage0 / "btc_frame_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["clip_row_index"] = 999
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="clip_row_index"):
        load_catalog_rows(load_stage0_bundle(stage0, require_full=False))


def test_catalog_rejects_duplicate_video_n(tmp_path: Path) -> None:
    stage0, _ = make_fixture(tmp_path)
    path = stage0 / "btc_frame_manifest.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join([rows[0], rows[0], *rows[2:]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_catalog_rows(load_stage0_bundle(stage0, require_full=False))


def test_build_memmap_offsets_norms_and_catalog(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    result = build_index(
        config(tmp_path, stage0, data, overwrite=True, build_git_commit="test-commit")
    )
    vectors = np.load(result.output_root / "index/clip_vectors.f16.npy", mmap_mode="r")
    norms = np.load(result.output_root / "index/vector_norms.f32.npy", mmap_mode="r")
    assert vectors.shape == (6, 512) and vectors.dtype == np.float16
    assert np.allclose(norms, 1)
    assert np.argmax(vectors, axis=1).tolist() == list(range(6))
    assert result.summary["self_retrieval_status"] == "PASS"
    run_manifest = json.loads((result.output_root / "run_manifest.json").read_text())
    assert result.index_manifest["build_git_commit"] == "test-commit"
    assert result.index_manifest["build_git_commit_source"] == "CLI"
    assert run_manifest["build_git_commit"] == result.index_manifest["build_git_commit"]
    assert run_manifest["build_git_commit_source"] == "CLI"
    report = json.loads(
        (result.output_root / "benchmark/self_retrieval_report.json").read_text()
    )
    assert set(report["classification_counts"]) == {
        "PASS_TOP1",
        "PASS_TOP_K",
        "TIE_SATURATION",
        "NEAR_TIE_RANKED_OUT",
        "SELF_SCORE_INVALID",
        "STRICTLY_BETTER_VECTOR_ANOMALY",
        "INDEX_CATALOG_ALIGNMENT_FAILURE",
        "QUERY_ROW_NOT_FOUND",
    }
    assert {"video_id", "n", "original_frame_idx"} <= set(
        report["query_diagnostics"][0]["diagnostic_top_candidates"][0]
    )


def test_build_maps_tie_warning_to_ready_with_tie_warnings(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    for path in data.glob("clip-features/*.npy"):
        matrix = np.load(path)
        matrix[:] = 0
        matrix[:, 0] = 1
        np.save(path, matrix)
    result = build_index(config(tmp_path, stage0, data, overwrite=True))
    assert result.summary["self_retrieval_status"] == "PASS_WITH_WARNINGS"
    assert result.summary["next_stage_readiness"]["corpus_index"] == (
        "READY_WITH_TIE_WARNINGS"
    )
    report = json.loads(
        (result.output_root / "benchmark/self_retrieval_report.json").read_text()
    )
    assert report["classification_counts"]["TIE_SATURATION"] == 1
    assert report["issues"][0]["code"] == "SELF_RETRIEVAL_TIE_SATURATION"


def test_catalog_array_length_mismatch_fails_self_retrieval(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    result = build_index(config(tmp_path, stage0, data, overwrite=True))
    path = result.output_root / "index/frame_n.npy"
    np.save(path, np.load(path)[:-1])
    from triage_eg.retrieval.stage1.benchmark import run_self_retrieval

    report = run_self_retrieval(result.output_root, samples=2)
    assert report["status"] == "FAIL"
    assert report["classification_counts"]["INDEX_CATALOG_ALIGNMENT_FAILURE"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda matrix: matrix.__setitem__((0, 0), np.nan), "finiteness"),
        (lambda matrix: matrix[:, :-1], "shape"),
    ],
)
def test_index_rejects_bad_source(tmp_path: Path, mutation, message: str) -> None:
    stage0, data = make_fixture(tmp_path)
    path = next(data.glob("clip-features/*.npy"))
    matrix = np.load(path)
    changed = mutation(matrix)
    np.save(path, matrix if changed is None else changed)
    with pytest.raises(ValueError, match=message):
        build_index(config(tmp_path, stage0, data, overwrite=True))
    assert not (tmp_path / "output/index").exists()


def test_index_rejects_clip_manifest_shape_mismatch(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    path = stage0 / "clip_manifest.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["shape"] = [999, 512]
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    with pytest.raises(ValueError, match="manifest shape"):
        build_index(config(tmp_path, stage0, data, overwrite=True))


def test_failed_overwrite_preserves_previous_complete_output(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    build_config = config(tmp_path, stage0, data, overwrite=True)
    first = build_index(build_config)
    previous_manifest = (first.output_root / "index/index_manifest.json").read_bytes()
    path = next(data.glob("clip-features/*.npy"))
    matrix = np.load(path)
    matrix[0, 0] = np.nan
    np.save(path, matrix)

    with pytest.raises(ValueError, match="finiteness"):
        build_index(build_config)

    assert (first.output_root / "index/index_manifest.json").read_bytes() == previous_manifest
    assert not (tmp_path / ".output.building").exists()
    assert not (tmp_path / ".output.previous").exists()


def test_index_reuse_and_stale_fingerprint(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    base = config(tmp_path, stage0, data, overwrite=True)
    build_index(base)
    reused = build_index(replace(base, overwrite=False, reuse_index=True))
    assert reused.reused
    path = next(data.glob("clip-features/*.npy"))
    matrix = np.load(path)
    np.save(path, np.concatenate((matrix, matrix[:1])))
    with pytest.raises(ValueError, match="stale"):
        build_index(replace(base, overwrite=False, reuse_index=True))


def backend(metric="cosine", chunk=2):
    values = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float16)
    norms = np.linalg.norm(values.astype(np.float32), axis=1)
    return NumPyMemmapExactIndex(values, norms, metric=metric, chunk_rows=chunk)


def test_exact_cosine_and_tie_break() -> None:
    scores, rows = backend().search(np.array([1, 0]), 3)
    assert rows[0].tolist() == [0, 1, 2]
    assert scores[0, 0] == 1


def test_exact_dot() -> None:
    scores, rows = backend("dot").search(np.array([0, 2]), 1)
    assert rows[0, 0] == 2 and scores[0, 0] == 2


@pytest.mark.parametrize("query", [np.zeros(2), np.array([np.nan, 0]), np.zeros(3)])
def test_query_validation(query) -> None:
    with pytest.raises(ValueError):
        backend().search(query, 2)


def test_chunked_equals_full_and_large_topk() -> None:
    query = np.array([0.5, 0.5])
    first = backend(chunk=1).search(query, 99)
    second = backend(chunk=99).search(query, 99)
    assert np.allclose(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert first[0].shape == (1, 3)


def test_bounded_topk_keeps_lowest_rows_across_large_tie() -> None:
    values = np.ones((10, 2), dtype=np.float16)
    norms = np.linalg.norm(values.astype(np.float32), axis=1)
    index = NumPyMemmapExactIndex(values, norms, chunk_rows=3)
    _, rows = index.search(np.ones(2, dtype=np.float32), 3)
    assert rows[0].tolist() == [0, 1, 2]


def candidates():
    return [
        {
            "rank": 1,
            "global_row": 1,
            "score": 0.8,
            "video_id": "v",
            "n": 2,
            "original_frame_idx": 7,
        },
        {
            "rank": 2,
            "global_row": 0,
            "score": 0.8,
            "video_id": "v",
            "n": 1,
            "original_frame_idx": 7,
        },
        {
            "rank": 3,
            "global_row": 2,
            "score": 0.7,
            "video_id": "w",
            "n": 1,
            "original_frame_idx": 3,
        },
    ]


def test_kis_dedup_preserves_internal_and_uses_frame_idx() -> None:
    source = candidates()
    kis, evidence = deduplicate_kis(source, 100)
    assert len(source) == 3 and kis == [
        {"video_id": "v", "frame_id": 7},
        {"video_id": "w", "frame_id": 3},
    ]
    assert evidence[("v", 7)] == [2, 1]


def test_kis_highest_score_and_max_predictions() -> None:
    values = candidates()
    values[1]["score"] = 0.9
    kis, _ = deduplicate_kis(values, 1)
    assert kis == [{"video_id": "v", "frame_id": 7}]


@pytest.mark.parametrize(("strategy", "expected"), [("max", 0.8), ("mean_top_k", 0.8)])
def test_video_grouping(strategy: str, expected: float) -> None:
    grouped = group_videos(candidates(), strategy=strategy)
    assert grouped[0]["video_id"] == "v" and grouped[0]["video_score"] == expected


def test_encoder_default_and_dimension_only_blocked() -> None:
    with pytest.raises(PermissionError):
        compatibility_gate(EncoderContract(compatibility_status="UNVERIFIED"))


def test_user_asserted_requires_override() -> None:
    contract = EncoderContract(
        compatibility_status="USER_ASSERTED", evidence_source="USER_ASSERTED"
    )
    with pytest.raises(PermissionError):
        compatibility_gate(contract)
    assert compatibility_gate(contract, allow_unverified=True) == "UNVERIFIED_OVERRIDE"


def test_verified_encoder_gate() -> None:
    assert (
        compatibility_gate(
            EncoderContract(compatibility_status="VERIFIED", evidence_source="AUTHORITATIVE")
        )
        == "VERIFIED"
    )


def test_verified_encoder_without_evidence_is_rejected() -> None:
    with pytest.raises(PermissionError, match="evidence"):
        compatibility_gate(EncoderContract(compatibility_status="VERIFIED"))


@pytest.mark.parametrize(
    "value", [np.zeros((1, 512)), np.full((1, 512), np.nan), np.zeros((1, 511))]
)
def test_encoder_output_rejected(value) -> None:
    with pytest.raises(ValueError):
        validate_encoder_output(value, 1)


def test_load_query_vector_shapes(tmp_path: Path) -> None:
    path = tmp_path / "query.npy"
    np.save(path, np.ones(512, dtype=np.float32))
    assert load_query_vector(path).shape == (1, 512)


def test_vector_search_maps_original_frame_and_writes_outputs(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    result = build_index(config(tmp_path, stage0, data, overwrite=True))
    search_config = SearchConfig(result.output_root, "demo", top_k=6)
    found, paths = search_vector(np.eye(1, 512, dtype=np.float32), search_config)
    assert found[0]["global_row"] == 0
    assert found[0]["original_frame_idx"] == 7
    assert all(path.is_file() for path in paths.values())


def test_report_and_index_zip_members(tmp_path: Path) -> None:
    stage0, data = make_fixture(tmp_path)
    result = build_index(config(tmp_path, stage0, data, overwrite=True))
    from triage_eg.retrieval.stage1.benchmark import run_benchmark

    run_benchmark(result.output_root, random_queries=1, self_queries=1, top_k=2)
    report_zip = create_report_bundle(result.output_root, tmp_path / "reports.zip")
    index_zip = create_index_bundle(result.output_root, tmp_path / "index.zip")
    stage1b_zip = create_stage1b_input_bundle(result.output_root, tmp_path / "stage1b.zip")
    with ZipFile(report_zip) as archive:
        assert set(archive.namelist()) == set(REPORT_MEMBERS)
        assert "index/clip_vectors.f16.npy" not in archive.namelist()
    with ZipFile(index_zip) as archive:
        assert set(archive.namelist()) == set(INDEX_MEMBERS)
        assert index_zip.name not in archive.namelist()
    with ZipFile(stage1b_zip) as archive:
        assert archive.namelist() == list(STAGE1B_INPUT_MEMBERS)
        assert "stage1_summary.json" in archive.namelist()
        assert "index/clip_vectors.f16.npy" in archive.namelist()
        assert stage1b_zip.name not in archive.namelist()


def test_source_has_no_disallowed_runtime_branches() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage1").glob("*.py")
    )
    for forbidden in (
        "import cv2",
        "VideoCapture",
        "extract_frame",
        "class FastLine",
        "EventGraph",
        "AgentRunner",
    ):
        assert forbidden not in source
