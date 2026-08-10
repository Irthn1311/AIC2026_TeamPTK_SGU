from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.experiments.reference_rt1.scoring import (
    build_video_row_groups,
    rank_unordered_event_max,
)
from triage_eg.experiments.reference_rt2 import (
    BENCHMARK_TYPE,
    RT2BenchmarkQuery,
    RT2ReferenceEvent,
    RT2RunnerConfig,
    RT2Settings,
    create_candidate_bundle,
    create_rt2_evaluation_bundle,
    resolve_benchmark_identities,
    run_reference_rt2_evaluation,
    select_lambda_from_dev,
    split_dev_holdout,
)
from triage_eg.experiments.reference_rt2 import evaluation as rt2_evaluation
from triage_eg.experiments.reference_rt2.benchmark import (
    GENERAL_ELIGIBLE,
    TEMPORALLY_DIVERSE,
    render_candidate_contact_sheet,
    select_candidate_videos,
)
from triage_eg.experiments.reference_rt2.evaluation import chain_collapse_metrics
from triage_eg.retrieval.stage2 import EncodedQueryBatch, Stage2RuntimeConfig


class Catalog:
    def __init__(self, video_count: int = 4, frames_per_video: int = 12) -> None:
        self.video_index = np.repeat(np.arange(video_count), frames_per_video).astype(np.int32)
        self.n = np.tile(np.arange(1, frames_per_video + 1), video_count).astype(np.int32)
        self.original_idx = self.n.astype(np.int64) * 10
        self.video_table = [
            {"video_id": f"V{index}", "keyframe_prefix": f"V{index}"}
            for index in range(video_count)
        ]

    def map_row(self, row: int) -> dict[str, object]:
        video_id = self.video_table[int(self.video_index[row])]["video_id"]
        n = int(self.n[row])
        return {
            "global_row": row,
            "video_id": video_id,
            "n": n,
            "original_frame_idx": int(self.original_idx[row]),
            "keyframe_relative_path": f"{video_id}/{n:03d}.jpg",
        }


def event(event_id: str, position: int, catalog: Catalog, video_id: str = "V0"):
    group = next(item for item in build_video_row_groups(catalog) if item.video_id == video_id)
    row = int(group.rows[position])
    mapped = catalog.map_row(row)
    return RT2ReferenceEvent(
        event_id,
        f"visible event {event_id}",
        f"S{position + 1:02d}",
        position,
        row,
        int(mapped["n"]),
        int(mapped["original_frame_idx"]),
    )


def query(query_id: str, catalog: Catalog, event_count: int = 2) -> RT2BenchmarkQuery:
    return RT2BenchmarkQuery(
        query_id,
        BENCHMARK_TYPE,
        "V0",
        "en",
        tuple(event(f"E{index + 1}", index, catalog) for index in range(event_count)),
        ("MULTI_EVENT",),
        "GPT-5.6 Sol",
        False,
    )


def test_candidate_selection_is_deterministic_and_balanced() -> None:
    catalog = Catalog(video_count=8)
    groups = build_video_row_groups(catalog)
    rng = np.random.default_rng(2026)
    vectors = rng.normal(size=(len(catalog.n), 8)).astype(np.float32)
    first = select_candidate_videos(groups, vectors, candidate_count=6, seed=2026)
    second = select_candidate_videos(groups, vectors, candidate_count=6, seed=2026)
    assert [item["video_id"] for item in first] == [item["video_id"] for item in second]
    assert sum(item["sampling_bucket"] == TEMPORALLY_DIVERSE for item in first) == 3
    assert sum(item["sampling_bucket"] == GENERAL_ELIGIBLE for item in first) == 3


def test_contact_sheet_manifest_is_chronological_and_canonical(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    catalog = Catalog(video_count=1, frames_per_video=20)
    group = build_video_row_groups(catalog)[0]
    dataset = tmp_path / "dataset"
    for row in group.rows:
        mapped = catalog.map_row(int(row))
        path = dataset / str(mapped["keyframe_relative_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        image_module.new("RGB", (80, 60), (int(row) * 5, 30, 90)).save(path)
    output = tmp_path / "sheet.jpg"
    rows = render_candidate_contact_sheet(
        output,
        dataset_root=dataset,
        catalog=catalog,
        group=group,
        sampling_bucket=TEMPORALLY_DIVERSE,
    )
    assert output.is_file()
    assert [item["sheet_slot"] for item in rows] == [f"S{i:02d}" for i in range(1, 17)]
    assert [item["catalog_position"] for item in rows] == sorted(
        item["catalog_position"] for item in rows
    )
    assert rows[0]["catalog_position"] == 0 and rows[-1]["catalog_position"] == 19


def test_benchmark_identity_resolves_to_canonical_stage1_rows() -> None:
    catalog = Catalog()
    value = query("rt2_identity", catalog)
    assert resolve_benchmark_identities([value], catalog) == [value]
    broken = RT2BenchmarkQuery(
        value.query_id,
        value.benchmark_type,
        value.source_video_id,
        value.language,
        (
            value.events[0],
            RT2ReferenceEvent("E2", "second", "S02", 1, 1, 2, 999),
        ),
        value.difficulty_tags,
        value.generator,
        value.human_reviewed,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_benchmark_identities([broken], catalog)


def test_non_monotonic_pseudo_gt_is_rejected() -> None:
    catalog = Catalog()
    with pytest.raises(ValueError, match="strictly increasing"):
        RT2BenchmarkQuery(
            "rt2_bad",
            BENCHMARK_TYPE,
            "V0",
            "en",
            (event("E1", 1, catalog), event("E2", 0, catalog)),
            ("MULTI_EVENT",),
            "GPT-5.6 Sol",
            False,
        )


class Backend:
    size = 4

    def __init__(self) -> None:
        self.score_calls = 0

    def score_many_all(self, embeddings: np.ndarray) -> np.ndarray:
        self.score_calls += 1
        assert len(embeddings) == 2
        return np.asarray([[0.9, 0.1, 0.1, 0.8], [0.1, 0.9, 0.8, 0.1]], dtype=np.float32)


class Runtime:
    def __init__(self) -> None:
        self.catalog = Catalog(video_count=2, frames_per_video=2)
        self.backend = Backend()
        self.preflight = {"stage1_index_fingerprint": "frozen"}

    def load(self):
        return self

    def encode_requests(self, requests):
        embeddings = np.zeros((len(requests), 512), dtype=np.float32)
        embeddings[:, 0] = 1.0
        resolutions = tuple(
            type("Resolution", (), {"as_dict": lambda self: {"resolved_language": "en"}})()
            for _ in requests
        )
        encodings = tuple(
            {"clip_input_text": request.text, "translation_applied": False} for request in requests
        )
        return EncodedQueryBatch(
            embeddings, resolutions, encodings, tuple({} for _ in requests), 1.0
        )

    def runtime_manifest(self):
        return {"ranking_policy": "FROZEN_STAGE1A_EXACT_COSINE_NO_RERANKING"}


def _runner_config(tmp_path: Path) -> RT2RunnerConfig:
    stage2 = Stage2RuntimeConfig(
        *(tmp_path / name for name in ("s1", "s1b", "s1e", "clip", "opus", "runtime", "s1d"))
    )
    return RT2RunnerConfig(
        stage2,
        tmp_path,
        tmp_path / "benchmark.jsonl",
        tmp_path / "output",
        RT2Settings(lambda_grid=(0.0, 0.001)),
    )


def test_runner_reuses_one_score_matrix_per_query(tmp_path: Path, monkeypatch) -> None:
    runtime = Runtime()
    values = [query(f"rt2_{index:03d}", runtime.catalog) for index in range(18)]
    monkeypatch.setattr(rt2_evaluation, "render_holdout_ab_sheet", lambda *args, **kwargs: [])
    summary = run_reference_rt2_evaluation(
        _runner_config(tmp_path),
        values,
        runtime=runtime,  # type: ignore[arg-type]
    )
    assert runtime.backend.score_calls == len(values)
    assert summary["calibration_status"] == "COMPLETE"
    assert summary["selected_lambda"] == 0.0
    manifest = json.loads((tmp_path / "output/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["score_matrix_computations_per_query"] == 1


def test_lambda_zero_is_supported_by_rt2_grid() -> None:
    settings = RT2Settings(lambda_grid=(0.0, 0.0001))
    assert settings.lambda_grid[0] == 0.0


def test_unordered_event_max_is_invariant_to_event_permutation() -> None:
    catalog = Catalog(video_count=2, frames_per_video=2)
    groups = build_video_row_groups(catalog)
    scores = np.asarray([[0.9, 0.1, 0.1, 0.8], [0.1, 0.9, 0.8, 0.1]])
    forward = rank_unordered_event_max(scores, ["E1", "E2"], groups, catalog)
    reversed_order = rank_unordered_event_max(scores[::-1], ["E2", "E1"], groups, catalog)
    assert [(item["video_id"], item["unordered_score"]) for item in forward] == [
        (item["video_id"], item["unordered_score"]) for item in reversed_order
    ]


def test_reversed_order_control_uses_reversed_event_order(tmp_path: Path) -> None:
    runtime = Runtime()
    run_reference_rt2_evaluation(
        _runner_config(tmp_path),
        [query("rt2_reverse", runtime.catalog)],
        runtime=runtime,  # type: ignore[arg-type]
    )
    result = json.loads(
        (tmp_path / "output/query_results/rt2_reverse.json").read_text(encoding="utf-8")
    )
    order = result["DANTE_DP"]["0"]["order_metrics"]
    assert order["correct_order_video_rank"] == 1
    assert order["reversed_order_video_rank"] == 2
    assert order["rank_improvement_correct_vs_reversed"] == 1


def test_chain_collapse_metrics_match_definition() -> None:
    catalog = Catalog(frames_per_video=12)
    value = query("rt2_collapse", catalog, event_count=3)
    source = {
        "chain": [
            {"event_id": "E1", "catalog_position": 4},
            {"event_id": "E2", "catalog_position": 5},
            {"event_id": "E3", "catalog_position": 7},
        ]
    }
    metrics = chain_collapse_metrics(value, source)
    assert metrics["predicted_span"] == 3
    assert metrics["reference_span"] == 2
    assert metrics["span_ratio"] == 1.5
    assert metrics["adjacent_step_distances"] == [1, 2]
    assert metrics["CHAIN_COLLAPSE_CANDIDATE"] is True


def test_dev_holdout_split_is_deterministic_stratified_and_disjoint() -> None:
    catalog = Catalog()
    values = [query(f"rt2_{index:03d}", catalog, 2 + index % 3) for index in range(21)]
    first = split_dev_holdout(values, 2026)
    second = split_dev_holdout(values, 2026)
    assert [[item.query_id for item in part] for part in first] == [
        [item.query_id for item in part] for part in second
    ]
    assert not ({item.query_id for item in first[0]} & {item.query_id for item in first[1]})
    assert len(first[0]) == 14 and len(first[1]) == 7
    assert {len(item.events) for item in first[0]} == {2, 3, 4}
    assert {len(item.events) for item in first[1]} == {2, 3, 4}


def test_lambda_selection_is_dev_only_and_lexicographic() -> None:
    def row(value: float, recall: float, mrr: float) -> dict[str, object]:
        return {
            "lambda": value,
            "split": "DEV",
            "video_metrics": {"INTERNAL_VIDEO_RECALL_AT_5": recall, "MRR": mrr},
            "anchor_metrics": {
                "AI_REFERENCE_ANCHOR_HIT_WITHIN_3": 0.5,
                "AI_REFERENCE_ANCHOR_MEAN_ABSOLUTE_CATALOG_POSITION_ERROR": 3.0,
            },
        }

    assert select_lambda_from_dev([row(0.0, 0.5, 0.9), row(0.001, 0.6, 0.1)]) == 0.001
    invalid = row(0.01, 1.0, 1.0)
    invalid["split"] = "HOLDOUT"
    with pytest.raises(ValueError, match="DEV-only"):
        select_lambda_from_dev([invalid])


def test_evaluation_bundle_excludes_heavy_assets(tmp_path: Path) -> None:
    root = tmp_path / "rt2"
    for name in (
        "rt2_summary.json",
        "rt2_metrics.json",
        "run_manifest.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "rt2_report.md").write_text("report\n", encoding="utf-8")
    (root / "issues.jsonl").write_text("", encoding="utf-8")
    benchmark = root / "benchmark/rt2_ai_benchmark.jsonl"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("{}\n", encoding="utf-8")
    (root / "_stage2_control").mkdir()
    np.save(root / "_stage2_control/index.npy", np.ones(2))
    archive = create_rt2_evaluation_bundle(root, tmp_path / "rt2.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "rt2_summary.json" in names
    assert not any(name.endswith((".npy", ".bin", ".pt", ".mp4")) for name in names)

    candidate_root = tmp_path / "candidates"
    (candidate_root / "candidates").mkdir(parents=True)
    (candidate_root / "candidates/V0.jpg").write_bytes(b"sheet")
    (candidate_root / "candidate_manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (candidate_root / "candidate_selection.json").write_text(
        json.dumps({"selected_video_count": 1}), encoding="utf-8"
    )
    (candidate_root / "README_AI_BENCHMARK_CREATION.md").write_text(
        "instructions\n", encoding="utf-8"
    )
    np.save(candidate_root / "forbidden.npy", np.ones(2))
    candidate_archive = create_candidate_bundle(candidate_root, tmp_path / "candidates.zip")
    with ZipFile(candidate_archive) as stream:
        candidate_names = stream.namelist()
    assert candidate_names == [
        "README_AI_BENCHMARK_CREATION.md",
        "candidate_manifest.jsonl",
        "candidate_selection.json",
        "candidates/V0.jpg",
    ]
