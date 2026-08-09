from __future__ import annotations

import inspect
import json
import time
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.experiments.reference_rt1 import (
    RT1Event,
    RT1Query,
    RT1RunnerConfig,
    RT1Settings,
    build_video_row_groups,
    create_rt1_bundle,
    dante_monotonic_dp,
    rank_dante_dp,
    rank_unordered_event_max,
    run_reference_rt1,
)
from triage_eg.experiments.reference_rt1 import runner as rt1_runner
from triage_eg.experiments.reference_rt1.visuals import render_rt1_visuals
from triage_eg.retrieval.numpy_index import NumPyMemmapExactIndex
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    EncodedQueryBatch,
    OperationalRetrievalRuntime,
    QueryRequest,
    QueryResult,
    Stage2RuntimeConfig,
)


class Catalog:
    def __init__(
        self,
        video_index: list[int],
        n: list[int],
        original: list[int] | None = None,
    ) -> None:
        self.video_index = np.asarray(video_index, dtype=np.int32)
        self.n = np.asarray(n, dtype=np.int32)
        self.original_idx = np.asarray(original or n, dtype=np.int64)
        self.video_table = [
            {"video_id": f"V{index}", "keyframe_prefix": f"V{index}"}
            for index in range(max(video_index) + 1)
        ]

    def map_row(self, row: int) -> dict[str, object]:
        video_id = self.video_table[int(self.video_index[row])]["video_id"]
        return {
            "global_row": row,
            "video_id": video_id,
            "n": int(self.n[row]),
            "original_frame_idx": int(self.original_idx[row]),
            "keyframe_relative_path": f"{video_id}/{int(self.n[row]):03d}.jpg",
        }


def _brute_force(scores: np.ndarray, distance_lambda: float) -> tuple[float, tuple[int, ...]]:
    best: tuple[float, tuple[int, ...]] | None = None
    for positions in combinations(range(scores.shape[1]), scores.shape[0]):
        score = sum(float(scores[event, position]) for event, position in enumerate(positions))
        score -= distance_lambda * sum(
            right - left for left, right in zip(positions[:-1], positions[1:], strict=True)
        )
        candidate = (score, positions)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best


def test_dante_recurrence_matches_brute_force_and_backtracks_strictly() -> None:
    scores = np.asarray(
        [[0.2, 0.8, 0.1, 0.4], [0.5, 0.2, 0.9, 0.3], [0.1, 0.2, 0.4, 0.95]],
        dtype=np.float32,
    )
    expected_score, expected_positions = _brute_force(scores, 0.001)
    result = dante_monotonic_dp(scores, 0.001)
    assert result is not None
    assert np.isclose(result.score, expected_score)
    assert result.positions == expected_positions
    assert all(
        left < right
        for left, right in zip(result.positions[:-1], result.positions[1:], strict=False)
    )


def test_dante_lambda_zero_is_valid_monotonic_dp() -> None:
    scores = np.asarray([[0.9, 0.1, 0.2], [0.3, 0.8, 0.4]], dtype=np.float32)
    result = dante_monotonic_dp(scores, 0.0)
    assert result is not None
    assert result.positions == (0, 1)
    assert np.isclose(result.score, 1.7)


def test_temporal_solver_changes_reversed_independent_winner() -> None:
    catalog = Catalog([0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3])
    groups = build_video_row_groups(catalog)
    scores = np.asarray(
        [
            [0.1, 0.2, 0.95, 0.80, 0.2, 0.1],
            [0.95, 0.2, 0.1, 0.1, 0.2, 0.80],
        ],
        dtype=np.float32,
    )
    unordered = rank_unordered_event_max(scores, ["E1", "E2"], groups, catalog)
    dante = rank_dante_dp(scores, ["E1", "E2"], groups, catalog, distance_lambda=0.001)
    assert unordered[0]["video_id"] == "V0"
    assert unordered[0]["independent_argmax_order_is_monotonic"] is False
    assert dante[0]["video_id"] == "V1"
    assert dante[0]["strictly_increasing_positions"] is True


def test_score_all_reproduces_stage1_topk_with_stable_ties() -> None:
    vectors = np.asarray([[1, 0], [1, 0], [0.5, 0.5], [0, 1]], dtype=np.float32)
    backend = NumPyMemmapExactIndex(
        vectors, np.linalg.norm(vectors, axis=1).astype(np.float32), chunk_rows=2
    )
    query = np.asarray([[1, 0]], dtype=np.float32)
    all_scores = backend.score_all(query)
    expected = np.lexsort((np.arange(len(all_scores)), -all_scores))[:3]
    top_scores, top_rows = backend.search(query, 3)
    assert np.array_equal(top_rows[0], expected)
    assert np.array_equal(top_scores[0], all_scores[expected])


def test_duplicate_submission_frames_remain_distinct_technical_positions() -> None:
    catalog = Catalog([0, 0], [1, 2], [100, 100])
    groups = build_video_row_groups(catalog)
    scores = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    ranked = rank_dante_dp(scores, ["E1", "E2"], groups, catalog, distance_lambda=0.0)
    chain = ranked[0]["chain"]
    assert [item["global_row"] for item in chain] == [0, 1]
    assert [item["original_frame_idx"] for item in chain] == [100, 100]


def test_temporal_order_uses_catalog_position_not_original_frame_idx() -> None:
    catalog = Catalog([0, 0], [10, 20], [900, 100])
    groups = build_video_row_groups(catalog)
    scores = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    ranked = rank_dante_dp(scores, ["E1", "E2"], groups, catalog, distance_lambda=0.0)
    assert [item["catalog_position"] for item in ranked[0]["chain"]] == [0, 1]
    assert [item["original_frame_idx"] for item in ranked[0]["chain"]] == [900, 100]


class Encoder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self.texts.extend(texts)
        output = np.zeros((len(texts), 512), dtype=np.float32)
        output[:, 0] = 1.0
        return output


class Translator:
    def __init__(self) -> None:
        self.loads = 0
        self.calls = 0

    def load(self) -> None:
        self.loads += 1

    def translate(self, texts: list[str]) -> list[dict[str, object]]:
        self.calls += 1
        return [
            {
                "translated_text_for_clip": f"translated:{text}",
                "translation_latency_ms": 1.0,
            }
            for text in texts
        ]


def _encoding_runtime(tmp_path: Path) -> tuple[OperationalRetrievalRuntime, Encoder, Translator]:
    config = Stage2RuntimeConfig(
        tmp_path / "s1",
        tmp_path / "s1b",
        tmp_path / "s1e",
        tmp_path / "clip",
        tmp_path / "opus",
        tmp_path / "output",
        tmp_path / "stage1d.yaml",
    )
    encoder, translator = Encoder(), Translator()
    runtime = OperationalRetrievalRuntime(config, translator_factory=lambda *_: translator)
    runtime.loaded = True
    runtime.encoder = encoder
    runtime.preflight = {"stage1b_candidate_id": "verified"}
    runtime.inputs = {
        "translator_asset": {"model_root": tmp_path},
        "translator_config": object(),
        "generation_config": object(),
    }
    return runtime, encoder, translator


def test_en_event_encoding_does_not_invoke_translator(tmp_path: Path) -> None:
    runtime, encoder, translator = _encoding_runtime(tmp_path)
    encoded = runtime.encode_requests([QueryRequest("q_E1", "A Green Field", "en", 1)])
    assert translator.loads == translator.calls == 0
    assert encoder.texts == ["A Green Field"]
    assert encoded.encodings[0]["translation_applied"] is False


def test_vi_event_encoding_uses_frozen_translator_path(tmp_path: Path) -> None:
    runtime, encoder, translator = _encoding_runtime(tmp_path)
    encoded = runtime.encode_requests([QueryRequest("q_E1", "một cánh đồng xanh", "vi", 1)])
    assert translator.loads == translator.calls == 1
    assert encoder.texts == ["translated:một cánh đồng xanh"]
    assert encoded.encodings[0]["translation_applied"] is True


class FakeBackend:
    size = 4

    def score_many_all(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8]], dtype=np.float32)


class FakeRuntime:
    def __init__(self) -> None:
        self.backend = FakeBackend()
        self.catalog = Catalog([0, 0, 1, 1], [1, 2, 1, 2])
        self.preflight = {"stage1_index_fingerprint": "frozen"}
        self.requests: list[QueryRequest] = []

    def load(self) -> FakeRuntime:
        return self

    def search_one(self, request: QueryRequest) -> QueryResult:
        self.requests.append(request)
        frames = [
            {
                "rank": rank,
                **self.catalog.map_row(row),
                "score": 1.0 / rank,
                "query_id": request.query_id,
            }
            for rank, row in enumerate(range(4), start=1)
        ]
        videos = [
            {"video_rank": 1, "video_id": "V0"},
            {"video_rank": 2, "video_id": "V1"},
        ]
        return QueryResult(request.query_id, {}, {}, {}, frames, videos, {}, Path("unused"))

    def encode_requests(self, requests: list[QueryRequest]) -> EncodedQueryBatch:
        embeddings = np.zeros((len(requests), 512), dtype=np.float32)
        embeddings[:, 0] = 1.0
        resolutions = tuple(
            type("Resolution", (), {"as_dict": lambda self: {"requested_language": "en"}})()
            for _ in requests
        )
        encodings = tuple(
            {
                "original_query_text": request.text,
                "translated_text": None,
                "clip_input_text": request.text,
                "translation_applied": False,
            }
            for request in requests
        )
        return EncodedQueryBatch(
            embeddings, resolutions, encodings, tuple({} for _ in requests), 1.0
        )

    def runtime_manifest(self) -> dict[str, object]:
        return {"ranking_policy": "FROZEN_STAGE1A_EXACT_COSINE_NO_RERANKING"}


def test_runner_keeps_whole_query_as_stage2a_control(tmp_path: Path, monkeypatch) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(
        rt1_runner,
        "render_rt1_visuals",
        lambda *args, **kwargs: (
            {},
            {"METHOD_A": "DANTE_DP", "METHOD_B": "UNORDERED_EVENT_MAX"},
            [],
        ),
    )
    query = RT1Query(
        "rt1_test",
        "en",
        "first event followed by second event",
        (RT1Event("E1", "first event"), RT1Event("E2", "second event")),
        "project_probe",
    )
    stage2 = Stage2RuntimeConfig(
        *(tmp_path / name for name in ("s1", "s1b", "s1e", "clip", "opus", "runtime", "s1d"))
    )
    output = tmp_path / "rt1"
    (output / "_stage2_control").mkdir(parents=True)
    run_reference_rt1(
        RT1RunnerConfig(stage2, tmp_path, tmp_path / "suite.jsonl", output, RT1Settings()),
        [query],
        runtime=runtime,  # type: ignore[arg-type]
    )
    assert len(runtime.requests) == 1
    assert runtime.requests[0].text == query.narrative_text
    written = [
        json.loads(line)
        for line in (output / "queries/rt1_test/whole_query/ranked_frames.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["global_row"] for item in written] == [0, 1, 2, 3]


def test_dante_scaling_sanity_is_linear_reference() -> None:
    scores = np.random.default_rng(2026).normal(size=(4, 20_000)).astype(np.float32)
    started = time.monotonic()
    result = dante_monotonic_dp(scores, 0.001)
    assert result is not None
    assert time.monotonic() - started < 2.5
    source = inspect.getsource(dante_monotonic_dp)
    assert "for predecessor" not in source and "for tau" not in source


def test_experiment_has_no_forbidden_retrieval_modules() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/experiments/reference_rt1").glob("*.py")
    ).lower()
    for forbidden in ("triage_eg.event_graph", "siglip", "beit", "nllb", "faiss"):
        assert forbidden not in source


def test_visual_outputs_and_blinded_mapping_use_canonical_keyframes(
    tmp_path: Path,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    dataset = tmp_path / "dataset"
    catalog = Catalog([0, 0, 1, 1], [1, 2, 1, 2])
    for row in range(4):
        mapped = catalog.map_row(row)
        path = dataset / str(mapped["keyframe_relative_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        image_module.new("RGB", (80, 60), (row * 40, 20, 100)).save(path)
    groups = build_video_row_groups(catalog)
    scores = np.asarray([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8]])
    unordered = rank_unordered_event_max(scores, ["E1", "E2"], groups, catalog)
    dante = rank_dante_dp(scores, ["E1", "E2"], groups, catalog, distance_lambda=0.001)
    whole = [{"rank": rank, **catalog.map_row(row)} for rank, row in enumerate(range(4), start=1)]
    paths, mapping, issues = render_rt1_visuals(
        tmp_path / "output",
        dataset_root=dataset,
        query_id="q",
        whole_frames=whole,
        unordered=unordered,
        dante=dante,
        top_k=2,
        review_seed=2026,
    )
    assert not issues
    assert set(mapping.values()) == {"UNORDERED_EVENT_MAX", "DANTE_DP"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_bundle_excludes_models_index_raw_media_and_cache(tmp_path: Path) -> None:
    root = tmp_path / "rt1"
    write_json(root / "run_manifest.json", {})
    write_json(root / "experiment_summary.json", {})
    write_jsonl(root / "query_suite/reference_rt1_queries.jsonl", [{"query_id": "q"}])
    write_json(root / "visuals/review_key.json", {})
    write_jsonl(root / "issues.jsonl", [])
    write_jsonl(root / "queries/q/dante_dp/ranked_videos.jsonl", [])
    (root / "visuals/q").mkdir(parents=True)
    (root / "visuals/q/ab_temporal_comparison.jpg").write_bytes(b"small-review")
    (root / "_stage2_control").mkdir()
    np.save(root / "_stage2_control/index.npy", np.ones(2))
    (root / "raw.mp4").write_bytes(b"raw")
    archive = create_rt1_bundle(root, tmp_path / "bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "visuals/q/ab_temporal_comparison.jpg" in names
    assert not any(name.endswith((".npy", ".pt", ".bin", ".mp4")) for name in names)
