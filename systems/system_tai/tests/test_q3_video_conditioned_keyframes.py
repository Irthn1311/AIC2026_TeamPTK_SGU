from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from system_tai.common.schemas import (
    CandidateFrame,
    FrameMappingRecord,
    KISResult,
    VideoFeatureStore,
)
from system_tai.data.corpus_discovery import CorpusManifest, DiscoveredVideo, _fingerprint
from system_tai.features.btc_clip_store import FeatureStoreRegistry, LoadedVideoFeatureStore
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.retrieval.video_restricted import (
    VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
    VideoConditionedKeyframeConfig,
    VideoConditionedKeyframeDiversity,
)


def _store(
    video_id: str,
    rows: list[tuple[int, float, tuple[float, float]]],
) -> LoadedVideoFeatureStore:
    mappings = tuple(
        FrameMappingRecord(
            clip_row=index,
            keyframe_order=index + 1,
            frame_id=frame_id,
            pts_time=pts_time,
            fps=30.0,
        )
        for index, (frame_id, pts_time, _vector) in enumerate(rows)
    )
    return LoadedVideoFeatureStore(
        descriptor=VideoFeatureStore(
            video_id=video_id,
            mapping_csv_path=Path(f"{video_id}.csv"),
            clip_npy_path=Path(f"{video_id}.npy"),
            row_count=len(rows),
            embedding_dimension=2,
            normalized=False,
        ),
        matrix=np.asarray([vector for _frame, _pts, vector in rows], dtype=np.float32),
        mappings=mappings,
    )


def _candidate(
    video_id: str,
    frame_id: int,
    clip_row: int,
    rank: int,
) -> CandidateFrame:
    return CandidateFrame(
        video_id=video_id,
        frame_id=frame_id,
        clip_row=clip_row,
        keyframe_order=clip_row + 1,
        score=1.0 / (60.0 + rank),
        rank=rank,
        source="weighted_rrf",
        diagnostic_metadata={"fusion_score": 1.0 / (60.0 + rank)},
    )


def test_disabled_path_and_default_bounds_are_unchanged() -> None:
    registry = FeatureStoreRegistry([_store("A", [(100, 0.0, (1.0, 0.0))])])
    original = KISResult("q", (_candidate("A", 100, 0, 1),))
    config = VideoConditionedKeyframeConfig()

    assert config.selected_video_global_rank_cap == 50
    assert config.max_selected_videos == 50
    outcome = VideoConditionedKeyframeDiversity(registry).condition(
        global_result=original,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=config,
    )

    assert outcome.result is original
    assert outcome.trace["enabled"] is False
    assert outcome.restricted_keyframe_rows_scored == 0


def test_selection_uses_first_occurrence_rank_50_and_respects_video_cap() -> None:
    a_rows = [(frame, float(frame), (1.0, 0.0)) for frame in range(1, 50)]
    registry = FeatureStoreRegistry(
        [
            _store("A", a_rows),
            _store("B", [(500, 50.0, (1.0, 0.0))]),
            _store("C", [(510, 51.0, (1.0, 0.0))]),
        ]
    )
    candidates = [
        _candidate("A", frame, frame - 1, rank)
        for rank, frame in enumerate(range(1, 50), start=1)
    ]
    candidates.extend((_candidate("B", 500, 0, 50), _candidate("C", 510, 0, 51)))
    result = KISResult("q", tuple(candidates))

    default_outcome = VideoConditionedKeyframeDiversity(registry).condition(
        global_result=result,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=VideoConditionedKeyframeConfig(enabled=True),
    )
    capped_outcome = VideoConditionedKeyframeDiversity(registry).condition(
        global_result=result,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=VideoConditionedKeyframeConfig(enabled=True, max_selected_videos=1),
    )

    assert default_outcome.trace["selected_video_ids"] == ["A", "B"]
    assert capped_outcome.trace["selected_video_ids"] == ["A"]


def test_restricted_ranking_nms_and_slot_substitution_are_deterministic() -> None:
    store_a = _store(
        "A",
        [
            (100, 100.0, (0.90, 0.10)),
            (200, 101.0, (0.80, 0.20)),
            (300, 102.0, (1.00, 0.00)),
            (300, 102.5, (1.00, 0.00)),
            (400, 104.0, (0.999, 0.001)),
            (500, 107.0, (0.99, 0.01)),
            (600, 112.0, (0.98, 0.02)),
            (700, 103.0, (0.70, 0.30)),
        ],
    )
    registry = FeatureStoreRegistry(
        [store_a, _store("B", [(10, 0.0, (0.1, 0.9))])]
    )
    original = KISResult(
        "q",
        (
            _candidate("A", 100, 0, 1),
            _candidate("B", 10, 0, 2),
            _candidate("A", 200, 1, 3),
            _candidate("A", 700, 7, 4),
        ),
    )
    conditioner = VideoConditionedKeyframeDiversity(registry)
    config = VideoConditionedKeyframeConfig(enabled=True)

    first = conditioner.condition(
        global_result=original,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=config,
        protected_prefix_rank=0,
    )
    second = conditioner.condition(
        global_result=original,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=config,
        protected_prefix_rank=0,
    )

    assert first.result == second.result
    assert [item.rank for item in first.result.ranked_candidates] == [1, 2, 3, 4]
    assert [item.video_id for item in first.result.ranked_candidates] == ["A", "B", "A", "A"]
    assert first.result.ranked_candidates[0] == original.ranked_candidates[0]
    assert [item.frame_id for item in first.result.ranked_candidates] == [100, 10, 300, 500]
    assert first.result.ranked_candidates[2].clip_row == 2
    assert first.result.ranked_candidates[2].frame_id == 300
    assert first.result.ranked_candidates[2].score == original.ranked_candidates[2].score
    assert first.result.ranked_candidates[2].diagnostic_metadata == {
        "fusion_score": original.ranked_candidates[2].score,
        "q3_policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
        "score_semantics": "ORIGINAL_GLOBAL_SLOT_SCORE",
        "original_global_rank": 3,
        "original_frame_id": 200,
        "restricted_cosine_score": 1.0,
        "restricted_rank": 1,
        "anchor_pts_time": 102.0,
        "anchor_index": 1,
    }
    identities = [(item.video_id, item.frame_id) for item in first.result.ranked_candidates]
    assert len(identities) == len(set(identities))
    video_trace = first.trace["videos"][0]
    assert [anchor["frame_id"] for anchor in video_trace["anchors"]] == [300, 500, 600]
    assert video_trace["anchors"][0]["pts_time"] - 100.0 < 5.0
    assert 400 not in [anchor["frame_id"] for anchor in video_trace["anchors"]]
    assert not {100, 200, 700}.intersection(
        anchor["frame_id"] for anchor in video_trace["anchors"]
    )
    assert video_trace["uninserted_anchor_count"] == 1


def test_refinement_prefix_protects_top3_but_allows_rank4_substitution() -> None:
    registry = FeatureStoreRegistry(
        [
            _store(
                "A",
                [
                    (100, 0.0, (0.90, 0.10)),
                    (200, 1.0, (0.80, 0.20)),
                    (300, 6.0, (1.00, 0.00)),
                    (700, 12.0, (0.70, 0.30)),
                ],
            ),
            _store("B", [(10, 0.0, (0.10, 0.90))]),
        ]
    )
    original = KISResult(
        "q",
        (
            _candidate("A", 100, 0, 1),
            _candidate("A", 200, 1, 2),
            _candidate("B", 10, 0, 3),
            _candidate("A", 700, 3, 4),
        ),
    )

    outcome = VideoConditionedKeyframeDiversity(registry).condition(
        global_result=original,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=VideoConditionedKeyframeConfig(enabled=True),
        protected_prefix_rank=3,
    )

    conditioned = outcome.result.ranked_candidates
    assert conditioned[:3] == original.ranked_candidates[:3]
    assert conditioned[3].frame_id == 300
    assert conditioned[3].frame_id != original.ranked_candidates[3].frame_id
    assert [item.rank for item in conditioned] == [1, 2, 3, 4]
    assert [item.video_id for item in conditioned] == ["A", "A", "B", "A"]
    assert len({(item.video_id, item.frame_id) for item in conditioned}) == 4
    assert outcome.trace["protected_prefix_rank"] == 3
    assert outcome.trace["protected_replacement_slot_count"] == 1
    assert outcome.trace["videos"][0]["available_replacement_slot_count"] == 1
    assert outcome.trace["videos"][0]["protected_replacement_slot_count"] == 1


def test_insufficient_same_video_slots_never_steals_another_video_slot() -> None:
    registry = FeatureStoreRegistry(
        [
            _store("A", [(100, 0.0, (0.9, 0.1)), (300, 5.0, (1.0, 0.0))]),
            _store("B", [(10, 0.0, (0.1, 0.9))]),
        ]
    )
    original = KISResult(
        "q",
        (_candidate("A", 100, 0, 1), _candidate("B", 10, 0, 2)),
    )
    outcome = VideoConditionedKeyframeDiversity(registry).condition(
        global_result=original,
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        config=VideoConditionedKeyframeConfig(enabled=True),
    )

    assert outcome.result == original
    assert outcome.substitution_count == 0
    assert outcome.selected_videos_with_no_replacement_capacity == 2
    assert outcome.total_same_video_replacement_slots == 0
    assert outcome.trace["videos"][0]["uninserted_anchor_count"] == 1


class _Encoder:
    dimension = 2
    identifiers = {"model": "ViT-B/32", "device": "cpu"}

    def encode(self, _text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_images(self, images: list[Any], *, batch_size: int = 32) -> np.ndarray:
        raise AssertionError((images, batch_size))


class _UnusedDecoder:
    def close(self) -> None:
        pass


def test_operational_runtime_exports_global_conditioned_and_trace_artifacts(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "A.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,0,30,0\n"
        "2,1,30,10\n"
        "3,2,30,20\n"
        "4,10,30,100\n",
        encoding="utf-8",
    )
    clip = tmp_path / "A.npy"
    np.save(
        clip,
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32),
    )
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    (keyframes / "1.jpg").touch()
    raw_video = tmp_path / "A.mp4"
    raw_video.touch()
    discovered = DiscoveredVideo(
        "A",
        mapping,
        clip,
        keyframes,
        raw_video,
        1,
        4,
        mapping.stat().st_size,
        clip.stat().st_size,
        4,
    )
    manifest = CorpusManifest(tmp_path, tmp_path, _fingerprint((discovered,)), (discovered,))
    manifest_path = tmp_path / "feature_manifest.json"
    manifest.write(manifest_path)
    output = tmp_path / "out"
    runtime = OperationalKISRuntime.bootstrap(
        SessionConfig(
            input_root=tmp_path,
            reuse_manifest=manifest_path,
            output_root=output,
            device="cpu",
            video_conditioned_keyframe_config=VideoConditionedKeyframeConfig(enabled=True),
        ),
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(
            path, expected_dimension=2
        ),
        encoder_factory=lambda **_kwargs: _Encoder(),
        decoder_factory=_UnusedDecoder,
    )
    try:
        response = runtime.handle_query(
            QueryRequest(
                "request-1",
                "q",
                "nguồn",
                query_en="english",
                include_vi_variant=False,
                top_k_per_variant=3,
                output_top_k=3,
                refine_top_n=0,
            )
        )
    finally:
        runtime.close()

    artifacts = response["artifacts"]
    assert "global_top100_jsonl" in artifacts
    assert "video_conditioned_keyframe_trace_json" in artifacts
    global_records = [
        json.loads(line)
        for line in (output / artifacts["global_top100_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    conditioned_records = [
        json.loads(line)
        for line in (output / artifacts["top100_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    trace = json.loads(
        (output / artifacts["video_conditioned_keyframe_trace_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert [row["frame_id"] for row in global_records] == [0, 10, 20]
    assert [row["frame_id"] for row in conditioned_records] == [0, 100, 20]
    assert [row["video_id"] for row in conditioned_records] == [
        row["video_id"] for row in global_records
    ]
    assert [row["rank"] for row in conditioned_records] == [1, 2, 3]
    assert response["retrieval_valid"] is True
    assert response["timings"]["q3_enabled"] is True
    assert trace["substitution_count"] == 1
    assert trace["protected_prefix_rank"] == 0
    assert "ground_truth" not in json.dumps(trace).casefold()
