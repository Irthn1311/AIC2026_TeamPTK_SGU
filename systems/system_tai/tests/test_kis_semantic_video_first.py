from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
from system_tai.data.corpus_discovery import (
    CorpusManifest,
    DiscoveredVideo,
    _fingerprint,
)
from system_tai.features.btc_clip_store import (
    FeatureStoreRegistry,
    LoadedVideoFeatureStore,
)
from system_tai.kis.session import build_parser, session_config_from_args
from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import QueryRequest, SessionConfig
from system_tai.kis.video_first import (
    KISVideoFirstConfig,
    build_kis_video_first_outcome,
    fuse_video_maxima,
)
from system_tai.refinement.video import DecodedFrame, DecodeResult, VideoProbe
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)
from system_tai.retrieval.semantic_query import (
    SemanticQueryConfig,
    SemanticUnitRole,
    compile_vietnamese_semantic_query,
    decompose_vietnamese_semantic_units,
)
from system_tai.retrieval.video_evidence import (
    FullCorpusVideoMaximaOutcome,
    VideoMaximumHit,
    VideoRestrictedFeatureSearcher,
)
from system_tai.retrieval.video_restricted import VideoConditionedKeyframeConfig


class _Provider:
    provider_name = "vinai-translate:fake@pinned"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def translate_many(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        self.calls.append(tuple(texts))
        translations = {
            texts[0]: "full visual description",
            texts[1]: "people exercise in a line touching their toes",
            texts[2]: "one person with glasses and three red hats",
        }
        return tuple(translations[text] for text in texts)


class _Guard:
    @staticmethod
    def split_for_clip(text: str) -> tuple[str, ...]:
        return (text,)

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split()) + 2


def _variant(variant_id: str, *, weight: float = 1.0) -> QueryVariant:
    return QueryVariant(
        variant_id=variant_id,
        text=variant_id,
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=weight,
    )


def _maximum(
    variant_id: str,
    video_id: str,
    rank: int,
    frame_id: int,
    cosine: float,
) -> VideoMaximumHit:
    return VideoMaximumHit(
        query_id=variant_id,
        video_id=video_id,
        frame_id=frame_id,
        clip_row=rank - 1,
        keyframe_order=rank,
        cosine_score=cosine,
        rank=rank,
    )


def _store(
    video_id: str,
    rows: list[tuple[int, tuple[float, float]]],
) -> LoadedVideoFeatureStore:
    mappings = tuple(
        FrameMappingRecord(
            clip_row=index,
            keyframe_order=index + 1,
            frame_id=frame_id,
            pts_time=frame_id / 10.0,
            fps=10.0,
        )
        for index, (frame_id, _vector) in enumerate(rows)
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
        matrix=np.asarray([vector for _frame, vector in rows], dtype=np.float32),
        mappings=mappings,
    )


def test_vietnamese_semantic_units_keep_full_query_and_soft_attributes() -> None:
    source = (
        "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
        "động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và "
        "ba người đội nón có màu đỏ."
    )
    units = decompose_vietnamese_semantic_units(query_id="p1-1", query_vi=source)

    assert [unit.role for unit in units] == [
        SemanticUnitRole.FULL_QUERY,
        SemanticUnitRole.PRIMARY_SCENE,
        SemanticUnitRole.SUPPORTING_ATTRIBUTE,
    ]
    assert units[0].text == source
    assert units[1].weight == 1.0
    assert units[2].weight == 0.35


def test_semantic_compiler_uses_one_vinai_batch_and_no_manual_english() -> None:
    provider = _Provider()
    source = (
        "Nhóm người xếp thành hàng đang tập thể dục. "
        "Trong nhóm chỉ có một người đeo kính và ba người đội nón màu đỏ."
    )
    compiled = compile_vietnamese_semantic_query(
        query_id="Q1",
        query_vi=source,
        provider=provider,
        token_budget_guard=_Guard(),
        config=SemanticQueryConfig(supporting_attribute_weight=0.25),
    )

    assert provider.calls == [tuple(unit.text for unit in compiled.units)]
    assert [item.query_variant.text for item in compiled.variants] == [
        "full visual description",
        "people exercise in a line touching their toes",
        "one person with glasses and three red hats",
    ]
    assert compiled.variants[-1].query_variant.weight == 0.25
    assert compiled.to_metadata()["was_truncated"] is False


def test_lossless_segments_share_unit_weight_instead_of_overweighting() -> None:
    class SplitGuard(_Guard):
        @staticmethod
        def split_for_clip(text: str) -> tuple[str, ...]:
            if text == "full visual description":
                return ("full part one", "full part two")
            return (text,)

    compiled = compile_vietnamese_semantic_query(
        query_id="Q1",
        query_vi=(
            "Nhóm người đang tập thể dục. "
            "Trong nhóm chỉ có một người đeo kính và ba người đội nón màu đỏ."
        ),
        provider=_Provider(),
        token_budget_guard=SplitGuard(),
    )
    full_segments = [
        item
        for item in compiled.variants
        if item.semantic_role is SemanticUnitRole.FULL_QUERY
    ]
    assert [item.query_variant.weight for item in full_segments] == [0.5, 0.5]


def test_video_level_rrf_accumulates_different_frames_without_cosine_mixing() -> None:
    variants = (_variant("scene-a"), _variant("scene-b"))
    maxima = FullCorpusVideoMaximaOutcome(
        rankings={
            "scene-a": (
                _maximum("scene-a", "A", 1, 10, -0.9),
                _maximum("scene-a", "B", 2, 30, 0.99),
            ),
            "scene-b": (
                _maximum("scene-b", "B", 1, 40, -0.8),
                _maximum("scene-b", "A", 2, 20, 0.98),
            ),
        },
        physical_rows_scored=4,
        video_store_scan_count=2,
    )
    selected = fuse_video_maxima(
        variants=variants,
        maxima=maxima,
        primary_variant_ids=frozenset({"scene-a", "scene-b"}),
        rrf_constant=60.0,
        nomination_depth=100,
        selected_video_cap=2,
    )

    assert [item.video_id for item in selected] == ["A", "B"]
    assert selected[0].rank == 1
    assert selected[0].fusion_score == selected[1].fusion_score
    assert {hit.maximum_frame_id for hit in selected[0].per_variant} == {10, 20}

    changed_cosines = FullCorpusVideoMaximaOutcome(
        rankings={
            "scene-a": (
                _maximum("scene-a", "A", 1, 10, 0.01),
                _maximum("scene-a", "B", 2, 30, -0.7),
            ),
            "scene-b": (
                _maximum("scene-b", "B", 1, 40, 0.02),
                _maximum("scene-b", "A", 2, 20, -0.6),
            ),
        },
        physical_rows_scored=4,
        video_store_scan_count=2,
    )
    repeated = fuse_video_maxima(
        variants=variants,
        maxima=changed_cosines,
        primary_variant_ids=frozenset({"scene-a", "scene-b"}),
        rrf_constant=60.0,
        nomination_depth=100,
        selected_video_cap=2,
    )
    assert [(item.video_id, item.fusion_score) for item in repeated] == [
        (item.video_id, item.fusion_score) for item in selected
    ]


def test_restricted_search_returns_unique_exact_frames_and_full_requested_depth() -> None:
    registry = FeatureStoreRegistry(
        [
            _store(
                "A",
                [
                    (0, (1.0, 0.0)),
                    (10, (0.9, 0.1)),
                    (20, (0.1, 0.9)),
                    (30, (0.0, 1.0)),
                ],
            ),
            _store("B", [(99, (0.7, 0.7))]),
        ]
    )
    searcher = VideoRestrictedFeatureSearcher(registry, chunk_size=2)
    variants = (_variant("x"), _variant("y"))
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    maxima = searcher.search_video_maxima(
        query_ids=("x", "y"),
        query_vectors=vectors,
    )
    selected = fuse_video_maxima(
        variants=variants,
        maxima=maxima,
        primary_variant_ids=frozenset({"x", "y"}),
        rrf_constant=60.0,
        nomination_depth=100,
        selected_video_cap=1,
    )
    assert [item.video_id for item in selected] == ["A"]
    restricted = searcher.search_selected_videos(
        video_ids=("A",),
        query_ids=("x", "y"),
        query_vectors=vectors,
        per_query_result_cap=4,
    )
    outcome = build_kis_video_first_outcome(
        query_id="Q",
        variants=variants,
        maxima=maxima,
        restricted=restricted,
        selected_videos=selected,
        weighted_rrf=WeightedRRFRetriever(object()),
        output_top_k=4,
        rrf_constant=60.0,
    )

    assert len(outcome.result.ranked_candidates) == 4
    assert [item.rank for item in outcome.result.ranked_candidates] == [1, 2, 3, 4]
    assert {item.frame_id for item in outcome.result.ranked_candidates} == {0, 10, 20, 30}
    assert all(item.video_id == "A" for item in outcome.result.ranked_candidates)
    assert all(item.frame_id != item.clip_row for item in outcome.result.ranked_candidates[1:])


def test_semantic_video_first_config_requires_vinai_and_cli_enables_it() -> None:
    with pytest.raises(ValueError, match="requires dynamic VinAI translation"):
        SessionConfig(kis_video_first_config=KISVideoFirstConfig(enabled=True))
    with pytest.raises(ValueError, match="requires vinai/vinai-translate"):
        SessionConfig(
            enable_dynamic_translation=True,
            translation_model_name="another/model",
            kis_video_first_config=KISVideoFirstConfig(enabled=True),
        )

    args = build_parser().parse_args(["--enable-kis-semantic-video-first"])
    config = session_config_from_args(args)
    assert config.enable_dynamic_translation is True
    assert config.kis_video_first_config.enabled is True


def test_cli_refine_top_n_zero_keeps_refinement_disabled_with_valid_template() -> None:
    args = build_parser().parse_args(
        ["--enable-kis-semantic-video-first", "--default-refine-top-n", "0"]
    )

    config = session_config_from_args(args)

    assert config.default_refine_top_n == 0
    assert config.refinement_config.top_candidates_to_refine == 1


def test_cli_wires_opt_in_semantic_multi_anchor_raw_refinement() -> None:
    args = build_parser().parse_args(
        [
            "--enable-kis-semantic-video-first",
            "--enable-kis-multi-anchor-refinement",
            "--default-refine-top-n",
            "5",
            "--kis-anchor-max-videos",
            "3",
            "--kis-anchors-per-video",
            "6",
            "--kis-max-extra-raw-anchors",
            "9",
        ]
    )

    config = session_config_from_args(args)

    assert config.video_conditioned_keyframe_config.enabled is True
    assert config.video_conditioned_keyframe_config.semantic_variant_coverage is True
    assert config.video_conditioned_keyframe_config.max_selected_videos == 3
    assert config.video_conditioned_keyframe_config.max_anchors_per_video == 6
    assert config.q3_anchor_refinement_config.enabled is True
    assert config.q3_anchor_refinement_config.max_extra_q3_anchors == 9


def test_cli_rejects_multi_anchor_without_video_first_or_raw_prefix() -> None:
    parser = build_parser()
    with pytest.raises(ValueError, match="requires --enable-kis-semantic-video-first"):
        session_config_from_args(
            parser.parse_args(["--enable-kis-multi-anchor-refinement"])
        )
    with pytest.raises(ValueError, match="default-refine-top-n greater than zero"):
        session_config_from_args(
            parser.parse_args(
                [
                    "--enable-kis-semantic-video-first",
                    "--enable-kis-multi-anchor-refinement",
                    "--default-refine-top-n",
                    "0",
                ]
            )
        )


class _Encoder:
    dimension = 2
    identifiers = {"model": "ViT-B/32", "device": "cpu"}

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "full visual description": (1.0, 0.0),
            "people exercise in a line touching their toes": (1.0, 0.0),
            "one person with glasses and three red hats": (0.0, 1.0),
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    def encode_images(self, images: list[Any], *, batch_size: int = 32) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _image in images], dtype=np.float32)


class _Decoder:
    backend_identifier = "fake"

    def probe(self, record: Any) -> VideoProbe:
        return VideoProbe(record.video_id, record.raw_video_path, "fake", 10.0, 4.0, 40, 1, 1)

    def decode(self, request: Any) -> DecodeResult:
        frames = tuple(
            DecodedFrame(frame_id, frame_id / request.probe.fps, frame_id)
            for frame_id in request.frame_ids
        )
        return DecodeResult(frames, len(frames), 0, 0, self.backend_identifier, ())

    def close(self) -> None:
        return None


def _make_manifest(tmp_path: Path) -> Path:
    videos: list[DiscoveredVideo] = []
    for video_id, vectors in (
        ("A", ((1.0, 0.0), (0.9, 0.1), (0.1, 0.9), (0.0, 1.0))),
        ("B", ((0.7, 0.7), (0.6, 0.8), (0.8, 0.6), (0.5, 0.5))),
    ):
        mapping = tmp_path / f"{video_id}.csv"
        mapping.write_text(
            "n,pts_time,fps,frame_idx\n"
            "1,0,10,0\n2,1,10,10\n3,2,10,20\n4,3,10,30\n",
            encoding="utf-8",
        )
        clip = tmp_path / f"{video_id}.npy"
        np.save(clip, np.asarray(vectors, dtype=np.float32))
        keyframes = tmp_path / f"keyframes_{video_id}"
        keyframes.mkdir()
        (keyframes / "1.jpg").touch()
        raw_video = tmp_path / f"{video_id}.mp4"
        raw_video.touch()
        videos.append(
            DiscoveredVideo(
                video_id,
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
        )
    manifest = CorpusManifest(tmp_path, tmp_path, _fingerprint(tuple(videos)), tuple(videos))
    path = tmp_path / "manifest.json"
    manifest.write(path)
    return path


def test_operational_session_uses_semantic_video_first_and_preserves_core_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import system_tai.translation.provider as translation_module

    provider = _Provider()

    class ProviderFactory:
        def __new__(cls, **_kwargs: Any) -> _Provider:
            return provider

    class GuardFactory(_Guard):
        def __init__(self, *, max_tokens: int) -> None:
            assert max_tokens == 75

    monkeypatch.setattr(translation_module, "VinAITranslateProvider", ProviderFactory)
    monkeypatch.setattr(translation_module, "TokenBudgetGuard", GuardFactory)

    manifest_path = _make_manifest(tmp_path)
    config = SessionConfig(
        input_root=tmp_path,
        reuse_manifest=manifest_path,
        output_root=tmp_path / "output",
        device="cpu",
        enable_dynamic_translation=True,
        default_refine_top_n=0,
        kis_video_first_config=KISVideoFirstConfig(
            enabled=True,
            selected_video_cap=1,
            restricted_frames_per_video_per_variant=4,
        ),
    )
    runtime = OperationalKISRuntime.bootstrap(
        config,
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(
            path,
            expected_dimension=2,
        ),
        encoder_factory=lambda **_kwargs: _Encoder(),
        decoder_factory=lambda: _Decoder(),
    )
    response = runtime.handle_query(
        QueryRequest(
            "request-1",
            "Q1",
            (
                "Nhóm người xếp thành hàng đang tập thể dục. "
                "Trong nhóm chỉ có một người đeo kính và ba người đội nón màu đỏ."
            ),
            output_top_k=4,
            refine_top_n=0,
        )
    )

    assert response["status"] == "SUCCESS"
    assert response["result_count"] == 4
    assert provider.calls and len(provider.calls) == 1
    assert response["timings"]["kis_video_first_enabled"] is True
    assert response["timings"]["video_first_full_corpus_store_scan_count"] == 2
    assert response["timings"]["video_first_restricted_store_scan_count"] == 1

    top100 = runtime.output_root / response["artifacts"]["top100_jsonl"]
    records = [json.loads(line) for line in top100.read_text(encoding="utf-8").splitlines()]
    assert all(set(record) == {"query_id", "rank", "video_id", "frame_id"} for record in records)
    assert [record["rank"] for record in records] == [1, 2, 3, 4]
    assert len({(record["video_id"], record["frame_id"]) for record in records}) == 4
    assert all(record["video_id"] == "A" for record in records)

    candidates = runtime.output_root / response["artifacts"]["candidates_json"]
    payload = json.loads(candidates.read_text(encoding="utf-8"))
    assert payload["translation"]["semantic_clause_compilation_enabled"] is True
    assert payload["translation"]["was_truncated"] is False
    assert payload["video_first"]["selected_video_count"] == 1
    assert payload["video_first"]["selected_videos"][0]["video_id"] == "A"


def test_manual_english_is_rejected_only_when_semantic_video_first_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import system_tai.translation.provider as translation_module

    monkeypatch.setattr(translation_module, "VinAITranslateProvider", lambda **_kw: _Provider())
    monkeypatch.setattr(translation_module, "TokenBudgetGuard", lambda **_kw: _Guard())
    manifest_path = _make_manifest(tmp_path)
    runtime = OperationalKISRuntime.bootstrap(
        SessionConfig(
            input_root=tmp_path,
            reuse_manifest=manifest_path,
            output_root=tmp_path / "out",
            device="cpu",
            enable_dynamic_translation=True,
            default_refine_top_n=0,
            kis_video_first_config=KISVideoFirstConfig(enabled=True),
        ),
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(
            path,
            expected_dimension=2,
        ),
        encoder_factory=lambda **_kwargs: _Encoder(),
        decoder_factory=lambda: _Decoder(),
    )
    with pytest.raises(ValueError, match="manual English variants are not allowed"):
        runtime.handle_query(
            QueryRequest(
                "request-manual-en",
                "Q1",
                "Nhóm người đang tập thể dục.",
                query_en="hard-coded English",
                refine_top_n=0,
            )
        )


def test_semantic_video_first_connects_existing_q3_and_raw_refinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import system_tai.translation.provider as translation_module

    monkeypatch.setattr(
        translation_module,
        "VinAITranslateProvider",
        lambda **_kwargs: _Provider(),
    )
    monkeypatch.setattr(
        translation_module,
        "TokenBudgetGuard",
        lambda **_kwargs: _Guard(),
    )
    manifest_path = _make_manifest(tmp_path)
    runtime = OperationalKISRuntime.bootstrap(
        SessionConfig(
            input_root=tmp_path,
            reuse_manifest=manifest_path,
            output_root=tmp_path / "out",
            device="cpu",
            enable_dynamic_translation=True,
            kis_video_first_config=KISVideoFirstConfig(
                enabled=True,
                selected_video_cap=1,
                restricted_frames_per_video_per_variant=4,
            ),
            video_conditioned_keyframe_config=VideoConditionedKeyframeConfig(
                enabled=True,
                max_selected_videos=1,
            ),
        ),
        registry_loader=lambda path: FeatureStoreRegistry.from_manifest(
            path,
            expected_dimension=2,
        ),
        encoder_factory=lambda **_kwargs: _Encoder(),
        decoder_factory=lambda: _Decoder(),
    )
    response = runtime.handle_query(
        QueryRequest(
            "request-q3-refine",
            "Q1",
            (
                "Nhóm người xếp thành hàng đang tập thể dục. "
                "Trong nhóm chỉ có một người đeo kính và ba người đội nón màu đỏ."
            ),
            output_top_k=4,
            refine_top_n=1,
        )
    )

    assert response["status"] == "SUCCESS"
    assert response["refinement_requested"] is True
    assert response["refinement_valid"] is True
    assert response["timings"]["q3_enabled"] is True
    assert "video_conditioned_keyframe_trace_json" in response["artifacts"]
    assert "refined_top100_jsonl" in response["artifacts"]


def test_disabled_config_preserves_existing_request_variants() -> None:
    request = QueryRequest(
        "request-old",
        "Q-old",
        "tiếng Việt",
        query_en="English",
        refine_top_n=0,
    )
    assert SessionConfig().kis_video_first_config.enabled is False
    assert [variant.text for variant in request.variants()] == ["tiếng Việt", "English"]
