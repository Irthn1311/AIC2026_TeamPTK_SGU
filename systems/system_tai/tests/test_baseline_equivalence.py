"""
Baseline Equivalence Test
=========================
Verifies that when all experimental feature flags are False (default state),
the system maintains exact equivalence to commit 5d9db1d.
"""

import pytest
from system_tai.kis.video_first import (
    KISVideoFirstConfig,
    fuse_restricted_frames,
    FusedVideoEvidence,
)
from system_tai.retrieval.video_evidence import (
    RestrictedFrameHit,
    VideoRestrictedSearchOutcome,
)
from system_tai.retrieval.semantic_query import (
    SemanticQueryConfig,
)
from system_tai.retrieval.multi_query import (
    QueryLanguage,
    QueryVariant,
    QueryVariantType,
    WeightedRRFRetriever,
)


def test_default_feature_flags_are_strictly_false():
    sq_cfg = SemanticQueryConfig()
    assert sq_cfg.enable_visual_content_filter is False

    vf_cfg = KISVideoFirstConfig()
    assert vf_cfg.enable_candidate_union is False
    assert vf_cfg.enable_score_normalization is False
    assert vf_cfg.enable_late_interaction is False
    assert vf_cfg.enable_positive_chain_bonus is False

    # Exact 5d9db1d hyperparameter defaults
    assert vf_cfg.video_nomination_depth == 100
    assert vf_cfg.restricted_frames_per_video_per_variant == 10
    assert vf_cfg.restricted_frame_min_gap == 0
    assert vf_cfg.top_m_evidence_cap == 3
    assert vf_cfg.top_m_min_frame_gap == 60
    assert vf_cfg.top_m_weights == (0.6, 0.3, 0.1)
    assert vf_cfg.coverage_threshold == 0.75
    assert vf_cfg.selected_video_cap == 32


def test_fuse_restricted_frames_equivalence():
    v1 = QueryVariant(variant_id="v1", text="query 1", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=1.0)
    v2 = QueryVariant(variant_id="v2", text="query 2", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=0.5)
    variants = (v1, v2)

    restricted_hits = {
        "v1": {
            "L30_V046": (
                RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=1, pts_time=97.0, cosine_score=0.45, rank=1),
                RestrictedFrameHit(video_id="L30_V046", frame_id=4865, clip_row=1, keyframe_order=2, pts_time=194.6, cosine_score=0.40, rank=2),
            )
        },
        "v2": {
            "L30_V046": (
                RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=1, pts_time=97.0, cosine_score=0.42, rank=1),
                RestrictedFrameHit(video_id="L30_V046", frame_id=4865, clip_row=1, keyframe_order=2, pts_time=194.6, cosine_score=0.38, rank=2),
            )
        },
    }

    restricted = VideoRestrictedSearchOutcome(
        rankings=restricted_hits,
        physical_rows_scored=4,
        video_store_scan_count=1,
    )

    selected_videos = (
        FusedVideoEvidence(
            video_id="L30_V046",
            rank=1,
            fusion_score=0.88,
            variant_hit_count=2,
            primary_coverage_count=2,
            best_individual_rank=1,
            per_variant=(),
        ),
    )

    weighted_rrf = WeightedRRFRetriever(object())

    result = fuse_restricted_frames(
        query_id="Q1",
        variants=variants,
        restricted=restricted,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=10,
        rrf_constant=60.0,
    )

    assert len(result.ranked_candidates) == 2
    # Because 2425 and 4865 are > 75 frames apart, both are retained in primary tier
    assert result.ranked_candidates[0].video_id == "L30_V046"
    assert result.ranked_candidates[0].frame_id == 2425
    assert result.ranked_candidates[1].frame_id == 4865
    assert "scores_by_variant" in result.ranked_candidates[0].diagnostic_metadata


def test_manual_reference_schema_and_tri_state_evaluator():
    import json
    from pathlib import Path
    
    ref_file = Path(__file__).resolve().parent.parent / "benchmarks" / "manual_kis_reference_v1.json"
    assert ref_file.exists(), "manual_kis_reference_v1.json must exist"
    
    data = json.loads(ref_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    queries = data["queries"]
    assert len(queries) == 5

    for q in queries:
        assert q["official_gt"] is None
        assert q["annotation_status"] in ("VIDEO_ONLY_VERIFIED", "FRAME_INTERVAL_VERIFIED")
        assert isinstance(q["human_verified_video_id"], str)
        assert len(q["human_verified_video_id"]) > 0

    p1_1 = next(q for q in queries if "p1-1" in q["query_id"])
    assert p1_1["annotation_status"] == "FRAME_INTERVAL_VERIFIED"
    assert p1_1["human_annotated_intervals"] == [[6600, 6850]]


def test_runner_py_compile_syntax():
    import py_compile
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    runner_script = repo_root / "scratch" / "run_kaggle_v2a_causal_closure.py"
    assert runner_script.exists(), "run_kaggle_v2a_causal_closure.py must exist"

    # Must compile cleanly without SyntaxError, IndentationError, or TabError
    py_compile.compile(str(runner_script), doraise=True)


def test_pts_time_and_variant_scores_preservation_through_rrf():
    """Verify that pts_time and scores_by_variant are strictly preserved across fusion and serialization."""
    from system_tai.common.schemas import CandidateFrame, KISResult

    v1 = QueryVariant(variant_id="vi_primary", text="nón bảo hiểm", language=QueryLanguage.VIETNAMESE, variant_type=QueryVariantType.VIETNAMESE_DIRECT, weight=1.0)
    v2 = QueryVariant(variant_id="en_translated", text="helmet", language=QueryLanguage.ENGLISH, variant_type=QueryVariantType.ENGLISH_TRANSLATION, weight=0.5)

    hit1 = CandidateFrame(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, score=0.45, rank=1, source="clip", diagnostic_metadata={"pts_time": 97.123})
    hit2 = CandidateFrame(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, score=0.42, rank=1, source="clip", diagnostic_metadata={"pts_time": 97.123})

    rrf = WeightedRRFRetriever(object())
    fused_res = rrf.fuse_rankings(
        query_id="Q-TEST",
        variants=(v1, v2),
        rankings={
            "vi_primary": KISResult(query_id="vi_primary", ranked_candidates=(hit1,)),
            "en_translated": KISResult(query_id="en_translated", ranked_candidates=(hit2,)),
        },
        output_top_k=10,
    )

    top_cand = fused_res.ranked_candidates[0]
    assert top_cand.diagnostic_metadata.get("pts_time") == 97.123, "pts_time must be preserved through WeightedRRFRetriever.fuse_rankings"

    # Now verify fuse_restricted_frames preserves pts_time and produces scores_by_variant
    restricted = VideoRestrictedSearchOutcome(
        rankings={
            "vi_primary": {"L30_V046": (RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, pts_time=97.123, cosine_score=0.45, rank=1),)},
            "en_translated": {"L30_V046": (RestrictedFrameHit(video_id="L30_V046", frame_id=2425, clip_row=0, keyframe_order=5, pts_time=97.123, cosine_score=0.42, rank=1),)},
        },
        physical_rows_scored=1,
        video_store_scan_count=1,
    )
    sel_vids = (FusedVideoEvidence(video_id="L30_V046", rank=1, fusion_score=0.9, variant_hit_count=2, primary_coverage_count=2, best_individual_rank=1, per_variant=()),)

    final_res = fuse_restricted_frames(
        query_id="Q-TEST",
        variants=(v1, v2),
        restricted=restricted,
        selected_videos=sel_vids,
        weighted_rrf=rrf,
        output_top_k=10,
        rrf_constant=60.0,
    )

    final_cand = final_res.ranked_candidates[0]
    diag = final_cand.diagnostic_metadata
    assert diag.get("pts_time") == 97.123, "pts_time must be preserved in final CandidateFrame"
    assert "scores_by_variant" in diag
    assert diag["scores_by_variant"] == {"vi_primary": 0.45, "en_translated": 0.42}


def test_coverage_audit_decoupled_branches():
    """Verify that run_gt_index_coverage_audit runs without NameError and separates legacy vs human reference."""
    import importlib.util
    from pathlib import Path
    from system_tai.common.schemas import FrameMappingRecord

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    runner_script = repo_root / "scratch" / "run_kaggle_v2a_causal_closure.py"
    tmp_path = repo_root / "scratch" / "test_tmp"

    spec = importlib.util.spec_from_file_location("runner_module", str(runner_script))
    runner_mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["runner_module"] = runner_mod
    try:
        spec.loader.exec_module(runner_mod)
    except Exception:
        sys.modules.pop("runner_module", None)
        raise

    # Mock Feature Store & Registry
    class MockStore:
        def __init__(self, vid, min_f, max_f, pts_min, pts_max):
            self.mappings = [
                FrameMappingRecord(clip_row=0, keyframe_order=1, frame_id=min_f, pts_time=pts_min, fps=25.0),
                FrameMappingRecord(clip_row=1, keyframe_order=2, frame_id=max_f, pts_time=pts_max, fps=25.0),
            ]

    stores = {
        "L30_V046": MockStore("L30_V046", 2400, 6784, 96.0, 271.36),
        "L21_V003": MockStore("L21_V003", 1000, 2500, 40.0, 100.0),
        "L29_V018": MockStore("L29_V018", 6050, 7000, 242.0, 280.0),
        "L22_V021": MockStore("L22_V021", 1000, 18000, 40.0, 720.0),
        "L28_V012": MockStore("L28_V012", 1000, 2000, 40.0, 80.0),
        "L26_V035": MockStore("L26_V035", 100, 6000, 4.0, 240.0),
        "L30_V021": MockStore("L30_V021", 100, 4000, 4.0, 160.0),
        "L22_V023": MockStore("L22_V023", 100, 5000, 4.0, 200.0),
    }

    class MockRegistry:
        def get(self, vid):
            if vid in stores:
                return stores[vid]
            raise KeyError(vid)

    class MockSearcher:
        registry = MockRegistry()

    class MockRuntime:
        video_restricted_searcher = MockSearcher()

    summary = runner_mod.run_gt_index_coverage_audit(MockRuntime(), tmp_path)

    assert "p1-1" in summary
    assert "p1-2" in summary

    # P1-1 has verified interval 264-274s, MockStore for L30_V046 has pts 271.36 -> PASS
    p1_1 = summary["p1-1"]
    assert p1_1["human_reference"]["video_id"] == "L30_V046"
    assert p1_1["human_reference"]["interval_status"] == "PASS"

    # P1-2 has distinct human (L21_V003) vs legacy (L29_V018)
    p1_2 = summary["p1-2"]
    assert p1_2["human_reference"]["video_id"] == "L21_V003"
    assert p1_2["legacy"]["video_id"] == "L29_V018"
    assert p1_2["human_reference"]["interval_status"] == "NOT_EVALUABLE"
    assert p1_2["legacy"]["coverage_pass"] is True
    assert "parity_passed" in p1_2["legacy"]
    assert "source_info" in p1_2["legacy"]
    assert "parity_passed" in p1_2["human_reference"]
    assert "source_info" in p1_2["human_reference"]

    # Test final summary table printer works without error
    runner_mod.print_final_summary_table(summary)


def test_simulate_b0_equal_budget_hybrid_logic():
    """Verify that the hybrid candidate selection logic guarantees exact equal budget allocation."""
    import numpy as np

    n_frames = 40
    timestamps = [10.0 + i * 0.2 for i in range(10)] + [20.0 + i * 3.0 for i in range(30)]
    scores = np.linspace(0.9, 0.1, n_frames)

    for total_budget in [15, 20, 25, 30]:
        sorted_rows = [int(r) for r in np.argsort(-scores)]
        raw_top10 = sorted_rows[:min(10, len(sorted_rows))]
        selected_pts = [timestamps[r] for r in raw_top10]
        selected_rows = list(raw_top10)

        for r in sorted_rows[10:]:
            if len(selected_rows) >= total_budget:
                break
            pts = timestamps[r]
            if not any(abs(pts - p) < 5.0 for p in selected_pts):
                selected_rows.append(r)
                selected_pts.append(pts)

        if len(selected_rows) < total_budget:
            selected_set = set(selected_rows)
            for r in sorted_rows:
                if r not in selected_set:
                    selected_rows.append(r)
                    selected_set.add(r)
                if len(selected_rows) >= total_budget:
                    break

        assert len(selected_rows) == min(total_budget, n_frames), f"Expected {total_budget} rows, got {len(selected_rows)}"
        assert len(set(selected_rows)) == len(selected_rows), "Duplicates detected in candidate selection"


def test_raw_and_hybrid_use_equal_nominal_budget_with_compulsory_extras():
    """Verify that Run B (raw K=20) and Run C (hybrid K=20) maintain equal nominal budgets, with compulsory as extras."""
    from pathlib import Path
    import numpy as np
    from system_tai.common.schemas import FrameMappingRecord, VideoFeatureStore
    from system_tai.features.btc_clip_store import LoadedVideoFeatureStore
    from system_tai.retrieval.video_evidence import (
        _UnrankedFrameHit,
        rank_store_frames,
        select_hybrid_candidates,
    )

    n_frames = 50
    mappings = [
        FrameMappingRecord(
            clip_row=i,
            keyframe_order=i,
            frame_id=1000 + i,
            pts_time=10.0 + (i * 0.2 if i < 10 else 20.0 + (i - 10) * 3.0),
            fps=25.0,
        )
        for i in range(n_frames)
    ]
    mat = np.zeros((n_frames, 512), dtype=np.float32)
    for i in range(n_frames):
        mat[i, 0] = 0.95 - i * 0.01
        mat[i, 1] = np.sqrt(max(0.0, 1.0 - mat[i, 0] ** 2))
    query_vec = np.zeros((1, 512), dtype=np.float32)
    query_vec[0, 0] = 1.0

    desc = VideoFeatureStore(
        video_id="L30_V046",
        mapping_csv_path=Path("dummy.csv"),
        clip_npy_path=Path("dummy.npy"),
        row_count=n_frames,
        embedding_dimension=512,
        normalized=True,
    )
    store = LoadedVideoFeatureStore(descriptor=desc, mappings=tuple(mappings), matrix=mat)

    # Compulsory frames from DP chain: frame 1000 (already in raw top 10), frame 1045 (at index 45, far out)
    compulsory = [1000, 1045]

    # 1. Run B: Depth Only Raw K=20
    raw_rankings, raw_telemetry = rank_store_frames(
        store,
        query_ids=("q1",),
        query_vectors=query_vec,
        expected_dimension=512,
        chunk_size=64,
        per_query_cap=20,
        enable_temporal_diversity=False,
        compulsory_frame_ids=compulsory,
        return_telemetry=True,
    )
    raw_hits = raw_rankings["q1"]
    raw_tel = raw_telemetry["q1"]

    # 2. Run C: Diversity at Equal Budget K=20, Gap 5s
    hybrid_rankings, hybrid_telemetry = rank_store_frames(
        store,
        query_ids=("q1",),
        query_vectors=query_vec,
        expected_dimension=512,
        chunk_size=64,
        per_query_cap=20,
        enable_temporal_diversity=True,
        temporal_diversity_gap_seconds=5.0,
        raw_top_k=10,
        compulsory_frame_ids=compulsory,
        return_telemetry=True,
    )
    hybrid_hits = hybrid_rankings["q1"]
    hybrid_tel = hybrid_telemetry["q1"]

    # Assert Equal Budget Contract
    assert raw_tel["nominal_budget"] == 20 == hybrid_tel["nominal_budget"]
    assert raw_tel["normal_candidate_count"] == 20 == hybrid_tel["normal_candidate_count"]
    assert raw_tel["compulsory_extra_count"] == 1 == hybrid_tel["compulsory_extra_count"]
    assert raw_tel["effective_candidate_count"] == 21 == hybrid_tel["effective_candidate_count"]
    assert len(raw_hits) == 21 == len(hybrid_hits)

    # In Run C, the 21st item is the compulsory frame 1045
    assert hybrid_hits[-1].frame_id == 1045
    assert hybrid_hits[-1].selection_source == "TEMPORAL_CHAIN_COMPULSORY"
    assert any(h.selection_source == "DIVERSE" for h in hybrid_hits)


def test_selection_by_variant_survives_rrf_and_candidate_serialization():
    """Verify that selection_by_variant survives RRF fusion and is preserved through real OperationalKISRuntime candidate JSON serialization."""
    import json
    import tempfile
    from pathlib import Path
    import numpy as np
    from unittest.mock import MagicMock
    from system_tai.kis.session_schema import (
        KISVideoFirstConfig,
        QueryRequest,
        SessionConfig,
    )
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.retrieval.multi_query import WeightedRRFRetriever
    from system_tai.retrieval.video_evidence import (
        FullCorpusVideoMaximaOutcome,
        VideoMaximumHit,
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )

    hit_v1 = RestrictedFrameHit(
        video_id="L30_V046",
        frame_id=6784,
        clip_row=50,
        keyframe_order=50,
        pts_time=271.36,
        cosine_score=0.25,
        rank=1,
        selection_source="DIVERSE",
        raw_local_rank=28,
    )

    def make_mock_runtime(is_exp: bool, out_dir: Path):
        mock_searcher = MagicMock()
        def fake_search_video_maxima(*, query_ids, query_vectors, **kwargs):
            return FullCorpusVideoMaximaOutcome(
                rankings={qid: (VideoMaximumHit(query_id=qid, video_id="L30_V046", frame_id=6784, clip_row=50, keyframe_order=50, cosine_score=0.9, rank=1),) for qid in query_ids},
                physical_rows_scored=10,
                video_store_scan_count=1,
            )
        mock_searcher.search_video_maxima.side_effect = fake_search_video_maxima

        def fake_search_selected_videos(*, video_ids, query_ids, query_vectors, **kwargs):
            return VideoRestrictedSearchOutcome(
                rankings={qid: {"L30_V046": (hit_v1,)} for qid in query_ids},
                physical_rows_scored=10,
                video_store_scan_count=1,
                candidate_selection_telemetry={qid: {"L30_V046": {"RAW": 10, "DIVERSE": 10}} for qid in query_ids},
            )
        mock_searcher.search_selected_videos.side_effect = fake_search_selected_videos

        cfg = SessionConfig(
            input_root=out_dir,
            output_root=out_dir,
            enable_dynamic_translation=True,
            kis_video_first_config=KISVideoFirstConfig(
                enabled=True,
                restricted_frames_per_video_per_variant=20 if is_exp else 10,
                enable_temporal_diverse_local_candidates=is_exp,
                temporal_diversity_gap_seconds=5.0,
                enable_vi_localization_variant=False,
            ),
        )

        mock_trans = MagicMock()
        mock_trans.provider_name = "mock_vinai"
        mock_trans.translate_many.return_value = ("A woman wearing a red jacket walking in a park.",)
        mock_trans.translate.return_value = "A woman wearing a red jacket walking in a park."
        mock_guard = MagicMock()
        mock_guard.split_for_clip.side_effect = lambda text: (text,)
        mock_guard.count_tokens.return_value = 15
        mock_guard.max_tokens = 75

        mock_encoder = MagicMock()
        mock_encoder.dimension = 512
        mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
        mock_encoder.encode_texts.return_value = np.ones((1, 512), dtype=np.float32)

        mock_registry = MagicMock()
        mock_registry.embedding_dimension = 512
        mock_registry.total_rows = 10
        mock_registry.stores = ()
        mock_registry.keys.return_value = ("L30_V046",)
        mock_registry.get_store.return_value = None

        mock_manifest = MagicMock()
        mock_manifest.identifiers = {}
        mock_manifest.dataset_root = out_dir
        mock_manifest.schema_version = "v1"
        mock_manifest.fingerprint = "dummy_fp"
        mock_manifest.videos = ()

        mock_raw_registry = MagicMock()
        mock_raw_registry.records = ()
        video_rec = MagicMock()
        video_rec.raw_video_path = str(out_dir / "video.mp4")
        video_rec.fps = 25.0
        video_rec.total_frames = 1000
        video_rec.duration_seconds = 40.0
        video_rec.codec = "h264"
        video_rec.width = 1280
        video_rec.height = 720
        video_rec.frame_index_base = 0
        mock_raw_registry.get.return_value = video_rec

        runtime = OperationalKISRuntime(
            config=cfg,
            manifest_path=out_dir / "manifest.json",
            manifest=mock_manifest,
            registry=mock_registry,
            raw_video_registry=mock_raw_registry,
            shared_encoder=mock_encoder,
            decoder=MagicMock(),
            translation_provider=mock_trans,
            token_budget_guard=mock_guard,
        )
        runtime.video_restricted_searcher = mock_searcher
        runtime.weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]
        return runtime

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # 1. Test Serialization with Feature Enabled (Run C / Experimental)
        out_dir_exp = tmp_path / "exp_run"
        out_dir_exp.mkdir()
        runtime_exp = make_mock_runtime(is_exp=True, out_dir=out_dir_exp)
        req_exp = QueryRequest(
            request_id="req_test_exp",
            query_id="query_p1_1",
            query_vi="Một người phụ nữ mặc áo khoác đỏ",
        )
        outcome_exp = runtime_exp.handle_query(req_exp)
        matches_exp = sorted((out_dir_exp / "requests").glob("*/candidates.json"))
        assert matches_exp
        cand_file_exp = matches_exp[-1]
        loaded_exp = json.loads(cand_file_exp.read_text(encoding="utf-8"))
        assert loaded_exp["query_id"] == "query_p1_1"
        assert "candidate_selection_telemetry" in loaded_exp
        rec_exp = loaded_exp["records"][0]
        sel_entry = list(rec_exp["selection_by_variant"].values())[0]
        assert sel_entry["source"] == "DIVERSE"
        assert sel_entry["raw_local_rank"] == 28

        # 2. Test Serialization with Feature Disabled (Run A Baseline Fidelity)
        out_dir_base = tmp_path / "base_run"
        out_dir_base.mkdir()
        runtime_base = make_mock_runtime(is_exp=False, out_dir=out_dir_base)
        req_base = QueryRequest(
            request_id="req_test_base",
            query_id="query_p1_1",
            query_vi="Một người phụ nữ mặc áo khoác đỏ",
        )
        outcome_base = runtime_base.handle_query(req_base)
        matches_base = sorted((out_dir_base / "requests").glob("*/candidates.json"))
        assert matches_base
        cand_file_base = matches_base[-1]
        loaded_base = json.loads(cand_file_base.read_text(encoding="utf-8"))
        assert loaded_base["query_id"] == "query_p1_1"
        assert "candidate_selection_telemetry" not in loaded_base
        rec_base = loaded_base["records"][0]
        assert "selection_by_variant" not in rec_base


def test_vi_variant_absent_from_video_maxima_query_ids():
    """Verify runtime interaction: when enable_vi_localization_variant is True, VI variant is NEVER passed to search_video_maxima."""
    import tempfile
    from pathlib import Path
    import numpy as np
    from unittest.mock import MagicMock
    from system_tai.kis.session_schema import (
        KISVideoFirstConfig,
        QueryRequest,
        SessionConfig,
    )
    from system_tai.kis.session_engine import OperationalKISRuntime
    from system_tai.retrieval.multi_query import WeightedRRFRetriever
    from system_tai.retrieval.video_evidence import (
        FullCorpusVideoMaximaOutcome,
        VideoMaximumHit,
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        maxima_query_ids_captured = []
        restricted_query_ids_captured = []

        mock_searcher = MagicMock()
        def fake_search_video_maxima(*, query_ids, query_vectors, **kwargs):
            maxima_query_ids_captured.extend(query_ids)
            return FullCorpusVideoMaximaOutcome(
                rankings={qid: (VideoMaximumHit(query_id=qid, video_id="L30_V046", frame_id=1, clip_row=0, keyframe_order=0, cosine_score=0.9, rank=1),) for qid in query_ids},
                physical_rows_scored=10,
                video_store_scan_count=1,
            )
        mock_searcher.search_video_maxima.side_effect = fake_search_video_maxima

        def fake_search_selected_videos(*, video_ids, query_ids, query_vectors, **kwargs):
            restricted_query_ids_captured.extend(query_ids)
            return VideoRestrictedSearchOutcome(
                rankings={qid: {"L30_V046": (RestrictedFrameHit(video_id="L30_V046", frame_id=1, clip_row=0, keyframe_order=0, pts_time=1.0, cosine_score=0.9, rank=1),)} for qid in query_ids},
                physical_rows_scored=10,
                video_store_scan_count=1,
            )
        mock_searcher.search_selected_videos.side_effect = fake_search_selected_videos

        cfg = SessionConfig(
            input_root=tmp_path,
            output_root=tmp_path,
            enable_dynamic_translation=True,
            kis_video_first_config=KISVideoFirstConfig(
                enabled=True,
                enable_vi_localization_variant=True,
                vi_localization_weight=0.5,
            ),
        )

        mock_trans = MagicMock()
        mock_trans.provider_name = "mock_vinai"
        mock_trans.translate_many.side_effect = lambda texts: tuple(f"English translation for {t}" for t in texts)
        mock_trans.translate.return_value = "A woman wearing a red jacket walking in a park."
        mock_guard = MagicMock()
        mock_guard.split_for_clip.side_effect = lambda text: (text,)
        mock_guard.count_tokens.return_value = 15
        mock_guard.max_tokens = 75

        mock_encoder = MagicMock()
        mock_encoder.dimension = 512
        mock_encoder.identifiers = {"device": "cpu", "model": "ViT-B/32"}
        mock_encoder.encode_texts.side_effect = lambda texts: np.ones((len(texts), 512), dtype=np.float32)

        mock_registry = MagicMock()
        mock_registry.embedding_dimension = 512
        mock_registry.total_rows = 10
        mock_registry.stores = ()
        mock_registry.keys.return_value = ("L30_V046",)
        mock_registry.get_store.return_value = None

        mock_manifest = MagicMock()
        mock_manifest.identifiers = {}
        mock_manifest.dataset_root = tmp_path
        mock_manifest.schema_version = "v1"
        mock_manifest.fingerprint = "dummy_fp"
        mock_manifest.videos = ()

        mock_raw_registry = MagicMock()
        mock_raw_registry.records = ()
        video_rec = MagicMock()
        video_rec.raw_video_path = str(tmp_path / "video.mp4")
        video_rec.fps = 25.0
        video_rec.total_frames = 1000
        video_rec.duration_seconds = 40.0
        video_rec.codec = "h264"
        video_rec.width = 1280
        video_rec.height = 720
        video_rec.frame_index_base = 0
        mock_raw_registry.get.return_value = video_rec

        runtime = OperationalKISRuntime(
            config=cfg,
            manifest_path=tmp_path / "manifest.json",
            manifest=mock_manifest,
            registry=mock_registry,
            raw_video_registry=mock_raw_registry,
            shared_encoder=mock_encoder,
            decoder=MagicMock(),
            translation_provider=mock_trans,
            token_budget_guard=mock_guard,
        )
        runtime.video_restricted_searcher = mock_searcher
        runtime.weighted_rrf = WeightedRRFRetriever(exact_retriever=None)  # type: ignore[arg-type]

        req = QueryRequest(
            request_id="req_test_1",
            query_id="query_p1_1",
            query_vi="Một người phụ nữ mặc áo khoác đỏ trong công viên",
        )

        outcome = runtime.handle_query(req)

        # 1. Assert search_video_maxima contains ONLY English nomination variants
        assert len(maxima_query_ids_captured) > 0
        assert not any("vi_local_query" in qid for qid in maxima_query_ids_captured), (
            f"VI query leaked into search_video_maxima: {maxima_query_ids_captured}"
        )

        # 2. Assert search_selected_videos DOES contain vi_local_query
        assert len(restricted_query_ids_captured) > 0
        assert any("vi_local_query" in qid for qid in restricted_query_ids_captured), (
            f"VI query missing from search_selected_videos: {restricted_query_ids_captured}"
        )





