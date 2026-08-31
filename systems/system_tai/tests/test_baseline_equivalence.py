"""
Baseline Equivalence Test
=========================
Verifies that when all experimental feature flags are False (default state),
the system maintains exact equivalence to commit 5d9db1d.
"""

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

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


def _create_mock_ablation_artifacts(
    base_dir: Path,
    runs: list[str],
    mutate_translation_run: str | None = None,
    missing_artifact_run: str | None = None,
    bad_query_id_run: str | None = None,
):
    import json
    query_order = ["p1-1", "p1-2", "p1-4", "p1-5", "p1-6"]
    for rk in runs:
        run_dir = base_dir / f"run_{rk}"
        req_dir = run_dir / "requests"
        req_dir.mkdir(parents=True, exist_ok=True)
        for q_short in query_order:
            if rk == missing_artifact_run and q_short == "p1-4":
                continue
            qid = f"query-{q_short}-kis"
            file_qid = "query-p1-1-wrong" if (rk == bad_query_id_run and q_short == "p1-1") else qid

            q_req_dir = req_dir / f"audit-top100-{q_short}-123456"
            q_req_dir.mkdir(parents=True, exist_ok=True)
            cand_file = q_req_dir / "candidates.json"

            text_suffix = " mutated" if rk == mutate_translation_run else ""
            units = [
                {
                    "segments": [
                        {"variant_id": f"{qid}::semantic_01", "text": f"English query for {q_short}{text_suffix}", "weight": 1.0},
                        {"variant_id": f"{qid}::semantic_02", "text": f"Primary scene for {q_short}", "weight": 1.0},
                    ]
                }
            ]
            cdata = {
                "query_id": file_qid,
                "request_id": f"audit-top100-{q_short}-123456",
                "translation": {"units": units},
                "records": [
                    {
                        "query_id": file_qid,
                        "rank": 1,
                        "video_id": "L30_V046" if q_short == "p1-1" else "L21_V003",
                        "frame_id": 6784 if q_short == "p1-1" else 100,
                        "pts_time": 271.36 if q_short == "p1-1" else 10.0,
                        "fusion_score": 0.95,
                    }
                ],
            }
            cand_file.write_text(json.dumps(cdata, indent=2), encoding="utf-8")


def _get_runner_module():
    import sys
    import importlib.util
    if "runner_module" in sys.modules:
        return sys.modules["runner_module"]
    runner_path = REPO_ROOT / "scratch" / "run_kaggle_v2a_causal_closure.py"
    spec = importlib.util.spec_from_file_location("runner_module", runner_path)
    assert spec and spec.loader
    runner_mod = importlib.util.module_from_spec(spec)
    sys.modules["runner_module"] = runner_mod
    spec.loader.exec_module(runner_mod)
    return runner_mod


def test_ablation_summary_table_generation_success():
    """Verify that generate_and_save_ablation_summary_table runs successfully and outputs valid summary matrix."""
    import tempfile
    import json
    from pathlib import Path

    runner_mod = _get_runner_module()

    with tempfile.TemporaryDirectory() as td:
        base_out = Path(td)
        runs = ["A", "B", "C"]
        _create_mock_ablation_artifacts(base_out, runs)

        ablation_configs = {
            "A": {"name": "Run A: Baseline", "restricted_frames_per_video_per_variant": 10},
            "B": {"name": "Run B: Depth", "restricted_frames_per_video_per_variant": 20},
            "C": {"name": "Run C: Diversity", "restricted_frames_per_video_per_variant": 20},
        }

        runner_mod.generate_and_save_ablation_summary_table(
            all_run_results={},
            base_out=base_out,
            runs_to_execute=runs,
            ablation_configs=ablation_configs,
        )

        sum_file = base_out / "ablation_matrix_summary.json"
        assert sum_file.exists()
        loaded = json.loads(sum_file.read_text(encoding="utf-8"))
        assert "generated_at" in loaded
        assert set(loaded["runs"].keys()) == {"A", "B", "C"}
        p1_1 = loaded["runs"]["A"]["queries"]["p1-1"]
        assert p1_1["valid_interval_hit"] is True
        assert p1_1["frame_evaluation_status"] == "VALID_MANUAL_INTERVAL_HIT"
        assert p1_1["first_human_video_rank"] == 1
        assert "source_assignment_breakdown" in p1_1
        assert "unique_candidates_with_diverse_source" in p1_1
        assert "unique_candidates_with_raw_source" in p1_1
        assert "multi_source_candidate_count" in p1_1

        # Unannotated query tri-state evaluation
        p1_2 = loaded["runs"]["A"]["queries"]["p1-2"]
        assert p1_2["valid_interval_hit"] is None
        assert p1_2["frame_evaluation_status"] == "NOT_EVALUABLE_NO_INTERVAL"


def test_ablation_summary_detects_translation_drift_and_raises():
    """Verify that translation drift across runs causes generate_and_save_ablation_summary_table to raise RuntimeError."""
    import tempfile
    import pytest
    from pathlib import Path

    runner_mod = _get_runner_module()

    with tempfile.TemporaryDirectory() as td:
        base_out = Path(td)
        runs = ["A", "B", "C"]
        _create_mock_ablation_artifacts(base_out, runs, mutate_translation_run="B")

        ablation_configs = {
            "A": {"name": "Run A: Baseline"},
            "B": {"name": "Run B: Depth"},
            "C": {"name": "Run C: Diversity"},
        }

        with pytest.raises(RuntimeError, match="Translation drift detected"):
            runner_mod.generate_and_save_ablation_summary_table(
                all_run_results={},
                base_out=base_out,
                runs_to_execute=runs,
                ablation_configs=ablation_configs,
            )


def test_ablation_summary_missing_artifact_raises():
    """Verify that a missing candidates.json artifact causes generate_and_save_ablation_summary_table to raise RuntimeError."""
    import tempfile
    import pytest
    from pathlib import Path

    runner_mod = _get_runner_module()

    with tempfile.TemporaryDirectory() as td:
        base_out = Path(td)
        runs = ["A", "B", "C"]
        _create_mock_ablation_artifacts(base_out, runs, missing_artifact_run="C")

        ablation_configs = {
            "A": {"name": "Run A: Baseline"},
            "B": {"name": "Run B: Depth"},
            "C": {"name": "Run C: Diversity"},
        }

        with pytest.raises(RuntimeError, match="Missing current candidate artifact"):
            runner_mod.generate_and_save_ablation_summary_table(
                all_run_results={},
                base_out=base_out,
                runs_to_execute=runs,
                ablation_configs=ablation_configs,
            )


def test_ablation_summary_mismatched_query_id_raises():
    """Verify that a mismatched query_id in candidates.json causes generate_and_save_ablation_summary_table to raise RuntimeError."""
    import tempfile
    import pytest
    from pathlib import Path

    runner_mod = _get_runner_module()

    with tempfile.TemporaryDirectory() as td:
        base_out = Path(td)
        runs = ["A", "B", "C"]
        _create_mock_ablation_artifacts(base_out, runs, bad_query_id_run="A")

        ablation_configs = {
            "A": {"name": "Run A: Baseline"},
            "B": {"name": "Run B: Depth"},
            "C": {"name": "Run C: Diversity"},
        }

        with pytest.raises(RuntimeError, match="Mismatched query_id"):
            runner_mod.generate_and_save_ablation_summary_table(
                all_run_results={},
                base_out=base_out,
                runs_to_execute=runs,
                ablation_configs=ablation_configs,
            )


def test_internal_rrf_depth_rescues_distinct_segment_beyond_rank_100():
    """Verify hypothesis: internal_rrf_candidate_depth=500 allows Segment Grouper to rescue

    a distinct segment candidate at rank 101 into Top 100, while depth=100 prunes it.
    """
    from system_tai.kis.video_first import (
        fuse_restricted_frames,
        FusedVideoEvidence,
        VariantVideoEvidence,
    )
    from system_tai.retrieval.video_evidence import (
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )
    from system_tai.retrieval.multi_query import (
        QueryVariant,
        QueryVariantType,
        QueryLanguage,
        WeightedRRFRetriever,
    )

    var = QueryVariant(
        variant_id="v1",
        text="exercising",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    weighted_rrf = WeightedRRFRetriever(object())

    # Create 120 hits in video L30_V046:
    # - Cluster A (Rank 1-75): frame 0..74
    # - Cluster B (Rank 76-100): frame 1000..1024
    # - Target Segment C (Rank 101): frame 2000
    # - Additional filler (Rank 102-120): frame 2001..2019
    hits = []
    # Cluster A: 75 frames (0..74)
    for i in range(75):
        hits.append(RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=i,
            clip_row=i,
            keyframe_order=i + 1,
            pts_time=float(i) / 25.0,
            cosine_score=0.99 - (i * 0.001),
            rank=i + 1,
            selection_source="RAW",
            raw_local_rank=i + 1,
        ))
    # Cluster B: 25 frames (1000..1024)
    for i in range(25):
        hits.append(RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=1000 + i,
            clip_row=75 + i,
            keyframe_order=76 + i,
            pts_time=float(1000 + i) / 25.0,
            cosine_score=0.85 - (i * 0.001),
            rank=76 + i,
            selection_source="RAW",
            raw_local_rank=76 + i,
        ))
    # Target Candidate (Rank 101): frame 2000 (Distinct segment > 75 frame gap)
    hits.append(RestrictedFrameHit(
        video_id="L30_V046",
        frame_id=2000,
        clip_row=100,
        keyframe_order=101,
        pts_time=80.0,
        cosine_score=0.80,
        rank=101,
        selection_source="DIVERSE",
        raw_local_rank=101,
    ))
    # Fillers: rank 102..120
    for i in range(19):
        hits.append(RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=2001 + i,
            clip_row=101 + i,
            keyframe_order=102 + i,
            pts_time=80.0 + float(i + 1) / 25.0,
            cosine_score=0.79 - (i * 0.001),
            rank=102 + i,
            selection_source="DIVERSE",
            raw_local_rank=102 + i,
        ))

    restricted_outcome = VideoRestrictedSearchOutcome(
        rankings={"v1": {"L30_V046": tuple(hits)}},
        physical_rows_scored=len(hits),
        video_store_scan_count=1,
    )
    selected_videos = (
        FusedVideoEvidence(
            video_id="L30_V046",
            rank=1,
            fusion_score=0.99,
            variant_hit_count=1,
            primary_coverage_count=1,
            best_individual_rank=1,
            per_variant=(
                VariantVideoEvidence(
                    variant_id="v1",
                    weight=1.0,
                    video_rank=1,
                    maximum_frame_id=0,
                    maximum_clip_row=0,
                    maximum_cosine_score=0.99,
                ),
            ),
        ),
    )

    # 1. Run with depth 100: Frame 2000 is PRUNED_BY_INTERNAL_RRF_CUTOFF
    res_100 = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=100,
        rrf_constant=60.0,
        internal_rrf_candidate_depth=100,
    )
    assert len(res_100.ranked_candidates) == 100
    frame_ids_100 = {c.frame_id for c in res_100.ranked_candidates}
    assert 2000 not in frame_ids_100, "Frame 2000 must be pruned by RRF cutoff when depth=100"

    # 2. Run with depth 500: Frame 2000 reaches Grouper, classified PRIMARY, promoted into Top 100!
    res_500 = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=100,
        rrf_constant=60.0,
        internal_rrf_candidate_depth=500,
    )
    assert len(res_500.ranked_candidates) == 100
    frame_ids_500 = {c.frame_id for c in res_500.ranked_candidates}
    assert 2000 in frame_ids_500, "Frame 2000 must be rescued into Top 100 by Segment Grouper when depth=500"

    cand_2000 = next(c for c in res_500.ranked_candidates if c.frame_id == 2000)
    assert cand_2000.rank <= 100


def test_collect_fusion_trace_does_not_change_output():
    """Verify invariance: collect_fusion_trace=True produces bit-for-bit identical candidates."""
    from system_tai.kis.video_first import (
        fuse_restricted_frames,
        FusedVideoEvidence,
        VariantVideoEvidence,
    )
    from system_tai.retrieval.video_evidence import (
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )
    from system_tai.retrieval.multi_query import (
        QueryVariant,
        QueryVariantType,
        QueryLanguage,
        WeightedRRFRetriever,
    )

    var = QueryVariant(
        variant_id="v1",
        text="test query",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    weighted_rrf = WeightedRRFRetriever(object())

    hits = [
        RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=i * 10,
            clip_row=i,
            keyframe_order=i + 1,
            pts_time=float(i * 10) / 25.0,
            cosine_score=0.9 - (i * 0.01),
            rank=i + 1,
            selection_source="RAW" if i % 2 == 0 else "DIVERSE",
            raw_local_rank=i + 1,
        )
        for i in range(50)
    ]
    restricted_outcome = VideoRestrictedSearchOutcome(
        rankings={"v1": {"L30_V046": tuple(hits)}},
        physical_rows_scored=len(hits),
        video_store_scan_count=1,
    )
    selected_videos = (
        FusedVideoEvidence(
            video_id="L30_V046",
            rank=1,
            fusion_score=0.9,
            variant_hit_count=1,
            primary_coverage_count=1,
            best_individual_rank=1,
            per_variant=(
                VariantVideoEvidence(
                    variant_id="v1",
                    weight=1.0,
                    video_rank=1,
                    maximum_frame_id=0,
                    maximum_clip_row=0,
                    maximum_cosine_score=0.9,
                ),
            ),
        ),
    )

    res_no_trace = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=20,
        rrf_constant=60.0,
        collect_fusion_trace=False,
    )

    res_with_trace, trace_map = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=20,
        rrf_constant=60.0,
        collect_fusion_trace=True,
        return_trace=True,
    )

    # Canonical projection comparison
    def to_canonical_projection(res):
        return [
            (c.video_id, c.frame_id, c.rank, c.score, c.source)
            for c in res.ranked_candidates
        ]

    assert to_canonical_projection(res_no_trace) == to_canonical_projection(res_with_trace)

    # Check trace map has 50 frame records + 1 summary record
    assert len(trace_map) == 51
    summary = trace_map[("__summary__", 0)]
    assert summary["fusion_trace_schema_version"] == "2.0.0"
    assert summary["output_top_k"] == 20
    assert summary["total_exported"] == 20

    target_info = trace_map[("L30_V046", 0)]
    assert target_info["untruncated_rrf_rank"] == 1
    assert target_info["group_bucket"] == "GROUP_BUCKET_PRIMARY"
    assert target_info["final_rank"] == 1
    assert target_info["final_lifecycle_status"] == "EXPORTED_AT_RANK_1"
    assert target_info["allocation_rejection_reason"] is None
    assert target_info["effective_cutoff_scope"] == "PRIMARY_BUCKET"


def test_allocation_diagnostics_score_cutoff_and_bucket_saturation():
    """Verify schema 2.0.0 allocation diagnostics: primary cutoff, score gap, and secondary saturation."""
    from system_tai.kis.video_first import (
        fuse_restricted_frames,
        FusedVideoEvidence,
        VariantVideoEvidence,
    )
    from system_tai.retrieval.video_evidence import (
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )
    from system_tai.retrieval.multi_query import (
        QueryVariant,
        QueryVariantType,
        QueryLanguage,
        WeightedRRFRetriever,
    )

    var = QueryVariant(
        variant_id="v1",
        text="query text",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    weighted_rrf = WeightedRRFRetriever(object())

    # Create 5 primary frames (gap >= 100) and 3 secondary frames (gap < 75 from frame 0)
    hits = []
    # Primary 1..5: frames 0, 100, 200, 300, 400
    for i, fid in enumerate([0, 100, 200, 300, 400], start=1):
        hits.append(RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=fid,
            clip_row=fid,
            keyframe_order=i,
            pts_time=float(fid) / 25.0,
            cosine_score=0.90 - (i * 0.01),
            rank=i,
            selection_source="RAW",
            raw_local_rank=i,
        ))
    # Secondary 1..3: frames 1, 2, 3 (close to frame 0)
    for i, fid in enumerate([1, 2, 3], start=6):
        hits.append(RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=fid,
            clip_row=fid,
            keyframe_order=i,
            pts_time=float(fid) / 25.0,
            cosine_score=0.80 - (i * 0.01),
            rank=i,
            selection_source="RAW",
            raw_local_rank=i,
        ))

    restricted_outcome = VideoRestrictedSearchOutcome(
        rankings={"v1": {"L30_V046": tuple(hits)}},
        physical_rows_scored=len(hits),
        video_store_scan_count=1,
    )
    selected_videos = (
        FusedVideoEvidence(
            video_id="L30_V046",
            rank=1,
            fusion_score=0.9,
            variant_hit_count=1,
            primary_coverage_count=1,
            best_individual_rank=1,
            per_variant=(
                VariantVideoEvidence(
                    variant_id="v1",
                    weight=1.0,
                    video_rank=1,
                    maximum_frame_id=0,
                    maximum_clip_row=0,
                    maximum_cosine_score=0.9,
                ),
            ),
        ),
    )

    # Test 1: output_top_k=3 (Primary candidates 1..3 exported, primary candidate 4 rejected by score, secondary saturated)
    res_top3, trace_top3 = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=3,
        rrf_constant=60.0,
        collect_fusion_trace=True,
        return_trace=True,
    )
    assert len(res_top3.ranked_candidates) == 3
    # Check primary candidate #4 (frame 300)
    p4 = trace_top3[("L30_V046", 300)]
    assert p4["group_bucket"] == "GROUP_BUCKET_PRIMARY"
    assert p4["pre_allocation_bucket_rank"] == 4
    assert p4["allocation_rejection_reason"] == "SCORE_BELOW_EFFECTIVE_CUTOFF"
    assert p4["effective_cutoff_scope"] == "PRIMARY_BUCKET"
    assert p4["effective_cutoff_candidate_key"] == "L30_V046::200"
    assert p4["score_gap_to_effective_cutoff"] > 0.0

    # Check secondary candidate (frame 1) - primary saturated output!
    s1 = trace_top3[("L30_V046", 1)]
    assert s1["group_bucket"] == "GROUP_BUCKET_SECONDARY"
    assert s1["allocation_rejection_reason"] == "PRIMARY_BUCKET_SATURATED_OUTPUT"
    assert s1["effective_cutoff_scope"] == "PRIMARY_BUCKET"
    assert s1["score_gap_to_effective_cutoff"] is None


def test_allocation_diagnostics_tie_break_resolution():
    """Verify tie-break reason recording when score matches cutoff exactly."""
    from system_tai.kis.video_first import (
        fuse_restricted_frames,
        FusedVideoEvidence,
        VariantVideoEvidence,
    )
    from system_tai.retrieval.video_evidence import (
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )
    from system_tai.retrieval.multi_query import (
        QueryVariant,
        QueryVariantType,
        QueryLanguage,
        WeightedRRFRetriever,
    )

    var1 = QueryVariant(
        variant_id="v1",
        text="query 1",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    var2 = QueryVariant(
        variant_id="v2",
        text="query 2",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    weighted_rrf = WeightedRRFRetriever(object())

    # Video A: rank 1 in v1 (cosine 0.95), rank 2 in v2 (cosine 0.85) -> RRF: 1/61 + 1/62
    # Video B: rank 2 in v1 (cosine 0.85), rank 1 in v2 (cosine 0.95) -> RRF: 1/62 + 1/61
    # Both have mathematically identical final score!
    hits_a_v1 = [RestrictedFrameHit(video_id="L10_V001", frame_id=100, clip_row=1, keyframe_order=1, pts_time=4.0, cosine_score=0.95, rank=1, selection_source="RAW", raw_local_rank=1)]
    hits_a_v2 = [RestrictedFrameHit(video_id="L10_V001", frame_id=100, clip_row=1, keyframe_order=1, pts_time=4.0, cosine_score=0.85, rank=2, selection_source="RAW", raw_local_rank=1)]

    hits_b_v1 = [RestrictedFrameHit(video_id="L20_V002", frame_id=200, clip_row=2, keyframe_order=1, pts_time=8.0, cosine_score=0.85, rank=2, selection_source="RAW", raw_local_rank=1)]
    hits_b_v2 = [RestrictedFrameHit(video_id="L20_V002", frame_id=200, clip_row=2, keyframe_order=1, pts_time=8.0, cosine_score=0.95, rank=1, selection_source="RAW", raw_local_rank=1)]

    restricted_outcome = VideoRestrictedSearchOutcome(
        rankings={
            "v1": {"L10_V001": tuple(hits_a_v1), "L20_V002": tuple(hits_b_v1)},
            "v2": {"L10_V001": tuple(hits_a_v2), "L20_V002": tuple(hits_b_v2)},
        },
        physical_rows_scored=4,
        video_store_scan_count=4,
    )
    selected_videos = (
        FusedVideoEvidence(
            video_id="L10_V001",
            rank=1,
            fusion_score=0.9,
            variant_hit_count=2,
            primary_coverage_count=2,
            best_individual_rank=1,
            per_variant=(
                VariantVideoEvidence(variant_id="v1", weight=1.0, video_rank=1, maximum_frame_id=100, maximum_clip_row=1, maximum_cosine_score=0.95),
                VariantVideoEvidence(variant_id="v2", weight=1.0, video_rank=2, maximum_frame_id=100, maximum_clip_row=1, maximum_cosine_score=0.85),
            ),
        ),
        FusedVideoEvidence(
            video_id="L20_V002",
            rank=1,
            fusion_score=0.9,
            variant_hit_count=2,
            primary_coverage_count=2,
            best_individual_rank=1,
            per_variant=(
                VariantVideoEvidence(variant_id="v1", weight=1.0, video_rank=2, maximum_frame_id=200, maximum_clip_row=2, maximum_cosine_score=0.85),
                VariantVideoEvidence(variant_id="v2", weight=1.0, video_rank=1, maximum_frame_id=200, maximum_clip_row=2, maximum_cosine_score=0.95),
            ),
        ),
    )

    res, trace = fuse_restricted_frames(
        query_id="q1",
        variants=(var1, var2),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=1,
        rrf_constant=60.0,
        collect_fusion_trace=True,
        return_trace=True,
    )

    assert len(res.ranked_candidates) == 1
    assert res.ranked_candidates[0].video_id == "L10_V001"

    cand_b = trace[("L20_V002", 200)]
    assert cand_b["allocation_rejection_reason"] == "TIE_BREAK_REJECTED"
    assert cand_b["score_gap_to_effective_cutoff"] == 0.0
    assert cand_b["tie_break_reason"] is not None
    assert "TIE_ON_SCORE" in cand_b["tie_break_reason"]
    assert "L10_V001::100" in cand_b["tie_break_reason"]


def test_candidate_pruned_before_allocation_has_null_allocation_reason():
    """Verify candidates pruned by internal RRF cutoff have allocation_rejection_reason=None."""
    from system_tai.kis.video_first import (
        fuse_restricted_frames,
        FusedVideoEvidence,
        VariantVideoEvidence,
    )
    from system_tai.retrieval.video_evidence import (
        VideoRestrictedSearchOutcome,
        RestrictedFrameHit,
    )
    from system_tai.retrieval.multi_query import (
        QueryVariant,
        QueryVariantType,
        QueryLanguage,
        WeightedRRFRetriever,
    )

    var = QueryVariant(
        variant_id="v1",
        text="query text",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    weighted_rrf = WeightedRRFRetriever(object())

    hits = [
        RestrictedFrameHit(
            video_id="L30_V046",
            frame_id=i * 100,
            clip_row=i,
            keyframe_order=i + 1,
            pts_time=float(i * 100) / 25.0,
            cosine_score=0.99 - (i * 0.01),
            rank=i + 1,
            selection_source="RAW",
            raw_local_rank=i + 1,
        )
        for i in range(5)
    ]
    restricted_outcome = VideoRestrictedSearchOutcome(
        rankings={"v1": {"L30_V046": tuple(hits)}},
        physical_rows_scored=5,
        video_store_scan_count=1,
    )
    selected_videos = (
        FusedVideoEvidence(
            video_id="L30_V046",
            rank=1,
            fusion_score=0.99,
            variant_hit_count=1,
            primary_coverage_count=1,
            best_individual_rank=1,
            per_variant=(
                VariantVideoEvidence(
                    variant_id="v1",
                    weight=1.0,
                    video_rank=1,
                    maximum_frame_id=0,
                    maximum_clip_row=0,
                    maximum_cosine_score=0.99,
                ),
            ),
        ),
    )

    # Set internal_rrf_candidate_depth=2. Frames 0, 100 pass, frames 200, 300, 400 are pruned before allocation.
    res, trace = fuse_restricted_frames(
        query_id="q1",
        variants=(var,),
        restricted=restricted_outcome,
        selected_videos=selected_videos,
        weighted_rrf=weighted_rrf,
        output_top_k=2,
        rrf_constant=60.0,
        internal_rrf_candidate_depth=2,
        collect_fusion_trace=True,
        return_trace=True,
    )

    pruned_cand = trace[("L30_V046", 400)]
    assert pruned_cand["rrf_cutoff_status"] == "PRUNED_BY_INTERNAL_RRF_CUTOFF"
    assert pruned_cand["group_bucket"] is None
    assert pruned_cand["final_selection_score"] is None
    assert pruned_cand["allocation_rejection_reason"] is None
    assert pruned_cand["score_gap_to_effective_cutoff"] is None
    assert pruned_cand["final_lifecycle_status"] == "PRUNED_BY_INTERNAL_RRF_CUTOFF"






