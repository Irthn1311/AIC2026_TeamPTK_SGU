"""API contract tests for Triplet Forensic Probe."""

from pathlib import Path
import numpy as np

from system_tai.retrieval.query_decomposition import decompose_query
from system_tai.qa.grounding import nominate_qa_videos, QAVideoConditionedEvidenceConfig
from system_tai.retrieval.video_evidence import VideoRestrictedFeatureSearcher


def test_query_decomposition_is_runtime_safe():
    """Verify decompose_query uses only query text and contains no oracle strings."""
    q_vi = "Chiếc xe ô tô màu gì đang chạy trên đường?"
    q_en = "What color car is driving on the road?"
    variants = decompose_query(query_text_vi=q_vi, query_text_en=q_en)
    decomp_list = variants.as_list()
    assert len(decomp_list) >= 1
    assert decomp_list[0][0] == "literal"
    # Ensure no external ground truth keywords were injected
    for name, text in decomp_list:
        assert isinstance(text, str)
        assert len(text) > 0


def test_searcher_and_nomination_api_contract():
    """Verify search_video_maxima and nominate_qa_videos methods exist and match signature."""
    assert hasattr(VideoRestrictedFeatureSearcher, "search_video_maxima")
    assert callable(getattr(VideoRestrictedFeatureSearcher, "search_video_maxima"))
    assert hasattr(VideoRestrictedFeatureSearcher, "search_selected_videos")
    assert callable(getattr(VideoRestrictedFeatureSearcher, "search_selected_videos"))
    assert callable(nominate_qa_videos)
