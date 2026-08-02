import pytest

from triage_eg.common.schemas import CandidateFrame
from triage_eg.retrieval.fusion import reciprocal_rank_fusion


def candidate(uid: str, rank: int, branch: str) -> CandidateFrame:
    return CandidateFrame("v", rank, rank * 40, uid, 1 / rank, rank, branch)


def test_reciprocal_rank_fusion_combines_branches():
    fused = reciprocal_rank_fusion(
        {"visual": [candidate("a", 1, "visual")], "text": [candidate("a", 2, "text")]},
        weights={"visual": 2.0, "text": 1.0},
        k=10,
    )
    assert fused[0].frame_uid == "a"
    assert fused[0].score == pytest.approx(2 / 11 + 1 / 12)
    assert fused[0].source_branch == "rrf"
