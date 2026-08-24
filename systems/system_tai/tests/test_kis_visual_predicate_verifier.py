"""Focused tests for bounded structured KIS visual verification."""

from __future__ import annotations

import pytest

from system_tai.refinement.engine import (
    LocalFrameFusion,
    rank_visually_verified_timeline_frames,
)
from system_tai.refinement.models import (
    SelectedVideoVisualVerifierConfig,
    VisualVerifierExecutionMode,
)
from system_tai.refinement.visual_verifier import (
    HuggingFaceStructuredVisualVerifier,
    VisualPredicateScore,
    VisualVerificationError,
    VisualVerificationInput,
    VisualVerificationResult,
    parse_visual_verification_json,
)


def _verification_result(frame_id: int) -> VisualVerificationResult:
    return VisualVerificationResult(
        video_id="V",
        absolute_frame_id=frame_id,
        match_score=0.9,
        requirement_coverage=0.8,
        all_visible_requirements_satisfied=False,
        predicates=(VisualPredicateScore("action", 0.9, True, "visible"),),
        summary="synthetic",
    )


class _CandidateIsolationVerifier(HuggingFaceStructuredVisualVerifier):
    """Exercise the production retry loop without loading a real model."""

    def __init__(self) -> None:
        self._max_new_tokens = 384
        self._progress_callback = None
        self._last_failures = ()
        self._last_recovered_retries = ()
        self.calls: list[tuple[int, int, int]] = []

    def _verify_candidate(
        self,
        *,
        query_vi,
        query_en,
        candidate,
        images,
        max_new_tokens,
    ):
        del query_vi, query_en
        self.calls.append(
            (candidate.absolute_frame_id, len(images), max_new_tokens)
        )
        if candidate.absolute_frame_id == 20 and len(images) > 1:
            raise VisualVerificationError("primary parse failed")
        if candidate.absolute_frame_id == 30:
            raise VisualVerificationError("persistent parse failure")
        return _verification_result(candidate.absolute_frame_id)


def test_structured_visual_result_parses_without_provenance_leak() -> None:
    result = parse_visual_verification_json(
        """```json
        {
          "match_score": 0.92,
          "requirement_coverage": 0.8,
          "all_visible_requirements_satisfied": false,
          "predicates": [
            {
              "requirement": "more than five people",
              "score": 1.0,
              "visible": true,
              "evidence": "seven people are visible"
            },
            {
              "requirement": "three red hats",
              "score": 0.6,
              "visible": true,
              "evidence": "two hats are unambiguous"
            }
          ],
          "summary": "partial visible conjunction"
        }
        ```""",
        video_id="V",
        absolute_frame_id=123,
    )

    assert result.video_id == "V"
    assert result.absolute_frame_id == 123
    assert result.match_score == pytest.approx(0.92)
    assert result.all_visible_requirements_satisfied is False
    trace = result.to_trace()
    assert "image" not in str(trace).casefold()
    assert "embedding" not in str(trace).casefold()


def test_structured_visual_result_rejects_duplicate_json_keys() -> None:
    with pytest.raises(VisualVerificationError, match="duplicate JSON key"):
        parse_visual_verification_json(
            """{
              "match_score": 0.2,
              "match_score": 0.9,
              "requirement_coverage": 0.5,
              "all_visible_requirements_satisfied": false,
              "predicates": [
                {"requirement": "x", "score": 0.5, "visible": true}
              ]
            }""",
            video_id="V",
            absolute_frame_id=1,
        )


def test_candidate_failure_retries_then_isolates_without_discarding_later_results() -> None:
    verifier = _CandidateIsolationVerifier()
    candidates = tuple(
        VisualVerificationInput("V", frame_id, frame_id / 10.0, (1, 2, 3))
        for frame_id in (10, 20, 30, 40)
    )

    results = verifier.verify(
        query_vi="mô tả",
        query_en="description",
        candidates=candidates,
    )

    assert tuple(item.absolute_frame_id for item in results) == (10, 20, 40)
    assert verifier.calls == [
        (10, 3, 384),
        (20, 3, 384),
        (20, 1, 384),
        (30, 3, 384),
        (30, 1, 384),
        (40, 3, 384),
    ]
    assert tuple(item.absolute_frame_id for item in verifier.last_failures) == (30,)
    assert tuple(
        item["absolute_frame_id"] for item in verifier.last_recovered_retries
    ) == (20,)


def test_compact_visual_result_parses_with_bounded_wire_schema() -> None:
    result = parse_visual_verification_json(
        '{"m":0.8,"c":0.75,"a":false,"p":['
        '["more than five",1.0,true],["three red hats",0.5,true],'
        '["one wears glasses",0.0,false]],"s":"partial conjunction"}',
        video_id="V",
        absolute_frame_id=6650,
    )

    assert result.match_score == pytest.approx(0.8)
    assert result.requirement_coverage == pytest.approx(0.75)
    assert result.predicate_bottleneck_score == 0.0
    assert result.predicates[0].evidence == ""
    assert result.to_trace()["predicate_bottleneck_score"] == 0.0


def test_visual_ranking_prioritizes_weakest_required_predicate() -> None:
    broad_match = LocalFrameFusion(10, 0.9, 1, 1, ())
    conjunction = LocalFrameFusion(20, 0.8, 1, 2, ())
    results = (
        VisualVerificationResult(
            "V",
            10,
            match_score=0.99,
            requirement_coverage=0.9,
            all_visible_requirements_satisfied=False,
            predicates=(
                VisualPredicateScore("group exercise", 1.0, True, ""),
                VisualPredicateScore("three red hats", 0.1, True, ""),
            ),
            summary="broad match",
        ),
        VisualVerificationResult(
            "V",
            20,
            match_score=0.8,
            requirement_coverage=0.8,
            all_visible_requirements_satisfied=False,
            predicates=(
                VisualPredicateScore("group exercise", 0.7, True, ""),
                VisualPredicateScore("three red hats", 0.7, True, ""),
            ),
            summary="balanced conjunction",
        ),
    )

    ranked = rank_visually_verified_timeline_frames(
        (broad_match, conjunction),
        results,
    )

    assert tuple(item.absolute_frame_id for item in ranked) == (20, 10)


def test_visual_verifier_config_is_bounded_and_default_off() -> None:
    assert SelectedVideoVisualVerifierConfig().enabled is False
    with pytest.raises(ValueError, match="coverage_bins"):
        SelectedVideoVisualVerifierConfig(
            enabled=True,
            shortlist_per_video=8,
            coverage_bins=9,
        )


def test_visual_verifier_auto_mode_bounds_cpu_work_without_changing_request() -> None:
    config = SelectedVideoVisualVerifierConfig(
        enabled=True,
        shortlist_per_video=16,
        coverage_bins=12,
        neighbor_sample_radius=1,
        max_new_tokens=384,
        device="cpu",
    )

    assert config.shortlist_per_video == 16
    assert config.coverage_bins == 12
    assert config.neighbor_sample_radius == 1
    assert config.max_new_tokens == 384
    assert config.cpu_fast_profile_applied is True
    assert config.effective_shortlist_per_video == 6
    assert config.effective_coverage_bins == 4
    assert config.effective_neighbor_sample_radius == 0
    assert config.effective_max_new_tokens == 192
    assert config.effective_max_image_pixels == 256 * 28 * 28


def test_visual_verifier_auto_cuda_and_explicit_full_preserve_requested_work() -> None:
    cuda = SelectedVideoVisualVerifierConfig(
        enabled=True,
        shortlist_per_video=16,
        coverage_bins=12,
        neighbor_sample_radius=1,
        max_new_tokens=384,
        device="cuda",
    )
    full_cpu = SelectedVideoVisualVerifierConfig(
        enabled=True,
        shortlist_per_video=16,
        coverage_bins=12,
        neighbor_sample_radius=1,
        max_new_tokens=384,
        device="cpu",
        execution_mode=VisualVerifierExecutionMode.FULL,
    )

    for config in (cuda, full_cpu):
        assert config.cpu_fast_profile_applied is False
        assert config.effective_shortlist_per_video == 16
        assert config.effective_coverage_bins == 12
        assert config.effective_neighbor_sample_radius == 1
        assert config.effective_max_new_tokens == 384
        assert config.effective_max_image_pixels is None
