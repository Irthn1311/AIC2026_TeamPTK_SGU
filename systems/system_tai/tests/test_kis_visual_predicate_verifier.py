"""Focused tests for bounded structured KIS visual verification."""

from __future__ import annotations

import pytest

from system_tai.refinement.models import (
    SelectedVideoVisualVerifierConfig,
    VisualVerifierExecutionMode,
)
from system_tai.refinement.visual_verifier import (
    VisualVerificationError,
    parse_visual_verification_json,
)


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
