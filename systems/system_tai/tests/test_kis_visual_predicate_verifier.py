"""Focused tests for bounded structured KIS visual verification."""

from __future__ import annotations

import json

import pytest

from system_tai.refinement.engine import (
    LocalFrameFusion,
    non_discriminative_visual_plateau_frame_ids,
    rank_visually_verified_timeline_frames,
)
from system_tai.refinement.models import (
    SelectedVideoVisualVerifierConfig,
    VisualVerifierExecutionMode,
)
from system_tai.refinement.visual_verifier import (
    HuggingFaceStructuredVisualVerifier,
    VisualPredicateRequirement,
    VisualPredicateScore,
    VisualVerificationError,
    VisualVerificationInput,
    VisualVerificationResult,
    compile_visual_predicate_contract,
    parse_visual_verification_json,
)

QUERY_VI = (
    "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
    "động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và "
    "ba người đội nón có màu đỏ."
)
QUERY_EN = (
    "The scene showed a group of more than five people in a row doing exercises, "
    "touching their toes with both hands. In the group there was only one person "
    "wearing glasses and three people wearing red hats."
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


def test_repeated_positive_visual_template_abstains_instead_of_promoting() -> None:
    shortlist = tuple(
        LocalFrameFusion(frame_id, score, 1, rank, ())
        for rank, (frame_id, score) in enumerate(
            ((10, 0.9), (20, 0.8), (30, 0.7), (40, 0.6)),
            start=1,
        )
    )

    def result(frame_id: int, *, bottleneck: float) -> VisualVerificationResult:
        return VisualVerificationResult(
            "V",
            frame_id,
            match_score=0.9,
            requirement_coverage=0.8,
            all_visible_requirements_satisfied=True,
            predicates=(
                VisualPredicateScore(
                    "group action",
                    bottleneck,
                    True,
                    f"frame {frame_id} people bending",
                ),
                VisualPredicateScore(
                    "three red hats",
                    bottleneck,
                    True,
                    f"frame {frame_id} three hats",
                ),
            ),
            summary="visible conjunction",
        )

    results = (
        result(10, bottleneck=0.8),
        result(20, bottleneck=0.8),
        result(30, bottleneck=0.8),
        result(40, bottleneck=0.6),
    )

    plateau = non_discriminative_visual_plateau_frame_ids(results)
    ranked = rank_visually_verified_timeline_frames(shortlist, results)

    assert plateau == frozenset({10, 20, 30})
    assert tuple(item.absolute_frame_id for item in ranked) == (40, 10, 20, 30)
    assert ranked[0].fusion_score == pytest.approx(0.9)
    assert ranked[1].fusion_score == pytest.approx(0.9)
    assert (
        ranked[1].per_variant_provenance[-1]["visual_verification"][
            "calibration_status"
        ]
        == "ABSTAIN_NON_DISCRIMINATIVE_POSITIVE_PLATEAU"
    )


def test_visual_verifier_prompt_requires_observed_predicate_evidence() -> None:
    prompt = HuggingFaceStructuredVisualVerifier._build_prompt(
        query_vi="mô tả",
        query_en="description",
    )

    assert (
        '["predicate_id",0.0,false,false,"observed_value",'
        '"image N: literal evidence"]'
    ) in prompt
    assert "observed_value must be only the visible integer" in prompt
    assert "never leave it empty" in prompt


def test_query_compiler_produces_stable_specific_predicate_ids() -> None:
    first = compile_visual_predicate_contract(
        query_vi=QUERY_VI,
        query_en=QUERY_EN,
    )
    second = compile_visual_predicate_contract(
        query_vi=QUERY_VI,
        query_en=QUERY_EN,
    )

    assert first == second
    assert tuple(item.predicate_id for item in first) == (
        "subject_count_1",
        "spatial_layout_1",
        "primary_action_1",
        "primary_action_2",
        "person_attribute_count_1",
        "person_attribute_count_2",
        "synchronized_action_1",
    )
    assert (first[0].comparison, first[0].expected_value) == ("gt", 5)
    assert (first[4].comparison, first[4].expected_value) == ("eq", 1)
    assert (first[5].comparison, first[5].expected_value) == ("eq", 3)


def _strict_contract_payload(
    contract: tuple[VisualPredicateRequirement, ...],
    *,
    red_hat_count: int = 3,
    action_evidence: str = "image 2: hands touch toes",
) -> str:
    predicates = []
    for item in contract:
        observed = "matched"
        evidence = "image 2: exact requirement visible"
        if item.predicate_id == "subject_count_1":
            observed = "7"
            evidence = "image 2: seven people visible"
        elif item.predicate_id == "person_attribute_count_1":
            observed = "1"
            evidence = "image 2: one person has glasses"
        elif item.predicate_id == "person_attribute_count_2":
            observed = str(red_hat_count)
            evidence = f"image 2: {red_hat_count} red hats visible"
        elif item.predicate_id == "primary_action_2":
            observed = "two-hand toe touch"
            evidence = action_evidence
        predicates.append(
            [item.predicate_id, 0.9, True, True, observed, evidence]
        )
    return json.dumps(
        {"m": 0.9, "c": 1.0, "a": True, "p": predicates, "s": "match"}
    )


def test_fixed_contract_accepts_only_grounded_complete_conjunction() -> None:
    contract = compile_visual_predicate_contract(
        query_vi=QUERY_VI,
        query_en=QUERY_EN,
    )
    result = parse_visual_verification_json(
        _strict_contract_payload(contract),
        video_id="V",
        absolute_frame_id=6650,
        predicate_contract=contract,
    )

    assert result.contract_validated is True
    assert result.all_visible_requirements_satisfied is True
    assert result.eligible_for_promotion is True
    assert result.requirement_coverage == pytest.approx(1.0)
    assert result.predicate_bottleneck_score == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("red_hat_count", "action_evidence"),
    [(2, "image 2: hands touch toes"), (3, "")],
)
def test_fixed_contract_fails_closed_on_count_mismatch_or_empty_evidence(
    red_hat_count: int,
    action_evidence: str,
) -> None:
    contract = compile_visual_predicate_contract(
        query_vi=QUERY_VI,
        query_en=QUERY_EN,
    )
    result = parse_visual_verification_json(
        _strict_contract_payload(
            contract,
            red_hat_count=red_hat_count,
            action_evidence=action_evidence,
        ),
        video_id="V",
        absolute_frame_id=4925,
        predicate_contract=contract,
    )

    assert result.all_visible_requirements_satisfied is False
    assert result.eligible_for_promotion is False
    assert result.predicate_bottleneck_score == 0.0
    assert result.requirement_coverage < 1.0


def test_fixed_contract_rejects_missing_or_renamed_predicate_id() -> None:
    contract = compile_visual_predicate_contract(
        query_vi=QUERY_VI,
        query_en=QUERY_EN,
    )
    payload = _strict_contract_payload(contract).replace(
        '"subject_count_1"',
        '"invented_count_1"',
        1,
    )

    with pytest.raises(VisualVerificationError, match="unexpected predicate ID"):
        parse_visual_verification_json(
            payload,
            video_id="V",
            absolute_frame_id=1,
            predicate_contract=contract,
        )


def test_unverified_fixed_contract_candidate_cannot_override_clip_order() -> None:
    contract = compile_visual_predicate_contract(
        query_vi=QUERY_VI,
        query_en=QUERY_EN,
    )
    clip_first = LocalFrameFusion(10, 0.9, 1, 1, ())
    claimed_match = LocalFrameFusion(20, 0.8, 1, 2, ())
    result = parse_visual_verification_json(
        _strict_contract_payload(contract, red_hat_count=2),
        video_id="V",
        absolute_frame_id=20,
        predicate_contract=contract,
    )

    ranked = rank_visually_verified_timeline_frames(
        (clip_first, claimed_match),
        (result,),
    )

    assert tuple(item.absolute_frame_id for item in ranked) == (10, 20)
    assert (
        ranked[1].per_variant_provenance[-1]["visual_verification"]
        ["status"]
        == "CANDIDATE_FALLBACK_CLIP"
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
