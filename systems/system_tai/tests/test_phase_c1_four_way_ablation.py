"""Unit tests for Phase C.1: Four-Way Controlled Ablation Benchmark."""

import json
from pathlib import Path
import pytest

from system_tai.translation.paraphrase_sidecar_provider import (
    ImmutableParaphraseEnsembleSidecarProvider,
)
from system_tai.translation.sidecar_provider import canonical_sidecar_sha256
from scratch.run_phase_c1_four_way_ablation import (
    CANONICAL_SHAM_SHA256,
    CANONICAL_TNEW_SHA256,
    CANONICAL_TOLD_SHA256,
    CANONICAL_PARAPHRASE_SHA256,
    classify_p15_findings,
    compute_pairwise_comparison,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SHAM_SIDECAR_PATH = REPO_ROOT / "scratch" / "benchmarks" / "translation_ablation" / "paraphrase_sham_duplicate_new_p1_focus_v1.json"


def test_sham_sidecar_canonical_sha256_and_loading():
    """Verify sham duplicate sidecar exists, matches canonical SHA, and loads cleanly."""
    assert SHAM_SIDECAR_PATH.exists(), f"Sham sidecar missing: {SHAM_SIDECAR_PATH}"
    actual_sha = canonical_sidecar_sha256(SHAM_SIDECAR_PATH)
    assert actual_sha == CANONICAL_SHAM_SHA256

    provider = ImmutableParaphraseEnsembleSidecarProvider(
        sidecar_path=SHAM_SIDECAR_PATH,
        expected_content_sha256=CANONICAL_SHAM_SHA256,
    )

    # For P1-5: must have group_sham_a and group_sham_b
    groups_p15 = provider.get_paraphrase_groups("query-p1-5-kis")
    assert len(groups_p15) == 2
    g_ids = [g["group_id"] for g in groups_p15]
    assert g_ids == ["group_sham_a", "group_sham_b"]

    exp_hashes = provider.expected_group_hashes("query-p1-5-kis")
    assert exp_hashes["group_sham_a"] == "243b0f915c63"
    assert exp_hashes["group_sham_b"] == "243b0f915c63"

    exp_counts = provider.expected_group_variant_counts("query-p1-5-kis")
    assert exp_counts["group_sham_a"] == 3
    assert exp_counts["group_sham_b"] == 3

    # Negative controls must have exactly 1 group (group_canonical_new)
    for qid in ["query-p1-1-kis", "query-p1-2-kis", "query-p1-4-kis", "query-p1-6-kis"]:
        groups = provider.get_paraphrase_groups(qid)
        assert len(groups) == 1
        assert groups[0]["group_id"] == "group_canonical_new"


def test_pairwise_comparison_math():
    """Verify compute_pairwise_comparison computes exact set metrics and rank shifts."""
    # Dummy candidates: 100 each
    records_a = [
        {"video_id": f"V{i:03d}", "frame_id": 100 + i, "rank": i + 1}
        for i in range(100)
    ]
    # Records B: shares 70 candidates, replaces 30
    records_b = [
        {"video_id": f"V{i:03d}", "frame_id": 100 + i, "rank": (i + 1) if i < 70 else (i + 1)}
        for i in range(70)
    ] + [
        {"video_id": f"NEW_{j:03d}", "frame_id": 900 + j, "rank": 71 + j}
        for j in range(30)
    ]

    res = compute_pairwise_comparison(records_a, records_b)
    assert res["intersection_count"] == 70
    assert res["union_count"] == 130
    assert round(res["jaccard_similarity"], 4) == round(70 / 130, 4)
    assert res["membership_replaced_count"] == 30
    assert res["membership_replaced_ratio"] == 0.30
    assert res["median_rank_shift"] == 0.0


def test_classify_p15_findings_destructive_interference():
    """Old is strong (#16), New is #25, Sham is #25, Mixed drops to #31 -> DESTRUCTIVE_INTER_GROUP_INTERFERENCE."""
    contrasts, verdict = classify_p15_findings(
        r_new=25,
        r_old=16,
        r_sham=25,
        r_mixed=31,
        selected_video_count_parity=True,
    )
    assert verdict == "DESTRUCTIVE_INTER_GROUP_INTERFERENCE"
    assert contrasts["findings"]["destructive_interference"] is True
    assert contrasts["findings"]["old_wording_stronger_than_new"] is True
    assert contrasts["findings"]["ensemble_mechanics_effect"] is False
    assert contrasts["rank_deltas"]["delta_mechanics_sham_minus_new"] == 0
    assert contrasts["rank_deltas"]["delta_wording_old_minus_new"] == -9
    assert contrasts["rank_deltas"]["delta_replacement_mixed_minus_sham"] == 6


def test_classify_p15_findings_ensemble_mechanics_confound():
    """Sham drops to #31 even without Old (identical duplicate) -> ENSEMBLE_MECHANICS_CONFOUND."""
    contrasts, verdict = classify_p15_findings(
        r_new=25,
        r_old=25,
        r_sham=31,
        r_mixed=31,
        selected_video_count_parity=True,
    )
    assert verdict == "ENSEMBLE_MECHANICS_CONFOUND"
    assert contrasts["findings"]["ensemble_mechanics_effect"] is True
    assert contrasts["rank_deltas"]["delta_mechanics_sham_minus_new"] == 6
    assert contrasts["rank_deltas"]["delta_replacement_mixed_minus_sham"] == 0


def test_classify_p15_findings_weak_old_dilution():
    """Old is weak (#60), Sham is #25, Mixed drops to #31 -> WEAK_OLD_GROUP_DILUTION_SUPPORTED."""
    contrasts, verdict = classify_p15_findings(
        r_new=25,
        r_old=60,
        r_sham=25,
        r_mixed=31,
        selected_video_count_parity=True,
    )
    assert verdict == "WEAK_OLD_GROUP_DILUTION_SUPPORTED"
    assert contrasts["findings"]["old_wording_weaker_than_new"] is True
    assert contrasts["findings"]["semantic_replacement_degradation"] is True
    assert contrasts["findings"]["ensemble_mechanics_effect"] is False


def test_classify_p15_findings_adaptive_confound():
    """Selected video counts differ across arms -> ADAPTIVE_VIDEO_BUDGET_CONFOUND."""
    contrasts, verdict = classify_p15_findings(
        r_new=25,
        r_old=16,
        r_sham=25,
        r_mixed=31,
        selected_video_count_parity=False,
    )
    assert verdict == "ADAPTIVE_VIDEO_BUDGET_CONFOUND"
    assert contrasts["findings"]["adaptive_video_budget_confound"] is True
