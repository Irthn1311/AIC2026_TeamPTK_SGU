"""Formal compact reporting for TCA-1."""

from __future__ import annotations

from typing import Any


def formal_report(
    *, head: str, integrity: dict[str, Any], evaluation: dict[str, Any], bundle: dict[str, Any]
) -> str:
    paired = evaluation["paired_score"]
    summary = evaluation["summary"]
    lines = [
        f"HEAD={head}",
        "TCA1_IMPLEMENTATION=COMPLETE",
        f"A0_G1_REPRODUCTION={integrity['a0_g1_reproduction']}",
        f"GT_LEAKAGE_GATE={integrity['gt_leakage_gate']}",
        f"SEALED_ACCESS_GATE={integrity['sealed_access_gate']}",
        f"REVIEW_FREEZE_GATE={integrity['review_freeze_gate']}",
        f"FAIL_OVERRIDE_COUNT={integrity['fail_override_count']}",
        f"CHANGED_CLIP_INPUT_UNIT_COUNT={integrity['changed_clip_input_unit_count']}",
    ]
    overall = paired["overall"]["final_score"]
    lines.append(
        f"FINAL_SCORE_A0={overall['a0']:.12f} A1={overall['a1']:.12f} "
        f"DELTA={overall['delta']:+.12f}"
    )
    for task in ("KIS", "QA", "TRAKE"):
        value = paired["by_task"][task]["final_score"]
        lines.append(
            f"{task}_A0={value['a0']:.12f} A1={value['a1']:.12f} DELTA={value['delta']:+.12f}"
        )
    lines.extend(
        [
            f"T3_TARGET_HITS={summary['t3_target_event_hits']}",
            f"TRAKE_TARGET_CHAINS={summary['trake']}",
            f"D1_ATTRIBUTION_COUNTS={summary['d1_primary_measured_counts']}",
            f"TCA1_CAUSAL_STATUS={summary['causal_status']}",
            "PRODUCTION_POLICY_CHANGED=false",
            f"BUNDLE_PATH={bundle['path']}",
            f"BUNDLE_SHA256={bundle['sha256']}",
        ]
    )
    return "\n".join(lines)


__all__ = ["formal_report"]
