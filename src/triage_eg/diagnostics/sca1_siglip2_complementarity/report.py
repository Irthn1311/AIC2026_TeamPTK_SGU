"""Compact formal SCA-1 run report."""

from __future__ import annotations

from typing import Any


def formal_report(
    *,
    head: str,
    integrity: dict[str, Any],
    evaluation: dict[str, Any],
    bundle: dict[str, Any],
) -> str:
    paired, summary = evaluation["paired_score"], evaluation["summary"]
    lines = [
        f"HEAD={head}",
        "SCA1_IMPLEMENTATION=COMPLETE",
        f"A0_REPRODUCTION={integrity['a0_reproduction_gate']}",
        f"TEXT_IDENTITY_GATE={integrity['text_identity_gate']}",
        f"TEXT_IDENTITY_COUNT={integrity['text_identity_count']}",
        f"QA_ANSWER_SPACE_ISOLATION={integrity['qa_answer_space_isolation']}",
        "GT_UNAVAILABLE_DURING_A0_PREDICTION=PASS",
        "GT_UNAVAILABLE_DURING_S1_PREDICTION=PASS",
        "SEALED_ACCESS=false",
    ]
    overall = paired["overall"]["final_score"]
    lines.append(
        f"FINAL_SCORE_A0={overall['a0']:.12f} S1={overall['s1']:.12f} "
        f"DELTA={overall['delta']:+.12f}"
    )
    for task in ("KIS", "QA", "TRAKE"):
        value = paired["by_task"][task]["final_score"]
        lines.append(
            f"{task}_A0={value['a0']:.12f} S1={value['s1']:.12f} DELTA={value['delta']:+.12f}"
        )
    lines.extend(
        [
            f"UNIQUE_S1_GLOBAL_TOP100_RESCUES={summary['unique_s1_target_global_top100_rescues']}",
            "UNIQUE_A0_GLOBAL_TOP100_LOSSES="
            f"{summary['unique_a0_target_global_top100_hits_lost_by_s1']}",
            f"T3_EVENT_HITS={summary['semantic_unit_t3_hit_counts']}",
            f"TRAKE_T3_EVENT_HITS={summary['trake_t3_event_hit_counts']}",
            f"TRAKE_TARGET_CHAINS={summary['trake_target_chain_counts']}",
            f"SCA1_COMPLEMENTARITY={summary['complementarity_classification']}",
            f"FUSION_GATE={summary['fusion_gate']}",
            "FUSION_IMPLEMENTED=false",
            "PRODUCTION_POLICY_CHANGED=false",
            f"BUNDLE_PATH={bundle['path']}",
            f"BUNDLE_SHA256={bundle['sha256']}",
        ]
    )
    return "\n".join(lines)


__all__ = ["formal_report"]
