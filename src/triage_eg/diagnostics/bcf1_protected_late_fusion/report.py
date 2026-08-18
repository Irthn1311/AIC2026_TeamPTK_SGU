"""Compact formal closeout for BCF-1 review."""

from __future__ import annotations

from typing import Any


def formal_report(
    *,
    head: str,
    integrity: dict[str, Any],
    evaluations: dict[str, Any],
    decision: dict[str, Any],
    bundle: dict[str, Any],
) -> str:
    lines = [
        "BCF1_STATUS=COMPLETE",
        f"HEAD={head}",
        "POLICY=A0_TOP5_PROTECTED_EQUAL_RRF60_LATE_FUSION",
        f"CROSS_F1_REPRODUCTION_GATE={integrity['cross_f1_reproduction_gate']}",
        f"L21_HASH_BEFORE_GT_GATE={integrity['l21_all_hashes_finalized_before_gt']}",
        f"INDEX_REUSE_GATE={integrity['siglip2_index_reused_without_rebuild']}",
        "SEALED_ACCESS=false",
        "PRODUCTION_POLICY_CHANGED=false",
        "AUTOMATIC_PRODUCTION_PROMOTION=false",
    ]
    for benchmark in ("cross", "l21"):
        for arm in ("A0", "S1", "F1"):
            summary = evaluations[benchmark]["arms"][arm]["summary"]
            lines.append(f"{benchmark.upper()}_{arm}_FINAL={summary['final_score']}")
        paired = evaluations[benchmark]["paired"]
        counts = paired["better_tie_worse"]
        lines.extend(
            [
                f"{benchmark.upper()}_F1_VS_A0_BETTER={counts.get('BETTER', 0)}",
                f"{benchmark.upper()}_F1_VS_A0_TIE={counts.get('TIE', 0)}",
                f"{benchmark.upper()}_F1_VS_A0_WORSE={counts.get('WORSE', 0)}",
            ]
        )
    lines.extend(
        [
            f"BCF1_CLASSIFICATION={decision['classification']}",
            f"BUNDLE_PATH={bundle['path']}",
            f"BUNDLE_SHA256={bundle['sha256']}",
            "STOP_CONDITION=REACHED",
        ]
    )
    return "\n".join(lines)


__all__ = ["formal_report"]
