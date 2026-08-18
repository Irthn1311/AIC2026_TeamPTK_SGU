"""Frozen post-GT B0/M0/M1 selection rule."""

from __future__ import annotations

from typing import Any


def _eligible(
    name: str, metrics: dict[str, Any], b0: dict[str, Any], integrity: bool
) -> tuple[bool, list[str]]:
    reasons = []
    cross, l21 = metrics["cross"], metrics["l21"]
    if not integrity:
        reasons.append("INTEGRITY_FAIL")
    if not l21["final_score"] > b0["l21"]["final_score"]:
        reasons.append("L21_NOT_STRICTLY_BETTER")
    if cross["final_score"] < b0["cross"]["final_score"] - 0.005:
        reasons.append("CROSS_REGRESSION_GATE")
    for benchmark in ("cross", "l21"):
        for task in ("KIS", "TRAKE"):
            for metric in ("R@1", "R@5"):
                if (
                    metrics[benchmark]["tasks"][task][metric]
                    != b0[benchmark]["tasks"][task][metric]
                ):
                    reasons.append(f"{benchmark}:{task}:{metric}_CHANGED")
        for task, values in metrics[benchmark]["tasks"].items():
            if values["final_score"] < b0[benchmark]["tasks"][task]["final_score"] - 0.02:
                reasons.append(f"{benchmark}:{task}_REGRESSED_GT_002")
    return not reasons, reasons


def select_arm(
    all_metrics: dict[str, dict[str, Any]], integrity: dict[str, bool]
) -> dict[str, Any]:
    b0 = all_metrics["B0"]
    checks = {
        name: _eligible(name, all_metrics[name], b0, integrity.get(name, False))
        for name in ("M0", "M1")
    }
    eligible = [name for name, (passed, _) in checks.items() if passed]
    selected = (
        max(
            eligible,
            key=lambda name: (
                all_metrics[name]["l21"]["final_score"],
                all_metrics[name]["cross"]["final_score"],
            ),
        )
        if eligible
        else "B0"
    )
    return {
        "selected_arm": selected,
        "eligible": eligible,
        "checks": {
            name: {"eligible": passed, "reasons": reasons}
            for name, (passed, reasons) in checks.items()
        },
        "production_promoted": False,
    }
