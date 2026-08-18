"""Pre-GT fusion provenance and post-GT paired BCF-1 diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from aic2026_eval.scoring import CUTOFFS

from .fusion import candidate_key


def fusion_diagnostics(
    queries: list[dict[str, Any]],
    a0: list[dict[str, Any]],
    s1: list[dict[str, Any]],
    f1: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    query_map = {str(row["query_id"]): row for row in queries}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: defaultdict(list) for arm in ("a0", "s1", "f1", "provenance")
    }
    for arm, rows in (("a0", a0), ("s1", s1), ("f1", f1), ("provenance", provenance)):
        for row in rows:
            grouped[arm][str(row["query_id"])].append(row)
    overlap_rows, prefix_rows = [], []
    source_counts: dict[str, Any] = {str(cutoff): Counter() for cutoff in CUTOFFS[2:]}
    source_by_task: dict[str, Any] = {
        task: {str(cutoff): Counter() for cutoff in CUTOFFS[2:]} for task in ("KIS", "QA", "TRAKE")
    }
    displaced = {str(cutoff): 0 for cutoff in CUTOFFS[2:]}
    s1_only = {str(cutoff): 0 for cutoff in CUTOFFS[2:]}
    for query_id, query in query_map.items():
        task = str(query["task"])
        a0_rows = sorted(grouped["a0"][query_id], key=lambda row: row["rank"])
        s1_rows = sorted(grouped["s1"][query_id], key=lambda row: row["rank"])
        f1_rows = sorted(grouped["f1"][query_id], key=lambda row: row["rank"])
        prov_rows = sorted(grouped["provenance"][query_id], key=lambda row: row["fused_rank"])
        a0_keys = {candidate_key(task, row) for row in a0_rows}
        s1_keys = {candidate_key(task, row) for row in s1_rows}
        intersection = len(a0_keys & s1_keys)
        union = len(a0_keys | s1_keys)
        overlap_rows.append(
            {
                "query_id": query_id,
                "task": task,
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else 0.0,
            }
        )
        prefix_identity = a0_rows[:5] == f1_rows[:5]
        prefix_byte_identity = [
            json.dumps(row, ensure_ascii=False).encode("utf-8") for row in a0_rows[:5]
        ] == [json.dumps(row, ensure_ascii=False).encode("utf-8") for row in f1_rows[:5]]
        prefix_rows.append(
            {
                "query_id": query_id,
                "task": task,
                "a0_top5_semantic_identical": prefix_identity,
                "a0_top5_byte_identical": prefix_byte_identity,
            }
        )
        if not prefix_identity or not prefix_byte_identity:
            raise RuntimeError(f"BCF1_A0_PROTECTED_PREFIX_IDENTITY_FAILED: {query_id}")
        for cutoff in CUTOFFS[2:]:
            label = str(cutoff)
            for row in prov_rows[:cutoff]:
                source_counts[label][str(row["source"])] += 1
                source_by_task[task][label][str(row["source"])] += 1
            a0_at_k = {candidate_key(task, row) for row in a0_rows[:cutoff]}
            f1_at_k = {candidate_key(task, row) for row in f1_rows[:cutoff]}
            displaced[label] += len(a0_at_k - f1_at_k)
            s1_only[label] += sum(row["source"] == "S1_ONLY" for row in prov_rows[:cutoff])
    overlap_summary = {}
    for task in ("KIS", "QA", "TRAKE"):
        rows = [row for row in overlap_rows if row["task"] == task]
        overlap_summary[task] = {
            "query_count": len(rows),
            "mean_intersection": mean(row["intersection"] for row in rows),
            "median_intersection": median(row["intersection"] for row in rows),
            "mean_union": mean(row["union"] for row in rows),
            "median_union": median(row["union"] for row in rows),
            "mean_jaccard": mean(row["jaccard"] for row in rows),
        }
    return {
        "status": "PASS",
        "query_count": len(queries),
        "protected_prefix_semantic_identity_count": sum(
            row["a0_top5_semantic_identical"] for row in prefix_rows
        ),
        "protected_prefix_byte_identity_count": sum(
            row["a0_top5_byte_identical"] for row in prefix_rows
        ),
        "protected_prefix_identity_gate": (
            "PASS"
            if all(
                row["a0_top5_semantic_identical"] and row["a0_top5_byte_identical"]
                for row in prefix_rows
            )
            else "FAIL"
        ),
        "overlap_by_task": overlap_summary,
        "overlap_per_query": overlap_rows,
        "candidate_source_counts": {
            cutoff: dict(counts) for cutoff, counts in source_counts.items()
        },
        "candidate_source_counts_by_task": {
            task: {cutoff: dict(counts) for cutoff, counts in values.items()}
            for task, values in source_by_task.items()
        },
        "a0_candidates_displaced": displaced,
        "s1_only_candidates_admitted": s1_only,
        "production_policy_changed": False,
    }


def _first_correct_rank(row: dict[str, Any]) -> int | None:
    ranks = [
        int(item["rank"])
        for item in row.get("prediction_diagnostics", [])
        if float(item.get("r_score", 0.0)) > 0.0
    ]
    return min(ranks) if ranks else None


def paired_evaluation(
    evaluations: dict[str, dict[str, Any]], *, benchmark_id: str
) -> dict[str, Any]:
    a0, s1, f1 = (evaluations[arm] for arm in ("A0", "S1", "F1"))
    by_arm = {
        arm: {row["query_id"]: row for row in value["per_query"]}
        for arm, value in evaluations.items()
    }
    counts: Counter[str] = Counter()
    per_query = []
    for query_id in sorted(by_arm["A0"]):
        rows = {arm: by_arm[arm][query_id] for arm in ("A0", "S1", "F1")}
        delta = rows["F1"]["final_score"] - rows["A0"]["final_score"]
        result = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "TIE"
        counts[result] += 1
        first = {arm: _first_correct_rank(row) for arm, row in rows.items()}
        per_query.append(
            {
                "query_id": query_id,
                "task": rows["A0"]["task"],
                "first_correct_rank_a0": first["A0"],
                "first_correct_rank_s1": first["S1"],
                "first_correct_rank_f1": first["F1"],
                "first_correct_rank_delta_f1_minus_a0": (
                    first["F1"] - first["A0"]
                    if first["F1"] is not None and first["A0"] is not None
                    else None
                ),
                "final_score_a0": rows["A0"]["final_score"],
                "final_score_s1": rows["S1"]["final_score"],
                "final_score_f1": rows["F1"]["final_score"],
                "final_score_delta_f1_minus_a0": delta,
                "result_f1_vs_a0": result,
            }
        )
    task_delta = {
        task: f1["slices"][f"task:{task}"]["final_score"]
        - a0["slices"][f"task:{task}"]["final_score"]
        for task in ("KIS", "QA", "TRAKE")
    }
    return {
        "benchmark_id": benchmark_id,
        "overall_delta_f1_minus_a0": f1["summary"]["final_score"] - a0["summary"]["final_score"],
        "task_delta_f1_minus_a0": task_delta,
        "better_tie_worse": dict(counts),
        "per_query": per_query,
    }


__all__ = ["fusion_diagnostics", "paired_evaluation"]
