"""Output contracts for the blind Prelim-1 actual-inference run."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ACTUAL_SYSTEM = "MY_PRELIM1_R5_QE"
CORE_FILES = frozenset(
    {
        "MY_SYSTEM_RESULTS.md",
        "my_system_primary.csv",
        "my_system_top5.csv",
        "my_system_top10.csv",
        "my_system_top100.jsonl",
        "query_manifest.json",
        "candidate_provenance.jsonl",
        "qa_hypotheses.csv",
        "trake_top20.json",
        "asset_status.json",
        "run_provenance.json",
    }
)


def confidence_bucket(row: dict[str, Any]) -> str:
    explicit = row.get("confidence_bucket")
    if explicit:
        return str(explicit).upper()
    tier = str(row.get("evidence_tier", ""))
    if tier.startswith("TIER_A"):
        return "HIGH"
    if tier.startswith("TIER_B"):
        return "MEDIUM"
    return "LOW"


def grouped_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["query_id"])].append(row)
    for values in output.values():
        values.sort(key=lambda row: int(row["candidate_rank"]))
    return dict(output)


def select_review_rows(
    queries: list[dict[str, Any]], rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    grouped = grouped_rows(rows)
    selected = []
    for query in queries:
        selected.extend(grouped[str(query["query_id"])][:limit])
    return selected


def build_primary_rows(
    queries: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped = grouped_rows(rows)
    max_events = max(int(query.get("event_count", 0)) for query in queries)
    output = []
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"])
        source = grouped[query_id][0]
        row: dict[str, Any] = {
            "query_id": query_id,
            "task_type": task,
            "video_id": source["video_id"],
            "frame_id": source.get("frame_id", ""),
            "answer": source.get("answer", ""),
            "confidence": confidence_bucket(source),
            "status": source.get("status", "READY"),
        }
        frames = source.get("frame_ids", [])
        for index in range(max_events):
            row[f"frame_{index + 1}"] = frames[index] if index < len(frames) else ""
        output.append(row)
    return output


def write_results_report(
    path: str | Path,
    queries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> Path:
    grouped = grouped_rows(rows)
    lines = [
        "# MY SYSTEM RESULTS — PRELIM 1",
        "",
        f"System: `{ACTUAL_SYSTEM}`",
        "",
        "Blind fresh-query inference. No GT, leaderboard feedback, or automatic upload.",
        "",
    ]
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"])
        selected = grouped[query_id][:5]
        lines.extend(
            [
                f"## {query_id} — {task}",
                "",
                str(query["normalized_text"]),
                "",
                "**Primary**",
                "",
            ]
        )
        for index, row in enumerate(selected):
            prefix = "- " if index == 0 else "- Alternative "
            if task == "TRAKE":
                coordinate = " / ".join(
                    f"E{event + 1}={frame}"
                    for event, frame in enumerate(row["frame_ids"])
                )
            else:
                coordinate = f"frame={row['frame_id']} t={float(row.get('video_time_sec', 0)):.1f}s"
            answer = f" / answer={row.get('answer')!r}" if task == "QA" else ""
            warning = (
                " / WARNING=MANUAL_REVIEW_REQUIRED"
                if row.get("status") == "MANUAL_REVIEW_REQUIRED"
                else ""
            )
            lines.append(
                f"{prefix}#{row['candidate_rank']} `{row['video_id']}` / {coordinate}{answer} / "
                f"confidence={confidence_bucket(row)} / {row.get('reason_short', '')}{warning}"
            )
        lines.append("")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def validate_actual_results(
    output_root: str | Path,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    qa_hypotheses: list[dict[str, Any]],
    trake_top20: list[dict[str, Any]],
) -> dict[str, Any]:
    root = Path(output_root).resolve(strict=True)
    queries = list(manifest["queries"])
    query_ids = [str(query["query_id"]) for query in queries]
    issues = []
    if len(query_ids) != 25 or len(query_ids) != len(set(query_ids)):
        issues.append("QUERY_ID_CARDINALITY")
    if manifest.get("task_counts") != {"KIS": 20, "QA": 4, "TRAKE": 1}:
        issues.append("TASK_COUNTS")
    grouped = grouped_rows(rows)
    if set(grouped) != set(query_ids):
        issues.append("RESULT_QUERY_SET")
    for query in queries:
        query_id, task = str(query["query_id"]), str(query["task"])
        values = grouped.get(query_id, [])
        expected = 20 if task == "TRAKE" else 100
        ranks = [int(row["candidate_rank"]) for row in values]
        if len(values) != expected or ranks != list(range(1, expected + 1)):
            issues.append(f"RANK_CARDINALITY:{query_id}:{len(values)}")
        if task == "TRAKE":
            seen = set()
            for row in values:
                frames = tuple(int(value) for value in row.get("frame_ids", []))
                key = str(row.get("video_id")), frames
                if (
                    len(frames) != int(query["event_count"])
                    or any(left >= right for left, right in zip(frames, frames[1:], strict=False))
                    or key in seen
                ):
                    issues.append(f"TRAKE_STRUCTURE:{query_id}:{row.get('candidate_rank')}")
                seen.add(key)
        else:
            for row in values:
                if not row.get("video_id") or row.get("frame_id") is None:
                    issues.append(f"FRAME_COORDINATE:{query_id}:{row.get('candidate_rank')}")
    qa_counts = Counter(str(row["query_id"]) for row in qa_hypotheses)
    if any(qa_counts[str(query["query_id"])] != 5 for query in queries if query["task"] == "QA"):
        issues.append("QA_TOP5_CARDINALITY")
    if len(trake_top20) != 20:
        issues.append("TRAKE_TOP20_CARDINALITY")
    missing = sorted(name for name in CORE_FILES if not (root / name).is_file())
    if missing:
        issues.append(f"MISSING_CORE_FILES:{missing}")
    forbidden = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(token in path.name.casefold() for token in ("ground_truth", "sealed_final_30"))
    ]
    if forbidden:
        issues.append(f"FORBIDDEN_ARTIFACTS:{forbidden}")
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "query_count": len(queries),
        "task_counts": manifest["task_counts"],
        "prediction_count": len(rows),
        "expected_prediction_count": 2420,
        "qa_hypothesis_count": len(qa_hypotheses),
        "trake_chain_count": len(trake_top20),
        "ground_truth_opened": False,
        "leaderboard_used": False,
        "submission_uploaded": False,
    }


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)


__all__ = [
    "ACTUAL_SYSTEM",
    "CORE_FILES",
    "build_primary_rows",
    "confidence_bucket",
    "grouped_rows",
    "jsonl_text",
    "select_review_rows",
    "validate_actual_results",
    "write_results_report",
]
