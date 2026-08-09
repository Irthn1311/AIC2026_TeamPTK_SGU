"""Fail-closed ingestion and scoring of human Stage 1C review CSV files."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1b.writers import write_json
from triage_eg.retrieval.stage1c.artifacts import REVIEW_FIELDS
from triage_eg.retrieval.stage1c.contracts import FAILURE_TAGS, REVIEW_LABELS

IDENTITY_FIELDS = tuple(
    field
    for field in REVIEW_FIELDS
    if field not in {"review_label", "review_notes", "failure_tags"}
)
GRADE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError("REVIEW_ROW_IDENTITY_MISMATCH: review CSV schema changed")
        return list(reader)


def _query_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cutoff in (1, 5, 10):
        selected = [row for row in rows if int(row["rank"]) <= cutoff]
        labels = [row["review_label"].strip().upper() for row in selected]
        judged = [label for label in labels if label]
        eligible = [label for label in judged if label in GRADE]
        result[f"judgments_completed_top{cutoff}"] = len(judged)
        result[f"uncertain_count_top{cutoff}"] = judged.count("UNCERTAIN")
        result[f"human_relevance_rate_top{cutoff}"] = (
            eligible.count("RELEVANT") / len(eligible) if eligible else None
        )
        result[f"human_graded_relevance_top{cutoff}"] = (
            sum(GRADE[label] for label in eligible) / len(eligible) if eligible else None
        )
    return result


def score_human_review(
    stage1c_root: str | Path,
    review_csv: str | Path,
) -> dict[str, Any]:
    root = Path(stage1c_root).expanduser().resolve(strict=True)
    template_path = root / "review/review_template.csv"
    template = _read_csv(template_path)
    reviewed = _read_csv(Path(review_csv).expanduser().resolve(strict=True))
    if len(reviewed) != len(template):
        raise ValueError("REVIEW_INCOMPLETE: review CSV row count differs from template")
    identities: set[tuple[str, ...]] = set()
    for expected, actual in zip(template, reviewed, strict=True):
        identity = tuple(actual[field] for field in IDENTITY_FIELDS)
        if identity in identities:
            raise ValueError("REVIEW_DUPLICATE_JUDGMENT: duplicate row identity")
        identities.add(identity)
        if any(expected[field] != actual[field] for field in IDENTITY_FIELDS):
            raise ValueError("REVIEW_ROW_IDENTITY_MISMATCH: identity fields were modified")
        label = actual["review_label"].strip().upper()
        if label and label not in REVIEW_LABELS:
            raise ValueError(f"REVIEW_LABEL_INVALID: {label!r}")
        tags = [item.strip() for item in actual["failure_tags"].split(";") if item.strip()]
        invalid_tags = sorted(set(tags) - FAILURE_TAGS)
        if invalid_tags:
            raise ValueError(f"REVIEW_LABEL_INVALID: failure tags {invalid_tags}")
        actual["review_label"] = label
    completed = sum(bool(row["review_label"]) for row in reviewed)
    status = "COMPLETE" if completed == len(reviewed) else (
        "PARTIALLY_REVIEWED" if completed else "NOT_REVIEWED"
    )
    by_query: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reviewed:
        by_query[row["query_id"]].append(row)
    per_query = {
        query_id: _query_metrics(sorted(rows, key=lambda item: int(item["rank"])))
        for query_id, rows in sorted(by_query.items())
    }
    groups: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in reviewed:
        groups[(row["language"], row["category"], row["difficulty"])].append(row)
    category_summary = []
    for (language, category, difficulty), rows in sorted(groups.items()):
        labels = [row["review_label"] for row in rows if row["review_label"]]
        counts = Counter(labels)
        eligible = [label for label in labels if label in GRADE]
        category_summary.append(
            {
                "language": language,
                "category": category,
                "difficulty": difficulty,
                "reviewed_count": len(labels),
                "relevant_count": counts["RELEVANT"],
                "partial_count": counts["PARTIAL"],
                "irrelevant_count": counts["IRRELEVANT"],
                "uncertain_count": counts["UNCERTAIN"],
                "human_relevance_rate": (
                    counts["RELEVANT"] / len(eligible) if eligible else None
                ),
                "human_graded_relevance": (
                    sum(GRADE[label] for label in eligible) / len(eligible)
                    if eligible
                    else None
                ),
            }
        )
    query_metadata = {row["query_id"]: row for row in reviewed}
    paired: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for query_id, metadata in query_metadata.items():
        paired[metadata["pair_id"]][metadata["language"]] = query_id
    pair_comparison = []
    for pair_id, languages in sorted(paired.items()):
        if set(languages) != {"en", "vi"}:
            continue
        en, vi = per_query[languages["en"]], per_query[languages["vi"]]
        record: dict[str, Any] = {"pair_id": pair_id, "interpretation": "OBSERVED_DIFFERENCE"}
        comparison_metrics = (
            (1, "human_relevance_rate"),
            (5, "human_relevance_rate"),
            (10, "human_graded_relevance"),
        )
        for cutoff, metric in comparison_metrics:
            key = f"{metric}_top{cutoff}"
            record[f"{key}_en_minus_vi"] = (
                en[key] - vi[key] if en[key] is not None and vi[key] is not None else None
            )
        pair_comparison.append(record)
    metrics = {
        "human_review_status": status,
        "retrieval_quality_status": (
            "QUALITATIVELY_EVALUATED" if status == "COMPLETE" else "NOT_REVIEWED"
        ),
        "judgments_expected": len(reviewed),
        "judgments_completed": completed,
        "per_query": per_query,
        "category_summary": category_summary,
        "paired_language_comparison": pair_comparison,
        "non_claims": [
            "Human qualitative rates are not competition Recall@K",
            "Observed language differences are not language causal effects",
            "UNCERTAIN judgments are reported separately and are not scored as zero",
        ],
    }
    review_root = root / "review"
    write_json(review_root / "review_metrics.json", metrics)
    lines = [
        "# Stage 1C Human Review Metrics",
        "",
        f"- Human review status: {status}",
        f"- Retrieval quality status: {metrics['retrieval_quality_status']}",
        f"- Judgments: {completed}/{len(reviewed)}",
        "- These are human qualitative metrics, not competition Recall@K.",
        "",
        "## Query metrics",
        "",
        "| Query | Human relevance Top-5 | Human graded Top-10 | Uncertain Top-10 |",
        "|---|---:|---:|---:|",
    ]
    for query_id, values in per_query.items():
        lines.append(
            f"| {query_id} | {values['human_relevance_rate_top5']} | "
            f"{values['human_graded_relevance_top10']} | {values['uncertain_count_top10']} |"
        )
    (review_root / "review_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
