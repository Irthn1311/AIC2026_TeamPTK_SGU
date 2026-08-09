"""Deterministic blinded three-arm review generation and scoring."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1b.writers import write_json

from .contracts import ARMS, REVIEW_LABELS, ReviewConfig

REVIEW_FIELDS = (
    "review_row_id",
    "pair_id",
    "condition_code",
    "en_reference_text",
    "vi_original_text",
    "rank",
    "video_id",
    "global_row",
    "n",
    "original_frame_idx",
    "score",
    "review_label",
    "review_notes",
)
IDENTITY_FIELDS = tuple(
    name for name in REVIEW_FIELDS if name not in {"review_label", "review_notes"}
)
GRADE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}


def _condition_map(pair_id: str, seed: int) -> dict[str, str]:
    arms = list(ARMS)
    random.Random(f"{seed}:{pair_id}").shuffle(arms)
    return {f"C{index:02d}": arm for index, arm in enumerate(arms, start=1)}


def write_blinded_review(
    output_root: Path,
    *,
    pairs: dict[str, dict[str, Any]],
    frames_by_pair_arm: dict[str, dict[str, list[dict[str, Any]]]],
    config: ReviewConfig,
) -> int:
    review_root = output_root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    key_pairs: list[dict[str, Any]] = []
    count = 0
    with (review_root / "review_template_blinded.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for pair_id, members in sorted(pairs.items()):
            mapping = _condition_map(pair_id, config.seed)
            key_pairs.append({"pair_id": pair_id, "conditions": mapping})
            for condition_code in sorted(mapping):
                arm = mapping[condition_code]
                for item in frames_by_pair_arm[pair_id][arm][: config.top_k]:
                    count += 1
                    writer.writerow(
                        {
                            "review_row_id": f"{pair_id}:{condition_code}:{item['rank']:02d}",
                            "pair_id": pair_id,
                            "condition_code": condition_code,
                            "en_reference_text": members["en"].text,
                            "vi_original_text": members["vi"].text,
                            "rank": item["rank"],
                            "video_id": item["video_id"],
                            "global_row": item["global_row"],
                            "n": item["n"],
                            "original_frame_idx": item["original_frame_idx"],
                            "score": item["score"],
                            "review_label": "",
                            "review_notes": "",
                        }
                    )
    write_json(
        review_root / "review_key.json",
        {
            "stage1d_review_version": "0.1.0",
            "seed": config.seed,
            "blinded": True,
            "pairs": key_pairs,
        },
    )
    (review_root / "review_instructions.md").write_text(
        review_instructions(), encoding="utf-8"
    )
    return count


def review_instructions() -> str:
    return """# Stage 1D Blinded Translation-Ablation Review

Judge each visible frame against the shared semantic intent shown by the English
reference and Vietnamese original query. Condition codes are deliberately opaque.

- RELEVANT: clearly satisfies the main intent.
- PARTIAL: matches part of the intent but misses an important component.
- IRRELEVANT: does not satisfy the main intent.
- UNCERTAIN: the frame is insufficient to decide.

Do not edit identity columns. Scores are diagnostics, not labels. This is a
qualitative ablation and must not be described as competition Recall@K.
"""


def _read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError("REVIEW_IDENTITY_MISMATCH: review schema changed")
        return list(reader)


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["review_label"] for row in rows if row["review_label"])
    result: dict[str, Any] = {
        "relevant_count": counts["RELEVANT"],
        "partial_count": counts["PARTIAL"],
        "irrelevant_count": counts["IRRELEVANT"],
        "uncertain_count": counts["UNCERTAIN"],
    }
    for cutoff in (1, 5):
        selected = [row for row in rows if int(row["rank"]) <= cutoff]
        eligible = [row["review_label"] for row in selected if row["review_label"] in GRADE]
        result[f"human_relevance_rate_top{cutoff}"] = (
            eligible.count("RELEVANT") / len(eligible) if eligible else None
        )
        result[f"human_graded_relevance_top{cutoff}"] = (
            sum(GRADE[label] for label in eligible) / len(eligible) if eligible else None
        )
        result[f"uncertain_count_top{cutoff}"] = sum(
            row["review_label"] == "UNCERTAIN" for row in selected
        )
    return result


def score_stage1d_review(
    stage1d_root: str | Path,
    review_csv: str | Path,
    review_key: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(stage1d_root).expanduser().resolve(strict=True)
    template = _read_review_csv(root / "review/review_template_blinded.csv")
    reviewed = _read_review_csv(Path(review_csv).expanduser().resolve(strict=True))
    if len(template) != len(reviewed):
        raise ValueError("REVIEW_INCOMPLETE: row count differs from template")
    key_path = (
        Path(review_key).expanduser().resolve(strict=True)
        if review_key is not None
        else root / "review/review_key.json"
    )
    key = json.loads(key_path.read_text(encoding="utf-8"))
    mappings = {
        item["pair_id"]: item["conditions"]
        for item in key.get("pairs", [])
        if isinstance(item, dict)
    }
    identities: set[tuple[str, ...]] = set()
    resolved: list[dict[str, Any]] = []
    for expected, actual in zip(template, reviewed, strict=True):
        identity = tuple(actual[name] for name in IDENTITY_FIELDS)
        if identity in identities:
            raise ValueError("REVIEW_DUPLICATE")
        identities.add(identity)
        if any(expected[name] != actual[name] for name in IDENTITY_FIELDS):
            raise ValueError("REVIEW_IDENTITY_MISMATCH")
        label = actual["review_label"].strip().upper()
        if label and label not in REVIEW_LABELS:
            raise ValueError(f"REVIEW_LABEL_INVALID: {label!r}")
        pair_id, condition = actual["pair_id"], actual["condition_code"]
        try:
            arm = mappings[pair_id][condition]
        except (KeyError, TypeError) as error:
            raise ValueError("REVIEW_IDENTITY_MISMATCH: unresolved condition") from error
        resolved.append({**actual, "review_label": label, "actual_arm": arm})
    completed = sum(bool(row["review_label"]) for row in resolved)
    status = (
        "COMPLETE"
        if completed == len(resolved)
        else "PARTIAL"
        if completed
        else "NOT_REVIEWED"
    )
    by_arm: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        by_arm[row["actual_arm"]].append(row)
    per_arm = {arm: _arm_metrics(by_arm[arm]) for arm in ARMS}
    comparisons = []
    by_pair_arm: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        by_pair_arm[(row["pair_id"], row["actual_arm"])].append(row)
    for pair_id in sorted({row["pair_id"] for row in resolved}):
        direct = _arm_metrics(by_pair_arm[(pair_id, "VI_DIRECT")])
        translated = _arm_metrics(by_pair_arm[(pair_id, "VI_TRANSLATED_EN")])
        en = _arm_metrics(by_pair_arm[(pair_id, "EN_DIRECT")])
        record: dict[str, Any] = {"pair_id": pair_id}
        for metric in (
            "human_relevance_rate_top1",
            "human_relevance_rate_top5",
            "human_graded_relevance_top5",
        ):
            left, right, reference = translated[metric], direct[metric], en[metric]
            record[f"translated_minus_vi_{metric}"] = (
                left - right if left is not None and right is not None else None
            )
            record[f"translated_minus_en_{metric}"] = (
                left - reference if left is not None and reference is not None else None
            )
        comparisons.append(record)
    metadata = {
        item["pair_id"]: item
        for item in (
            json.loads(line)
            for line in (root / "comparisons/pair_comparisons.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        item = metadata[row["pair_id"]]
        grouped[(row["actual_arm"], item["category"], item["difficulty"])].append(row)
    category_difficulty = [
        {
            "arm": arm,
            "category": category,
            "difficulty": difficulty,
            **_arm_metrics(rows),
        }
        for (arm, category, difficulty), rows in sorted(grouped.items())
    ]
    metrics = {
        "human_review_status": status,
        "language_bridge_quality_status": (
            "QUALITATIVELY_EVALUATED" if status == "COMPLETE" else "NOT_REVIEWED"
        ),
        "judgments_expected": len(resolved),
        "judgments_completed": completed,
        "per_arm": per_arm,
        "category_difficulty": category_difficulty,
        "ablation_deltas": comparisons,
        "non_claims": [
            "Human qualitative rates are not competition Recall@K",
            "UNCERTAIN is reported separately and excluded from graded denominators",
            "No production route is selected automatically",
        ],
    }
    write_json(root / "review/review_metrics.json", metrics)
    lines = [
        "# Stage 1D Human Review Metrics",
        "",
        f"- Human review status: {status}",
        f"- Language bridge quality: {metrics['language_bridge_quality_status']}",
        f"- Judgments: {completed}/{len(resolved)}",
        "",
        "| Arm | Relevant@1 | Relevant@5 | Graded@5 | Uncertain |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = per_arm[arm]
        lines.append(
            f"| {arm} | {values['human_relevance_rate_top1']} | "
            f"{values['human_relevance_rate_top5']} | "
            f"{values['human_graded_relevance_top5']} | "
            f"{values['uncertain_count']} |"
        )
    (root / "review/review_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return metrics

