"""AI-specific ingestion and scoring for frozen Stage 1D review judgments."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from triage_eg.retrieval.stage1d.contracts import ARMS, REVIEW_LABELS
from triage_eg.retrieval.stage1d.review import (
    IDENTITY_FIELDS,
    REVIEW_FIELDS,
    _arm_metrics,
)

from .contracts import (
    AI_REVIEW_STATUS,
    EVALUATION_MODE,
    EXPECTED_JUDGMENTS,
    EXPECTED_PAIRS,
    HUMAN_REVIEW_STATUS,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    PAIR_METRIC_FIELDS,
)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        return fields, list(reader)


def _score_is_one_ulp_or_closer(expected: str, actual: str) -> bool:
    try:
        left, right = float(expected), float(actual)
    except ValueError:
        return False
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= math.ulp(left)


def _ai_arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _arm_metrics(rows)
    result = {
        key: raw[key]
        for key in (
            "relevant_count",
            "partial_count",
            "irrelevant_count",
            "uncertain_count",
        )
    }
    for cutoff in (1, 5):
        result[f"ai_relevance_rate_top{cutoff}"] = raw[f"human_relevance_rate_top{cutoff}"]
        result[f"ai_graded_relevance_top{cutoff}"] = raw[f"human_graded_relevance_top{cutoff}"]
        result[f"uncertain_count_top{cutoff}"] = raw[f"uncertain_count_top{cutoff}"]
    return result


def _load_review_key(path: Path) -> dict[str, dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    mappings = {
        item["pair_id"]: item["conditions"]
        for item in value.get("pairs", [])
        if isinstance(item, dict)
    }
    if len(mappings) != EXPECTED_PAIRS:
        raise ValueError("AI_REVIEW_KEY_INVALID: expected 14 pair mappings")
    for pair_id, mapping in mappings.items():
        if set(mapping) != {"C01", "C02", "C03"} or set(mapping.values()) != set(ARMS):
            raise ValueError(f"AI_REVIEW_KEY_INVALID: invalid mapping for {pair_id}")
    return mappings


def _load_pair_metadata(stage1d_root: Path) -> dict[str, dict[str, Any]]:
    path = stage1d_root / "comparisons/pair_comparisons.jsonl"
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    metadata = {item["pair_id"]: item for item in values}
    if len(metadata) != EXPECTED_PAIRS:
        raise ValueError("AI_REVIEW_METADATA_INVALID: expected 14 pair records")
    return metadata


def validate_and_score_ai_review(
    stage1d_root: str | Path,
    judged_csv: str | Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]]]:
    """Validate AI labels against frozen identities and recompute all metrics.

    Frozen identity strings are required verbatim except for a bounded CSV
    serialization repair: an AI score differing by at most one IEEE-754 ULP is
    replaced with its frozen string. Scores are never used in relevance metrics.
    """

    root = Path(stage1d_root).expanduser().resolve(strict=True)
    template_path = root / "review/review_template_blinded.csv"
    key_path = root / "review/review_key.json"
    template_fields, template = _read_csv(template_path)
    judged_fields, judged = _read_csv(Path(judged_csv).expanduser().resolve(strict=True))
    if template_fields != REVIEW_FIELDS or judged_fields != REVIEW_FIELDS:
        raise ValueError("AI_REVIEW_IDENTITY_MISMATCH: review schema changed")
    if len(template) != EXPECTED_JUDGMENTS or len(judged) != EXPECTED_JUDGMENTS:
        raise ValueError("AI_REVIEW_INCOMPLETE: expected exactly 210 rows")
    template_by_id = {row["review_row_id"]: row for row in template}
    if len(template_by_id) != EXPECTED_JUDGMENTS:
        raise ValueError("AI_REVIEW_DUPLICATE: frozen template IDs are not unique")
    judged_ids = [row["review_row_id"] for row in judged]
    if len(set(judged_ids)) != len(judged_ids):
        raise ValueError("AI_REVIEW_DUPLICATE")
    if set(judged_ids) != set(template_by_id):
        raise ValueError("AI_REVIEW_IDENTITY_MISMATCH: review_row_id set differs")

    mappings = _load_review_key(key_path)
    normalized: list[dict[str, str]] = []
    resolved: list[dict[str, Any]] = []
    score_normalizations = 0
    max_score_delta = 0.0
    for actual in judged:
        expected = template_by_id[actual["review_row_id"]]
        canonical = dict(actual)
        for field in IDENTITY_FIELDS:
            if actual[field] == expected[field]:
                continue
            if field == "score" and _score_is_one_ulp_or_closer(expected[field], actual[field]):
                score_normalizations += 1
                max_score_delta = max(
                    max_score_delta, abs(float(expected[field]) - float(actual[field]))
                )
                canonical[field] = expected[field]
                continue
            raise ValueError(
                f"AI_REVIEW_IDENTITY_MISMATCH: {actual['review_row_id']} field={field}"
            )
        label = actual["review_label"].strip().upper()
        if not label:
            raise ValueError(f"AI_REVIEW_INCOMPLETE: missing label {actual['review_row_id']}")
        if label not in REVIEW_LABELS:
            raise ValueError(f"AI_REVIEW_LABEL_INVALID: {label!r}")
        pair_id, condition = canonical["pair_id"], canonical["condition_code"]
        try:
            arm = mappings[pair_id][condition]
        except (KeyError, TypeError) as error:
            raise ValueError("AI_REVIEW_IDENTITY_MISMATCH: unresolved condition") from error
        canonical["review_label"] = label
        normalized.append(canonical)
        resolved.append({**canonical, "actual_arm": arm})

    by_arm: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair_arm: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resolved:
        by_arm[row["actual_arm"]].append(row)
        by_pair_arm[(row["pair_id"], row["actual_arm"])].append(row)
    per_arm = {arm: _ai_arm_metrics(by_arm[arm]) for arm in ARMS}
    metadata = _load_pair_metadata(root)
    pair_metrics: list[dict[str, Any]] = []
    for pair_id in sorted(metadata):
        item: dict[str, Any] = {
            "pair_id": pair_id,
            "category": metadata[pair_id]["category"],
            "difficulty": metadata[pair_id]["difficulty"],
        }
        for arm in ARMS:
            values = _ai_arm_metrics(by_pair_arm[(pair_id, arm)])
            item[f"{arm}_relevance_top5"] = values["ai_relevance_rate_top5"]
            item[f"{arm}_graded_top5"] = values["ai_graded_relevance_top5"]
        item["translated_minus_vi_graded_top5"] = (
            item["VI_TRANSLATED_EN_graded_top5"] - item["VI_DIRECT_graded_top5"]
        )
        item["translated_minus_en_graded_top5"] = (
            item["VI_TRANSLATED_EN_graded_top5"] - item["EN_DIRECT_graded_top5"]
        )
        pair_metrics.append(item)
    deltas = [item["translated_minus_vi_graded_top5"] for item in pair_metrics]
    epsilon = 1e-12
    pair_comparison = {
        "translated_better_than_vi_direct_pairs_by_graded_top5": sum(
            value > epsilon for value in deltas
        ),
        "translated_tied_vi_direct_pairs_by_graded_top5": sum(
            abs(value) <= epsilon for value in deltas
        ),
        "translated_worse_than_vi_direct_pairs_by_graded_top5": sum(
            value < -epsilon for value in deltas
        ),
    }
    metrics = {
        "evaluation_mode": EVALUATION_MODE,
        "judge": {"provider": JUDGE_PROVIDER, "model": JUDGE_MODEL},
        "ai_review_status": AI_REVIEW_STATUS,
        "human_review_status": HUMAN_REVIEW_STATUS,
        "judgments_expected": EXPECTED_JUDGMENTS,
        "judgments_completed": len(resolved),
        "identity_validation": {
            "status": "VALID",
            "exact_non_score_identity_matches": EXPECTED_JUDGMENTS,
            "score_strings_canonicalized_within_one_ulp": score_normalizations,
            "max_score_absolute_delta": max_score_delta,
            "canonical_identity_source": "FROZEN_STAGE1D_REVIEW_TEMPLATE",
        },
        "per_arm": per_arm,
        "pair_comparison": pair_comparison,
        "non_claims": [
            "AI judgments are not human judgments",
            "This qualitative gate is not competition Recall@K",
            "The selected translator is not claimed globally optimal",
            "Final competition retrieval quality is not proven",
        ],
    }
    return metrics, normalized, pair_metrics


def validate_supplied_ai_metrics(
    recomputed: dict[str, Any],
    recomputed_pairs: list[dict[str, Any]],
    supplied_metrics_path: str | Path,
    supplied_pair_metrics_path: str | Path,
) -> None:
    """Require supplied AI summaries to agree with independently recomputed values."""

    supplied = json.loads(Path(supplied_metrics_path).read_text(encoding="utf-8"))
    if supplied.get("evaluation_mode") != EVALUATION_MODE:
        raise ValueError("AI_REVIEW_PROVENANCE_MISMATCH: evaluation mode")
    if supplied.get("judge") != JUDGE_MODEL:
        raise ValueError("AI_REVIEW_PROVENANCE_MISMATCH: judge model")
    if supplied.get("judgments_completed") != EXPECTED_JUDGMENTS:
        raise ValueError("AI_REVIEW_INCOMPLETE: supplied metrics")
    if (
        supplied.get("ai_review_status") != AI_REVIEW_STATUS
        or supplied.get("human_review_status") != HUMAN_REVIEW_STATUS
    ):
        raise ValueError("AI_REVIEW_PROVENANCE_MISMATCH: review statuses")
    for arm in ARMS:
        for name, value in recomputed["per_arm"][arm].items():
            supplied_value = supplied.get("per_arm", {}).get(arm, {}).get(name)
            if isinstance(value, float):
                if supplied_value is None or not math.isclose(
                    value, float(supplied_value), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"AI_REVIEW_METRIC_MISMATCH: {arm}/{name}")
            elif supplied_value != value:
                raise ValueError(f"AI_REVIEW_METRIC_MISMATCH: {arm}/{name}")
    for name, value in recomputed["pair_comparison"].items():
        if supplied.get("pair_comparison", {}).get(name) != value:
            raise ValueError(f"AI_REVIEW_METRIC_MISMATCH: pair comparison/{name}")
    supplied_fields, supplied_pairs = _read_csv(Path(supplied_pair_metrics_path))
    if len(supplied_pairs) != EXPECTED_PAIRS or supplied_fields != PAIR_METRIC_FIELDS:
        raise ValueError("AI_REVIEW_METRIC_MISMATCH: pair metrics")
    expected_by_pair = {item["pair_id"]: item for item in recomputed_pairs}
    if set(expected_by_pair) != {item["pair_id"] for item in supplied_pairs}:
        raise ValueError("AI_REVIEW_METRIC_MISMATCH: pair IDs")
    for supplied_pair in supplied_pairs:
        expected_pair = expected_by_pair[supplied_pair["pair_id"]]
        for name in PAIR_METRIC_FIELDS:
            if name in {"pair_id", "category", "difficulty"}:
                if supplied_pair[name] != str(expected_pair[name]):
                    raise ValueError(
                        f"AI_REVIEW_METRIC_MISMATCH: {supplied_pair['pair_id']}/{name}"
                    )
            elif not math.isclose(
                float(supplied_pair[name]),
                float(expected_pair[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"AI_REVIEW_METRIC_MISMATCH: {supplied_pair['pair_id']}/{name}")


__all__ = ["validate_and_score_ai_review", "validate_supplied_ai_metrics"]
