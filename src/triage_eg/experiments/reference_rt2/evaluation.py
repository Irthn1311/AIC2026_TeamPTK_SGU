"""Evaluation and reporting for the bounded RT2 DANTE calibration experiment."""

from __future__ import annotations

import hashlib
import math
import shutil
import textwrap
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml

from triage_eg.experiments.reference_rt1.scoring import (
    build_video_row_groups,
    rank_dante_dp,
    rank_unordered_event_max,
)
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage1d.artifacts import _paste_frame
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
)

from .benchmark import (
    BENCHMARK_TYPE,
    RT2_VERSION,
    RT2BenchmarkQuery,
    resolve_benchmark_identities,
    stable_query_order,
)

DEFAULT_LAMBDA_GRID = (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01)
VIDEO_CUTOFFS = (1, 5, 20)
MIN_USABLE_BENCHMARK = 18
FORBIDDEN_BUNDLE_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".mp4"}


@dataclass(frozen=True)
class RT2Settings:
    candidate_count: int = 36
    frames_per_sheet: int = 16
    seed: int = 2026
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    minimum_usable_queries: int = MIN_USABLE_BENCHMARK
    visual_top_k: int = 3

    def __post_init__(self) -> None:
        if self.candidate_count < 2 or not 1 <= self.frames_per_sheet <= 16:
            raise ValueError("invalid RT2 candidate settings")
        if not self.lambda_grid or len(set(self.lambda_grid)) != len(self.lambda_grid):
            raise ValueError("RT2 lambda grid must be non-empty and unique")
        if any(not math.isfinite(value) or value < 0 for value in self.lambda_grid):
            raise ValueError("RT2 lambda values must be finite and non-negative")
        if self.minimum_usable_queries < 1 or self.visual_top_k != 3:
            raise ValueError("invalid RT2 evaluation settings")


@dataclass(frozen=True)
class RT2RunnerConfig:
    stage2: Stage2RuntimeConfig
    dataset_root: Path
    benchmark_path: Path
    output_root: Path
    settings: RT2Settings


def load_rt2_settings(path: str | Path) -> RT2Settings:
    source = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("reference_experiment") != "RT2":
        raise ValueError("Invalid RT2 experiment configuration")
    candidate = value.get("candidate_preparation", {})
    evaluation = value.get("evaluation", {})
    return RT2Settings(
        candidate_count=int(candidate.get("candidate_count", 36)),
        frames_per_sheet=int(candidate.get("frames_per_sheet", 16)),
        seed=int(value.get("seed", 2026)),
        lambda_grid=tuple(float(item) for item in evaluation.get("lambda_grid", [])),
        minimum_usable_queries=int(evaluation.get("minimum_usable_queries", 18)),
        visual_top_k=int(evaluation.get("visual_top_k", 3)),
    )


def split_dev_holdout(
    queries: list[RT2BenchmarkQuery], seed: int = 2026
) -> tuple[list[RT2BenchmarkQuery], list[RT2BenchmarkQuery]]:
    """Create a deterministic approximately 2/3 DEV, 1/3 HOLDOUT split."""

    if len(queries) < 2:
        return list(queries), []
    grouped: dict[int, list[RT2BenchmarkQuery]] = {}
    for query in queries:
        grouped.setdefault(len(query.events), []).append(query)
    for values in grouped.values():
        values.sort(key=lambda query: stable_query_order(query, seed))
    holdout_target = max(1, len(queries) // 3)
    raw = {count: len(values) * holdout_target / len(queries) for count, values in grouped.items()}
    allocation = {
        count: min(int(math.floor(raw[count])), max(len(values) - 1, 0))
        for count, values in grouped.items()
    }
    remaining = holdout_target - sum(allocation.values())
    priority = sorted(
        grouped,
        key=lambda count: (-(raw[count] - math.floor(raw[count])), count),
    )
    while remaining > 0:
        changed = False
        for count in priority:
            capacity = len(grouped[count]) - allocation[count]
            if capacity <= 0:
                continue
            allocation[count] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            raise RuntimeError("Unable to allocate deterministic RT2 holdout")
    dev, holdout = [], []
    for count in sorted(grouped):
        values = grouped[count]
        holdout.extend(values[: allocation[count]])
        dev.extend(values[allocation[count] :])
    dev.sort(key=lambda query: query.query_id)
    holdout.sort(key=lambda query: query.query_id)
    if {item.query_id for item in dev} & {item.query_id for item in holdout}:
        raise RuntimeError("RT2 DEV and HOLDOUT overlap")
    return dev, holdout


def _lambda_key(value: float) -> str:
    return format(value, ".10g")


def _source_item(ranked: list[dict[str, Any]], source_video_id: str) -> dict[str, Any]:
    for item in ranked:
        if item["video_id"] == source_video_id:
            return item
    raise RuntimeError(f"Source video missing from ranking: {source_video_id}")


def video_rank_metrics(
    ranked: list[dict[str, Any]], source_video_id: str
) -> dict[str, float | int | bool]:
    source = _source_item(ranked, source_video_id)
    rank = int(source["video_rank"])
    return {
        "correct_video_rank": rank,
        "reciprocal_rank": 1.0 / rank,
        **{f"INTERNAL_VIDEO_RECALL_AT_{cutoff}": rank <= cutoff for cutoff in VIDEO_CUTOFFS},
    }


def aggregate_video_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return {
            "query_count": 0,
            "INTERNAL_VIDEO_RECALL_AT_1": None,
            "INTERNAL_VIDEO_RECALL_AT_5": None,
            "INTERNAL_VIDEO_RECALL_AT_20": None,
            "MRR": None,
        }
    return {
        "query_count": len(records),
        **{
            f"INTERNAL_VIDEO_RECALL_AT_{cutoff}": mean(
                float(item[f"INTERNAL_VIDEO_RECALL_AT_{cutoff}"]) for item in records
            )
            for cutoff in VIDEO_CUTOFFS
        },
        "MRR": mean(float(item["reciprocal_rank"]) for item in records),
    }


def ai_reference_anchor_metrics(
    query: RT2BenchmarkQuery, source_dante: dict[str, Any]
) -> dict[str, Any]:
    predicted = {item["event_id"]: int(item["catalog_position"]) for item in source_dante["chain"]}
    per_event = []
    for event in query.events:
        position = predicted[event.event_id]
        error = abs(position - event.reference_catalog_position)
        per_event.append(
            {
                "event_id": event.event_id,
                "reference_catalog_position": event.reference_catalog_position,
                "predicted_catalog_position": position,
                "absolute_catalog_position_error": error,
            }
        )
    errors = [item["absolute_catalog_position_error"] for item in per_event]
    return {
        "per_event": per_event,
        "AI_REFERENCE_ANCHOR_MEAN_ABSOLUTE_CATALOG_POSITION_ERROR": mean(errors),
        "AI_REFERENCE_ANCHOR_MEDIAN_ABSOLUTE_CATALOG_POSITION_ERROR": median(errors),
        "AI_REFERENCE_ANCHOR_HIT_WITHIN_1": mean(error <= 1 for error in errors),
        "AI_REFERENCE_ANCHOR_HIT_WITHIN_3": mean(error <= 3 for error in errors),
        "AI_REFERENCE_ANCHOR_HIT_WITHIN_5": mean(error <= 5 for error in errors),
    }


def aggregate_anchor_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        int(event["absolute_catalog_position_error"])
        for record in records
        for event in record["per_event"]
    ]
    if not errors:
        return {
            "event_count": 0,
            "AI_REFERENCE_ANCHOR_MEAN_ABSOLUTE_CATALOG_POSITION_ERROR": None,
            "AI_REFERENCE_ANCHOR_MEDIAN_ABSOLUTE_CATALOG_POSITION_ERROR": None,
            "AI_REFERENCE_ANCHOR_HIT_WITHIN_1": None,
            "AI_REFERENCE_ANCHOR_HIT_WITHIN_3": None,
            "AI_REFERENCE_ANCHOR_HIT_WITHIN_5": None,
        }
    return {
        "event_count": len(errors),
        "AI_REFERENCE_ANCHOR_MEAN_ABSOLUTE_CATALOG_POSITION_ERROR": mean(errors),
        "AI_REFERENCE_ANCHOR_MEDIAN_ABSOLUTE_CATALOG_POSITION_ERROR": median(errors),
        "AI_REFERENCE_ANCHOR_HIT_WITHIN_1": mean(error <= 1 for error in errors),
        "AI_REFERENCE_ANCHOR_HIT_WITHIN_3": mean(error <= 3 for error in errors),
        "AI_REFERENCE_ANCHOR_HIT_WITHIN_5": mean(error <= 5 for error in errors),
    }


def chain_collapse_metrics(
    query: RT2BenchmarkQuery, source_dante: dict[str, Any]
) -> dict[str, Any]:
    positions = [int(item["catalog_position"]) for item in source_dante["chain"]]
    reference = [event.reference_catalog_position for event in query.events]
    span = positions[-1] - positions[0]
    reference_span = reference[-1] - reference[0]
    return {
        "event_count": len(positions),
        "predicted_span": span,
        "reference_span": reference_span,
        "span_ratio": span / max(reference_span, 1),
        "adjacent_step_distances": [
            right - left for left, right in zip(positions[:-1], positions[1:], strict=True)
        ],
        "CHAIN_COLLAPSE_CANDIDATE": span <= len(positions),
    }


def aggregate_collapse_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "chain_count": 0,
            "CHAIN_COLLAPSE_CANDIDATE_COUNT": 0,
            "CHAIN_COLLAPSE_CANDIDATE_RATE": None,
            "span_ratio_distribution": None,
        }
    ratios = [float(item["span_ratio"]) for item in records]
    collapsed = sum(bool(item["CHAIN_COLLAPSE_CANDIDATE"]) for item in records)
    return {
        "chain_count": len(records),
        "CHAIN_COLLAPSE_CANDIDATE_COUNT": collapsed,
        "CHAIN_COLLAPSE_CANDIDATE_RATE": collapsed / len(records),
        "span_ratio_distribution": {
            "min": min(ratios),
            "max": max(ratios),
            "mean": mean(ratios),
            "median": median(ratios),
        },
    }


def order_discrimination_metrics(
    correct: dict[str, Any], reversed_order: dict[str, Any]
) -> dict[str, Any]:
    correct_rank = int(correct["video_rank"])
    reversed_rank = int(reversed_order["video_rank"])
    correct_score = float(correct["dante_score"])
    reversed_score = float(reversed_order["dante_score"])
    return {
        "correct_order_video_rank": correct_rank,
        "reversed_order_video_rank": reversed_rank,
        "correct_order_source_score": correct_score,
        "reversed_order_source_score": reversed_score,
        "rank_improvement_correct_vs_reversed": reversed_rank - correct_rank,
        "score_margin_correct_vs_reversed": correct_score - reversed_score,
        "correct_order_ranked_better": correct_rank < reversed_rank,
    }


def aggregate_order_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(records),
        "ORDER_DISCRIMINATION_RATE": (
            mean(bool(item["correct_order_ranked_better"]) for item in records) if records else None
        ),
        "mean_rank_improvement_correct_vs_reversed": (
            mean(float(item["rank_improvement_correct_vs_reversed"]) for item in records)
            if records
            else None
        ),
        "mean_score_margin_correct_vs_reversed": (
            mean(float(item["score_margin_correct_vs_reversed"]) for item in records)
            if records
            else None
        ),
    }


def select_lambda_from_dev(table: list[dict[str, Any]]) -> float:
    """Apply the frozen transparent lexicographic DEV-only selection rule."""

    if not table or any(item.get("split") != "DEV" for item in table):
        raise ValueError("Lambda selection accepts a non-empty DEV-only table")

    def key(item: dict[str, Any]) -> tuple[float, ...]:
        video = item["video_metrics"]
        anchor = item["anchor_metrics"]
        mae = anchor["AI_REFERENCE_ANCHOR_MEAN_ABSOLUTE_CATALOG_POSITION_ERROR"]
        return (
            float(video["INTERNAL_VIDEO_RECALL_AT_5"]),
            float(video["MRR"]),
            float(anchor["AI_REFERENCE_ANCHOR_HIT_WITHIN_3"]),
            -float(mae),
            -abs(float(item["lambda"])),
        )

    return float(max(table, key=key)["lambda"])


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _blinded_mapping(query_id: str, seed: int) -> dict[str, str]:
    digest = hashlib.sha256(f"RT2:{seed}:{query_id}".encode()).digest()
    methods = (
        ("DANTE_SELECTED_LAMBDA", "UNORDERED_EVENT_MAX")
        if digest[0] & 1
        else ("UNORDERED_EVENT_MAX", "DANTE_SELECTED_LAMBDA")
    )
    return {"METHOD_A": methods[0], "METHOD_B": methods[1]}


def render_holdout_ab_sheet(
    output_path: Path,
    *,
    dataset_root: Path,
    query: RT2BenchmarkQuery,
    unordered: list[dict[str, Any]],
    dante: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    event_count = len(query.events)
    tile_width, image_height, label_height = 300, 170, 54
    method_width = event_count * tile_width
    header_height = 150
    sheet = Image.new(
        "RGB", (method_width * 2, header_height + 3 * (image_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{query.query_id} · HOLDOUT BLIND REVIEW", fill="black", font=_font(21))
    event_line = " | ".join(f"{event.event_id}: {event.text}" for event in query.events)
    wrapped_events = "\n".join(textwrap.wrap(event_line, width=max(80, event_count * 45)))
    draw.multiline_text((10, 40), wrapped_events, fill="black", font=_font(17), spacing=3)
    issues = []
    by_method = {"UNORDERED_EVENT_MAX": unordered, "DANTE_SELECTED_LAMBDA": dante}
    for method_index, blind_name in enumerate(("METHOD_A", "METHOD_B")):
        method = mapping[blind_name]
        draw.text(
            (method_index * method_width + 10, 112),
            blind_name,
            fill="black",
            font=_font(21),
        )
        for row_index, video in enumerate(by_method[method][:3]):
            anchors = video["event_best"] if method == "UNORDERED_EVENT_MAX" else video["chain"]
            for column, anchor in enumerate(anchors):
                x = method_index * method_width + column * tile_width
                y = header_height + row_index * (image_height + label_height)
                issue = _paste_frame(
                    sheet,
                    draw,
                    item={**anchor, "query_id": query.query_id},
                    dataset_root=dataset_root,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=image_height,
                )
                if issue:
                    issues.append(issue)
                draw.multiline_text(
                    (x + 6, y + image_height + 4),
                    (
                        f"{video['video_id']} · {anchor['event_id']}\n"
                        f"original_frame_idx: {anchor['original_frame_idx']}"
                    ),
                    fill="black",
                    font=_font(15),
                    spacing=2,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return issues


def _aggregate_lambda(records: list[dict[str, Any]], split_ids: set[str]) -> dict[str, Any]:
    selected = [item for item in records if item["query_id"] in split_ids]
    return {
        "video_metrics": aggregate_video_metrics([item["video_metrics"] for item in selected]),
        "anchor_metrics": aggregate_anchor_metrics([item["anchor_metrics"] for item in selected]),
        "collapse_metrics": aggregate_collapse_metrics(
            [item["collapse_metrics"] for item in selected]
        ),
        "order_metrics": aggregate_order_metrics([item["order_metrics"] for item in selected]),
    }


def _report_markdown(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Reference Experiment RT2 Report",
        "",
        "RT2 evaluates an AI-curated internal pseudo-GT benchmark. It is not an official "
        "competition evaluation and no human review has been performed.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Benchmark queries: `{summary['benchmark_query_count']}`",
        f"- Calibration status: `{summary['calibration_status']}`",
        f"- Selected lambda: `{summary['selected_lambda']}`",
        "- DANTE quality decision: `NOT_EVALUATED`",
        "",
        "## DEV lambda table",
        "",
        "| lambda | R@5 | MRR | anchor hit ±3 | anchor MAE | collapse rate | order rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["dev_lambda_table"]:
        video, anchor = item["video_metrics"], item["anchor_metrics"]
        collapse, order = item["collapse_metrics"], item["order_metrics"]
        lines.append(
            "| {lambda_value:g} | {r5:.4f} | {mrr:.4f} | {hit:.4f} | {mae:.4f} | "
            "{collapse:.4f} | {order:.4f} |".format(
                lambda_value=item["lambda"],
                r5=video["INTERNAL_VIDEO_RECALL_AT_5"] or 0.0,
                mrr=video["MRR"] or 0.0,
                hit=anchor["AI_REFERENCE_ANCHOR_HIT_WITHIN_3"] or 0.0,
                mae=anchor["AI_REFERENCE_ANCHOR_MEAN_ABSOLUTE_CATALOG_POSITION_ERROR"] or 0.0,
                collapse=collapse["CHAIN_COLLAPSE_CANDIDATE_RATE"] or 0.0,
                order=order["ORDER_DISCRIMINATION_RATE"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "Lambda is selected on DEV only by R@5, MRR, anchor hit ±3, anchor MAE, then "
            "smaller lambda. HOLDOUT is never used for selection.",
            "",
            "## Non-claims",
            "",
            "- INTERNAL_VIDEO_RECALL_AT_K is not an official competition score.",
            "- AI_REFERENCE_ANCHOR metrics are approximate because labels come from sparse sheets.",
            "- RT2 does not automatically KEEP, REDESIGN, or DROP DANTE.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_reference_rt2_evaluation(
    config: RT2RunnerConfig,
    queries: list[RT2BenchmarkQuery],
    *,
    runtime: OperationalRetrievalRuntime | None = None,
) -> dict[str, Any]:
    output = config.output_root.expanduser().resolve(strict=False)
    dataset = config.dataset_root.expanduser().resolve(strict=True)
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        if runtime is None or existing - {"_stage2_control"}:
            raise FileExistsError(f"RT2 evaluation output already exists: {output}")
    else:
        output.mkdir(parents=True)
    write_jsonl(output / "benchmark/rt2_ai_benchmark.jsonl", [item.as_dict() for item in queries])
    stage2_config = replace(config.stage2, output_root=output / "_stage2_control")
    active_runtime = runtime or OperationalRetrievalRuntime(stage2_config)
    owns_runtime = runtime is None
    issues: list[dict[str, Any]] = []
    started = monotonic()
    try:
        active_runtime.load()
        resolve_benchmark_identities(queries, active_runtime.catalog)
        groups = build_video_row_groups(active_runtime.catalog)
        dev, holdout = split_dev_holdout(queries, config.settings.seed)
        dev_ids = {item.query_id for item in dev}
        holdout_ids = {item.query_id for item in holdout}
        lambda_records: dict[str, list[dict[str, Any]]] = {
            _lambda_key(value): [] for value in config.settings.lambda_grid
        }
        unordered_records = []
        visual_cache: dict[str, dict[str, Any]] = {}
        timing_records = []
        for query in queries:
            query_started = monotonic()
            requests = [
                QueryRequest(f"{query.query_id}__{event.event_id}", event.text, query.language, 1)
                for event in query.events
            ]
            encoding_started = monotonic()
            encoded = active_runtime.encode_requests(requests)
            encoding_ms = (monotonic() - encoding_started) * 1000
            scoring_started = monotonic()
            full_scores = np.asarray(
                active_runtime.backend.score_many_all(encoded.embeddings), dtype=np.float32
            )
            scoring_ms = (monotonic() - scoring_started) * 1000
            expected_shape = (len(query.events), active_runtime.backend.size)
            if full_scores.shape != expected_shape or not np.isfinite(full_scores).all():
                raise RuntimeError(f"Invalid RT2 score matrix for {query.query_id}")
            event_ids = [event.event_id for event in query.events]
            unordered_started = monotonic()
            unordered = rank_unordered_event_max(
                full_scores, event_ids, groups, active_runtime.catalog
            )
            unordered_reversed = rank_unordered_event_max(
                full_scores[::-1], event_ids[::-1], groups, active_runtime.catalog
            )
            left = [(item["video_id"], item["unordered_score"]) for item in unordered]
            right = [(item["video_id"], item["unordered_score"]) for item in unordered_reversed]
            if any(
                a[0] != b[0] or not np.isclose(a[1], b[1]) for a, b in zip(left, right, strict=True)
            ):
                raise RuntimeError("UNORDERED_EVENT_MAX is not permutation invariant")
            unordered_ms = (monotonic() - unordered_started) * 1000
            unordered_video = video_rank_metrics(unordered, query.source_video_id)
            unordered_records.append({"query_id": query.query_id, **unordered_video})
            per_lambda_output = {}
            lambda_sweep_started = monotonic()
            for value in config.settings.lambda_grid:
                key = _lambda_key(value)
                correct = rank_dante_dp(
                    full_scores,
                    event_ids,
                    groups,
                    active_runtime.catalog,
                    distance_lambda=value,
                )
                reversed_ranked = rank_dante_dp(
                    full_scores[::-1],
                    event_ids[::-1],
                    groups,
                    active_runtime.catalog,
                    distance_lambda=value,
                )
                source_correct = _source_item(correct, query.source_video_id)
                source_reversed = _source_item(reversed_ranked, query.source_video_id)
                record = {
                    "query_id": query.query_id,
                    "video_metrics": video_rank_metrics(correct, query.source_video_id),
                    "anchor_metrics": ai_reference_anchor_metrics(query, source_correct),
                    "collapse_metrics": chain_collapse_metrics(query, source_correct),
                    "order_metrics": order_discrimination_metrics(source_correct, source_reversed),
                }
                lambda_records[key].append(record)
                per_lambda_output[key] = {
                    **record,
                    "top3_videos": correct[: config.settings.visual_top_k],
                }
            lambda_sweep_ms = (monotonic() - lambda_sweep_started) * 1000
            query_output = {
                "query_id": query.query_id,
                "split": "DEV" if query.query_id in dev_ids else "HOLDOUT",
                "source_video_id": query.source_video_id,
                "event_count": len(query.events),
                "event_encoding": [
                    {
                        "event_id": event.event_id,
                        "language_resolution": encoded.resolutions[index].as_dict(),
                        **encoded.encodings[index],
                    }
                    for index, event in enumerate(query.events)
                ],
                "UNORDERED_EVENT_MAX": {
                    "video_metrics": unordered_video,
                    "permutation_invariant": True,
                    "top3_videos": unordered[: config.settings.visual_top_k],
                },
                "DANTE_DP": per_lambda_output,
            }
            write_json(output / "query_results" / f"{query.query_id}.json", query_output)
            visual_cache[query.query_id] = {
                "unordered": unordered[: config.settings.visual_top_k],
                "dante": {key: value["top3_videos"] for key, value in per_lambda_output.items()},
            }
            timing_records.append(
                {
                    "query_id": query.query_id,
                    "event_encoding_ms": encoding_ms,
                    "full_score_matrix_ms": scoring_ms,
                    "unordered_with_invariance_check_ms": unordered_ms,
                    "dante_lambda_and_reverse_sweep_ms": lambda_sweep_ms,
                    "total_ms": (monotonic() - query_started) * 1000,
                    "score_matrix_computations": 1,
                }
            )
        unordered_by_split = {
            "DEV": aggregate_video_metrics(
                [item for item in unordered_records if item["query_id"] in dev_ids]
            ),
            "HOLDOUT": aggregate_video_metrics(
                [item for item in unordered_records if item["query_id"] in holdout_ids]
            ),
        }
        lambda_metrics = {}
        dev_table = []
        for value in config.settings.lambda_grid:
            key = _lambda_key(value)
            by_split = {
                "DEV": _aggregate_lambda(lambda_records[key], dev_ids),
                "HOLDOUT": _aggregate_lambda(lambda_records[key], holdout_ids),
            }
            lambda_metrics[key] = by_split
            dev_table.append({"lambda": value, "split": "DEV", **by_split["DEV"]})
        enough = len(queries) >= config.settings.minimum_usable_queries
        selected_lambda = select_lambda_from_dev(dev_table) if enough else None
        selected_key = _lambda_key(selected_lambda) if selected_lambda is not None else None
        holdout_comparison = None
        review_key = []
        if selected_key is not None:
            baseline = unordered_by_split["HOLDOUT"]
            selected = lambda_metrics[selected_key]["HOLDOUT"]
            holdout_comparison = {
                "UNORDERED_EVENT_MAX": baseline,
                "DANTE_DP_SELECTED_LAMBDA": {
                    "lambda": selected_lambda,
                    **selected,
                },
                "delta_dante_minus_unordered": {
                    metric: selected["video_metrics"][metric] - baseline[metric]
                    for metric in (
                        "INTERNAL_VIDEO_RECALL_AT_1",
                        "INTERNAL_VIDEO_RECALL_AT_5",
                        "INTERNAL_VIDEO_RECALL_AT_20",
                        "MRR",
                    )
                },
            }
            for query in holdout:
                mapping = _blinded_mapping(query.query_id, config.settings.seed)
                issues.extend(
                    render_holdout_ab_sheet(
                        output / "visuals" / f"{query.query_id}_ab.jpg",
                        dataset_root=dataset,
                        query=query,
                        unordered=visual_cache[query.query_id]["unordered"],
                        dante=visual_cache[query.query_id]["dante"][selected_key],
                        mapping=mapping,
                    )
                )
                review_key.append({"query_id": query.query_id, **mapping})
            write_json(
                output / "visuals/review_key.json",
                {
                    "seed": config.settings.seed,
                    "selected_lambda": selected_lambda,
                    "queries": review_key,
                },
            )
        metrics = {
            "BENCHMARK_TYPE": BENCHMARK_TYPE,
            "metric_scope": "INTERNAL_PSEUDO_GT_NOT_OFFICIAL_COMPETITION_SCORE",
            "lambda_grid": list(config.settings.lambda_grid),
            "split": {
                "seed": config.settings.seed,
                "DEV": sorted(dev_ids),
                "HOLDOUT": sorted(holdout_ids),
                "stratification": "EVENT_COUNT_WHEN_POSSIBLE",
            },
            "UNORDERED_EVENT_MAX": unordered_by_split,
            "DANTE_DP": lambda_metrics,
            "dev_lambda_table": dev_table,
            "lambda_selection": {
                "uses_split": "DEV_ONLY",
                "procedure": [
                    "MAXIMIZE_INTERNAL_VIDEO_RECALL_AT_5",
                    "MAXIMIZE_MRR",
                    "MAXIMIZE_AI_REFERENCE_ANCHOR_HIT_WITHIN_3",
                    "MINIMIZE_AI_REFERENCE_ANCHOR_MAE",
                    "PREFER_SMALLER_ABSOLUTE_LAMBDA",
                ],
                "selected_lambda": selected_lambda,
                "holdout_used_for_selection": False,
            },
            "holdout_comparison": holdout_comparison,
        }
        calibration_status = "COMPLETE" if enough else "INSUFFICIENT_BENCHMARK"
        summary = {
            "reference_experiment": "RT2",
            "status": "COMPLETE",
            "calibration_status": calibration_status,
            "benchmark_query_count": len(queries),
            "dev_query_count": len(dev),
            "holdout_query_count": len(holdout),
            "selected_lambda": selected_lambda,
            "BENCHMARK_TYPE": BENCHMARK_TYPE,
            "HUMAN_REVIEW_STATUS": "NOT_PERFORMED",
            "OFFICIAL_GT_STATUS": "NOT_AVAILABLE",
            "DANTE_QUALITY_DECISION": "NOT_EVALUATED",
            "issues": len(issues),
        }
        manifest = {
            **summary,
            "rt2_version": RT2_VERSION,
            "completed_at": datetime.now(UTC).isoformat(),
            "build_git_commit": config.stage2.build_git_commit,
            "stage1_index_fingerprint": active_runtime.preflight["stage1_index_fingerprint"],
            "stage2a_runtime_manifest": active_runtime.runtime_manifest(),
            "arms": ["UNORDERED_EVENT_MAX", "DANTE_DP_LAMBDA_GRID"],
            "lambda_grid": list(config.settings.lambda_grid),
            "event_score_matrix_reused_across_lambdas": True,
            "score_matrix_computations_per_query": 1,
            "canonical_temporal_order": "N_ASCENDING_THEN_GLOBAL_ROW",
            "no_stage1_rebuild": True,
            "no_model_change": True,
            "no_new_scoring_formula": True,
            "network_required": False,
            "timings": {
                "queries": timing_records,
                "experiment_total_ms": (monotonic() - started) * 1000,
            },
        }
        write_json(output / "rt2_summary.json", summary)
        write_json(output / "rt2_metrics.json", metrics)
        (output / "rt2_report.md").write_text(_report_markdown(summary, metrics), encoding="utf-8")
        write_jsonl(output / "issues.jsonl", issues)
        write_json(output / "run_manifest.json", manifest)
        return summary
    finally:
        if owns_runtime:
            active_runtime.close()


def create_rt2_evaluation_bundle(root: str | Path, zip_path: str | Path) -> Path:
    source = Path(root).expanduser().resolve(strict=True)
    target = Path(zip_path).expanduser().resolve(strict=False)
    if source in target.parents:
        raise ValueError("RT2 evaluation ZIP must be outside the output root")
    required = (
        "rt2_summary.json",
        "rt2_metrics.json",
        "rt2_report.md",
        "issues.jsonl",
        "run_manifest.json",
        "benchmark/rt2_ai_benchmark.jsonl",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RT2 evaluation artifacts: {missing}")
    members = [source / name for name in required]
    members.extend(sorted((source / "query_results").glob("*.json")))
    if (source / "visuals/review_key.json").is_file():
        members.append(source / "visuals/review_key.json")
    members.extend(sorted((source / "visuals").glob("*.jpg")))
    relative = [path.relative_to(source).as_posix() for path in members]
    if any(
        Path(name).suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES
        or name.startswith(("_stage2_control/", "cache/", "logs/", "index/", "models/"))
        for name in relative
    ):
        raise ValueError("RT2 evaluation bundle contains a forbidden artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".building")
    staging.unlink(missing_ok=True)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for path, name in sorted(zip(members, relative, strict=True), key=lambda item: item[1]):
                archive.write(path, arcname=name)
        shutil.move(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target


__all__ = [
    "DEFAULT_LAMBDA_GRID",
    "MIN_USABLE_BENCHMARK",
    "RT2RunnerConfig",
    "RT2Settings",
    "aggregate_anchor_metrics",
    "aggregate_collapse_metrics",
    "aggregate_order_metrics",
    "aggregate_video_metrics",
    "ai_reference_anchor_metrics",
    "chain_collapse_metrics",
    "create_rt2_evaluation_bundle",
    "load_rt2_settings",
    "order_discrimination_metrics",
    "render_holdout_ab_sheet",
    "run_reference_rt2_evaluation",
    "select_lambda_from_dev",
    "split_dev_holdout",
    "video_rank_metrics",
]
