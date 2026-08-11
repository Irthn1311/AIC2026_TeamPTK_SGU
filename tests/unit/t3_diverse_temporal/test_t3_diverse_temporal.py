from __future__ import annotations

import inspect
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.experiments.reference_rt2 import (
    RT2BenchmarkQuery,
    RT2ReferenceEvent,
    split_dev_holdout,
)
from triage_eg.experiments.t3_diverse_temporal import (
    DiverseTemporalPath,
    EventCandidate,
    build_diverse_event_pool,
    create_t3_bundle,
    enumerate_feasible_paths,
    event_region_novelty,
    select_coverage_aware,
    select_score_top_k,
    validate_a0_reproduction,
)
from triage_eg.experiments.t3_diverse_temporal import hypotheses as hypotheses_module
from triage_eg.experiments.t3_diverse_temporal.runner import _select_delta, run_t3
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl


def _candidate(event: str, region: int, position: int, score: float) -> EventCandidate:
    return EventCandidate(
        event,
        f"{event}:R{region:02d}:P{position:06d}",
        position,
        position * 100,
        score,
    )


def _path(score: float, positions: tuple[int, int], regions: tuple[str, str]):
    return DiverseTemporalPath(score, positions, regions, (score / 2, score / 2))


def test_a0_reproduction_requires_exactly_54_of_74() -> None:
    rows = [
        {"by_k": {"5": {"reachable_seconds": {"6": index < 54}}}}
        for index in range(74)
    ]
    assert validate_a0_reproduction(rows) == 54
    rows[54]["by_k"]["5"]["reachable_seconds"]["6"] = True
    with pytest.raises(RuntimeError, match="T3_A0_REPRODUCTION_FAILED"):
        validate_a0_reproduction(rows)


def test_candidate_pool_ranking_is_deterministic() -> None:
    scores = np.asarray([0.5, 0.9, 0.9, 0.4], dtype=np.float32)
    frames = np.asarray([0, 100, 500, 900], dtype=np.int64)
    first = build_diverse_event_pool("E1", scores, frames, fps=10.0)
    second = build_diverse_event_pool("E1", scores, frames, fps=10.0)
    assert first == second
    assert first[0].catalog_position == 1


def test_temporal_nms_uses_original_frames_over_fps() -> None:
    scores = np.asarray([1.0, 0.9, 0.8], dtype=np.float32)
    frames = np.asarray([0, 10, 1_000], dtype=np.int64)
    pool = build_diverse_event_pool("E1", scores, frames, fps=10.0)
    assert [item.catalog_position for item in pool] == [0, 2]


def test_retained_regions_are_more_than_three_seconds_apart() -> None:
    scores = np.linspace(1.0, 0.1, 20, dtype=np.float32)
    frames = np.arange(20, dtype=np.int64) * 20
    pool = build_diverse_event_pool("E1", scores, frames, fps=10.0)
    assert all(
        abs(left.original_frame_idx - right.original_frame_idx) / 10.0 > 3.0
        for index, left in enumerate(pool)
        for right in pool[index + 1 :]
    )


def test_strict_monotonic_enumeration_is_correct() -> None:
    pools = (
        (_candidate("E1", 1, 0, 0.9), _candidate("E1", 2, 2, 0.8)),
        (_candidate("E2", 1, 1, 0.7), _candidate("E2", 2, 3, 0.95)),
    )
    paths, raw_count = enumerate_feasible_paths(pools)
    assert raw_count == 4
    assert {path.positions for path in paths} == {(0, 1), (0, 3), (2, 3)}
    assert all(path.positions[0] < path.positions[1] for path in paths)


def test_enumeration_contains_true_score_best_path_and_exact_sum() -> None:
    pools = (
        (_candidate("E1", 1, 0, 0.9), _candidate("E1", 2, 1, 0.8)),
        (_candidate("E2", 1, 2, 0.95), _candidate("E2", 2, 3, 0.7)),
    )
    paths, _ = enumerate_feasible_paths(pools)
    assert paths[0].positions == (0, 2)
    assert paths[0].score == pytest.approx(sum(paths[0].event_scores))
    assert paths[0].score == pytest.approx(1.85)


def test_a1_returns_score_top_five() -> None:
    paths = tuple(
        _path(score, (index, index + 10), (f"E1:R{index}", f"E2:R{index}"))
        for index, score in enumerate((0.7, 1.0, 0.6, 0.9, 0.8, 0.5))
    )
    selected = select_score_top_k(paths)
    assert [path.score for path in selected] == [1.0, 0.9, 0.8, 0.7, 0.6]


def test_a2_always_retains_global_best_path() -> None:
    paths = tuple(
        _path(score, (index, index + 10), (f"E1:R{index}", f"E2:R{index}"))
        for index, score in enumerate((1.0, 0.99, 0.98, 0.97, 0.96, 0.95))
    )
    selected = select_coverage_aware(paths, 0.05)
    assert selected[0] == paths[0]


def test_a2_novelty_is_per_event_position() -> None:
    selected = (_path(1.0, (0, 10), ("E1:R1", "E2:R1")),)
    e2_new = _path(0.99, (0, 11), ("E1:R1", "E2:R2"))
    both_seen = _path(0.98, (1, 10), ("E1:R1", "E2:R1"))
    assert event_region_novelty(e2_new, selected) == 1
    assert event_region_novelty(both_seen, selected) == 0


def test_a2_paths_remain_unique_and_strictly_monotonic() -> None:
    paths = tuple(
        _path(1.0 - index * 0.005, (index, index + 10), (f"E1:R{index}", f"E2:R{index}"))
        for index in range(8)
    )
    selected = select_coverage_aware(paths, 0.05)
    assert len(selected) == 5
    assert len({path.positions for path in selected}) == 5
    assert all(path.positions[0] < path.positions[1] for path in selected)


def test_generation_module_has_no_pseudo_reference_inputs() -> None:
    source = inspect.getsource(hypotheses_module).lower()
    assert "reference_original_frame_idx" not in source
    for function in (
        build_diverse_event_pool,
        enumerate_feasible_paths,
        select_score_top_k,
        select_coverage_aware,
    ):
        assert "reference" not in inspect.signature(function).parameters


def _synthetic_query(index: int, event_count: int) -> RT2BenchmarkQuery:
    events = tuple(
        RT2ReferenceEvent(
            f"E{event + 1}",
            f"event {event + 1}",
            f"S{event + 1:02d}",
            event,
            index * 10 + event,
            event + 1,
            event * 100,
        )
        for event in range(event_count)
    )
    return RT2BenchmarkQuery(
        f"q{index:02d}",
        "AI_CURATED_INTERNAL_PSEUDO_GT",
        f"L01_V{index + 1:03d}",
        "en",
        events,
        ("MULTI_EVENT",),
        "GPT-5.6 Sol",
        False,
    )


def test_dev_holdout_are_disjoint_and_selection_precedes_holdout_processing() -> None:
    queries = [_synthetic_query(index, 2 + index % 3) for index in range(24)]
    dev, holdout = split_dev_holdout(queries, 2026)
    assert len(dev) == 16 and len(holdout) == 8
    assert {query.query_id for query in dev}.isdisjoint(
        {query.query_id for query in holdout}
    )
    source = inspect.getsource(run_t3)
    assert source.index("selected_delta = _select_delta(dev_sweep)") < source.index(
        "holdout_results = []"
    )


def test_dev_delta_selection_obeys_lexicographic_rule_and_smaller_tie() -> None:
    rows = [
        {
            "delta": delta,
            "EVENT_WINDOW_RECALL@5": 0.8,
            "SINGLE_PATH_ALL_EVENTS_REACHABLE@5": 0.6,
            "EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY@5": 2.0,
            "mean_selected_path_relative_score_gap": 0.01,
        }
        for delta in (0.01, 0.03, 0.05)
    ]
    assert _select_delta(rows) == 0.01
    rows[1]["EVENT_WINDOW_RECALL@5"] = 0.9
    assert _select_delta(rows) == 0.03


def test_bundle_excludes_heavy_assets(tmp_path: Path) -> None:
    output = tmp_path / "t3"
    for name in (
        "t3_summary.json",
        "t3_metrics.json",
        "dev_delta_sweep.json",
        "holdout_comparison.json",
        "run_manifest.json",
    ):
        write_json(output / name, {})
    for name in (
        "event_candidate_pools.jsonl",
        "query_hypotheses.jsonl",
        "event_reachability.jsonl",
        "issues.jsonl",
    ):
        write_jsonl(output / name, [])
    (output / "t3_report.md").write_text("report\n", encoding="utf-8")
    (output / "cache").mkdir()
    np.save(output / "cache/scores.npy", np.ones(2))
    (output / "video.mp4").write_bytes(b"heavy")
    bundle = create_t3_bundle(output, tmp_path / "bundle.zip")
    with ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert names == {
        "t3_summary.json",
        "t3_metrics.json",
        "dev_delta_sweep.json",
        "event_candidate_pools.jsonl",
        "query_hypotheses.jsonl",
        "event_reachability.jsonl",
        "holdout_comparison.json",
        "run_manifest.json",
        "issues.jsonl",
        "t3_report.md",
    }
    assert not any(name.endswith((".npy", ".mp4")) for name in names)
