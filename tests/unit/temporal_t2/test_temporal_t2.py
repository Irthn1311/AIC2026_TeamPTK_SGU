from __future__ import annotations

import ast
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.reference_rt1 import dante_monotonic_dp
from triage_eg.experiments.temporal_t2 import (
    build_t2_metrics,
    create_t2_bundle,
    k_best_monotonic_paths,
    query_all_events_reachable,
    validate_recall_monotonicity,
)
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl


def _brute_k(scores: np.ndarray, k: int) -> list[tuple[float, tuple[int, ...]]]:
    candidates = []
    for positions in combinations(range(scores.shape[1]), scores.shape[0]):
        score = sum(float(scores[index, position]) for index, position in enumerate(positions))
        candidates.append((score, positions))
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[:k]


def test_k1_exactly_reproduces_existing_lambda_zero_dp() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(20):
        scores = rng.normal(size=(4, 18)).astype(np.float32)
        expected = dante_monotonic_dp(scores, distance_lambda=0.0)
        actual = k_best_monotonic_paths(scores, 1)
        assert expected is not None and len(actual) == 1
        assert actual[0].positions == expected.positions
        assert np.isclose(actual[0].score, expected.score)


def test_paths_are_strict_unique_and_match_bounded_brute_force() -> None:
    scores = np.asarray(
        [[0.9, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.9]], dtype=np.float32
    )
    paths = k_best_monotonic_paths(scores, 5)
    brute = _brute_k(scores, 5)
    assert [(path.score, path.positions) for path in paths] == brute
    assert len({path.positions for path in paths}) == len(paths)
    assert all(
        left < right
        for path in paths
        for left, right in zip(path.positions[:-1], path.positions[1:], strict=True)
    )


def test_kbest_matches_brute_force_on_seeded_small_matrices() -> None:
    rng = np.random.default_rng(2027)
    for _ in range(10):
        scores = rng.normal(size=(3, 7)).astype(np.float32)
        actual = k_best_monotonic_paths(scores, 5)
        expected = _brute_k(scores, 5)
        assert [(path.score, path.positions) for path in actual] == expected


def test_deterministic_ties_use_lexicographic_positions() -> None:
    scores = np.zeros((2, 4), dtype=np.float32)
    first = k_best_monotonic_paths(scores, 5)
    second = k_best_monotonic_paths(scores, 5)
    expected = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3))
    assert tuple(path.positions for path in first) == expected
    assert first == second
    baseline = dante_monotonic_dp(scores, distance_lambda=0.0)
    assert baseline is not None and first[0].positions == baseline.positions


def test_synthetic_alternate_valid_path_is_retained() -> None:
    scores = np.asarray([[0.9, 0.8, 0.0], [0.0, 0.8, 0.9]], dtype=np.float32)
    paths = k_best_monotonic_paths(scores, 3)
    assert paths[0].positions == (0, 2)
    assert {path.positions for path in paths} >= {(0, 1), (1, 2)}


def _event_row(query_id: str, event_count: int, reachable: list[bool]) -> dict[str, object]:
    return {
        "query_id": query_id,
        "event_count": event_count,
        "by_k": {
            str(k): {"reachable_seconds": {str(t): reachable[index] for t in (3, 6, 9, 12)}}
            for index, k in enumerate((1, 3, 5))
        },
    }


def _query_row(query_id: str, event_count: int, reachable: list[bool]) -> dict[str, object]:
    return {
        "query_id": query_id,
        "event_count": event_count,
        "by_k": {
            str(k): {
                "all_events_reachable_seconds": {
                    str(t): reachable[index] for t in (3, 6, 9, 12)
                }
            }
            for index, k in enumerate((1, 3, 5))
        },
        "path_diversity": {
            str(k): {
                "unique_path_count": k,
                "duplicate_path_rate": 0.0,
                "anchor_position_diversity_per_event": [k, k],
            }
            for k in (1, 3, 5)
        },
    }


def test_recall_at_k_is_monotonic_non_decreasing() -> None:
    events = [
        _event_row("q1", 2, [False, True, True]),
        _event_row("q1", 2, [True, True, True]),
    ]
    queries = [_query_row("q1", 2, [False, True, True])]
    metrics = build_t2_metrics(events, queries)
    validate_recall_monotonicity(metrics)
    primary = metrics["OVERALL"]["PRIMARY_6_SECONDS"]
    assert primary["EVENT_WINDOW_RECALL@1"] <= primary["EVENT_WINDOW_RECALL@3"]
    assert primary["EVENT_WINDOW_RECALL@3"] <= primary["EVENT_WINDOW_RECALL@5"]


def test_query_all_events_reachable_requires_every_event() -> None:
    rows = [
        {"by_k": {"3": {"reachable_seconds": {"6": True}}}},
        {"by_k": {"3": {"reachable_seconds": {"6": False}}}},
    ]
    assert not query_all_events_reachable(rows, 3, 6)
    rows[1]["by_k"]["3"]["reachable_seconds"]["6"] = True
    assert query_all_events_reachable(rows, 3, 6)


def test_t2_package_imports_no_forbidden_modules() -> None:
    forbidden = (
        "triage_eg.experiments.moment_m1",
        "triage_eg.event_graph",
        "cv2",
        "siglip",
        "faiss",
    )
    for path in Path("src/triage_eg/experiments/temporal_t2").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not any(module.startswith(forbidden) for module in modules)


def test_bundle_is_allowlisted_and_excludes_heavy_assets(tmp_path: Path) -> None:
    output = tmp_path / "t2"
    for name in ("t2_summary.json", "t2_metrics.json", "run_manifest.json"):
        write_json(output / name, {})
    for name in (
        "event_reachability.jsonl",
        "query_results.jsonl",
        "top_paths.jsonl",
        "issues.jsonl",
    ):
        write_jsonl(output / name, [])
    (output / "vectors.npy").write_bytes(b"heavy")
    (output / "raw.mp4").write_bytes(b"heavy")
    bundle = create_t2_bundle(output, tmp_path / "t2.zip")
    with ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert names == {
        "t2_summary.json",
        "t2_metrics.json",
        "event_reachability.jsonl",
        "query_results.jsonl",
        "top_paths.jsonl",
        "run_manifest.json",
        "issues.jsonl",
    }
    assert not any(name.endswith((".npy", ".mp4")) for name in names)
