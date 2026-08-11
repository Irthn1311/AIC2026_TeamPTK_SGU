from __future__ import annotations

import ast
import inspect
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.experiments.reference_rt1 import dante_monotonic_dp
from triage_eg.experiments.reference_rt2 import RT2BenchmarkQuery, RT2ReferenceEvent
from triage_eg.experiments.t2d_ceiling import (
    build_t2d_metrics,
    create_t2d_bundle,
    diagnose_source_query,
    reference_neighborhood_mask,
    stable_event_ranking,
    validate_expected_t2_reproduction,
)
from triage_eg.experiments.t2d_ceiling.runner import run_t2d
from triage_eg.experiments.temporal_t2 import k_best_monotonic_paths
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl


class Catalog:
    def __init__(self) -> None:
        self.original_idx = np.asarray([0, 100, 200, 300, 400, 500], dtype=np.int64)
        self.mapping_fps = np.full(6, 10.0, dtype=np.float32)
        self.n = np.arange(1, 7, dtype=np.int32)

    def map_row(self, row: int) -> dict[str, object]:
        return {
            "global_row": row,
            "video_id": "L01_V001",
            "n": int(self.n[row]),
            "original_frame_idx": int(self.original_idx[row]),
        }


def _query() -> RT2BenchmarkQuery:
    return RT2BenchmarkQuery(
        "rt2_test",
        "AI_CURATED_INTERNAL_PSEUDO_GT",
        "L01_V001",
        "en",
        (
            RT2ReferenceEvent("E1", "first event", "S01", 1, 1, 2, 100),
            RT2ReferenceEvent("E2", "second event", "S02", 4, 4, 5, 400),
        ),
        ("MULTI_EVENT",),
        "GPT-5.6 Sol",
        False,
    )


def _scores() -> np.ndarray:
    return np.asarray(
        [
            [0.1, 0.8, 0.2, 0.3, 0.4, 0.9],
            [0.1, 0.2, 0.3, 0.95, 0.7, 0.4],
        ],
        dtype=np.float32,
    )


def _diagnose():
    scores = _scores()
    paths = k_best_monotonic_paths(scores, 5)
    return diagnose_source_query(_query(), scores, np.arange(6), Catalog(), paths), paths


def test_event_only_ranking_is_score_desc_then_position_asc() -> None:
    scores = np.asarray([0.5, 0.8, 0.8, 0.1], dtype=np.float32)
    assert stable_event_ranking(scores).tolist() == [1, 2, 0, 3]


def test_best_reference_neighborhood_rank_is_computed_correctly() -> None:
    result, _ = _diagnose()
    ceiling = result[0]
    assert ceiling[0]["best_neighborhood_rank"] == 2
    assert ceiling[0]["best_neighborhood_catalog_position"] == 1
    assert ceiling[1]["best_neighborhood_rank"] == 2
    assert ceiling[1]["best_neighborhood_catalog_position"] == 4


def test_reference_neighborhood_uses_raw_frames_and_fps_not_catalog_distance() -> None:
    frames = np.asarray([0, 1_000], dtype=np.int64)
    mask = reference_neighborhood_mask(frames, 0, fps=10.0, tolerance_seconds=6)
    assert mask.tolist() == [True, False]


def test_unconstrained_path_exactly_reproduces_t2_k1_lambda_zero() -> None:
    result, paths = _diagnose()
    oracle_row = result[2]
    baseline = dante_monotonic_dp(_scores(), distance_lambda=0.0)
    assert baseline is not None
    assert paths[0].positions == baseline.positions
    assert oracle_row["unconstrained_path_positions"] == list(baseline.positions)


def test_window_oracle_anchors_are_inside_own_windows_and_strict() -> None:
    result, _ = _diagnose()
    oracle = result[2]
    assert oracle["oracle_path_feasible"]
    assert oracle["all_oracle_anchors_inside_event_neighborhoods"]
    assert oracle["strictly_monotonic"]
    assert [item["catalog_position"] for item in oracle["oracle_path_anchors"]] == [1, 4]


def test_forced_event_constrains_only_requested_event() -> None:
    result, _ = _diagnose()
    forced_e1 = result[1][0]
    positions = [item["catalog_position"] for item in forced_e1["full_forced_path"]]
    assert positions[0] == 1
    assert positions[1] == 3  # E2 remains unconstrained and may sit outside its own window.
    assert forced_e1["forced_anchor"]["catalog_position"] == 1


def _reproduction_inputs():
    event_ceiling = []
    forced = []
    t2_join = []
    for index in range(74):
        query_id = f"q{min(index // 3, 23):02d}"
        event_id = f"e{index:02d}"
        by_tolerance = {
            str(tolerance): {"best_neighborhood_rank": (index % 60) + 1}
            for tolerance in (3, 6, 9, 12)
        }
        event_ceiling.append(
            {
                "query_id": query_id,
                "event_id": event_id,
                "best_neighborhood_rank": (index % 60) + 1,
                "by_tolerance_seconds": by_tolerance,
            }
        )
        forced.append(
            {
                "query_id": query_id,
                "event_id": event_id,
                "forced_event_path_feasible": True,
                "forced_event_relative_score_gap": index / 1_000,
            }
        )
        t2_join.append(
            {
                "query_id": query_id,
                "event_id": event_id,
                "t2_k5_reachable_6s": index < 54,
            }
        )
    query_t2 = []
    for index in range(24):
        diversity = [1, 1, 1, 1] if index < 2 else [2, 3, 4]
        query_t2.append(
            {
                "query_id": f"q{index:02d}",
                "single_path_all_events_reachable_6s": {
                    "1": index < 5,
                    "3": index < 7,
                    "5": index < 9,
                },
                "k5_anchor_diversity_per_event": diversity,
            }
        )
    oracles = [{"query_id": f"q{index:02d}", "oracle_path_feasible": True} for index in range(24)]
    return event_ceiling, forced, oracles, t2_join, query_t2


def test_t2_k5_and_single_path_reproduction_gates_accept_54_20_and_5_7_9() -> None:
    metrics = build_t2d_metrics(*_reproduction_inputs())
    validate_expected_t2_reproduction(metrics)
    reproduced = metrics["T2_REPRODUCTION"]
    assert reproduced["k5_reachable_6s_count"] == 54
    assert reproduced["k5_missed_6s_count"] == 20
    assert reproduced["single_path_all_events_reachable_6s_counts"] == {
        "1": 5,
        "3": 7,
        "5": 9,
    }


def test_event_weighted_diversity_is_independent_from_query_weighted() -> None:
    metrics = build_t2d_metrics(*_reproduction_inputs())
    diversity = metrics["K5_PATH_DIVERSITY"]
    assert diversity["QUERY_WEIGHTED_MEAN_ANCHOR_DIVERSITY"] != diversity[
        "EVENT_WEIGHTED_MEAN_ANCHOR_DIVERSITY"
    ]
    assert sum(diversity["event_level_unique_anchor_distribution"].values()) == 74


def test_runner_has_one_score_matrix_call_and_no_forbidden_imports() -> None:
    assert inspect.getsource(run_t2d).count("score_many_all(") == 1
    forbidden = (
        "triage_eg.experiments.moment_m1",
        "triage_eg.event_graph",
        "cv2",
        "siglip",
        "faiss",
    )
    for path in Path("src/triage_eg/experiments/t2d_ceiling").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not any(module.startswith(forbidden) for module in modules)


def test_bundle_excludes_heavy_assets_and_cache(tmp_path: Path) -> None:
    output = tmp_path / "t2d"
    for name in ("t2d_summary.json", "t2d_metrics.json", "run_manifest.json"):
        write_json(output / name, {})
    for name in (
        "event_candidate_ceiling.jsonl",
        "forced_event_diagnostics.jsonl",
        "query_oracle_diagnostics.jsonl",
        "k5_failure_analysis.jsonl",
        "issues.jsonl",
    ):
        write_jsonl(output / name, [])
    (output / "t2d_report.md").write_text("diagnostic\n", encoding="utf-8")
    (output / "cache").mkdir()
    (output / "cache/vectors.npy").write_bytes(b"heavy")
    (output / "raw.mp4").write_bytes(b"heavy")
    archive = create_t2d_bundle(output, tmp_path / "bundle.zip")
    with ZipFile(archive) as stream:
        names = set(stream.namelist())
    assert names == {
        "t2d_summary.json",
        "t2d_metrics.json",
        "event_candidate_ceiling.jsonl",
        "forced_event_diagnostics.jsonl",
        "query_oracle_diagnostics.jsonl",
        "k5_failure_analysis.jsonl",
        "run_manifest.json",
        "issues.jsonl",
        "t2d_report.md",
    }
    assert not any(name.endswith((".npy", ".mp4")) for name in names)
