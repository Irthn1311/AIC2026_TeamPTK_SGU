from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from aic2026_eval.census import build_corpus_inventory, build_usage_census
from aic2026_eval.contracts import contract_document, validate_query
from aic2026_eval.discovery import resolve_dataset_root, resolve_named_file
from aic2026_eval.evaluate_predictions import main as evaluate_main
from aic2026_eval.holdout import select_heldout_candidates
from aic2026_eval.mapping import audit_l21_bootstrap, read_mapping
from aic2026_eval.render import (
    render_dense_requests,
    render_overview_atlas,
    requested_frame_ids,
)
from aic2026_eval.report import BOOTSTRAP_REQUIRED, create_bundle
from aic2026_eval.scoring import evaluate, score_prediction
from aic2026_eval.validation import validate_predictions


def video_id(level: int, index: int) -> str:
    return f"L{level}_V{index:03d}"


def mapping_file(path: Path, rows: list[tuple[int, float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "n,pts_time,fps,frame_idx\n"
        + "".join(f"{n},{pts},{fps},{frame}\n" for n, pts, fps, frame in rows),
        encoding="utf-8",
    )


def inventory_row(level: int, index: int, root: Path | None = None) -> dict:
    identity = video_id(level, index)
    base = root or Path("/synthetic")
    return {
        "video_id": identity,
        "top_level": f"L{level}",
        "source_group": f"Videos_L{level}_a",
        "video_path": str(base / f"{identity}.mp4"),
        "mapping_path": str(base / f"{identity}.csv"),
        "keyframe_directory": str(base / identity),
        "fps": 25.0,
        "total_frames": 1000,
        "valid_frame_metadata": True,
        "mapping_available": True,
        "keyframes_available": True,
    }


def holdout_fixture() -> tuple[list[dict], list[dict]]:
    inventory = [inventory_row(level, offset + 1) for level in range(22, 31) for offset in range(5)]
    census = [{"video_id": row["video_id"], "usage_tier": "T0_UNREFERENCED"} for row in inventory]
    return inventory, census


def query(task: str = "KIS", **extra) -> dict:
    value = {"query_id": "query_1", "task": task, "query": "synthetic query", **extra}
    return value


def prediction(task: str = "KIS", **extra) -> dict:
    value = {
        "query_id": "query_1",
        "rank": 1,
        "video_id": video_id(22, 1),
        **extra,
    }
    if task in {"KIS", "QA"}:
        value.setdefault("frame_id", 10)
    return value


def test_video_inventory_parses_layout_and_metadata(tmp_path: Path) -> None:
    identity = video_id(22, 1)
    video = tmp_path / "Videos_L22_a/video" / f"{identity}.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video-placeholder")
    metadata = tmp_path / "media-info-aic25-b1/media-info" / f"{identity}.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"fps": 25, "total_frames": 250, "duration_sec": 10}),
        encoding="utf-8",
    )
    mapping_file(
        tmp_path / "map-keyframes-aic25-b1/map-keyframes" / f"{identity}.csv",
        [(1, 0.0, 25.0, 0)],
    )
    (tmp_path / "keyframes/keyframes/Keyframes_L22_a/keyframes" / identity).mkdir(parents=True)
    rows, summary, issues = build_corpus_inventory(tmp_path)
    assert (len(rows), summary["status"], issues) == (1, "PASS", [])
    assert rows[0]["source_group"] == "Videos_L22_a"
    assert rows[0]["total_frames"] == 250
    assert rows[0]["keyframes_available"] is True


def test_mapping_preserves_duplicate_original_frames(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    mapping_file(path, [(1, 0.0, 25.0, 7), (2, 0.04, 25.0, 7)])
    assert [row["frame_idx"] for row in read_mapping(path)] == [7, 7]


def test_l21_is_never_selected() -> None:
    inventory, census = holdout_fixture()
    excluded = inventory_row(21, 1)
    inventory.append(excluded)
    census.append({"video_id": excluded["video_id"], "usage_tier": "T0_UNREFERENCED"})
    selected, _ = select_heldout_candidates(inventory, census)
    assert all(row["top_level"] != "L21" for row in selected)


def test_t2_and_t3_are_excluded() -> None:
    inventory, census = holdout_fixture()
    blocked = {inventory[0]["video_id"], inventory[1]["video_id"]}
    census[0]["usage_tier"] = "T2_EXPERIMENT_USED"
    census[1]["usage_tier"] = "T3_GT_OR_QC_USED"
    selected, report = select_heldout_candidates(inventory, census)
    assert blocked.isdisjoint(row["video_id"] for row in selected)
    assert report["t2_t3_used"] is False


def test_manual_exclusion_is_t3(tmp_path: Path) -> None:
    identity = video_id(24, 2)
    manual = tmp_path / "manual.txt"
    manual.write_text(identity + " # external use\n", encoding="utf-8")
    rows, _ = build_usage_census(
        [inventory_row(24, 2)],
        tmp_path,
        manual_exclude_path=manual,
        tracked_files=[],
    )
    assert rows[0]["usage_tier"] == "T3_GT_OR_QC_USED"


def test_usage_census_chooses_more_contaminated_tier(tmp_path: Path) -> None:
    identity = video_id(25, 3)
    infra = tmp_path / "docs/notes.md"
    benchmark = tmp_path / "configs/benchmark/used.json"
    infra.parent.mkdir(parents=True)
    benchmark.parent.mkdir(parents=True)
    infra.write_text(identity, encoding="utf-8")
    benchmark.write_text(identity, encoding="utf-8")
    rows, _ = build_usage_census(
        [inventory_row(25, 3)],
        tmp_path,
        tracked_files=["docs/notes.md", "configs/benchmark/used.json"],
    )
    assert rows[0]["usage_tier"] == "T3_GT_OR_QC_USED"
    assert len(rows[0]["evidence_paths"]) == 2


def test_heldout_selection_is_deterministic() -> None:
    inventory, census = holdout_fixture()
    first, _ = select_heldout_candidates(inventory, census, seed=17)
    second, _ = select_heldout_candidates(inventory, census, seed=17)
    assert first == second


def test_blind_and_sealed_roles_are_disjoint_and_sized() -> None:
    inventory, census = holdout_fixture()
    rows, report = select_heldout_candidates(inventory, census)
    assert report["blind_candidate_count"] == 24
    assert report["sealed_candidate_count"] == 12
    assert report["blind_sealed_video_overlap"] == 0
    assert len({row["video_id"] for row in rows}) == 36


def test_holdout_fails_closed_before_t2() -> None:
    inventory, census = holdout_fixture()
    for item in census[10:]:
        item["usage_tier"] = "T2_EXPERIMENT_USED"
    with pytest.raises(RuntimeError, match="FAIL_CLOSED"):
        select_heldout_candidates(inventory, census)


def test_common_query_schemas_and_neutral_contract() -> None:
    values = [
        query("KIS"),
        query("QA", question="What is visible?"),
        query("TRAKE", event_count=3),
    ]
    assert [validate_query(value)["task"] for value in values] == ["KIS", "QA", "TRAKE"]
    contract = contract_document()
    assert contract["team_neutral"] is True
    assert contract["frame_id_semantics"] == "original_frame_idx"
    assert contract["maximum_predictions_per_query"] == 100


def test_query_schema_rejects_missing_qa_question() -> None:
    with pytest.raises(ValueError, match="question"):
        validate_query(query("QA"))


def test_prediction_rank_must_be_unique_and_sorted() -> None:
    rows = [prediction(rank=2), prediction(rank=1), prediction(rank=1)]
    summary, issues = validate_predictions([query()], rows)
    assert summary["status"] == "FAIL"
    assert {item["code"] for item in issues} >= {"RANK_DUPLICATE", "RANK_NOT_SORTED"}


def test_prediction_maximum_is_100() -> None:
    rows = [prediction(rank=rank) for rank in range(1, 102)]
    _, issues = validate_predictions([query()], rows)
    assert "TOO_MANY_PREDICTIONS" in {item["code"] for item in issues}


def test_prediction_frame_bounds() -> None:
    rows = [prediction(frame_id=1000)]
    _, issues = validate_predictions([query()], rows, inventory=[inventory_row(22, 1)])
    assert "FRAME_OUT_OF_BOUNDS" in {item["code"] for item in issues}


def test_qa_prediction_requires_non_empty_answer() -> None:
    _, issues = validate_predictions(
        [query("QA", question="Question?")],
        [prediction("QA", answer=" ")],
    )
    assert "QA_ANSWER_INVALID" in {item["code"] for item in issues}


def test_trake_prediction_requires_event_count() -> None:
    _, issues = validate_predictions(
        [query("TRAKE", event_count=2)],
        [prediction("TRAKE", frame_ids=[10])],
    )
    assert "TRAKE_EVENT_COUNT_MISMATCH" in {item["code"] for item in issues}


def test_kis_interval_scoring() -> None:
    score, diagnostics = score_prediction(
        query(),
        prediction(frame_id=12),
        {"correct_video": video_id(22, 1), "intervals": [[10, 20]]},
    )
    assert score == 1.0 and diagnostics["grounding_correct"] is True


def test_qa_alias_scoring_is_deterministic_but_not_official_semantics() -> None:
    score, diagnostics = score_prediction(
        query("QA", question="Question?"),
        prediction("QA", answer="  Red-car! "),
        {
            "correct_video": video_id(22, 1),
            "intervals": [[0, 20]],
            "accepted_answers": ["red car"],
        },
    )
    assert score == 1.0
    assert diagnostics["alias_matching_is_official_btc_semantics"] is False


def test_trake_fractional_scoring() -> None:
    score, diagnostics = score_prediction(
        query("TRAKE", event_count=3),
        prediction("TRAKE", frame_ids=[5, 99, 25]),
        {
            "correct_video": video_id(22, 1),
            "event_intervals": [[0, 10], [11, 20], [21, 30]],
        },
    )
    assert score == pytest.approx(2 / 3)
    assert diagnostics["per_event_hit"] == [True, False, True]


def test_r_at_k_final_score_and_scoreboard_cli(tmp_path: Path) -> None:
    predictions = [prediction(rank=1, frame_id=99), prediction(rank=2, frame_id=10)]
    queries = [query()]
    ground_truth = [
        {
            "query_id": "query_1",
            "correct_video": video_id(22, 1),
            "intervals": [[5, 15]],
        }
    ]
    summary, by_query, _, _ = evaluate(
        queries,
        predictions,
        ground_truth,
    )
    assert by_query[0]["R@1"] == 0.0 and by_query[0]["R@5"] == 1.0
    assert summary["final_score"] == pytest.approx(0.8)
    paths = {
        "queries": tmp_path / "queries.jsonl",
        "predictions": tmp_path / "predictions.jsonl",
        "ground_truth": tmp_path / "ground_truth.jsonl",
    }
    for name, rows in (
        ("queries", queries),
        ("predictions", predictions),
        ("ground_truth", ground_truth),
    ):
        paths[name].write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "scoreboard"
    exit_code = evaluate_main(
        [
            "--queries",
            str(paths["queries"]),
            "--predictions",
            str(paths["predictions"]),
            "--ground-truth",
            str(paths["ground_truth"]),
            "--output",
            str(output),
            "--system-id",
            "synthetic",
            "--config-id",
            "test",
            "--dataset-version",
            "v0",
        ]
    )
    assert exit_code == 0
    written = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert written["prediction_sha256"] and written["system_id"] == "synthetic"


def test_dense_requested_actual_frame_identity(tmp_path: Path) -> None:
    request = {
        "anchor_id": "anchor_1",
        "video_id": video_id(22, 1),
        "approx_original_frame_idx": 40,
        "mode": "MOMENT_DENSE",
        "radius_seconds": 0.2,
    }

    def decoder(_: Path, frame_ids: list[int]):
        return [(frame_id, np.zeros((16, 24, 3), dtype=np.uint8)) for frame_id in frame_ids]

    manifests, issues = render_dense_requests(
        [request], [inventory_row(22, 1)], tmp_path, decoder=decoder
    )
    assert issues == []
    assert manifests[0]["requested_frame_ids"] == manifests[0]["actual_frame_ids"]
    assert manifests[0]["frame_identity_exact"] is True
    frames = requested_frame_ids(
        {
            "approx_original_frame_idx": 500,
            "mode": "INTERVAL",
            "radius_seconds": 4,
        },
        inventory_row(22, 1),
    )
    assert len(frames) == 28 and frames == sorted(set(frames))


def test_overview_atlas_uses_btc_mapping_identity(tmp_path: Path) -> None:
    from PIL import Image

    mapping = tmp_path / "mapping.csv"
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir()
    mapping_file(mapping, [(1, 0.4, 25.0, 10), (2, 0.8, 25.0, 20)])
    for n in (1, 2):
        Image.new("RGB", (24, 16), "navy").save(keyframes / f"{n:03d}.jpg")
    candidate = {
        **inventory_row(22, 1),
        "mapping_path": str(mapping),
        "keyframe_directory": str(keyframes),
        "role": "BLIND_CANDIDATE_POOL",
    }
    manifest, index, issues = render_overview_atlas([candidate], tmp_path / "output")
    assert issues == [] and index["status"] == "READY"
    assert [tile["original_frame_idx"] for tile in manifest[0]["tiles"]] == [10, 20]


def test_bundle_allow_list_excludes_raw_model_and_cache(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    for name in BOOTSTRAP_REQUIRED:
        (root / name).write_text("{}\n", encoding="utf-8")
    atlas = root / "atlas"
    atlas.mkdir()
    (atlas / "sheet.jpg").write_bytes(b"jpeg")
    (root / "raw.mp4").write_bytes(b"raw")
    (root / "model.pt").write_bytes(b"model")
    (root / "runtime_cache").mkdir()
    target = create_bundle(root, tmp_path / "bundle.zip")
    with zipfile.ZipFile(target) as archive:
        members = archive.namelist()
    assert "atlas/sheet.jpg" in members
    assert not any(name.endswith((".mp4", ".pt")) or "cache" in name for name in members)


def test_l21_bootstrap_skip_and_coordinate_audit(tmp_path: Path) -> None:
    rows, summary = audit_l21_bootstrap(None, [], temporary_root=tmp_path)
    assert rows == [] and summary["status"] == "SKIPPED_NO_INPUT"
    identity = video_id(21, 1)
    mapping = tmp_path / "mapping.csv"
    mapping_file(mapping, [(1, 0.0, 25.0, 10), (2, 0.04, 25.0, 10)])
    archive = tmp_path / "bootstrap.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "nested/l21_anchor_index.jsonl",
            json.dumps(
                {
                    "anchor_id": "anchor_1",
                    "video_id": identity,
                    "start_frame": 9,
                    "end_frame": 11,
                }
            )
            + "\n",
        )
    record = inventory_row(21, 1)
    record["mapping_path"] = str(mapping)
    audit, audit_summary = audit_l21_bootstrap(
        archive,
        [record],
        temporary_root=tmp_path / "extract",
    )
    assert audit_summary["status"] == "PASS"
    assert audit[0]["status"] == "COORDINATE_SUPPORTED"
    assert audit[0]["duplicate_frame_idx_groups"] == {"10": 2}


def test_nested_input_discovery(tmp_path: Path) -> None:
    dataset = tmp_path / "mount/nested/dataset"
    (dataset / "Videos_L22_a/video").mkdir(parents=True)
    (dataset / "map-keyframes-aic25-b1/map-keyframes").mkdir(parents=True)
    marker = tmp_path / "requests/deeper/anchor_requests.jsonl"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    assert resolve_dataset_root(tmp_path / "mount") == dataset.resolve()
    assert resolve_named_file(tmp_path / "requests", "anchor_requests.jsonl") == marker.resolve()
