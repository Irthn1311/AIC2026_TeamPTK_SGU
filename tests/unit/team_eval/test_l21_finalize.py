from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from aic2026_eval.l21_finalize import (
    REGISTRY,
    aliases_from_source,
    audit_anchors,
    create_development_bundle,
    create_l21_bundle,
    create_review_bundle,
    materialize_ground_truth,
    normalize_queries,
    validate_draft_integrity,
    verify_frame_coordinate_contract,
)
from aic2026_eval.mapping import read_mapping
from aic2026_eval.scoring import evaluate


def video_id() -> str:
    return f"L{21}_V{1:03d}"


def mapping_file(path: Path, rows: list[tuple[int, float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "n,pts_time,fps,frame_idx\n"
        + "".join(f"{n},{pts},{fps},{frame}\n" for n, pts, fps, frame in rows),
        encoding="utf-8",
    )


def query_rows() -> list[dict]:
    rows = []
    for index in range(1, 51):
        rows.append(
            {
                "query_id": f"L21-KIS-{index:02d}",
                "task": "KIS",
                "query": f"KIS source {index}",
                "language": "VI",
                "difficulty": " Easy ",
                "tags": ["Visual"],
                "source_query_id": f"KIS-{index:02d}",
            }
        )
        rows.append(
            {
                "query_id": f"L21-QA-{index:02d}",
                "task": "QA",
                "query": f"QA combined source {index}",
                "question": "will be preserved from query",
                "qa_prompt_split_status": "SOURCE_COMBINED_PROMPT_NOT_SPLIT",
                "language": "vi",
                "difficulty": "medium",
                "tags": ["OCR"],
                "source_query_id": f"QA-{index:02d}",
            }
        )
        rows.append(
            {
                "query_id": f"L21-TR-{index:02d}",
                "task": "TRAKE",
                "query": f"TRAKE source {index}",
                "language": "vi",
                "difficulty": "hard",
                "tags": ["state"],
                "source_query_id": f"TR-{index:02d}",
                "event_count": 3,
                "event_descriptions": [
                    {"event_id": f"E{event}", "description": f"event {event}"}
                    for event in range(1, 4)
                ],
            }
        )
    return rows


def anchor_rows(*, unresolved: str | None = None) -> list[dict]:
    rows = []
    for index in range(1, 100):
        start = index * 20
        sources = [f"KIS-{index:02d}"] if index <= 50 else [f"QA-{index - 50:02d}"]
        if index == 99:
            sources = ["QA-49", "QA-50"]
        rows.append(
            {
                "anchor_id": f"anchor_{index:03d}",
                "video_id": video_id(),
                "provisional_raw_interval": [start, start + 12],
                "provisional_raw_reference_frame": start + 6,
                "source_query_ids": sources,
                "status": "NEEDS_VISUAL_REVIEW"
                if unresolved == f"anchor_{index:03d}"
                else "RESOLVED",
                "canonical_interval": [start, start + 12],
            }
        )
    return rows


def provisional_rows() -> list[dict]:
    rows = []
    anchors = anchor_rows()
    by_source = {source: anchor for anchor in anchors for source in anchor["source_query_ids"]}
    for query in query_rows():
        row = {
            "query_id": query["query_id"],
            "task": query["task"],
            "correct_video": video_id(),
            "human_reviewed": True,
        }
        if query["task"] in {"KIS", "QA"}:
            anchor = by_source[query["source_query_id"]]
            row["provisional_raw_interval"] = anchor["provisional_raw_interval"]
            if query["task"] == "QA":
                row["provisional_accepted_answers"] = ["Song sắt / khung cửa sắt"]
        else:
            row["provisional_event_intervals"] = [
                anchors[event]["provisional_raw_interval"] for event in (0, 1, 2)
            ]
            row["event_count"] = 3
        rows.append(row)
    return rows


def inventory(tmp_path: Path, *, total_frames: int = 5000) -> dict:
    mapping = tmp_path / "mapping.csv"
    mapping_file(mapping, [(1, 0.4, 25.0, 10), (2, 0.8, 25.0, 20), (3, 1.2, 25.0, 30)])
    keyframes = tmp_path / "keyframes"
    keyframes.mkdir(exist_ok=True)
    for n in range(1, 4):
        Image.new("RGB", (64, 36), (0, 0, 128)).save(keyframes / f"{n:03d}.jpg")
    return {
        "video_id": video_id(),
        "video_path": str(tmp_path / "video.mp4"),
        "mapping_path": str(mapping),
        "keyframe_directory": str(keyframes),
        "fps": 25.0,
        "total_frames": total_frames,
        "mapping_available": True,
        "keyframes_available": True,
    }


def test_l21_query_count_task_counts_schema_and_normalization() -> None:
    rows = normalize_queries(query_rows())
    assert len(rows) == 150
    assert {task: sum(row["task"] == task for row in rows) for task in ("KIS", "QA", "TRAKE")} == {
        "KIS": 50,
        "QA": 50,
        "TRAKE": 50,
    }
    assert rows[0]["difficulty"] == "easy" and rows[0]["tags"] == ["visual"]
    qa = next(row for row in rows if row["task"] == "QA")
    assert qa["question"] == qa["query"]


def test_unknown_difficulty_and_tag_fail_closed() -> None:
    rows = query_rows()
    rows[0]["difficulty"] = "impossible"
    with pytest.raises(ValueError, match="difficulty"):
        normalize_queries(rows)
    rows = query_rows()
    rows[0]["tags"] = ["private_architecture_tag"]
    with pytest.raises(ValueError, match="TEAM-EVAL tag"):
        normalize_queries(rows)


def test_qa_aliases_only_split_explicit_slash() -> None:
    assert aliases_from_source(["Song sắt / khung cửa sắt"]) == [
        "Song sắt / khung cửa sắt",
        "Song sắt",
        "khung cửa sắt",
    ]
    assert aliases_from_source(["xe hơi"]) == ["xe hơi"]


def test_anchor_reuse_materializes_150_gt_once() -> None:
    queries = normalize_queries(query_rows())
    anchors = anchor_rows()
    provisional = provisional_rows()
    validate_draft_integrity(queries, provisional, anchors)
    gt, issues = materialize_ground_truth(queries, provisional, anchors)
    assert len(anchors) == 99 and len(gt) == 150 and issues == []
    assert len({row["anchor_id"] for row in gt if row["task"] != "TRAKE"}) == 99


def test_trake_uses_linked_semantic_intervals_not_plus_minus_four() -> None:
    gt, issues = materialize_ground_truth(
        normalize_queries(query_rows()), provisional_rows(), anchor_rows()
    )
    assert issues == []
    trake = next(row for row in gt if row["task"] == "TRAKE")
    assert trake["event_intervals"] == [[20, 32], [40, 52], [60, 72]]
    assert all(end - start == 12 for start, end in trake["event_intervals"])


def test_unresolved_anchor_blocks_affected_queries_without_discarding_rest() -> None:
    gt, issues = materialize_ground_truth(
        normalize_queries(query_rows()),
        provisional_rows(),
        anchor_rows(unresolved="anchor_001"),
    )
    assert len(gt) < 150 and gt
    assert {row["code"] for row in issues} >= {
        "QUERY_ANCHOR_UNRESOLVED",
        "TRAKE_LINKED_ANCHOR_UNRESOLVED",
    }


def test_mapping_duplicate_frame_idx_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "mapping.csv"
    mapping_file(path, [(1, 0.4, 25.0, 10), (2, 0.44, 25.0, 10)])
    assert [row["frame_idx"] for row in read_mapping(path)] == [10, 10]


def test_anchor_frame_bounds_fail_closed(tmp_path: Path) -> None:
    source = anchor_rows()[0]
    source["provisional_raw_interval"] = [90, 100]
    source["provisional_raw_reference_frame"] = 95
    rows = audit_anchors([source], [inventory(tmp_path, total_frames=100)])
    assert rows[0]["status"] == "NEEDS_VISUAL_REVIEW"
    assert rows[0]["reason"] == "OUT_OF_BOUNDS"


def test_anchor_decode_preserves_requested_actual_frame_identity(tmp_path: Path) -> None:
    source = anchor_rows()[0]

    def decoder(_: Path, frame_ids: list[int]):
        return [(frame_id, np.zeros((36, 64, 3), dtype=np.uint8)) for frame_id in frame_ids]

    rows = audit_anchors([source], [inventory(tmp_path)], decoder=decoder)
    assert rows[0]["status"] == "RESOLVED"
    assert rows[0]["decoded_actual_frame_ids"] == rows[0]["suggested_raw_frame_ids"]
    assert rows[0]["timestamp_fps_final_reconstruction_used"] is False


def test_frame_coordinate_contract_passes_matching_raw_and_btc(tmp_path: Path) -> None:
    blue = np.zeros((36, 64, 3), dtype=np.uint8)
    blue[:, :, 2] = 128

    def decoder(_: Path, frame_ids: list[int]):
        return [(frame_id, blue.copy()) for frame_id in frame_ids]

    summary, checks = verify_frame_coordinate_contract([inventory(tmp_path)], decoder=decoder)
    assert summary["status"] == "PASS" and checks[0]["status"] == "PASS"
    assert summary["timestamp_fps_reconstruction_used"] is False


def test_frame_coordinate_contract_fails_visual_mismatch(tmp_path: Path) -> None:
    white = np.full((36, 64, 3), 255, dtype=np.uint8)

    def decoder(_: Path, frame_ids: list[int]):
        return [(frame_id, white.copy()) for frame_id in frame_ids]

    summary, _ = verify_frame_coordinate_contract([inventory(tmp_path)], decoder=decoder)
    assert summary["status"] == "FAIL"


def write_benchmark_files(root: Path, benchmark_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "gt.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "annotation_audit.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"benchmark_id": benchmark_id, "scoring_ready": True}), encoding="utf-8"
    )


def test_l21_bundle_has_exact_scoring_contract(tmp_path: Path) -> None:
    root = tmp_path / "l21"
    write_benchmark_files(root, "DEV_L21_150")
    target = create_l21_bundle(root, tmp_path / "l21.zip")
    with ZipFile(target) as archive:
        assert set(archive.namelist()) == {
            "queries.jsonl",
            "gt.jsonl",
            "manifest.json",
            "annotation_audit.jsonl",
        }
    (root / "review_requests.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "raw.mp4").write_bytes(b"raw")
    review = create_review_bundle(root, tmp_path / "review.zip")
    with ZipFile(review) as archive:
        review_names = archive.namelist()
    assert "review_requests.jsonl" in review_names
    assert "gt.jsonl" not in review_names and "raw.mp4" not in review_names


def test_development_bundle_excludes_sealed_content(tmp_path: Path) -> None:
    l21 = tmp_path / "l21"
    write_benchmark_files(l21, "DEV_L21_150")
    cross_source = tmp_path / "cross"
    write_benchmark_files(cross_source, "DEV_CROSS_60")
    cross_zip = tmp_path / "cross.zip"
    with ZipFile(cross_zip, "w") as archive:
        for path in cross_source.iterdir():
            archive.write(path, path.name)
        archive.writestr("SEALED_FINAL_30/gt.jsonl", "secret")
    target = create_development_bundle(l21, cross_zip, tmp_path / "work", tmp_path / "dev.zip")
    with ZipFile(target) as archive:
        names = archive.namelist()
    assert not any("sealed" in name.lower() for name in names)
    assert "benchmarks/dev_l21_150/gt.jsonl" in names
    assert "benchmarks/dev_cross_60/gt.jsonl" in names


def test_registry_has_exact_roles_and_separate_dev_scores() -> None:
    roles = {row["benchmark_id"]: row["role"] for row in REGISTRY["benchmarks"]}
    assert roles == {
        "DEV_L21_150": "PUBLIC_REGRESSION_DEBUG",
        "DEV_CROSS_60": "PUBLIC_CROSS_LEVEL_DEVELOPMENT",
        "SEALED_FINAL_30": "FINAL_HELDOUT_GATE",
    }
    assert REGISTRY["combined_unweighted_dev_score_allowed"] is False


def test_shared_evaluator_accepts_finalized_l21_contract() -> None:
    queries = normalize_queries(query_rows())
    gt, issues = materialize_ground_truth(queries, provisional_rows(), anchor_rows())
    assert issues == []
    predictions = []
    for query, truth in zip(queries, gt, strict=True):
        row = {
            "query_id": query["query_id"],
            "rank": 1,
            "video_id": truth["correct_video"],
        }
        if query["task"] == "TRAKE":
            row["frame_ids"] = [interval[0] for interval in truth["event_intervals"]]
        else:
            row["frame_id"] = truth["acceptable_intervals"][0][0]
            if query["task"] == "QA":
                row["answer"] = truth["accepted_answers"][0]
        predictions.append(row)
    summary, by_query, _, _ = evaluate(queries, predictions, gt)
    assert summary["query_count"] == len(by_query) == 150
    assert summary["final_score"] == 1.0
