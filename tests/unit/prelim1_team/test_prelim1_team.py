from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from triage_eg.prelim1_team.consensus import consensus_rows
from triage_eg.prelim1_team.parser import parse_prelim1_zip
from triage_eg.prelim1_team.ranking import build_qa_review_rows, fuse_team_frames


def _package(path: Path, *, event_lines: bool = True) -> Path:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for index in range(1, 21):
            text = "Nội dung KIS trùng nhau" if index in {8, 14} else f"Nội dung KIS {index}"
            archive.writestr(f"query-p1-{index}-kis.txt", text)
        for index in range(21, 25):
            archive.writestr(f"query-p1-{index}-qa.txt", f"Con số câu {index} là bao nhiêu?")
        trake = "Bối cảnh\nE1 sự kiện một\nE2 sự kiện hai\nE3 sự kiện ba"
        if not event_lines:
            trake = "Bối cảnh không có event"
        archive.writestr("query-p1-25-trake.txt", trake)
    return path


def test_prelim_parser_is_dynamic_and_preserves_duplicate_text(tmp_path: Path) -> None:
    manifest = parse_prelim1_zip(
        _package(tmp_path / "queries.zip"),
        expected_sha256="",
        expected_content_sha256="",
    )
    assert manifest["query_count"] == 25
    assert manifest["task_counts"] == {"KIS": 20, "QA": 4, "TRAKE": 1}
    assert manifest["duplicate_text_groups"] == [["query-p1-8-kis", "query-p1-14-kis"]]
    trake = next(row for row in manifest["queries"] if row["task"] == "TRAKE")
    assert trake["event_count"] == 3
    assert [row["event_index"] for row in trake["events"]] == [0, 1, 2]


def test_prelim_parser_fails_closed_on_hash_and_trake_content(tmp_path: Path) -> None:
    package = _package(tmp_path / "queries.zip")
    with pytest.raises(ValueError, match="PACKAGE_SHA256_MISMATCH"):
        parse_prelim1_zip(package, expected_sha256="0" * 64)
    broken = _package(tmp_path / "broken.zip", event_lines=False)
    with pytest.raises(ValueError, match="TRAKE_NO_EVENTS"):
        parse_prelim1_zip(broken, expected_sha256="", expected_content_sha256="")


def test_consensus_counts_distinct_members_and_selects_medoid() -> None:
    members = {
        "m1": [
            {
                "query_id": "q1",
                "candidate_rank": 1,
                "video_id": "L01_V001",
                "frame_id": 100,
                "embedding_key_a0": "a",
            }
        ],
        "m2": [
            {
                "query_id": "q1",
                "candidate_rank": 2,
                "video_id": "L01_V001",
                "frame_id": 104,
                "embedding_key_a0": "b",
            }
        ],
    }
    embeddings = {
        "m1": {"a": np.asarray([1.0, 0.0], dtype=np.float32)},
        "m2": {"b": np.asarray([0.99, 0.01], dtype=np.float32)},
    }
    rows = consensus_rows(members, embeddings=embeddings, near_frame_tolerance=8)
    assert rows[0]["distinct_member_support"] == 2
    assert rows[0]["same_video_vote_count"] == 2
    assert rows[0]["embedding_cluster_member_count"] == 2
    assert rows[0]["automatic_submission"] is False


class _Catalog:
    def map_row(self, global_row: int) -> dict[str, object]:
        return {
            "original_frame_idx": global_row,
            "pts_time": global_row / 25.0,
        }


class _Resolver:
    catalog = _Catalog()

    def nearest_row(self, _video_id: str, frame_id: int) -> int:
        return frame_id

    def nearest_time_row(self, _video_id: str, seconds: float) -> int:
        return round(seconds * 25)


def _visual(video: str, frame: int, rank: int) -> dict[str, object]:
    return {"query_id": "q", "video_id": video, "frame_id": frame, "rank": rank}


def test_visual_agreement_keeps_weak_object_out_of_top1() -> None:
    query = {"query_id": "q", "task": "KIS", "query": "người áo đỏ"}
    a0 = [
        _visual("L01_V001", 100, 1),
        _visual("L01_V002", 200, 2),
        _visual("L01_V004", 400, 3),
        _visual("L01_V005", 500, 4),
    ]
    s1 = [_visual("L01_V001", 100, 3), _visual("L01_V003", 300, 1)]
    provenance = [
        {
            "query_id": "q",
            "branch": branch,
            "candidate_key": ["L01_V001", 100],
            "view_ranks": {"ORIGINAL_VI": 1},
        }
        for branch in ("A0", "S1")
    ]
    objects = [
        {
            "query_id": "q",
            "video_id": "L01_V999",
            "frame_id": 1,
            "rank": 1,
            "text": "người áo đỏ",
        }
    ]
    rows, _ = fuse_team_frames(
        query,
        a0=a0,
        s1=s1,
        a0_provenance=[provenance[0]],
        s1_provenance=[provenance[1]],
        asr_lexical=[],
        asr_e5=[],
        ocr=[],
        objects=objects,
        resolver=_Resolver(),
        limit=5,
    )
    assert (rows[0]["video_id"], rows[0]["frame_id"]) == ("L01_V001", 100)
    assert rows[0]["evidence_tier"] == "TIER_A_VISUAL_AGREEMENT"


def test_qa_never_invents_answer_without_local_evidence() -> None:
    query = {"query_id": "qa", "task": "QA", "answer_type": "NUMBER_OR_COUNT"}
    contexts = [
        {
            "query_id": "qa",
            "task_type": "QA",
            "candidate_rank": rank,
            "video_id": f"L01_V00{rank}",
            "frame_id": rank,
            "primary_candidate": rank == 1,
        }
        for rank in range(1, 6)
    ]
    rows, audit = build_qa_review_rows(query, contexts, asr_rows=[], ocr_rows=[])
    assert len(rows) == 5
    assert all(row["answer"] == "" for row in rows)
    assert all(row["status"] == "MANUAL_REVIEW_REQUIRED" for row in rows)
    assert all(row["ground_truth_used"] is False for row in audit)


def test_visible_number_uses_ocr_not_asr_and_visual_count_stays_manual() -> None:
    contexts = [
        {
            "query_id": "qa",
            "task_type": "QA",
            "candidate_rank": rank,
            "video_id": "L01_V001" if rank == 1 else f"L01_V{rank:03d}",
            "frame_id": rank,
            "primary_candidate": rank == 1,
        }
        for rank in range(1, 6)
    ]
    evidence = [{"video_id": "L01_V001", "text": "biển báo 17"}]
    visible, _ = build_qa_review_rows(
        {"query_id": "qa", "task": "QA", "answer_type": "VISIBLE_NUMBER"},
        contexts,
        asr_rows=[{"video_id": "L01_V001", "text": "đội số 99"}],
        ocr_rows=evidence,
    )
    assert visible[0]["answer"] == "17"
    assert visible[0]["evidence_type"] == "OCR"
    count, _ = build_qa_review_rows(
        {"query_id": "qa", "task": "QA", "answer_type": "VISUAL_COUNT"},
        contexts,
        asr_rows=evidence,
        ocr_rows=evidence,
    )
    assert all(row["answer"] == "" for row in count)
    assert all(row["status"] == "MANUAL_REVIEW_REQUIRED" for row in count)
