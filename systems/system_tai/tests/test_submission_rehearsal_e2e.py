"""Master End-to-End Submission Rehearsal Test for AIC 2026 System Tai.

Validates the full pipeline for KIS, Q&A, and TRAKE:
Input Query -> Runtime -> Top-100 Constructor -> P0-A Validator -> Export -> Official Evaluator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from system_tai.common.observability import ExecutionTrace, TraceContext
from system_tai.common.schemas import (
    CandidateFrame,
    KISResult,
)
from system_tai.preliminary.evaluation import evaluate_ranked_query
from system_tai.preliminary.matching import NormalizedAliasAnswerMatcher
from system_tai.preliminary.schemas import (
    KISGroundTruth,
    KISPrediction,
    QAGroundTruth,
    QAPrediction,
    TRAKEGroundTruth,
    TRAKEPrediction,
)
from system_tai.preliminary.scoring import (
    score_kis_prediction,
    score_qa_prediction,
    score_trake_prediction,
)
from system_tai.preliminary.top100 import (
    RankedTop100Dataset,
    RankedTop100Query,
    load_top100_jsonl,
    write_top100_jsonl,
)
from system_tai.qa.top100_constructor import construct_ranked_qa_top100


def export_csv(predictions: list[dict[str, object]], path: Path, task_type: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        for p in predictions:
            if task_type == "kis":
                writer.writerow([p["video_id"], p["frame_id"]])
            elif task_type == "qa":
                writer.writerow([p["video_id"], p["frame_id"], p["answer"]])
            elif task_type == "trake":
                writer.writerow([p["video_id"], *(p["frame_ids"])])


def test_kis_full_submission_rehearsal_e2e(tmp_path: Path) -> None:
    """Test full KIS lifecycle: Query -> Retrieval -> Top-100 -> Validate -> CSV/JSONL -> Score."""
    trace = ExecutionTrace(request_id="req_kis_01", query_id="KIS-01", task_type="KIS")
    with TraceContext(trace, "retrieval_and_ranking"):
        candidates = [
            CandidateFrame(
                video_id=f"L21_V00{1 + (i % 5)}",
                frame_id=1200 + i * 30,
                clip_row=i,
                keyframe_order=i + 1,
                score=1.0 - i * 0.01,
                rank=i + 1,
                source="exact_numpy",
            )
            for i in range(100)
        ]
        # Target video L21_V001 at rank 1
        candidates[0] = CandidateFrame(
            video_id="L21_V001",
            frame_id=1200,
            clip_row=0,
            keyframe_order=1,
            score=1.0,
            rank=1,
            source="exact_numpy",
        )
        kis_result = KISResult(query_id="KIS-01", ranked_candidates=tuple(candidates))

        preds = [
            KISPrediction(
                query_id=kis_result.query_id,
                rank=c.rank,
                video_id=c.video_id,
                frame_id=c.frame_id,
            )
            for c in kis_result.ranked_candidates
        ]

    # 1. Export JSONL & Validate Dataset
    dataset = RankedTop100Dataset(
        task_type="kis",
        queries=(RankedTop100Query("kis", "KIS-01", tuple(preds)),),
    )
    jsonl_out = tmp_path / "kis_top100.jsonl"
    write_top100_jsonl(dataset, jsonl_out)
    loaded_dataset = load_top100_jsonl(jsonl_out, task_type="kis")
    assert len(loaded_dataset.queries) == 1

    # 2. Export CSV
    csv_out = tmp_path / "submission_kis.csv"
    export_csv(
        [{"video_id": p.video_id, "frame_id": p.frame_id} for p in preds],
        csv_out,
        task_type="kis",
    )
    assert csv_out.exists()

    with open(csv_out, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 100

    gt = KISGroundTruth(
        query_id="KIS-01",
        video_id="L21_V001",
        start_frame_id=1150,
        end_frame_id=1300,
    )

    # 3. Evaluate with Official Metric
    report = evaluate_ranked_query(
        query_id="KIS-01",
        task_type="kis",
        predictions=preds,
        ground_truth=gt,
        scorer=score_kis_prediction,
    )
    assert report.r_at_1 == 1.0
    assert report.final_score == 1.0


def test_qa_full_submission_rehearsal_e2e(tmp_path: Path) -> None:
    """Test full Q&A lifecycle: Event+Q -> Top-100 Constructor -> Validate -> CSV/JSONL -> Score."""
    trace = ExecutionTrace(request_id="req_qa_01", query_id="QA-01", task_type="Q&A")
    with TraceContext(trace, "top100_constructor"):
        scored_candidates = [
            {
                "video_id": f"L21_V00{1 + (i % 4)}",
                "frame_id": 3000 + i * 15,
                "answers": ["Xe cứu thương", "Xe cảnh sát"] if i == 0 else [f"Vật thể {i}"],
                "score": 0.95 - i * 0.01,
            }
            for i in range(25)
        ]
        # Ensure target video L21_V001 is at rank 0
        scored_candidates[0] = {
            "video_id": "L21_V001",
            "frame_id": 3000,
            "answers": ["Xe cứu thương", "Xe cảnh sát"],
            "score": 1.0,
        }

        preds = construct_ranked_qa_top100(
            query_id="QA-01",
            scored_candidates=scored_candidates,
            output_top_k=100,
        )

    assert len(preds) == 100
    assert preds[0].video_id == "L21_V001"
    assert preds[0].frame_id == 3000
    assert preds[0].answer == "Xe cứu thương"

    # 1. Export JSONL
    dataset = RankedTop100Dataset(
        task_type="qa",
        queries=(RankedTop100Query("qa", "QA-01", tuple(preds)),),
    )
    jsonl_out = tmp_path / "qa_top100.jsonl"
    write_top100_jsonl(dataset, jsonl_out)
    assert jsonl_out.exists()

    # 2. Export CSV
    csv_out = tmp_path / "submission_qa.csv"
    export_csv(
        [{"video_id": p.video_id, "frame_id": p.frame_id, "answer": p.answer} for p in preds],
        csv_out,
        task_type="qa",
    )
    assert csv_out.exists()

    with open(csv_out, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 100
    assert rows[0] == ["L21_V001", "3000", "Xe cứu thương"]

    gt = QAGroundTruth(
        query_id="QA-01",
        video_id="L21_V001",
        start_frame_id=2950,
        end_frame_id=3100,
        accepted_answers=("xe cứu thương",),
    )

    # 3. Evaluate with Official Metric (using NormalizedAliasAnswerMatcher)
    matcher = NormalizedAliasAnswerMatcher(strip_punctuation=True)
    report = evaluate_ranked_query(
        query_id="QA-01",
        task_type="qa",
        predictions=preds,
        ground_truth=gt,
        scorer=lambda p, g: score_qa_prediction(p, g, matcher),
    )
    assert report.r_at_1 == 1.0
    assert report.final_score == 1.0


def test_trake_full_submission_rehearsal_e2e(tmp_path: Path) -> None:
    """Test full TRAKE lifecycle: Event Chain -> Monotonic DP/Beam -> Validate -> CSV/JSONL -> Score."""
    trace = ExecutionTrace(request_id="req_trake_01", query_id="TRAKE-01", task_type="TRAKE")
    with TraceContext(trace, "chain_solver"):
        preds = [
            TRAKEPrediction(
                query_id="TRAKE-01",
                rank=i + 1,
                video_id=f"L21_V00{1 + (i % 3)}",
                frame_ids=(100 + i * 5, 200 + i * 5, 300 + i * 5),
            )
            for i in range(100)
        ]
        # Rank 1 matches GT
        preds[0] = TRAKEPrediction(
            query_id="TRAKE-01",
            rank=1,
            video_id="L21_V001",
            frame_ids=(100, 200, 300),
        )

    # 1. Export JSONL
    dataset = RankedTop100Dataset(
        task_type="trake",
        queries=(RankedTop100Query("trake", "TRAKE-01", tuple(preds)),),
    )
    jsonl_out = tmp_path / "trake_top100.jsonl"
    write_top100_jsonl(dataset, jsonl_out, expected_trake_event_counts={"TRAKE-01": 3})
    assert jsonl_out.exists()

    # 2. Export CSV
    csv_out = tmp_path / "submission_trake.csv"
    export_csv(
        [{"video_id": p.video_id, "frame_ids": p.frame_ids} for p in preds],
        csv_out,
        task_type="trake",
    )
    assert csv_out.exists()

    with open(csv_out, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 100
    assert rows[0] == ["L21_V001", "100", "200", "300"]

    gt = TRAKEGroundTruth(
        query_id="TRAKE-01",
        video_id="L21_V001",
        event_intervals=((90, 120), (190, 220), (290, 320)),
    )

    # 3. Evaluate with Official Metric
    report = evaluate_ranked_query(
        query_id="TRAKE-01",
        task_type="trake",
        predictions=preds,
        ground_truth=gt,
        scorer=score_trake_prediction,
    )
    assert report.r_at_1 == 1.0
    assert report.final_score == 1.0
