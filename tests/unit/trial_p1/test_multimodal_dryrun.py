from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_eg.trial_p1.multimodal_dryrun import (
    build_asr_candidate_evidence,
    build_external_parquet_evidence,
    build_xclip_event_evidence,
    candidate_comparison,
    normalize_trial_plans,
    prioritize_qa_sufficient,
    qa_evidence_summary,
    validate_trial_contract,
    write_blocked_artifacts,
)


def _queries() -> list[dict]:
    rows = []
    for value in range(1, 19):
        rows.append(
            {
                "query_id": f"query-p1-{value}-kis",
                "task": "KIS",
                "language": "vi",
                "query": f"kis {value}",
            }
        )
    for value in (15, 19, 22):
        rows.append(
            {
                "query_id": f"query-p1-{value}-qa",
                "task": "QA",
                "language": "vi",
                "query": f"qa {value}",
            }
        )
    rows.extend(
        {
            "query_id": query_id,
            "task": "TRAKE",
            "language": "vi",
            "query": "four events",
            "event_count": 4,
            "event_descriptions": ["one", "two", "three", "four"],
            "raw_event_labels": labels,
        }
        for query_id, labels in (
            ("query-p1-4-trake", ["E1", "E2", "E3", "E4"]),
            ("query-p1-18-trake", ["E1", "E2", "E2", "E4"]),
            ("query-p1-26-trake", ["E1", "E2", "E3", "E4"]),
        )
    )
    # IDs intentionally overlap numerically but remain unique by task.
    return rows


def _rows(queries: list[dict], source: str = "b0") -> list[dict]:
    output = []
    for query in queries:
        for rank in range(1, 101):
            common = {
                "query_id": query["query_id"],
                "rank": rank,
                "video_id": f"L{21 + rank % 10:02d}_V{rank % 100:03d}",
                "source": source,
            }
            if query["task"] == "TRAKE":
                common["frame_ids"] = [rank, rank + 1, rank + 2, rank + 3]
            else:
                common["frame_id"] = rank
            if query["task"] == "QA":
                common["answer"] = "không đủ bằng chứng"
                common["evidence_sufficient"] = False
            output.append(common)
    return output


def test_trial_contract_preserves_duplicated_raw_e2() -> None:
    assert validate_trial_contract(_queries())["status"] == "PASS"
    broken = _queries()
    next(row for row in broken if row["query_id"] == "query-p1-18-trake")["raw_event_labels"] = [
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    with pytest.raises(RuntimeError, match="DUPLICATED_RAW_E2"):
        validate_trial_contract(broken)


def test_normalize_frozen_plans_keeps_four_ordinal_events() -> None:
    plans = []
    for query in _queries():
        plan = {
            "query_id": query["query_id"],
            "task": query["task"],
            "raw_text": query["query"],
            "team_query": {key: value for key, value in query.items() if key != "raw_event_labels"},
        }
        if query["task"] == "TRAKE":
            plan["events"] = [
                {
                    "description": description,
                    "raw_event_label": query["raw_event_labels"][index],
                }
                for index, description in enumerate(query["event_descriptions"])
            ]
        plans.append(plan)
    normalized = normalize_trial_plans(plans)
    p18 = next(row for row in normalized if row["query_id"] == "query-p1-18-trake")
    assert p18["event_count"] == 4
    assert p18["raw_event_labels"] == ["E1", "E2", "E2", "E4"]


def test_qa_sufficient_rows_are_ranked_before_unsupported_and_generic_is_rejected() -> None:
    baseline = _rows([_queries()[18]])
    qwen = [
        {
            "query_id": "query-p1-15-qa",
            "video_id": "L30_V072",
            "frame_id": 20,
            "answer": "ghế",
            "evidence_sufficient": True,
            "rank": 1,
        },
        {
            "query_id": "query-p1-15-qa",
            "video_id": "L25_V001",
            "frame_id": 21,
            "answer": "FANA",
            "evidence_sufficient": True,
            "rank": 2,
        },
    ]
    ranked = prioritize_qa_sufficient(baseline, qwen, baseline)
    assert ranked[0]["answer"] == "FANA"
    assert ranked[0]["evidence_sufficient"] is True
    assert all(row.get("answer") != "ghế" for row in ranked)


def test_qa_hard_gate_blocks_zero_sufficient_candidates() -> None:
    queries = _queries()
    rows = _rows(queries)
    candidates = {name: rows for name in ("TRIAGEEG_M0_FULL", "TRIAGEEG_M1_FULL", "TRIAGEEG_SAFE")}
    assert qa_evidence_summary(queries, candidates)["hard_gate"] == "QA_BLOCK_SUBMISSION_2"


def test_comparison_has_all_24_queries() -> None:
    queries = _queries()
    baseline = _rows(queries)
    candidates = {
        name: _rows(queries, name)
        for name in ("TRIAGEEG_M0_FULL", "TRIAGEEG_M1_FULL", "TRIAGEEG_SAFE")
    }
    comparison = candidate_comparison(queries, baseline, candidates)
    assert len(comparison["rows"]) == 24
    assert comparison["top1_changed_vs_bcf1"] == 0


def test_blocked_artifacts_are_explicit_and_do_not_fabricate_candidate_zips(tmp_path: Path) -> None:
    output = write_blocked_artifacts(
        tmp_path,
        ["XCLIP_EVIDENCE_MISSING", "QWEN_ASSET_MISSING"],
        {"gt_opened": False},
    )
    decision = (output / "SUBMISSION_2_DECISION.md").read_text(encoding="utf-8")
    assert "DO_NOT_SUBMIT_2_YET" in decision
    assert "XCLIP_EVIDENCE_MISSING" in decision
    assert not list(output.glob("TRIAGEEG_*.zip"))
    assert (
        json.loads((output / "qa_evidence_summary.json").read_text())["status"]
        == "NOT_RUN_FAIL_CLOSED"
    )


def test_asr_evidence_maps_seconds_only_spans_to_frozen_frame_identities() -> None:
    query = _queries()[0]
    baseline = _rows([query])

    class Loader:
        @staticmethod
        def retrieve_spans(text: str, max_spans: int) -> list[dict]:
            assert text and max_spans == 200
            return [
                {
                    "video_id": baseline[3]["video_id"],
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "text": "strong topic",
                }
            ]

    evidence = build_asr_candidate_evidence([query], baseline, Loader())
    row = evidence[query["query_id"]][0]
    assert row["frame_id"] == baseline[3]["frame_id"]
    assert "frame_id" not in row["asr_span"]


def test_external_ocr_evidence_is_evidence_only(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "ocr.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video_id": "L21_V001",
                    "frame_idx": 10,
                    "corrected_text": "Japan Fest",
                    "combined_text": "Japan Fest",
                    "mean_confidence": 0.9,
                }
            ]
        ),
        path,
    )
    query = {**_queries()[0], "query": "Japan Fest"}
    rows = build_external_parquet_evidence([query], path, "ocr")[query["query_id"]]
    assert rows[0]["text"] == "Japan Fest"
    assert rows[0]["evidence_only"] is True
    assert "answer" not in rows[0]


def test_xclip_is_scored_for_every_ordinal_event() -> None:
    query = next(row for row in _queries() if row["query_id"] == "query-p1-18-trake")
    baseline = _rows([query])

    def score(text: str, video_id: str, frame_id: int) -> dict:
        return {
            "score": float(frame_id),
            "finite": True,
            "text": text,
            "video_id": video_id,
        }

    rows = build_xclip_event_evidence([query], baseline, score, candidates_per_event=3)[
        query["query_id"]
    ]
    assert {row["event_index"] for row in rows} == {0, 1, 2, 3}
    assert len(rows) == 12
