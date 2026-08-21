from __future__ import annotations

from pathlib import Path

from triage_eg.prelim1_team.actual import (
    CORE_FILES,
    build_primary_rows,
    select_review_rows,
    validate_actual_results,
)
from triage_eg.prelim1_team.ranking import fuse_team_frames


def _queries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, 21):
        rows.append({"query_id": f"q{index}", "task": "KIS"})
    for index in range(21, 25):
        rows.append({"query_id": f"q{index}", "task": "QA"})
    rows.append({"query_id": "q25", "task": "TRAKE", "event_count": 3})
    return rows


def _results(queries: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for query in queries:
        task = str(query["task"])
        count = 20 if task == "TRAKE" else 100
        for rank in range(1, count + 1):
            row: dict[str, object] = {
                "query_id": query["query_id"],
                "task_type": task,
                "candidate_rank": rank,
                "video_id": f"L01_V{rank:03d}",
                "primary_candidate": rank == 1,
                "evidence_tier": "TIER_A_VISUAL_AGREEMENT",
            }
            if task == "TRAKE":
                row["frame_ids"] = [rank, rank + 100, rank + 200]
            else:
                row["frame_id"] = rank
            if task == "QA":
                row.update({"answer": "7" if rank <= 5 else "", "status": "EVIDENCE_SUPPORTED"})
            output.append(row)
    return output


def test_actual_packet_contract_accepts_top100_and_trake_top20(tmp_path: Path) -> None:
    queries = _queries()
    rows = _results(queries)
    for name in CORE_FILES:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    manifest = {
        "queries": queries,
        "task_counts": {"KIS": 20, "QA": 4, "TRAKE": 1},
    }
    qa = [row for row in rows if row["task_type"] == "QA" and row["candidate_rank"] <= 5]
    trake = [row for row in rows if row["task_type"] == "TRAKE"]
    validation = validate_actual_results(tmp_path, manifest, rows, qa, trake)
    assert validation["status"] == "PASS"
    assert validation["prediction_count"] == 2420
    assert len(build_primary_rows(queries, rows)) == 25
    assert len(select_review_rows(queries, rows, 5)) == 125
    assert len(select_review_rows(queries, rows, 10)) == 250


class _Catalog:
    def map_row(self, global_row: int) -> dict[str, object]:
        return {"original_frame_idx": global_row, "pts_time": global_row / 25.0}


class _Resolver:
    catalog = _Catalog()

    def nearest_row(self, _video_id: str, frame_id: int) -> int:
        return frame_id

    def nearest_time_row(self, _video_id: str, seconds: float) -> int:
        return round(seconds * 25)


def test_high_asr_specificity_corroborates_but_does_not_replace_visual_head() -> None:
    query = {"query_id": "q", "task": "KIS", "query": "biển hiệu đặc biệt"}
    a0 = [
        {"query_id": "q", "video_id": "L01_V001", "frame_id": 100, "rank": 1},
        {"query_id": "q", "video_id": "L01_V002", "frame_id": 200, "rank": 2},
        {"query_id": "q", "video_id": "L01_V003", "frame_id": 300, "rank": 3},
        {"query_id": "q", "video_id": "L01_V004", "frame_id": 400, "rank": 4},
        {"query_id": "q", "video_id": "L01_V005", "frame_id": 500, "rank": 5},
    ]
    s1 = [
        {"query_id": "q", "video_id": "L01_V001", "frame_id": 100, "rank": 5},
        {"query_id": "q", "video_id": "L01_V002", "frame_id": 200, "rank": 1},
    ]
    rows, _ = fuse_team_frames(
        query,
        a0=a0,
        s1=s1,
        a0_provenance=[],
        s1_provenance=[],
        asr_lexical=[],
        asr_e5=[],
        ocr=[],
        objects=[],
        resolver=_Resolver(),
        asr_specificity=[
            {
                "video_id": "L01_V999",
                "frame_id": 999,
                "rank": 1,
                "specificity_tier": "HIGH",
                "text": "biển hiệu đặc biệt",
            }
        ],
        limit=5,
    )
    assert (rows[0]["video_id"], rows[0]["frame_id"]) in {
        ("L01_V001", 100),
        ("L01_V002", 200),
    }
    assert rows[0]["video_id"] != "L01_V999"
