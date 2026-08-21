from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import triage_eg.trial_p1.submit2_hybrid as hybrid


def _queries() -> list[dict]:
    plans = []
    for index in range(18):
        plans.append(
            {
                "query_id": f"kis-{index}",
                "task": "KIS",
                "team_query": {"query_id": f"kis-{index}", "task": "KIS"},
            }
        )
    for index in range(3):
        plans.append(
            {
                "query_id": f"trake-{index}",
                "task": "TRAKE",
                "events": [{}, {}, {}],
                "team_query": {
                    "query_id": f"trake-{index}",
                    "task": "TRAKE",
                    "event_count": 3,
                },
            }
        )
    for index in range(3):
        plans.append(
            {
                "query_id": f"qa-{index}",
                "task": "QA",
                "team_query": {"query_id": f"qa-{index}", "task": "QA"},
            }
        )
    return plans


def _rows(plans: list[dict], branch: str) -> list[dict]:
    rows = []
    for query_index, query in enumerate(plans, 1):
        task = query["task"]
        for rank in range(1, 101):
            safe_tail = branch == "SAFE_R4" and rank > 5 and task != "QA"
            video_number = query_index * 1000 + rank + (20000 if safe_tail else 0)
            row = {
                "query_id": query["query_id"],
                "video_id": f"L21_V{video_number:05d}",
                "rank": rank,
            }
            if task == "KIS":
                row["frame_id"] = rank + (5000 if safe_tail else 0)
            elif task == "QA":
                row["frame_id"] = rank
                row["answer"] = f"answer-{rank}"
            else:
                offset = 5000 if safe_tail else 0
                row["frame_ids"] = [rank + offset, rank + offset + 100, rank + offset + 200]
            rows.append(row)
    return rows


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def test_hybrid_preserves_bcf1_and_forces_qualified_asr_by_top20() -> None:
    plans = _queries()
    bcf1 = _rows(plans, "BCF1")
    safe = _rows(plans, "SAFE_R4")
    strong = [
        {
            "query_id": query["query_id"],
            "qualified_direct_count": 1,
            "best_strong_asr_video": next(
                row["video_id"]
                for row in safe
                if row["query_id"] == query["query_id"] and row["rank"] == 100
            ),
        }
        for query in plans
        if query["task"] == "KIS"
    ]
    result = hybrid.build_hybrid(plans, bcf1, safe, strong)

    assert len(result["predictions"]) == 2400
    assert result["exact_audit"]["HYBRID_KIS_TOP5_EXACT_BCF1"] is True
    assert result["exact_audit"]["HYBRID_TRAKE_TOP5_EXACT_BCF1"] is True
    assert result["exact_audit"]["HYBRID_QA_EXACT_BCF1_100"] is True
    assert all(row["final_hybrid_rank"] <= 20 for row in result["strong_asr_audit"])
    assert all(row["intervention"] == "INSERT_AT_RANK_20" for row in result["strong_asr_audit"])
    qa = result["hybrid_grouped"]["qa-0"]
    qa_source = [row for row in bcf1 if row["query_id"] == "qa-0"]
    assert qa == qa_source


def test_rrf_tie_break_prefers_bcf1_then_canonical_coordinate() -> None:
    protected = [
        {"query_id": "q", "video_id": f"L21_V00{rank}", "frame_id": rank, "rank": rank}
        for rank in range(1, 6)
    ]
    bcf1 = protected + [
        {
            "query_id": "q",
            "video_id": f"L21_V{100 + rank:03d}",
            "frame_id": rank,
            "rank": rank,
        }
        for rank in range(6, 101)
    ]
    safe = protected + [
        {
            "query_id": "q",
            "video_id": f"L22_V{100 + rank:03d}",
            "frame_id": rank,
            "rank": rank,
        }
        for rank in range(6, 101)
    ]
    tail = hybrid._fuse_tail("KIS", bcf1, safe, protected)
    assert hybrid.identity("KIS", tail[0]) == hybrid.identity("KIS", bcf1[5])
    assert hybrid.identity("KIS", tail[1]) == hybrid.identity("KIS", safe[5])


def test_package_writes_exact_oj_zip_and_provenance(tmp_path: Path, monkeypatch) -> None:
    plans = _queries()
    bcf1 = _rows(plans, "BCF1")
    safe = _rows(plans, "SAFE_R4")
    strong = [
        {
            "query_id": query["query_id"],
            "qualified_direct_count": 1,
            "best_strong_asr_video": next(
                row["video_id"]
                for row in safe
                if row["query_id"] == query["query_id"] and row["rank"] == 100
            ),
        }
        for query in plans
        if query["task"] == "KIS"
    ]
    bcf1_payload, query_payload = _jsonl(bcf1), _jsonl(plans)
    safe_payload, strong_payload = _jsonl(safe), _jsonl(strong)
    bcf1_hash = hashlib.sha256(bcf1_payload).hexdigest()
    query_hash = hashlib.sha256(query_payload).hexdigest()
    monkeypatch.setattr(hybrid, "EXPECTED_BCF1_SHA256", bcf1_hash)
    bcf1_zip, r4_zip = tmp_path / "bcf1.zip", tmp_path / "r4.zip"
    with zipfile.ZipFile(bcf1_zip, "w") as archive:
        archive.writestr(hybrid.BCF1_MEMBER, bcf1_payload)
        archive.writestr(hybrid.QUERY_MEMBER, query_payload)
        archive.writestr(hybrid.BCF1_PROVENANCE_MEMBER, json.dumps({"mode": "fixture"}))
        archive.writestr(hybrid.BCF1_SUMMARY_MEMBER, json.dumps({"f1_sha256": bcf1_hash}))
    with zipfile.ZipFile(r4_zip, "w") as archive:
        archive.writestr(hybrid.SAFE_R4_MEMBER, safe_payload)
        archive.writestr(hybrid.STRONG_ASR_MEMBER, strong_payload)
        archive.writestr(
            hybrid.R4_PROVENANCE_MEMBER,
            json.dumps(
                {
                    "candidate_hashes": {"SAFE_R4": hybrid._semantic_content_hash(safe)},
                    "asset_hashes": {
                        "query_plans": query_hash,
                        "bcf1_predictions": bcf1_hash,
                    },
                    "canonical_inventory": {"status": "PASS"},
                    "gt_opened": False,
                    "whisper_run": False,
                    "submission_uploaded": False,
                }
            ),
        )

    provenance = hybrid.package_submit2_hybrid(
        bcf1_zip, r4_zip, tmp_path / "output", head="fixture"
    )
    output_zip = Path(provenance["output_zip"])
    with zipfile.ZipFile(output_zip) as archive:
        assert len(archive.namelist()) == 24
        assert all(name.startswith("submission/") for name in archive.namelist())
    assert provenance["decision"] == "READY_FOR_HUMAN_SUBMIT_2"
    assert provenance["gt_opened"] is False
    assert provenance["model_inference_run"] is False
    assert provenance["submission_uploaded"] is False
