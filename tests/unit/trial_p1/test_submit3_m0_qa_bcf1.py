from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import triage_eg.trial_p1.submit3_m0_qa_bcf1 as submit3
from triage_eg.trial_p1.submit2_hybrid import _semantic_content_hash


def _plans() -> list[dict]:
    values = []
    for index in range(18):
        values.append(
            {
                "query_id": f"kis-{index}",
                "task": "KIS",
                "team_query": {"event_count": None},
            }
        )
    for index in range(3):
        values.append(
            {
                "query_id": f"trake-{index}",
                "task": "TRAKE",
                "events": [{}, {}, {}],
                "team_query": {"event_count": 3},
            }
        )
    for index in range(3):
        values.append({"query_id": f"qa-{index}", "task": "QA", "team_query": {}})
    return values


def _arm_rows(plans: list[dict], arm: str) -> list[dict]:
    rows = []
    for query_number, query in enumerate(plans, 1):
        for rank in range(1, 101):
            row = {
                "query_id": query["query_id"],
                "video_id": f"L21_V{query_number:03d}",
                "rank": rank,
            }
            if query["task"] == "KIS":
                row["frame_id"] = rank + (1000 if arm != "BCF1" else 0)
            elif query["task"] == "TRAKE":
                offset = 1000 if arm != "BCF1" else 0
                row["frame_ids"] = [rank + offset, rank + offset + 200, rank + offset + 400]
            else:
                row["frame_id"] = rank
                row["answer"] = f"{arm}-answer-{rank}"
            rows.append(row)
    return rows


def _payload(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _source() -> dict:
    plans = _plans()
    return {
        "queries": plans,
        "bcf1": _arm_rows(plans, "BCF1"),
        "m0_r4": _arm_rows(plans, "M0_R4"),
        "m1_r4": _arm_rows(plans, "M1_R4"),
        "safe_r4": _arm_rows(plans, "SAFE_R4"),
    }


def test_submit3_copies_exact_task_specific_frozen_arms() -> None:
    source = _source()
    result = submit3.build_submit3_predictions(source)
    grouped = {}
    for row in result["predictions"]:
        grouped.setdefault(row["query_id"], []).append(row)

    assert len(result["predictions"]) == 2400
    assert grouped["kis-0"] == [row for row in source["m0_r4"] if row["query_id"] == "kis-0"]
    assert grouped["trake-0"] == [row for row in source["m0_r4"] if row["query_id"] == "trake-0"]
    assert grouped["qa-0"] == [row for row in source["bcf1"] if row["query_id"] == "qa-0"]
    assert all(result["construction_gates"].values())


def test_submit3_packages_and_validates_exact_24_csvs(tmp_path: Path, monkeypatch) -> None:
    source = _source()
    plans_payload = _payload(source["queries"])
    bcf1_payload = _payload(source["bcf1"])
    m0_payload = _payload(source["m0_r4"])
    m1_payload = _payload(source["m1_r4"])
    safe_payload = _payload(source["safe_r4"])
    bcf1_hash = hashlib.sha256(bcf1_payload).hexdigest()
    monkeypatch.setattr(submit3, "EXPECTED_BCF1_SHA256", bcf1_hash)
    bcf1_zip = tmp_path / "bcf1.zip"
    r4_zip = tmp_path / "r4.zip"
    with zipfile.ZipFile(bcf1_zip, "w") as archive:
        archive.writestr(submit3.BCF1_MEMBER, bcf1_payload)
        archive.writestr(submit3.QUERY_MEMBER, plans_payload)
        archive.writestr(submit3.BCF1_PROVENANCE_MEMBER, json.dumps({"GT_OPENED": False}))
        archive.writestr(
            submit3.BCF1_SUMMARY_MEMBER,
            json.dumps({"f1_sha256": bcf1_hash, "gt_opened": False}),
        )
    with zipfile.ZipFile(r4_zip, "w") as archive:
        archive.writestr(submit3.M0_R4_MEMBER, m0_payload)
        archive.writestr(submit3.M1_R4_MEMBER, m1_payload)
        archive.writestr(submit3.SAFE_R4_MEMBER, safe_payload)
        archive.writestr(
            submit3.R4_PROVENANCE_MEMBER,
            json.dumps(
                {
                    "asset_hashes": {
                        "bcf1_predictions": bcf1_hash,
                        "query_plans": hashlib.sha256(plans_payload).hexdigest(),
                    },
                    "candidate_hashes": {
                        "M0_R4": _semantic_content_hash(source["m0_r4"]),
                        "M1_R4": _semantic_content_hash(source["m1_r4"]),
                        "SAFE_R4": _semantic_content_hash(source["safe_r4"]),
                    },
                    "canonical_inventory": {"status": "PASS"},
                    "gt_opened": False,
                    "whisper_run": False,
                    "submission_uploaded": False,
                }
            ),
        )

    provenance = submit3.package_submit3(bcf1_zip, r4_zip, tmp_path / "output", head="fixture")
    with zipfile.ZipFile(provenance["output_zip"]) as archive:
        assert len(archive.namelist()) == 24
        assert all(name.startswith("submission/") for name in archive.namelist())
    assert provenance["decision"] == "READY_FOR_HUMAN_SUBMIT_3"
    assert provenance["gt_opened"] is False
    assert provenance["model_inference_run"] is False
    assert provenance["submission_uploaded"] is False
