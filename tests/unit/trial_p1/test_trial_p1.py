from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from triage_eg.fs1.router import classify_answer_type, route_events, route_query
from triage_eg.fs1_v11.qa import exact_text_variants
from triage_eg.trial_p1 import compile_queries, parse_trial_zip, run_b0_safe

REPO = Path(__file__).resolve().parents[3]
OFFICIAL = Path(os.environ.get("AIC_TRIAL_P1_ZIP", REPO / "outputs/Task/THUNGHIEM-bo-de-thi.zip"))


def test_official_manifest_and_ordinal_trake_edge_case() -> None:
    manifest = parse_trial_zip(OFFICIAL)
    assert manifest["query_count"] == 24
    assert manifest["task_counts"] == {"KIS": 18, "QA": 3, "TRAKE": 3}
    trake = {row["query_id"]: row for row in manifest["queries"] if row["task"] == "TRAKE"}
    assert {key: value["event_count"] for key, value in trake.items()} == {
        "query-p1-4-trake": 4,
        "query-p1-16-trake": 4,
        "query-p1-18-trake": 4,
    }
    assert trake["query-p1-18-trake"]["raw_event_labels"] == ["E1", "E2", "E2", "E4"]
    assert [event["event_id"] for event in trake["query-p1-18-trake"]["events"]] == [
        "E1",
        "E2",
        "E3",
        "E4",
    ]


def test_compiler_routes_all_trial_qa_and_trake() -> None:
    plans = {row["query_id"]: row for row in compile_queries(parse_trial_zip(OFFICIAL))}
    assert plans["query-p1-15-qa"]["answer_type"] == "LOCATION_NAME"
    assert plans["query-p1-19-qa"]["answer_type"] == "QUOTE_OR_VISIBLE_TEXT"
    assert plans["query-p1-22-qa"]["answer_type"] == "TITLE"
    for query_id in ("query-p1-15-qa", "query-p1-19-qa", "query-p1-22-qa"):
        modalities = plans[query_id]["routing"][0]["modalities"]
        assert {"b0_visual", "ocr", "asr"} <= set(modalities)
    for query_id in ("query-p1-4-trake", "query-p1-16-trake", "query-p1-18-trake"):
        assert len(plans[query_id]["routing"]) == 4
        assert all("action" in route["modalities"] for route in plans[query_id]["routing"])
    assert {"ocr", "asr"} <= set(plans["query-p1-23-kis"]["routing"][0]["modalities"])


def test_generic_knowledge_expansion_is_text_based() -> None:
    plans = {row["query_id"]: row for row in compile_queries(parse_trial_zip(OFFICIAL))}
    expansion = plans["query-p1-23-kis"]["knowledge_expansions"]
    assert expansion and expansion[0]["source"] == "QUERY_KNOWLEDGE_EXPANSION"
    assert "Jaws" in expansion[0]["text"]


def test_router_generic_contract() -> None:
    assert classify_answer_type("Hỏi tiêu đề của công thức là gì?") == "TITLE"
    qa = route_query(
        "QA", "Hỏi tên xã là gì?", available=("ocr", "asr"), answer_type="LOCATION_NAME"
    )
    assert qa.modalities == ("b0_visual", "ocr", "asr")
    routes = route_events("TRAKE", ("cắt nấm", "chảo nóng"), available=("action", "object"))
    assert len(routes) == 2 and all("action" in route.modalities for route in routes)


def test_exact_text_variants_fail_instead_of_silent_truncation() -> None:
    assert exact_text_variants("  Hai câu thơ.  ") == ("Hai câu thơ.", "Hai câu thơ")
    assert exact_text_variants("ấ" * 60) == ("ấ" * 60,)
    with pytest.raises(ValueError, match="REQUIRES_REVIEW"):
        exact_text_variants("ấ" * 101)


def test_package_rejects_extra_member(tmp_path: Path) -> None:
    target = tmp_path / "invalid.zip"
    with zipfile.ZipFile(OFFICIAL) as source, zipfile.ZipFile(target, "w") as output:
        for info in source.infolist():
            output.writestr(info.filename, source.read(info))
        output.writestr("README.md", io.BytesIO(b"extra").getvalue())
    with pytest.raises(ValueError, match="FILE_COUNT_MISMATCH"):
        parse_trial_zip(target)


def test_b0_runner_creates_and_validates_official_zip(tmp_path: Path) -> None:
    class FakePipeline:
        def predict_query(self, query, variant):
            assert variant == "P0_COARSE"
            task = query["task"]
            row = {"query_id": query["query_id"], "rank": 1, "video_id": "L01_V001"}
            if task == "TRAKE":
                row["frame_ids"] = [10, 20, 30, 40]
            else:
                row["frame_id"] = 10
                if task == "QA":
                    row["answer"] = "bằng chứng chưa đủ"
            return SimpleNamespace(predictions=(row,), diagnostics=())

    compiled = compile_queries(parse_trial_zip(OFFICIAL))
    result = run_b0_safe(
        FakePipeline(),
        compiled,
        tmp_path / "artifacts",
        tmp_path / "trial_p1_B0_SAFE_submission.zip",
    )
    assert result["submission_validation"] == {
        "status": "PASS",
        "query_count": 24,
        "prediction_count": 24,
    }
    with zipfile.ZipFile(result["submission_zip"]) as archive:
        assert len(archive.namelist()) == 24
        assert "submission/query-p1-18-trake.csv" in archive.namelist()
