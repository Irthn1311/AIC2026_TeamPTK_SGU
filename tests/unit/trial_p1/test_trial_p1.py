from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from triage_eg.e2e1.planning import plan_query
from triage_eg.e2e1.qa import (
    dynamic_object_candidates,
    garbage_reason,
    reconstruct_ocr_lines,
    select_text_preserving_answer,
)
from triage_eg.fs1.router import classify_answer_type, route_events, route_query
from triage_eg.fs1_v11.qa import exact_text_variants
from triage_eg.trial_p1 import compile_queries, parse_trial_zip, run_b0_safe, run_true_bcf1

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


def test_compiled_qa_contract_reaches_query_plan() -> None:
    compiled = {row["query_id"]: row for row in compile_queries(parse_trial_zip(OFFICIAL))}
    plan = plan_query(compiled["query-p1-19-qa"]["team_query"])
    assert plan.answer_type == "QUOTE_OR_VISIBLE_TEXT"
    assert plan.answer_policy == "TEXT_PRESERVING"
    assert {"b0_visual", "ocr", "asr"} <= set(plan.compiled_routing)
    assert plan.evidence_provenance == ("TRIAL_P1_DETERMINISTIC_QUERY_COMPILER_V2",)


def test_openimages_mid_and_ocr_junk_are_rejected() -> None:
    candidates = dynamic_object_candidates(["/m/01ww8y", "01ww8y", "03jm5", "traffic_light"])
    assert [row.en_output for row in candidates] == ["traffic light"]
    for value in ("/m/01ww8y", "01ww8y", "03jm5", "|", ">", "_Y", "Re"):
        assert garbage_reason(value, "LOCATION_NAME") is not None


def test_contextual_ocr_line_reconstruction_and_location_selection() -> None:
    rows = [
        {"text": "Xã", "confidence": 91, "block_num": 1, "par_num": 1, "line_num": 1, "left": 1},
        {"text": "Vạn", "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1, "left": 20},
        {
            "text": "Thạnh",
            "confidence": 89,
            "block_num": 1,
            "par_num": 1,
            "line_num": 1,
            "left": 40,
        },
    ]
    assert reconstruct_ocr_lines(rows)[0]["text"] == "Xã Vạn Thạnh"
    answer, diagnostic = select_text_preserving_answer(rows, "LOCATION_NAME")
    assert answer == "Xã Vạn Thạnh" and diagnostic["garbage_rejection"] is None


def test_true_bcf1_runner_uses_g1_and_frozen_protected_rrf(tmp_path: Path) -> None:
    compiled = compile_queries(parse_trial_zip(OFFICIAL))

    class FakeArm:
        def __init__(self, offset: int):
            self.offset = offset

        def predict_queries(self, queries, variant):
            assert variant == "G1_COVERAGE_COARSE"
            output = []
            for query in queries:
                rows = []
                for rank in range(1, 101):
                    row = {
                        "query_id": query["query_id"],
                        "rank": rank,
                        "video_id": (
                            f"L{(rank + self.offset) % 9 + 1:02d}_V{rank + self.offset:03d}"
                        ),
                    }
                    if query["task"] == "TRAKE":
                        base = 10 * rank + self.offset
                        row["frame_ids"] = [base, base + 1, base + 2, base + 3]
                    else:
                        row["frame_id"] = rank + self.offset
                        if query["task"] == "QA":
                            row["answer"] = "không đủ bằng chứng"
                    rows.append(row)
                diagnostics = ()
                if query["task"] == "QA":
                    diagnostics = (
                        {
                            "query_id": query["query_id"],
                            "compiled_answer_type": query["answer_type"],
                            "intent": "OCR_TEXT",
                            "intent_source": "COMPILED_ANSWER_TYPE",
                            "answer_policy": query["answer_policy"],
                            "compiled_routing": query["compiled_routing"],
                            "asr_status": "ASR_PENDING",
                        },
                    )
                output.append(SimpleNamespace(predictions=tuple(rows), diagnostics=diagnostics))
            return output

    result = run_true_bcf1(
        FakeArm(0),
        FakeArm(100),
        compiled,
        tmp_path / "true",
        tmp_path / "trial_p1_TRUE_BCF1_submission.zip",
    )
    assert result["status"] == "PASS"
    assert result["policy"] == "A0_TOP5_PROTECTED_EQUAL_RRF60_LATE_FUSION"
    assert result["submission_validation"]["query_count"] == 24
