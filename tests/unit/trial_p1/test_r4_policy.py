from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from triage_eg.trial_p1.r3_policy import derive_r3_anchor_profiles
from triage_eg.trial_p1.r4_policy import (
    build_bounded_local_ocr_rescue,
    build_r4_candidates,
    needs_local_ocr_r4,
    verify_answer_r4,
    write_r4_artifacts,
)


def _profile(query: str, answer_type: str) -> dict:
    rows = [{"query_id": "Q", "task": "QA", "query": query, "answer_type": answer_type}]
    plans = [{"query_id": "Q", "raw_text": query, "action_anchors": [], "knowledge_expansions": []}]
    return derive_r3_anchor_profiles(rows, plans)["Q"]


def _extraction(answer: str, span: str, *, sufficient: bool = True) -> dict:
    return {
        "video_id": "L21_V001",
        "frame_id": 10,
        "grounding_rank": 1,
        "answer": answer,
        "answer_type": "IGNORED_MODEL_FIELD",
        "supporting_source_ids": ["asr:1"],
        "supporting_spans": [span],
        "evidence_sufficient": sufficient,
    }


def test_r4_location_requires_exact_answer_and_requested_granularity() -> None:
    profile = _profile("FANA trao quà ở xã nào tại Khánh Hòa?", "LOCATION_NAME")
    evidence = [
        {
            "source_id": "asr:1",
            "modality": "asr",
            "text": "FANA trao quà tại xã Phú Thuận, tỉnh Khánh Hòa",
        }
    ]
    wrong = verify_answer_r4(
        _extraction("Khánh Hòa", evidence[0]["text"]),
        "LOCATION_NAME",
        evidence,
        profile,
        grounding_plausibility=0.9,
    )
    correct = verify_answer_r4(
        _extraction("Phú Thuận", evidence[0]["text"]),
        "LOCATION_NAME",
        evidence,
        profile,
        grounding_plausibility=0.9,
    )
    assert wrong["answer_type_pass"] is False
    assert wrong["final_evidence_sufficient"] is False
    assert correct["answer_support_pass"] is True
    assert correct["answer_type_pass"] is True
    assert correct["final_evidence_sufficient"] is True


def test_r4_answer_support_rejects_world_knowledge_inference() -> None:
    profile = _profile("FANA ở xã An Lập thuộc tỉnh nào?", "LOCATION_NAME")
    evidence = [
        {
            "source_id": "asr:1",
            "modality": "asr",
            "text": "FANA ở xã An Lập, huyện Dầu Tiếng",
        }
    ]
    result = verify_answer_r4(
        _extraction("Bình Dương", evidence[0]["text"]),
        "LOCATION_NAME",
        evidence,
        profile,
        grounding_plausibility=0.8,
    )
    assert result["answer_support_pass"] is False
    assert result["answer_support_reason"] == "ANSWER_TEXT_ABSENT_FROM_SUPPORT"
    assert result["final_evidence_sufficient"] is False


def test_r4_title_keeps_original_and_applies_only_generic_wrapper() -> None:
    profile = _profile("Món gồm mực chiên giòn và sốt tắc có tên gì?", "TITLE")
    text = "mực chiên giòn và sốt tắc, món ăn có tên gọi là mực chiên giòn sốt tắc"
    result = verify_answer_r4(
        _extraction("món ăn có tên gọi là mực chiên giòn sốt tắc", text),
        "TITLE",
        [{"source_id": "asr:1", "modality": "asr", "text": text}],
        profile,
        grounding_plausibility=0.9,
    )
    assert result["unstripped_answer"].startswith("món ăn")
    assert result["answer"] == "mực chiên giòn sốt tắc"
    assert result["canonicalization_rule"] == "STRIP_GENERIC_TITLE_WRAPPER"
    assert result["final_evidence_sufficient"] is True


def test_r4_quote_two_lines_requires_two_supported_spans() -> None:
    profile = _profile("Hai câu thơ về Nguyễn Trung Trực là gì?", "QUOTE_OR_VISIBLE_TEXT")
    profile["requested_quote_line_count"] = 2
    extraction = {
        **_extraction(
            "Bao giờ người Tây nhổ hết cỏ nước Nam; Thì mới hết người Nam đánh Tây",
            "Bao giờ người Tây nhổ hết cỏ nước Nam",
        ),
        "supporting_source_ids": ["asr:1", "asr:2"],
        "supporting_spans": [
            "Bao giờ người Tây nhổ hết cỏ nước Nam",
            "Thì mới hết người Nam đánh Tây",
        ],
    }
    evidence = [
        {
            "source_id": "asr:1",
            "modality": "asr",
            "text": "Nguyễn Trung Trực: Bao giờ người Tây nhổ hết cỏ nước Nam",
        },
        {"source_id": "asr:2", "modality": "asr", "text": "Thì mới hết người Nam đánh Tây"},
    ]
    result = verify_answer_r4(
        extraction,
        "QUOTE_OR_VISIBLE_TEXT",
        evidence,
        profile,
        grounding_plausibility=0.8,
    )
    assert result["answer_support_pass"] is True
    assert result["answer_type_pass"] is True


def test_local_ocr_is_bounded_to_shortlisted_candidate_neighbors() -> None:
    query = {
        "query_id": "Q",
        "task": "QA",
        "query": "FANA trao quà ở xã nào?",
        "answer_type": "LOCATION_NAME",
    }
    profile = _profile(query["query"], "LOCATION_NAME")
    calls = []

    def provider(video_id: str, frames: list[int]) -> list[dict]:
        calls.append((video_id, frames))
        return [{"frame_id": frames[0], "text": "xã Phú Thuận", "engine": "fixture"}]

    rows, audit = build_bounded_local_ocr_rescue(
        [query],
        {"Q": profile},
        {
            "asr": {
                "Q": [{"video_id": "L21_V001", "frame_id": 100, "rank": 1, "text": "FANA trao quà"}]
            },
            "ocr": {"Q": []},
        },
        [{"query_id": "Q", "selected_context_videos": ["L21_V001"]}],
        provider,
        max_frames_per_query=3,
    )
    assert calls == [("L21_V001", [52, 76, 100])]
    assert rows["Q"][0]["source"] == "local_ocr_r4_bounded_candidate_neighbor"
    assert audit[0]["corpus_job_launched"] is False


def test_text_heavy_location_still_gets_bounded_local_ocr_confirmation() -> None:
    query = {"answer_type": "LOCATION_NAME"}
    profile = {"location_granularity": "xã"}
    needed, reason = needs_local_ocr_r4(query, profile, [{"text": "FANA tại xã Phú Thuận"}])
    assert needed is True
    assert reason == "TEXT_HEAVY_LOCATION_BOUNDED_LOCAL_CONFIRMATION"


def test_qwen_r4_schema_has_no_generated_answer_type(tmp_path: Path) -> None:
    import torch

    from triage_eg.fs1.qa import GroundingCandidate
    from triage_eg.fs1.qwen_adapter import QwenEvidenceAdapter

    class Inputs(dict):
        def __init__(self) -> None:
            super().__init__(input_ids=torch.ones((1, 3), dtype=torch.long))
            self.input_ids = self["input_ids"]

        def to(self, _: str):
            return self

    class Processor:
        prompt = ""

        def apply_chat_template(self, messages, **kwargs):
            self.prompt = messages[0]["content"][1]["text"]
            return "prompt"

        def __call__(self, **kwargs):
            return Inputs()

        def batch_decode(self, *args, **kwargs):
            return [
                json.dumps(
                    {
                        "answer": "Phú Thuận",
                        "supporting_source_ids": ["asr:1"],
                        "supporting_spans": ["xã Phú Thuận"],
                        "evidence_sufficient": True,
                    }
                )
            ]

    class Model:
        def generate(self, **kwargs):
            return torch.ones((1, 5), dtype=torch.long)

    processor = Processor()
    adapter = QwenEvidenceAdapter(tmp_path, device="cpu")
    adapter.processor, adapter.model = processor, Model()
    parsed, audit = adapter.answer_extraction(
        GroundingCandidate("L21_V001", 1, 1, {}),
        object(),
        description="FANA",
        question="xã nào?",
        evidence_rows=[{"source_id": "asr:1", "text": "xã Phú Thuận"}],
        answer_type="LOCATION_NAME",
        answer_policy="TEXT_PRESERVING",
    )
    assert "answer_type (string)" not in processor.prompt
    assert parsed["answer_type"] == "LOCATION_NAME"
    assert audit["qwen_parse_pass"] is True


def test_safe_r4_replaces_same_video_neighbor_with_exact_frozen_tuple(monkeypatch) -> None:
    import triage_eg.trial_p1.r4_policy as policy

    queries, baseline = [], []
    for query_number in range(24):
        query_id = f"Q{query_number:02d}"
        queries.append({"query_id": query_id, "task": "KIS"})
        baseline.extend(
            {
                "query_id": query_id,
                "video_id": f"L21_V{rank:03d}",
                "frame_id": rank,
                "rank": rank,
            }
            for rank in range(1, 101)
        )
    wrong_safe = [
        {**row, "frame_id": row["frame_id"] + 1} if row["rank"] <= 5 else dict(row)
        for row in baseline
    ]

    def fake_r3(*args, **kwargs):
        return {
            "candidates": {
                "M0_R3": [dict(row) for row in baseline],
                "M1_R3": [dict(row) for row in baseline],
                "SAFE_R3": wrong_safe,
            },
            "validation": {},
            "strong_asr_audit": [],
            "trake_checks": {},
        }

    monkeypatch.setattr(policy, "build_r3_candidates", fake_r3)
    monkeypatch.setattr(
        policy,
        "validate_predictions",
        lambda queries, rows, inventory: ({"status": "PASS"}, []),
    )
    result = build_r4_candidates(queries, baseline, {}, {}, object(), inventory=[])
    safe = result["candidates"]["SAFE_R4"]
    for query in queries:
        query_id = query["query_id"]
        frozen = [row for row in baseline if row["query_id"] == query_id][:5]
        actual = [row for row in safe if row["query_id"] == query_id][:5]
        assert [(row["video_id"], row["frame_id"]) for row in actual] == [
            (row["video_id"], row["frame_id"]) for row in frozen
        ]
    assert all(row["pass"] for row in result["safe_top5_audit"])


def test_r4_writer_emits_official_submission_zips_only_after_all_gates(tmp_path: Path) -> None:
    queries = []
    for index in range(24):
        if index < 18:
            task = "KIS"
        elif index < 21:
            task = "QA"
        else:
            task = "TRAKE"
        query = {"query_id": f"Q{index:02d}", "task": task}
        if task == "TRAKE":
            query["event_count"] = 2
        queries.append(query)
    baseline, candidates = [], {name: [] for name in ("M0_R4", "M1_R4", "SAFE_R4")}
    for query in queries:
        for rank in range(1, 101):
            row = {
                "query_id": query["query_id"],
                "video_id": f"L21_V{rank:03d}",
                "rank": rank,
            }
            if query["task"] == "TRAKE":
                row["frame_ids"] = [rank, rank + 100]
            else:
                row["frame_id"] = rank
            if query["task"] == "QA":
                row["answer"] = "Phú Thuận"
                row.update(
                    qwen_parse_pass=True,
                    qwen_claims_sufficient=True,
                    context_relevance_pass=True,
                    answer_support_pass=True,
                    answer_type_pass=True,
                    final_evidence_sufficient=True,
                )
            baseline.append(dict(row))
            for name in candidates:
                candidates[name].append({**row, "system_variant": name})
    qa_rows = [
        dict(row)
        for row in candidates["M0_R4"]
        if row["query_id"] in {"Q18", "Q19", "Q20"} and row["rank"] == 1
    ]
    result = {
        "candidates": candidates,
        "validation": {
            name: {"status": "PASS", "exact_100_per_query": True} for name in candidates
        },
        "safe_top5_audit": [
            {"query_id": query["query_id"], "pass": True}
            for query in queries
            if query["task"] in {"KIS", "TRAKE"}
        ],
        "strong_asr_audit": [],
        "trake_checks": {
            query["query_id"]: {"graph_causal_gate_pass": True, "arms": {}}
            for query in queries
            if query["task"] == "TRAKE"
        },
    }
    report = write_r4_artifacts(
        tmp_path,
        queries,
        baseline,
        result,
        {},
        [{"query_id": row["query_id"], "audit": {"qwen_parse_pass": True}} for row in qa_rows],
        qa_rows,
        [],
        provenance={
            "runtime_candidate_failure_count": 0,
            "gt_opened": False,
            "submission_uploaded": False,
            "whisper_run": False,
            "asset_hashes": {},
        },
    )
    assert report["recommendation"] == "SUBMIT_2_SAFE_R4"
    assert set(report["oj_ready_submissions"]) == {"M0_R4", "M1_R4", "SAFE_R4"}
    for item in report["oj_ready_submissions"].values():
        with zipfile.ZipFile(item["path"]) as archive:
            assert len(archive.namelist()) == 24
            assert all(name.startswith("submission/") for name in archive.namelist())


def test_notebook_41_local_ocr_uses_decoded_frame_contract() -> None:
    notebook = Path(__file__).parents[3] / "notebooks/41_trial_p1_multimodal_dryrun.ipynb"
    source = notebook.read_text(encoding="utf-8")
    assert "item.actual_frame_idx" in source
    assert re.search(r"item\.actual_frame_id\b", source) is None
