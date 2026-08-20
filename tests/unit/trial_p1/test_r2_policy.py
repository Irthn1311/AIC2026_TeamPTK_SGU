from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_eg.fs1.qa import GroundingCandidate
from triage_eg.fs1.qwen_adapter import QwenEvidenceAdapter
from triage_eg.trial_p1.r2_policy import (
    TIER_A,
    TIER_C,
    _complete_ranked_rows,
    classify_asr_specificity,
    derive_anchor_profiles,
    rank_qa_r2,
    repair_kis_ranking,
    tier_evidence,
    verify_answer_extraction,
    write_r2_artifacts,
)


def _baseline(query_id: str, task: str = "KIS") -> list[dict]:
    rows = []
    for rank in range(1, 101):
        row = {
            "query_id": query_id,
            "video_id": f"L21_V{rank:03d}",
            "rank": rank,
        }
        if task == "TRAKE":
            row["frame_ids"] = [rank, rank + 10]
        else:
            row["frame_id"] = rank
        if task == "QA":
            row["answer"] = "không đủ bằng chứng"
        rows.append(row)
    return rows


def test_anchor_profile_and_asr_specificity_are_query_generic() -> None:
    queries = [{"query_id": "Q1", "task": "KIS", "query": "SpaceX Dragon crew"}]
    plans = [
        {
            "query_id": "Q1",
            "raw_text": "SpaceX Dragon đưa phi hành đoàn bốn người lên quỹ đạo",
            "semantic_core": "SpaceX Dragon phi hành đoàn",
            "text_entity_anchors": ["SpaceX Dragon"],
            "action_anchors": [],
        }
    ]
    profile = derive_anchor_profiles(queries, plans)["Q1"]
    direct = classify_asr_specificity(
        {
            "video_id": "L21_V900",
            "frame_id": 1,
            "rank": 1,
            "asr_span": {"text": "SpaceX Dragon chở phi hành đoàn bốn người"},
            "asr_source_ranks": [
                {"branch": "LEXICAL", "rank": 1},
                {"branch": "E5", "rank": 2},
            ],
        },
        profile,
    )
    weak = classify_asr_specificity(
        {
            "video_id": "L21_V901",
            "frame_id": 2,
            "rank": 1,
            "asr_span": {"text": "một chương trình nghiên cứu"},
            "asr_source_ranks": [{"branch": "LEXICAL", "rank": 1}],
        },
        profile,
    )
    assert direct["evidence_tier"] == TIER_A
    assert direct["asr_specificity"]["lexical_e5_agreement"] is True
    assert weak["evidence_tier"] == TIER_C


def test_anchor_profile_extracts_text_from_compiler_knowledge_expansion() -> None:
    queries = [{"query_id": "Q1", "task": "KIS", "query": "Jaws"}]
    plans = [
        {
            "query_id": "Q1",
            "raw_text": "Bộ phim năm 1975 của Steven Spielberg",
            "semantic_core": "phim Steven Spielberg",
            "text_entity_anchors": [],
            "action_anchors": [],
            "knowledge_expansions": [
                {
                    "text": "cá mập nguy hiểm, bộ phim Jaws",
                    "source": "QUERY_KNOWLEDGE_EXPANSION",
                }
            ],
        }
    ]
    profile = derive_anchor_profiles(queries, plans)["Q1"]
    assert "cá mập nguy hiểm, bộ phim Jaws" in profile["distinctive_phrases"]


def test_safe_r2_never_allows_object_only_top5_override() -> None:
    query = {"query_id": "Q1", "task": "KIS", "query": "specific topic"}
    baseline = _baseline("Q1")
    object_only = {
        "video_id": "L99_V999",
        "frame_id": 7,
        "rank": 1,
        "evidence_tier": TIER_C,
        "evidence_tier_reasons": ["OBJECT_ONLY_STANDALONE_WEAK"],
        "matched_query_anchors": [],
    }
    _, safe, diagnostics = repair_kis_ranking(
        query, baseline, {"asr": [], "ocr": [], "object": [object_only]}
    )
    assert [row["video_id"] for row in safe[:5]] == [row["video_id"] for row in baseline[:5]]
    assert diagnostics["weak_modality_override"] is False


def test_kis_r2_preserves_100_frame_coordinates_when_videos_repeat() -> None:
    query = {"query_id": "Q1", "task": "KIS", "query": "specific topic"}
    baseline = [
        {
            "query_id": "Q1",
            "video_id": f"L21_V{((rank - 1) // 10) + 1:03d}",
            "frame_id": rank,
            "rank": rank,
        }
        for rank in range(1, 101)
    ]
    full, safe, _ = repair_kis_ranking(
        query, baseline, {"asr": [], "ocr": [], "object": []}
    )
    assert len(full) == len(safe) == 100
    assert [row["rank"] for row in full] == list(range(1, 101))
    assert [
        (row["video_id"], row["frame_id"]) for row in safe[:5]
    ] == [
        (row["video_id"], row["frame_id"]) for row in baseline[:5]
    ]


def test_ocr_confidence_is_normalized_from_percentage_scale() -> None:
    query = {"query_id": "Q1", "task": "KIS", "query": "SpaceX"}
    profile = {
        "Q1": {
            "important_tokens": ["SpaceX"],
            "distinctive_phrases": [],
        }
    }
    evidence = {
        "ocr": {
            "Q1": [
                {
                    "video_id": "L21_V001",
                    "frame_id": 1,
                    "rank": 1,
                    "text": "SpaceX",
                    "source_confidence": 20.0,
                },
                {
                    "video_id": "L21_V002",
                    "frame_id": 2,
                    "rank": 2,
                    "text": "SpaceX",
                    "source_confidence": 80.0,
                },
            ]
        }
    }
    tiered, _ = tier_evidence([query], evidence, profile)
    by_video = {row["video_id"]: row for row in tiered["ocr"]["Q1"]}
    assert by_video["L21_V001"]["evidence_tier"] == TIER_C
    assert by_video["L21_V002"]["evidence_tier"] != TIER_C


def test_safe_r2_logs_direct_asr_override() -> None:
    query = {"query_id": "Q1", "task": "KIS", "query": "specific topic"}
    baseline = _baseline("Q1")
    direct = {
        "video_id": "L99_V999",
        "frame_id": 7,
        "rank": 1,
        "evidence_tier": TIER_A,
        "evidence_tier_reasons": ["EXACT_DISTINCTIVE_PHRASE_COVERAGE"],
        "matched_query_anchors": ["specific topic"],
    }
    full, safe, diagnostics = repair_kis_ranking(
        query, baseline, {"asr": [direct], "ocr": [], "object": []}
    )
    assert full[0]["video_id"] == "L99_V999"
    assert safe[0]["video_id"] == "L99_V999"
    assert diagnostics["safe_top5_overrides"][0]["evidence_tier"] == TIER_A


def test_location_verifier_requires_location_phrase_not_fana_clue() -> None:
    valid = verify_answer_extraction(
        {
            "answer": "xã Phú Thuận",
            "evidence_sufficient": True,
            "supporting_source_ids": ["asr:1"],
            "supporting_spans": ["tại xã Phú Thuận"],
        },
        "LOCATION_NAME",
        [
            {
                "source_id": "asr:1",
                "modality": "asr",
                "text": "chương trình được tổ chức tại xã Phú Thuận",
            }
        ],
        grounding_plausibility=0.8,
    )
    invalid = verify_answer_extraction(
        {
            "answer": "FANA",
            "evidence_sufficient": True,
            "supporting_source_ids": ["asr:2"],
            "supporting_spans": ["chương trình FANA"],
        },
        "LOCATION_NAME",
        [{"source_id": "asr:2", "modality": "asr", "text": "chương trình FANA"}],
        grounding_plausibility=0.9,
    )
    assert valid["final_evidence_sufficient"] is True
    assert invalid["final_evidence_sufficient"] is False
    assert "LOCATION_NAME_SEMANTIC_SUPPORT_FAILED" in invalid["verifier_reasons"]


@pytest.mark.parametrize(
    ("kind", "answer", "text", "expected"),
    [
        (
            "QUOTE_OR_VISIBLE_TEXT",
            "câu thơ gợi nhắc Nguyễn Trung Trực",
            "Nguyễn Trung Trực người anh hùng bất khuất",
            False,
        ),
        ("TITLE", "Canh nấm măng tây", "Công thức: Canh nấm măng tây", True),
        ("TITLE", "món ăn", "hôm nay giới thiệu một món ăn", False),
    ],
)
def test_text_preserving_verifier_rules(kind: str, answer: str, text: str, expected: bool) -> None:
    result = verify_answer_extraction(
        {
            "answer": answer,
            "evidence_sufficient": True,
            "supporting_source_ids": ["ocr:1"],
            "supporting_spans": [text],
        },
        kind,
        [{"source_id": "ocr:1", "modality": "ocr", "text": text}],
        grounding_plausibility=0.5,
    )
    assert result["final_evidence_sufficient"] is expected


def test_qa_r2_ranks_final_sufficiency_before_grounding() -> None:
    query = {"query_id": "Q1", "task": "QA", "answer_type": "TITLE"}
    baseline = _baseline("Q1", "QA")
    verified = [
        {
            "video_id": "L99_V001",
            "frame_id": 9,
            "answer": "Canh nấm",
            "final_evidence_sufficient": True,
            "evidence_type_compatible": True,
            "corroborating_source_count": 1,
            "grounding_plausibility": 0.1,
            "grounding_rank": 20,
        },
        {
            "video_id": "L99_V002",
            "frame_id": 10,
            "answer": "món ăn",
            "final_evidence_sufficient": False,
            "evidence_type_compatible": False,
            "corroborating_source_count": 0,
            "grounding_plausibility": 1.0,
            "grounding_rank": 1,
        },
    ]
    rows = rank_qa_r2(query, verified, baseline)
    assert rows[0]["video_id"] == "L99_V001"
    assert rows[0]["final_evidence_sufficient"] is True
    assert all(row.get("answer") != "món ăn" for row in rows)


def test_qa_r2_uses_only_structural_fallback_when_no_verified_answer() -> None:
    query = {"query_id": "Q1", "task": "QA", "answer_type": "TITLE"}
    rows = rank_qa_r2(
        query,
        [
            {
                "video_id": "L99_V001",
                "frame_id": 9,
                "answer": "unverified title",
                "final_evidence_sufficient": False,
            }
        ],
        _baseline("Q1", "QA"),
    )
    assert len(rows) == 100
    assert {row["answer"] for row in rows} == {"không đủ bằng chứng"}


def test_qa_r2_structural_fill_keeps_100_when_frozen_coordinates_repeat() -> None:
    query = {"query_id": "Q1", "task": "QA", "answer_type": "TITLE"}
    baseline = [
        {
            "query_id": "Q1",
            "video_id": "L21_V001",
            "frame_id": 1,
            "rank": rank,
            "answer": "legacy",
        }
        for rank in range(1, 101)
    ]
    rows = rank_qa_r2(query, [], baseline)
    assert len(rows) == 100
    assert [row["rank"] for row in rows] == list(range(1, 101))
    assert {row["answer"] for row in rows} == {"không đủ bằng chứng"}


def test_trake_structural_fill_keeps_unique_graph_chains_then_b0_tail() -> None:
    primary = [
        {
            "query_id": "Q1",
            "video_id": "L21_V001",
            "frame_ids": [rank, rank + 100, rank + 200, rank + 300],
            "rank": rank,
            "graph_candidate": True,
        }
        for rank in range(1, 21)
    ]
    fallback = [
        {
            "query_id": "Q1",
            "video_id": f"L21_V{rank:03d}",
            "frame_ids": [rank, rank + 10, rank + 20, rank + 30],
            "rank": rank,
            "graph_candidate": False,
        }
        for rank in range(1, 101)
    ]
    rows = _complete_ranked_rows(
        primary,
        fallback,
        query_id="Q1",
        variant="M1_R2",
    )
    assert len(rows) == 100
    assert all(row["graph_candidate"] for row in rows[:20])
    assert all(
        all(
            left < right
            for left, right in zip(row["frame_ids"], row["frame_ids"][1:], strict=False)
        )
        for row in rows
    )


def test_qwen_r2_extraction_persists_sources_but_not_final_verdict(tmp_path: Path) -> None:
    import torch

    class Inputs(dict):
        def __init__(self) -> None:
            super().__init__(input_ids=torch.ones((1, 3), dtype=torch.long))
            self.input_ids = self["input_ids"]

        def to(self, _: str):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return Inputs()

        def batch_decode(self, *args, **kwargs):
            return [
                json.dumps(
                    {
                        "answer": "xã Phú Thuận",
                        "answer_type": "LOCATION_NAME",
                        "supporting_source_ids": ["asr:1"],
                        "supporting_spans": ["tại xã Phú Thuận"],
                        "evidence_sufficient": True,
                    },
                    ensure_ascii=False,
                )
            ]

    class Model:
        def generate(self, **kwargs):
            return torch.ones((1, 5), dtype=torch.long)

    adapter = QwenEvidenceAdapter(tmp_path, device="cpu")
    adapter.processor, adapter.model = Processor(), Model()
    extraction, audit = adapter.answer_extraction(
        GroundingCandidate("L21_V001", 1, 1, {}),
        object(),
        description="location",
        question="Tên xã là gì?",
        evidence_rows=[
            {"source_id": "asr:1", "modality": "asr", "text": "tại xã Phú Thuận"}
        ],
        answer_type="LOCATION_NAME",
        answer_policy="TEXT_PRESERVING",
    )
    assert extraction["supporting_source_ids"] == ["asr:1"]
    assert "final_evidence_sufficient" not in extraction
    assert audit["final_evidence_sufficient_assigned_here"] is False


def test_qwen_r2_accepts_explicit_empty_insufficient_extraction(tmp_path: Path) -> None:
    import torch

    class Inputs(dict):
        def __init__(self) -> None:
            super().__init__(input_ids=torch.ones((1, 3), dtype=torch.long))
            self.input_ids = self["input_ids"]

        def to(self, _: str):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return Inputs()

        def batch_decode(self, *args, **kwargs):
            return [
                json.dumps(
                    {
                        "answer": "",
                        "answer_type": "TITLE",
                        "supporting_source_ids": [],
                        "supporting_spans": [],
                        "evidence_sufficient": False,
                    }
                )
            ]

    class Model:
        def generate(self, **kwargs):
            return torch.ones((1, 5), dtype=torch.long)

    adapter = QwenEvidenceAdapter(tmp_path, device="cpu")
    adapter.processor, adapter.model = Processor(), Model()
    extraction, audit = adapter.answer_extraction(
        GroundingCandidate("L21_V001", 1, 1, {}),
        object(),
        description="title",
        question="Tên món ăn là gì?",
        evidence_rows=[],
        answer_type="TITLE",
        answer_policy="TEXT_PRESERVING",
    )
    assert extraction is not None
    assert extraction["evidence_sufficient"] is False
    assert extraction["answer"] == ""
    assert audit["parse_reason"] is None


def test_r2_writer_emits_required_review_packet_and_candidate_zips(tmp_path: Path) -> None:
    query = {"query_id": "Q1", "task": "KIS"}
    b0 = [{"query_id": "Q1", "video_id": "L21_V001", "frame_id": 1, "rank": 1}]
    row = {
        **b0[0],
        "evidence_tier": TIER_A,
        "evidence_tier_reasons": ["DIRECT"],
        "dominant_modalities": ["asr"],
        "matched_query_anchors": ["anchor"],
        "corroborating_sources": ["asr:1"],
    }
    previous = {
        "TRIAGEEG_M0_FULL": [{**row, "system_variant": "TRIAGEEG_M0_FULL"}],
        "TRIAGEEG_SAFE": [{**row, "system_variant": "TRIAGEEG_SAFE"}],
    }
    result = {
        "candidates": {
            name: [{**row, "system_variant": name}]
            for name in ("M0_R2", "M1_R2", "SAFE_R2")
        },
        "validation": {
            name: {"status": "PASS"} for name in ("M0_R2", "M1_R2", "SAFE_R2")
        },
        "kis_diagnostics": [
            {
                "query_id": "Q1",
                "weak_modality_override": False,
                "strong_asr_dropped": [],
            }
        ],
        "m1_diagnostics": [],
        "trake_checks": {},
    }
    report = write_r2_artifacts(
        tmp_path,
        [query],
        b0,
        previous,
        result,
        [],
        [],
        [],
        provenance={"gt_opened": False, "submission_uploaded": False},
    )
    expected = {
        "trial_p1_r2_human_review.md",
        "trial_p1_r2_candidate_comparison.csv",
        "trial_p1_r2_candidate_comparison.json",
        "kis_evidence_tier_diagnostics.jsonl",
        "asr_specificity_diagnostics.jsonl",
        "qa_extractions.jsonl",
        "qa_semantic_verifier.jsonl",
        "qa_r2_readiness.json",
        "trake_r2_graph_summary.json",
        "M0_R2.zip",
        "M1_R2.zip",
        "SAFE_R2.zip",
        "SUBMISSION_2_R2_DECISION.md",
        "run_provenance.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert report["recommendation"] == "DO_NOT_SUBMIT_2_YET"
