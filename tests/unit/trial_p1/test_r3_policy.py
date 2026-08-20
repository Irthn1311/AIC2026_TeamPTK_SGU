from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_eg.fs1.qa import GroundingCandidate
from triage_eg.fs1.qwen_adapter import QwenEvidenceAdapter, _first_complete_json_object
from triage_eg.trial_p1.r2_policy import TIER_A, TIER_C
from triage_eg.trial_p1.r3_policy import (
    augment_qa_context_r3,
    build_bounded_qa_rescue_evidence,
    canonicalize_title,
    classify_asr_r3,
    derive_r3_anchor_profiles,
    evaluate_context_relevance,
    repair_kis_r3,
    tier_evidence_r3,
    verify_answer_r3,
    write_r3_artifacts,
)


def _profile(raw: str, *, answer_type: str = "OTHER") -> dict:
    queries = [{"query_id": "Q1", "task": "QA", "query": raw, "answer_type": answer_type}]
    plans = [
        {
            "query_id": "Q1",
            "raw_text": raw,
            "action_anchors": [],
            "knowledge_expansions": [],
        }
    ]
    return derive_r3_anchor_profiles(queries, plans)["Q1"]


def _baseline(query_id: str = "Q1", task: str = "KIS") -> list[dict]:
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


def _asr(text: str, *, branches: bool = True) -> dict:
    return {
        "query_id": "Q1",
        "video_id": "L99_V001",
        "frame_id": 7,
        "rank": 1,
        "asr_span": {"text": text},
        "asr_source_ranks": (
            [{"branch": "LEXICAL", "rank": 1}, {"branch": "E5", "rank": 2}]
            if branches
            else [{"branch": "E5", "rank": 1}]
        ),
    }


def test_generic_folded_tokens_never_create_asr_tier_a() -> None:
    profile = _profile("Nhiệm vụ hay thực hiện trong nhà vào buổi sáng")
    row = classify_asr_r3(_asr("nhiem sang hay thuc ten nha"), profile)
    assert row["evidence_tier"] == TIER_C
    assert not {
        "nhiem",
        "sang",
        "hay",
        "thuc",
        "ten",
        "nha",
    }.intersection(row["matched_query_anchors"])


def test_exact_named_entity_with_lexical_e5_agreement_is_tier_a() -> None:
    profile = _profile("Nguyễn Trung Trực tại Kiên Giang")
    row = classify_asr_r3(
        _asr("Câu chuyện về Nguyễn Trung Trực được kể tại Kiên Giang"), profile
    )
    assert row["evidence_tier"] == TIER_A
    assert row["asr_specificity"]["lexical_rank"] == 1
    assert row["asr_specificity"]["e5_rank"] == 2
    assert "ENTITY_PHRASE_HIGH" in row["matched_anchor_classes"]


def test_lexical_e5_agreement_alone_is_not_tier_a() -> None:
    profile = _profile("một người đang nghiên cứu trong nhà")
    row = classify_asr_r3(_asr("người đang ở trong nhà"), profile)
    assert row["evidence_tier"] == TIER_C


def test_object_never_tier_a_and_ocr_needs_high_multiword_phrase() -> None:
    query = {"query_id": "Q1", "task": "KIS", "query": "Nguyễn Trung Trực"}
    profile = _profile("Nguyễn Trung Trực tại Kiên Giang")
    evidence = {
        "object": {
            "Q1": [
                {
                    "video_id": "L99_V001",
                    "frame_id": 1,
                    "rank": 1,
                    "text": "Nguyễn Trung Trực",
                }
            ]
        },
        "ocr": {
            "Q1": [
                {
                    "video_id": "L99_V002",
                    "frame_id": 2,
                    "rank": 1,
                    "text": "Nguyễn Trung Trực",
                    "source_confidence": 80.0,
                },
                {
                    "video_id": "L99_V003",
                    "frame_id": 3,
                    "rank": 2,
                    "text": "Nguyễn",
                    "source_confidence": 99.0,
                },
            ]
        },
    }
    tiered, _ = tier_evidence_r3([query], evidence, {"Q1": profile})
    assert tiered["object"]["Q1"][0]["evidence_tier"] == TIER_C
    by_video = {row["video_id"]: row for row in tiered["ocr"]["Q1"]}
    assert by_video["L99_V002"]["evidence_tier"] == TIER_A
    assert by_video["L99_V003"]["evidence_tier"] != TIER_A


def test_strong_asr_is_in_top20_and_safe_top5_stays_exact() -> None:
    baseline = _baseline()
    strong = {
        **_asr("Nguyễn Trung Trực tại Kiên Giang"),
        "evidence_tier": TIER_A,
        "evidence_tier_reasons": ["EXACT_HIGH_ENTITY_OR_DISTINCTIVE_PHRASE"],
        "matched_query_anchors": ["Nguyễn Trung Trực"],
        "matched_phrase_anchors": ["Nguyễn Trung Trực"],
        "asr_specificity": {
            "exact_high_phrase": True,
            "lexical_e5_agreement": True,
            "meaningful_anchor_count": 2,
        },
    }
    full, safe, diagnostic = repair_kis_r3(
        {"query_id": "Q1", "task": "KIS"},
        baseline,
        {"asr": [strong], "ocr": [], "object": []},
    )
    assert next(row["rank"] for row in full if row["video_id"] == "L99_V001") <= 20
    assert [(row["video_id"], row["frame_id"]) for row in safe[:5]] == [
        (row["video_id"], row["frame_id"]) for row in baseline[:5]
    ]
    assert diagnostic["strong_asr_inclusion"]["inclusion_status"] != "DROPPED"


def test_context_gate_rejects_faithful_but_unrelated_quote() -> None:
    profile = _profile("Hai câu thơ về Nguyễn Trung Trực tại Kiên Giang", answer_type="QUOTE")
    unrelated = evaluate_context_relevance(
        profile,
        [{"source_id": "asr:1", "modality": "asr", "text": "trăng sáng trên quê hương"}],
    )
    related = evaluate_context_relevance(
        profile,
        [
            {
                "source_id": "asr:2",
                "modality": "asr",
                "text": "Nguyễn Trung Trực, người anh hùng Kiên Giang",
            }
        ],
    )
    assert unrelated["context_relevant"] is False
    assert related["context_relevant"] is True


def test_r3_context_augmentation_keeps_local_answer_and_adds_same_video_entity() -> None:
    profile = _profile("FANA tổ chức tại Khánh Hòa", answer_type="LOCATION_NAME")
    local = [
        {
            "source_id": "asr:answer",
            "modality": "asr",
            "video_id": "L21_V001",
            "frame_id": 500,
            "text": "tổ chức tại xã Phú Thuận",
        }
    ]
    rows = augment_qa_context_r3(
        profile,
        "L21_V001",
        local,
        {
            "asr": [
                {
                    "video_id": "L21_V001",
                    "frame_id": 100,
                    "rank": 4,
                    "asr_span": {"chunk_id": "context", "text": "FANA tại Khánh Hòa"},
                },
                {
                    "video_id": "L21_V999",
                    "frame_id": 1,
                    "rank": 1,
                    "asr_span": {"chunk_id": "wrong", "text": "FANA tại Khánh Hòa"},
                },
            ],
            "ocr": [],
        },
    )
    assert rows[0]["source_id"] == "asr:answer"
    assert {row["source_id"] for row in rows} == {"asr:answer", "asr:context"}


def test_location_rescue_scans_only_shortlisted_video_rows(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = tmp_path / "ocr.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video_id": "L21_V001",
                    "frame_idx": 10,
                    "corrected_text": "tại xã Phú Thuận",
                    "combined_text": "tại xã Phú Thuận",
                    "mean_confidence": 0.9,
                },
                {
                    "video_id": "L99_V999",
                    "frame_idx": 20,
                    "corrected_text": "tại xã ngoài shortlist",
                    "combined_text": "tại xã ngoài shortlist",
                    "mean_confidence": 0.9,
                },
            ]
        ),
        parquet,
    )

    class Loader:
        transcripts = [
            {
                "video_id": "L21_V001",
                "chunk_id": "context",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "text": "FANA hoạt động tại Khánh Hòa",
            },
            {
                "video_id": "L21_V001",
                "chunk_id": "answer",
                "start_seconds": 80.0,
                "end_seconds": 85.0,
                "text": "đoàn trao quà tại xã Phú Thuận",
            },
            {
                "video_id": "L99_V999",
                "chunk_id": "wrong",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "text": "FANA tại xã ngoài shortlist",
            },
        ]

        @staticmethod
        def map_span_to_frame(row, mapper):
            return {**row, "frame_id": 100}

    query = {
        "query_id": "Q1",
        "task": "QA",
        "query": "FANA trao quà ở xã nào tại Khánh Hòa?",
        "answer_type": "LOCATION_NAME",
    }
    profile = _profile(query["query"], answer_type="LOCATION_NAME")
    rescue, diagnostics = build_bounded_qa_rescue_evidence(
        [query],
        {"Q1": profile},
        Loader(),
        parquet,
        object(),
        {
            "asr": {
                "Q1": [
                    {
                        "video_id": "L21_V001",
                        "frame_id": 1,
                        "rank": 1,
                        "asr_span": {"text": "FANA tại Khánh Hòa"},
                    }
                ]
            },
            "ocr": {"Q1": []},
        },
        _baseline("Q1", "QA"),
        max_videos=1,
    )
    assert {row["video_id"] for row in rescue["asr"]["Q1"]} == {"L21_V001"}
    assert {row["video_id"] for row in rescue["ocr"]["Q1"]} == {"L21_V001"}
    assert diagnostics[0]["corpus_job_launched"] is False


def test_location_granularity_and_title_canonicalization() -> None:
    profile = _profile("FANA tổ chức từ thiện ở xã nào tại Khánh Hòa?", answer_type="LOCATION_NAME")
    invalid = verify_answer_r3(
        {
            "answer": "Khánh Hòa",
            "answer_type": "LOCATION_NAME",
            "supporting_source_ids": ["asr:1"],
            "supporting_spans": ["tại tỉnh Khánh Hòa"],
            "evidence_sufficient": True,
        },
        "LOCATION_NAME",
        [
            {
                "source_id": "asr:1",
                "modality": "asr",
                "text": "FANA tổ chức tại tỉnh Khánh Hòa",
            }
        ],
        profile,
        grounding_plausibility=0.9,
    )
    assert invalid["final_evidence_sufficient"] is False
    assert "LOCATION_GRANULARITY_NOT_SUPPORTED" in invalid["answer_type_verifier_reasons"]
    assert canonicalize_title("món ăn có tên gọi là mực chiên giòn sốt tắc") == (
        "mực chiên giòn sốt tắc",
        True,
    )


def test_qwen_balanced_parser_accepts_fence_and_rejects_truncation() -> None:
    value = _first_complete_json_object('```json\n{"answer":"ok"}\n``` trailing')
    assert value == {"answer": "ok"}
    with pytest.raises(ValueError, match="TRUNCATED"):
        _first_complete_json_object('{"answer":"unfinished"')


def test_qwen_rejects_more_than_three_supports(tmp_path: Path) -> None:
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
                        "supporting_source_ids": ["a", "b", "c", "d"],
                        "supporting_spans": ["xã Phú Thuận"],
                        "evidence_sufficient": True,
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
        description="location",
        question="xã nào?",
        evidence_rows=[],
        answer_type="LOCATION_NAME",
        answer_policy="TEXT_PRESERVING",
    )
    assert extraction is None
    assert "QWEN_R2_EXTRACTION_CONTRACT_INVALID" in audit["parse_reason"]


def test_r3_writer_emits_required_packet(tmp_path: Path) -> None:
    query = {"query_id": "Q1", "task": "KIS"}
    baseline = _baseline()[:1]
    candidates = {
        name: [{**baseline[0], "system_variant": name}]
        for name in ("M0_R3", "M1_R3", "SAFE_R3")
    }
    report = write_r3_artifacts(
        tmp_path,
        [query],
        baseline,
        {"M0_R2": [{**baseline[0], "system_variant": "M0_R2"}]},
        {
            "candidates": candidates,
            "validation": {name: {"status": "PASS"} for name in candidates},
            "kis_diagnostics": [
                {
                    "query_id": "Q1",
                    "safe_top5_exact": True,
                    "strong_asr_inclusion": {
                        "query_id": "Q1",
                        "inclusion_status": "NO_QUALIFIED_DIRECT_ASR",
                        "best_strong_asr_video": None,
                        "final_best_rank": None,
                    },
                }
            ],
            "strong_asr_audit": [
                {
                    "query_id": "Q1",
                    "inclusion_status": "NO_QUALIFIED_DIRECT_ASR",
                    "best_strong_asr_video": None,
                    "final_best_rank": None,
                }
            ],
            "trake_checks": {},
        },
        {"Q1": {"anchors": []}},
        [],
        [],
        [],
        [],
        provenance={"runtime_candidate_failure_count": 0, "asset_hashes": {}},
    )
    assert report["recommendation"] == "DO_NOT_SUBMIT_2_YET"
    for name in (
        "SUBMISSION_2_R3_DECISION.md",
        "trial_p1_r3_human_review.md",
        "trial_p1_r3_candidate_comparison.json",
        "trial_p1_r3_candidate_comparison.csv",
        "query_anchor_diagnostics.jsonl",
        "asr_r3_specificity_diagnostics.jsonl",
        "strong_asr_inclusion_audit.jsonl",
        "qa_r3_extractions.jsonl",
        "qa_context_relevance.jsonl",
        "qa_r3_semantic_verifier.jsonl",
        "qa_r3_readiness.json",
        "trake_r3_graph_summary.json",
        "asset_hashes.json",
        "run_provenance.json",
    ):
        assert (tmp_path / name).is_file(), name
