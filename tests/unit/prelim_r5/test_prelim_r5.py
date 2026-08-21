from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from triage_eg.prelim_r5 import (
    R5Settings,
    build_deterministic_qa_rows,
    build_query_views,
    build_r5_query_candidates,
    finalize_pre_gt_predictions,
    fuse_asr_multiview,
    fuse_multiview_branch,
    materialize_view_queries,
    select_production_policy,
)


def _kis_rows(query_id: str, *, offset: int = 0, variant: str = "TEST") -> list[dict]:
    return [
        {
            "query_id": query_id,
            "video_id": f"L{(rank + offset) // 10:02d}_V{rank + offset:03d}",
            "frame_id": rank + offset,
            "rank": rank,
            "system_variant": variant,
        }
        for rank in range(1, 101)
    ]


def _qa_rows(query_id: str) -> list[dict]:
    return [
        {
            "query_id": query_id,
            "video_id": f"L00_V{rank:03d}",
            "frame_id": rank,
            "answer": f"fallback {rank}",
            "rank": rank,
            "system_variant": "BCF1",
        }
        for rank in range(1, 101)
    ]


def test_views_are_traceable_and_trake_ordinals_do_not_collapse() -> None:
    query = {
        "query_id": "TR-1",
        "task": "TRAKE",
        "query": "Chuỗi hai sự kiện",
        "language": "vi",
        "event_count": 2,
        "event_descriptions": [
            {"event_id": "E1", "description": "Một xe tải màu đỏ chạy qua cầu."},
            {"event_id": "E2", "description": "Người dẫn chương trình mở cửa."},
        ],
    }
    views = build_query_views(query, translator=lambda text: f"EN {text}")
    assert len(views) == 10
    assert {row["event_index"] for row in views} == {0, 1}
    assert all(row["source_spans"] and row["gt_used"] is False for row in views)
    transformed = materialize_view_queries(query, views)
    assert all(value["event_count"] == 2 for value in transformed.values())
    assert all(len(value["event_descriptions"]) == 2 for value in transformed.values())


def test_multiview_rrf_retains_view_provenance() -> None:
    query = {"query_id": "K-1", "task": "KIS"}
    shared = _kis_rows("K-1__r5__original_vi")[:3]
    translated = [
        {**row, "query_id": "K-1__r5__translated_en", "rank": rank}
        for rank, row in enumerate(reversed(shared), 1)
    ]
    fused, provenance = fuse_multiview_branch(
        query,
        {"ORIGINAL_VI": shared, "TRANSLATED_EN": translated},
        branch="A0",
    )
    assert len(fused) == 3
    assert all(row["query_id"] == "K-1" for row in fused)
    assert all(row["views_agreeing"] == 2 for row in provenance)
    assert all(row["rrf_k"] == 60 for row in provenance)


def test_safe_r5_protection_strong_asr_and_single_gated_override() -> None:
    query = {"query_id": "K-1", "task": "KIS"}
    bcf1 = _kis_rows("K-1")
    safe = _kis_rows("K-1", offset=100)
    a0 = _kis_rows("K-1", offset=200)
    s1 = _kis_rows("K-1", offset=300)
    representative = a0[0]
    result = build_r5_query_candidates(
        query,
        bcf1=bcf1,
        safe_r4_tail_source=safe,
        a0_multiview=a0,
        s1_multiview=s1,
        asr_multiview=_kis_rows("K-1", offset=400),
        r5_strong_asr={
            "qualified": True,
            "video_id": representative["video_id"],
            "representative": representative,
            "tier": "TIER_A_DIRECT",
        },
        gated_candidates=[
            {
                "rank": 1,
                "tier": "TIER_A_DIRECT",
                "high_anchor_match_count": 1,
                "visual_support": True,
                "lexical_e5_agreement": True,
                "ocr_exact_high_with_visual": False,
                "object_only": False,
                "weak_ocr_only": False,
                "representative": representative,
            }
        ],
    )
    live = result["SAFE_R4_LIVE_WINNER"]
    qe = result["SAFE_R5_QE"]
    gated = result["SAFE_R5_GATED"]
    assert [(row["video_id"], row["frame_id"]) for row in live[:5]] == [
        (row["video_id"], row["frame_id"]) for row in bcf1[:5]
    ]
    assert [(row["video_id"], row["frame_id"]) for row in qe[:5]] == [
        (row["video_id"], row["frame_id"]) for row in bcf1[:5]
    ]
    assert len(qe) == len(gated) == 100
    assert gated[0]["video_id"] == bcf1[0]["video_id"]
    assert gated[4]["video_id"] == representative["video_id"]
    assert result["head_override_audit"]["override"] is True


def test_asr_multiview_requires_canonical_representative_and_records_gates() -> None:
    query = {"query_id": "K-1", "task": "KIS", "query": "Hội An", "language": "vi"}
    views = build_query_views(query, translator=lambda _: "Hoi An")
    representatives = _kis_rows("K-1")
    target = representatives[0]["video_id"]
    lexical = {
        name: [
            {
                "video_id": target,
                "text": "Thành phố Hội An",
                "asr_rank": 1,
                "chunk_id": f"{name}-L",
            }
        ]
        for name in ("ORIGINAL_VI", "ENTITY_DISTINCTIVE")
    }
    e5 = {
        "ORIGINAL_VI": [
            {
                "video_id": target,
                "text": "Thành phố Hội An",
                "rank": 1,
                "chunk_id": "E5",
            }
        ]
    }
    result = fuse_asr_multiview(
        query,
        views,
        lexical,
        e5,
        a0_multiview=representatives,
        s1_multiview=_kis_rows("K-1", offset=100),
        fallback_rows=representatives,
        safe_r4_rows=representatives,
    )
    assert result["rows"][0]["video_id"] == target
    assert result["provenance"][0]["lexical_e5_agreement"] is True
    assert result["provenance"][0]["visual_support"] is True
    assert result["strong"]["qualified"] is True


def test_deterministic_qa_extracts_supported_title_or_copies_exact_fallback() -> None:
    query = {
        "query_id": "Q-1",
        "task": "QA",
        "query": "Cảnh tờ báo",
        "question": "Tiêu đề là gì?",
        "answer_type": "TITLE",
    }
    fallback = _qa_rows("Q-1")
    output, audit = build_deterministic_qa_rows(
        query,
        [
            {
                "video_id": fallback[0]["video_id"],
                "frame_id": fallback[0]["frame_id"],
                "text": 'Tờ báo có tên là "COVIDIA".',
                "source": "OCR",
            }
        ],
        fallback,
        context_videos=[fallback[0]["video_id"]],
    )
    assert output[0]["answer"] == "COVIDIA"
    assert len(output) == 100
    copied, copied_audit = build_deterministic_qa_rows(query, [], fallback, context_videos=[])
    assert copied == fallback
    assert copied_audit[-1]["decision"].startswith("EXACT_BCF1_FALLBACK")
    assert audit[-1]["decision"] == "DETERMINISTIC_EVIDENCE_FIRST_WITH_BCF1_TAIL"


def _all_predictions(queries: dict[str, list[dict[str, Any]]]) -> dict:
    return {
        benchmark: {
            arm: _kis_rows(query_rows[0]["query_id"], variant=arm)
            for arm in ("TRUE_BCF1", "SAFE_R4_LIVE_WINNER", "SAFE_R5_QE", "SAFE_R5_GATED")
        }
        for benchmark, query_rows in queries.items()
    }


def test_pre_gt_freezes_all_arms_and_rejects_forbidden_reference(tmp_path: Path) -> None:
    queries = {
        "cross": [{"query_id": "C-1", "task": "KIS", "query": "x", "language": "vi"}],
        "l21": [{"query_id": "L-1", "task": "KIS", "query": "y", "language": "vi"}],
    }
    predictions = _all_predictions(queries)
    frozen = finalize_pre_gt_predictions(tmp_path, queries, predictions, config={"r5": True})
    assert frozen["all_predictions_finalized_before_gt"] is True
    assert len(frozen["prediction_hashes"]["cross"]) == 4
    payload = json.loads((tmp_path / "pre_gt_prediction_hashes.json").read_text())
    assert payload["gt_opened"] is False
    with pytest.raises(RuntimeError, match="R5_SEALED_FINAL_REFERENCE_FORBIDDEN"):
        finalize_pre_gt_predictions(
            tmp_path / "bad", queries, predictions, config={"forbidden": "SEALED_FINAL_30"}
        )


def _evaluation(score: float, query_scores: list[float]) -> dict[str, Any]:
    return {
        "summary": {"final_score": score},
        "per_query": [
            {"query_id": f"Q-{index}", "final_score": value}
            for index, value in enumerate(query_scores)
        ],
    }


def test_frozen_decision_defaults_to_live_and_promotes_only_when_gates_pass() -> None:
    evaluations = {
        benchmark: {
            "TRUE_BCF1": _evaluation(0.1, [0.0, 0.0, 0.0]),
            "SAFE_R4_LIVE_WINNER": _evaluation(0.2, [0.1, 0.1, 0.1]),
            "SAFE_R5_QE": _evaluation(0.21, [0.2, 0.2, 0.2]),
            "SAFE_R5_GATED": _evaluation(0.2, [0.1, 0.1, 0.1]),
        }
        for benchmark in ("cross", "l21")
    }
    fallback = select_production_policy(
        evaluations,
        {"all_structural_gates_pass": False, "coverage_improved": True},
    )
    assert fallback["production_policy"] == "PRODUCTION_SAFE_R4_LIVE_WINNER"
    promoted = select_production_policy(
        evaluations,
        {
            "all_structural_gates_pass": True,
            "coverage_improved": True,
            "override_audit_sane": True,
        },
    )
    assert promoted["production_policy"] == "PRODUCTION_SAFE_R5_QE"
    assert promoted["frozen_thresholds"] == R5Settings().__dict__
