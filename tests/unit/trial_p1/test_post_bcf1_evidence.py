from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_eg.fs1_v11.contracts import QWEN_ID, QWEN_REVISION, WHISPER_ID, WHISPER_REVISION
from triage_eg.trial_p1.asr_v12_loader import ASRV12Loader
from triage_eg.trial_p1.post_bcf1 import prepare_post_bcf1_artifacts
from triage_eg.trial_p1.qa_evidence import (
    BoundedEvidencePackage,
    BoundedQwenExecutor,
    assess_answer_evidence,
    rank_full_qa,
)

REPO = Path(__file__).resolve().parents[3]


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _asr_fixture(root: Path) -> None:
    root.mkdir()
    inventory = [
        {"video_id": "L01_V001", "has_audio": True, "probe_status": "PASS"},
        {"video_id": "L01_V002", "has_audio": True, "probe_status": "PASS"},
        {"video_id": "L01_V003", "has_audio": False, "probe_status": "NO_AUDIO"},
    ]
    transcripts = []
    for video_id, text in (("L01_V001", "xã Vạn Thạnh"), ("L01_V002", "tin thể thao")):
        transcripts.append(
            {
                "video_id": video_id,
                "status": "PASS",
                "model_id": WHISPER_ID,
                "model_revision": WHISPER_REVISION,
                "segments": [
                    {
                        "start_seconds": 1.0,
                        "end_seconds": 3.0,
                        "normalized_text": text,
                    }
                ],
            }
        )
    _jsonl(root / "asr_audio_inventory_v12.jsonl", inventory)
    _jsonl(root / "asr_transcripts_v12.jsonl", transcripts)
    (root / "asr_lexical_index_v12.json").write_text(
        json.dumps(
            {
                "xã": [
                    {
                        "video_id": "L01_V001",
                        "start_seconds": 1.0,
                        "end_seconds": 3.0,
                        "text": "xã Vạn Thạnh",
                    }
                ],
                "thể": [
                    {
                        "video_id": "L01_V002",
                        "start_seconds": 1.0,
                        "end_seconds": 3.0,
                        "text": "tin thể thao",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "asr_performance_report_v12.json").write_text(
        json.dumps({"status": "PASS", "pass_rate": 1.0}), encoding="utf-8"
    )
    (root / "full_shard_0_manifest.json").write_text(
        json.dumps({"shard_id": 0, "videos": ["L01_V001", "L01_V002"]}),
        encoding="utf-8",
    )


def test_asr_v12_loader_validates_and_retrieves_seconds_only(tmp_path: Path) -> None:
    root = tmp_path / "asr"
    _asr_fixture(root)
    loader = ASRV12Loader(root)
    assert loader.validation.as_dict() == {
        "status": "PASS",
        "model_id": WHISPER_ID,
        "model_revision": WHISPER_REVISION,
        "inventory_audio_video_count": 2,
        "transcript_video_count": 2,
        "pass_video_count": 2,
        "lexical_term_count": 2,
        "manifest_file_count": 1,
        "coverage_complete": True,
        "performance_status": "PASS",
    }
    spans = loader.retrieve_spans("Tên xã là gì?", video_ids={"L01_V001"})
    assert spans[0]["video_id"] == "L01_V001"
    assert "frame_id" not in spans[0]
    mapped = loader.map_span_to_frame(spans[0], lambda video_id, seconds: 321)
    assert mapped["frame_id"] == 321
    assert mapped["frame_mapping_source"] == "INJECTED_CANONICAL_FRAMEMAP"
    assert loader.rank_video_hypotheses("tin thể thao")[0]["fusion_contract"] == (
        "RANK_LEVEL_COMPLEMENTARY_BRANCH_ONLY"
    )


def test_asr_v12_loader_rejects_wrong_revision_and_inferred_frames(tmp_path: Path) -> None:
    root = tmp_path / "asr"
    _asr_fixture(root)
    rows = [
        json.loads(line) for line in (root / "asr_transcripts_v12.jsonl").read_text().splitlines()
    ]
    rows[0]["model_revision"] = "floating-main"
    _jsonl(root / "asr_transcripts_v12.jsonl", rows)
    with pytest.raises(RuntimeError, match="MODEL_PROVENANCE_MISMATCH"):
        ASRV12Loader(root)
    rows[0]["model_revision"] = WHISPER_REVISION
    rows[0]["segments"][0]["frame_id"] = 99
    _jsonl(root / "asr_transcripts_v12.jsonl", rows)
    with pytest.raises(RuntimeError, match="INFERRED_FRAME_FIELD_FORBIDDEN"):
        ASRV12Loader(root)


def test_text_preserving_sufficiency_rejects_generic_and_fragment() -> None:
    empty = BoundedEvidencePackage("L01_V001", 10, 1, ("A0_OPENAI_CLIP",))
    generic = assess_answer_evidence("người", "LOCATION_NAME", empty)
    assert generic.syntax_pass and not generic.evidence_sufficient
    fragment = assess_answer_evidence(
        "beet",
        "TITLE",
        BoundedEvidencePackage(
            "L01_V001",
            10,
            1,
            ("A0_OPENAI_CLIP",),
            ocr_lines=({"text": "beet", "confidence": 99},),
        ),
    )
    assert not fragment.evidence_sufficient
    contextual = assess_answer_evidence(
        "Xã Vạn Thạnh",
        "LOCATION_NAME",
        BoundedEvidencePackage(
            "L01_V001",
            10,
            1,
            ("A0_OPENAI_CLIP",),
            ocr_lines=({"text": "Ủy ban Xã Vạn Thạnh", "confidence": 91},),
        ),
    )
    assert contextual.evidence_sufficient
    assert contextual.evidence_sources == ("OCR_CONTEXTUAL_PHRASE",)


def test_qwen_executor_is_bounded_pinned_and_cannot_invent_ids() -> None:
    empty = BoundedEvidencePackage("L01_V001", 10, 1, ("S1_SIGLIP2",))
    executor = BoundedQwenExecutor(lambda prompt, package: "not called")
    result, diagnostic = executor.execute("Tên xã?", "LOCATION_NAME", empty)
    assert result is None and diagnostic["reason"] == "QWEN_BOUNDED_EVIDENCE_PACKAGE_EMPTY"
    assert (diagnostic["model_id"], diagnostic["model_revision"]) == (QWEN_ID, QWEN_REVISION)

    package = BoundedEvidencePackage(
        "L01_V001",
        10,
        1,
        ("S1_SIGLIP2",),
        visual_context_present=True,
        ocr_lines=({"text": "Xã Vạn Thạnh", "confidence": 92},),
    )
    valid = BoundedQwenExecutor(
        lambda prompt, bounded: '{"answer":"Xã Vạn Thạnh","evidence_sufficient":true}'
    )
    result, diagnostic = valid.execute("Tên xã?", "LOCATION_NAME", package)
    assert result == {
        "video_id": "L01_V001",
        "frame_id": 10,
        "answer": "Xã Vạn Thạnh",
        "answer_type": "LOCATION_NAME",
        "evidence_sufficient": True,
    }
    assert diagnostic["qwen_verified"] is True
    invented = BoundedQwenExecutor(
        lambda prompt, bounded: '{"answer":"L01_V001","evidence_sufficient":true}'
    )
    result, diagnostic = invented.execute("Tên xã?", "LOCATION_NAME", package)
    assert result is None and diagnostic["reason"] == "QWEN_OUTPUT_CONTRACT_INVALID"


def test_full_qa_ranking_has_no_protected_prefix() -> None:
    rows = [
        {
            "video_id": "L01_V001",
            "frame_id": 1,
            "answer": "weak",
            "grounding_rank": 1,
            "grounding_plausibility": 0.7,
            "evidence_sufficient": False,
            "evidence_sources": [],
        },
        {
            "video_id": "L01_V002",
            "frame_id": 2,
            "answer": "verified",
            "grounding_rank": 8,
            "grounding_plausibility": 0.9,
            "evidence_sufficient": True,
            "evidence_sources": ["OCR_CONTEXTUAL_PHRASE", "ASR_LOCAL_SPAN"],
        },
    ]
    ranked = rank_full_qa(rows)
    assert ranked[0]["answer"] == "verified"
    assert [row["rank"] for row in ranked] == [1, 2]


def test_actual_trial_bundle_freezes_visual_and_fails_qa_closed(tmp_path: Path) -> None:
    bundle = REPO / "outputs/trial_p1_TRUE_BCF1_bundle.zip"
    if not bundle.is_file():
        pytest.skip("downloaded Trial BCF1 bundle is not present")
    result = prepare_post_bcf1_artifacts(bundle, tmp_path / "post")
    assert result["freeze"]["a0_top5_preserved_all_21"] is True
    assert result["freeze"]["historical_exactness_status"].startswith("NOT_PROVEN")
    assert result["readiness"]["not_ready_query_count"] == 3
    assert result["readiness"]["kis_trake_packaging_blocked"] is False
