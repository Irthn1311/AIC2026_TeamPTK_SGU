from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_eg.trial_p1.asr_v12_loader import (
    ASR_EXTERNAL_V3_SOURCE_TYPE,
    ASR_V12_SOURCE_TYPE,
    ASRExternalV3Loader,
    load_asr_evidence,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(root: Path) -> None:
    root.mkdir()
    rows = [
        {
            "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
            "video_id": "L01_V001",
            "chunk_id": "L01_V001_chunk_0000",
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "text": "xã Vạn Thạnh",
            "status": "PASS",
            "is_no_speech": False,
        },
        {
            "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
            "video_id": "L01_V002",
            "chunk_id": "L01_V002_nospeech_0000",
            "start_seconds": 0.0,
            "end_seconds": 0.0,
            "text": "[NO_SPEECH]",
            "status": "NO_SPEECH",
            "is_no_speech": True,
        },
    ]
    (root / "asr_external_v3_transcripts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(
        root / "asr_external_v3_lexical_index.json",
        {
            "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
            "chunks": {
                "L01_V001_chunk_0000": {
                    "video_id": "L01_V001",
                    "start_seconds": 1.0,
                    "end_seconds": 3.0,
                    "text": "xã Vạn Thạnh",
                    "e5_row_index": 0,
                }
            },
            "postings": {"xã": ["L01_V001_chunk_0000"]},
        },
    )
    _write_json(
        root / "asr_external_v3_e5_manifest.json",
        {"alignment_status": "PASS", "vector_count": 1},
    )
    _write_json(
        root / "asr_external_v3_video_coverage.json",
        {"coverage_complete": True, "canonical_video_count": 2},
    )
    _write_json(root / "asr_external_v3_timestamp_audit.json", {"status": "PASS"})
    _write_json(
        root / "asr_external_v3_provenance.json",
        {
            "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
            "provenance_level": "VALIDATED_EXTERNAL_NOT_REPRODUCIBLE",
            "is_whisper_v12_exact": False,
        },
    )


def test_external_loader_preserves_explicit_source_and_normalized_interface(tmp_path: Path) -> None:
    root = tmp_path / "external"
    _fixture(root)
    loader = ASRExternalV3Loader(root)
    assert loader.validation.status == "PASS"
    spans = loader.retrieve_spans("Tên xã là gì?")
    assert spans[0]["source_type"] == ASR_EXTERNAL_V3_SOURCE_TYPE
    assert spans[0]["chunk_id"] == "L01_V001_chunk_0000"
    assert spans[0]["provenance"]["transcription_exact_revision"] is None
    assert "frame_id" not in spans[0]
    mapped = loader.map_span_to_frame(spans[0], lambda video_id, seconds: 99)
    assert mapped["frame_id"] == 99
    assert mapped["frame_mapping_source"] == "INJECTED_CANONICAL_FRAMEMAP"


def test_source_type_is_explicit_and_never_inferred_or_aliased(tmp_path: Path) -> None:
    root = tmp_path / "external"
    _fixture(root)
    assert isinstance(load_asr_evidence(root, ASR_EXTERNAL_V3_SOURCE_TYPE), ASRExternalV3Loader)
    with pytest.raises(RuntimeError, match="ASR_SOURCE_TYPE_UNSUPPORTED"):
        load_asr_evidence(root, "AUTO")
    with pytest.raises(RuntimeError, match="ASR_V12_REQUIRED_FILES_MISSING"):
        load_asr_evidence(root, ASR_V12_SOURCE_TYPE)


def test_external_loader_rejects_whisper_v12_alias(tmp_path: Path) -> None:
    root = tmp_path / "external"
    _fixture(root)
    provenance_path = root / "asr_external_v3_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["is_whisper_v12_exact"] = True
    _write_json(provenance_path, provenance)
    with pytest.raises(RuntimeError, match="WHISPER_ALIAS_FORBIDDEN"):
        ASRExternalV3Loader(root)
