from __future__ import annotations

import json

import numpy as np
import pytest

from system_tai.checkpointing.exporter import CheckpointExporter
from system_tai.common.schemas import CandidateFrame, KISQuery, KISResult
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.retrieval.vector_search import ExactNumpyRetriever
from system_tai.validation.checkpoint_validator import CheckpointValidator
from tests.phase2_helpers import make_store
from tests.test_phase2_retrieval import FakeEncoder


def _candidate(rank: int, frame: int, *, video: str = "video") -> CandidateFrame:
    return CandidateFrame(
        video_id=video,
        frame_id=frame,
        clip_row=rank - 1,
        keyframe_order=rank,
        score=0.9,
        rank=rank,
        source="clip_exact",
        diagnostic_metadata={"test": True},
    )


def test_export_core_mode_and_internal_mode(tmp_path) -> None:
    result = KISResult(query_id="q", ranked_candidates=(_candidate(1, 7),))
    core_path = tmp_path / "core.jsonl"
    summary = CheckpointExporter().export(result, core_path)
    record = json.loads(core_path.read_text(encoding="utf-8"))
    assert record == {"query_id": "q", "rank": 1, "video_id": "video", "frame_id": 7}
    assert summary.record_count == 1
    assert CheckpointValidator().validate(core_path).valid

    internal_path = tmp_path / "internal.jsonl"
    CheckpointExporter().export(result, internal_path, include_internal=True)
    assert "_internal" in json.loads(internal_path.read_text(encoding="utf-8"))


def test_export_rejects_duplicates_and_more_than_100(tmp_path) -> None:
    duplicate = KISResult(
        query_id="q",
        ranked_candidates=(_candidate(1, 7), _candidate(2, 7)),
    )
    with pytest.raises(ValueError, match="duplicate query/video/frame"):
        CheckpointExporter().export(duplicate, tmp_path / "duplicate.jsonl")
    too_many = KISResult(
        query_id="q",
        ranked_candidates=tuple(_candidate(rank, rank) for rank in range(1, 102)),
    )
    with pytest.raises(ValueError, match="maximum 100"):
        CheckpointExporter().export(too_many, tmp_path / "too-many.jsonl")


def test_validator_reports_invalid_rank_duplicate_and_limit(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    records = [
        {"query_id": "q", "rank": 2, "video_id": "v", "frame_id": 1},
        {"query_id": "q", "rank": 2, "video_id": "v", "frame_id": 1},
    ]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    result = CheckpointValidator().validate(path)
    codes = {error.code for error in result.errors}
    assert {"DUPLICATE_RANK", "DUPLICATE_VIDEO_FRAME", "NON_CONTIGUOUS_RANKS"} <= codes

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_text(
        "\n".join(
            json.dumps({"query_id": "q", "rank": rank, "video_id": "v", "frame_id": rank})
            for rank in range(1, 102)
        ),
        encoding="utf-8",
    )
    result = CheckpointValidator().validate(oversized)
    assert "TOO_MANY_RESULTS" in {error.code for error in result.errors}


def test_validator_reports_utf8_json_shape_fields_and_types(tmp_path) -> None:
    invalid_utf8 = tmp_path / "invalid-utf8.jsonl"
    invalid_utf8.write_bytes(b"\xff\xfe")
    assert [
        error.code for error in CheckpointValidator().validate(invalid_utf8).errors
    ] == ["INVALID_UTF8"]

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps(["not", "an", "object"]),
                json.dumps({"query_id": "q"}),
                json.dumps(
                    {
                        "query_id": "",
                        "rank": True,
                        "video_id": 7,
                        "frame_id": -1,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    codes = {error.code for error in CheckpointValidator().validate(mixed).errors}
    assert {
        "INVALID_JSON",
        "NOT_AN_OBJECT",
        "MISSING_FIELD",
        "INVALID_QUERY_ID",
        "INVALID_RANK",
        "INVALID_VIDEO_ID",
        "INVALID_FRAME_ID",
    } <= codes


def test_validator_can_check_registry_coordinates(tmp_path) -> None:
    registry = FeatureStoreRegistry(
        [make_store("v", np.asarray([[1, 0, 0]], dtype=np.float32), [42])]
    )
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        json.dumps({"query_id": "q", "rank": 1, "video_id": "v", "frame_id": 43}),
        encoding="utf-8",
    )
    result = CheckpointValidator().validate(path, registry=registry)
    assert [error.code for error in result.errors] == ["UNKNOWN_FRAME_ID"]


def test_fixture_integration_fake_encoder_to_valid_jsonl(tmp_path) -> None:
    registry = FeatureStoreRegistry(
        [
            make_store(
                "L21_V001",
                np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
                [12345, 23456],
            )
        ]
    )
    result = ExactNumpyRetriever(registry, FakeEncoder(), chunk_size=1).retrieve(
        KISQuery(query_id="q001", text="motorcycle", top_k=100)
    )
    destination = tmp_path / "result.jsonl"
    CheckpointExporter().export(result, destination)
    validation = CheckpointValidator().validate(destination, registry=registry)
    assert validation.valid
    assert json.loads(destination.read_text(encoding="utf-8").splitlines()[0]) == {
        "query_id": "q001",
        "rank": 1,
        "video_id": "L21_V001",
        "frame_id": 12345,
    }
