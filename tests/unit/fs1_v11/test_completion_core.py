from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from triage_eg.fs1_v11.asr import decode_mp4_waveform, lexical_index, transcribe_video
from triage_eg.fs1_v11.events import compile_query_events
from triage_eg.fs1_v11.graph_runtime import Candidate, ExecutableEventGraph, build_graph_chains
from triage_eg.fs1_v11.pipeline import (
    assert_graph_exercised,
    build_completion_arm,
    semantic_content_hash,
)
from triage_eg.fs1_v11.xclip import uniform_indices
from triage_eg.submission.aic26_prelim import create_submission_zip, validate_submission_zip


def trake_query() -> dict:
    return {"query_id": "Q-T", "task": "TRAKE", "query": "walk then sit", "event_count": 2}


def test_mp4_is_decoded_by_ffmpeg_before_whisper(monkeypatch, tmp_path: Path) -> None:
    waveform = np.asarray([0.1, -0.2, 0.3], dtype="<f4")

    def fake_run(command, **kwargs):
        assert command[0] == "ffmpeg" and "f32le" in command and "pipe:1" in command
        return subprocess.CompletedProcess(command, 0, stdout=waveform.tobytes(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    decoded, command = decode_mp4_waveform(tmp_path / "video.mp4")
    assert np.allclose(decoded, waveform) and command[-1] == "pipe:1"


def test_asr_segments_are_monotonic_and_index_nonempty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "triage_eg.fs1_v11.asr.ffprobe_audio_stream",
        lambda path: {"command": ["ffprobe", str(path)], "stream": {"codec_name": "aac"}},
    )
    monkeypatch.setattr(
        "triage_eg.fs1_v11.asr.decode_mp4_waveform",
        lambda path: (np.ones(16000, np.float32), ["ffmpeg", str(path)]),
    )

    def transcriber(*args, **kwargs):
        return {
            "language": "vi",
            "chunks": [{"timestamp": (0.0, 1.0), "text": " Xin chào "}],
        }

    row = transcribe_video("L01_V001", tmp_path / "v.mp4", 25.0, transcriber)
    assert row["segments"][0]["end_frame"] == 25
    assert lexical_index([row])["xin"][0]["video_id"] == "L01_V001"


def test_trake_compiles_exact_n_events_from_prediction_side() -> None:
    query = {"query_id": "Q", "task": "TRAKE", "query": "a then b", "event_count": 2}
    rows = [{"frame_ids": [1, 2]}]
    events = compile_query_events(query, rows)
    assert [event.event_index for event in events] == [0, 1]
    assert all(event.event_count == 2 for event in events)


def test_graph_revision_adds_evidence_and_feeds_t3() -> None:
    events = compile_query_events(trake_query(), [{"frame_ids": [1, 2]}])
    graph = ExecutableEventGraph("Q-T", events)
    graph.add(Candidate(0, "L01_V001", 10, 1, "b0", {"rank": 1}))
    graph.add(Candidate(1, "L01_V001", 20, 1, "b0", {"rank": 1}))
    graph.revise_once(
        "EXPLOIT",
        1,
        [Candidate(1, "L01_V001", 30, 2, "xclip", {"rank": 2})],
    )
    chains = build_graph_chains(graph)
    assert chains and graph.revision["evidence_added"] == 1
    assert any(edge["type"] == "PRECEDES" for edge in graph.edges)
    assert all(row["frame_ids"][0] < row["frame_ids"][1] for row in chains)


def test_graph_activation_gate_rejects_identical_content() -> None:
    rows = [{"query_id": "Q", "video_id": "L01_V001", "frame_ids": [1, 2], "rank": 1}]
    assert semantic_content_hash(rows) == semantic_content_hash([{**rows[0], "metadata": 1}])
    with pytest.raises(RuntimeError, match="GRAPH_NOT_EXERCISED"):
        assert_graph_exercised({"cross": rows}, {"cross": rows}, [])


def test_m0_and_m1_use_temporal_pools_and_m1_revision() -> None:
    query = trake_query()
    baseline = [
        {
            "query_id": "Q-T",
            "video_id": "L01_V001",
            "frame_ids": [10 + rank, 30 + rank],
            "rank": rank,
        }
        for rank in range(1, 6)
    ]
    evidence = {"action": {"Q-T": []}}
    m0, _, m0_diagnostics = build_completion_arm("M0_v11", [query], baseline, evidence, {"action"})

    def revise(query_row, event, action):
        return [
            {
                "query_id": "Q-T",
                "event_index": event.event_index,
                "video_id": "L01_V001",
                "frame_id": 50,
                "rank": 1,
                "source": "xclip_revision",
            }
        ]

    m1, _, m1_diagnostics = build_completion_arm(
        "M1_v11",
        [query],
        baseline,
        evidence,
        {"action"},
        revision_provider=revise,
    )
    assert m0 and m1 and m0_diagnostics[0]["graph"] is None
    assert m1_diagnostics[0]["graph"]["revision"]["evidence_added"] == 1


def test_xclip_uniform_contract() -> None:
    values = uniform_indices(0, 70)
    assert len(values) == 8 and values[0] == 0 and values[-1] == 70


def test_qwen_answer_updates_only_matching_grounded_candidate() -> None:
    query = {"query_id": "Q1", "task": "QA", "query": "red car", "question": "color?"}
    baseline = [
        {"query_id": "Q1", "video_id": "L01_V001", "frame_id": 10, "answer": "x", "rank": 1},
        {"query_id": "Q1", "video_id": "L01_V002", "frame_id": 20, "answer": "y", "rank": 2},
    ]
    evidence = {
        "qwen": {
            "Q1": [
                {
                    "video_id": "L01_V002",
                    "frame_id": 20,
                    "answer": "đỏ",
                    "evidence_sufficient": True,
                }
            ]
        }
    }
    rows, _, _ = build_completion_arm("M0_v11", [query], baseline, evidence, set())
    assert [(row["video_id"], row["answer"]) for row in rows] == [
        ("L01_V001", "x"),
        ("L01_V002", "đỏ"),
    ]


def test_submission_zip_contract_and_root(tmp_path: Path) -> None:
    queries = [
        {"query_id": "K1", "task": "KIS", "query": "x"},
        {"query_id": "Q1", "task": "QA", "query": "x", "question": "y"},
        {"query_id": "T1", "task": "TRAKE", "query": "x", "event_count": 2},
    ]
    predictions = [
        {"query_id": "K1", "video_id": "L01_V001", "frame_id": 1, "rank": 1},
        {"query_id": "Q1", "video_id": "L01_V001", "frame_id": 2, "answer": "đỏ", "rank": 1},
        {"query_id": "T1", "video_id": "L01_V001", "frame_ids": [3, 4], "rank": 1},
    ]
    path = create_submission_zip(queries, predictions, tmp_path / "submission.zip")
    assert validate_submission_zip(path, queries)["status"] == "PASS"
    with zipfile.ZipFile(path) as archive:
        assert all(name.startswith("submission/") for name in archive.namelist())


def test_submission_validator_rejects_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("submission/K1.csv", "video_id,frame_id\n")
    with pytest.raises(ValueError):
        validate_submission_zip(path, [{"query_id": "K1", "task": "KIS", "query": "x"}])
