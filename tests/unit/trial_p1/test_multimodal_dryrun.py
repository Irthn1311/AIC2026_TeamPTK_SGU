from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_eg.trial_p1 import multimodal_dryrun as dryrun
from triage_eg.trial_p1.multimodal_dryrun import (
    CanonicalBTCFrameMapper,
    build_asr_candidate_evidence,
    build_external_parquet_evidence,
    build_qwen_context,
    build_xclip_event_evidence,
    candidate_comparison,
    normalize_trial_plans,
    prioritize_qa_sufficient,
    qa_evidence_summary,
    run_causal_graph_fixture,
    select_novel_graph_revision,
    select_qwen_grounding_rows,
    validate_trial_contract,
    validate_trial_runtime_assets,
    write_blocked_artifacts,
)


def _queries() -> list[dict]:
    rows = []
    for value in range(1, 19):
        rows.append(
            {
                "query_id": f"query-p1-{value}-kis",
                "task": "KIS",
                "language": "vi",
                "query": f"kis {value}",
            }
        )
    for value in (15, 19, 22):
        rows.append(
            {
                "query_id": f"query-p1-{value}-qa",
                "task": "QA",
                "language": "vi",
                "query": f"qa {value}",
            }
        )
    rows.extend(
        {
            "query_id": query_id,
            "task": "TRAKE",
            "language": "vi",
            "query": "four events",
            "event_count": 4,
            "event_descriptions": ["one", "two", "three", "four"],
            "raw_event_labels": labels,
        }
        for query_id, labels in (
            ("query-p1-4-trake", ["E1", "E2", "E3", "E4"]),
            ("query-p1-18-trake", ["E1", "E2", "E2", "E4"]),
            ("query-p1-26-trake", ["E1", "E2", "E3", "E4"]),
        )
    )
    # IDs intentionally overlap numerically but remain unique by task.
    return rows


def _rows(queries: list[dict], source: str = "b0") -> list[dict]:
    output = []
    for query in queries:
        for rank in range(1, 101):
            common = {
                "query_id": query["query_id"],
                "rank": rank,
                "video_id": f"L{21 + rank % 10:02d}_V{rank % 100:03d}",
                "source": source,
            }
            if query["task"] == "TRAKE":
                common["frame_ids"] = [rank, rank + 1, rank + 2, rank + 3]
            else:
                common["frame_id"] = rank
            if query["task"] == "QA":
                common["answer"] = "không đủ bằng chứng"
                common["evidence_sufficient"] = False
            output.append(common)
    return output


def test_trial_contract_preserves_duplicated_raw_e2() -> None:
    assert validate_trial_contract(_queries())["status"] == "PASS"
    broken = _queries()
    next(row for row in broken if row["query_id"] == "query-p1-18-trake")["raw_event_labels"] = [
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    with pytest.raises(RuntimeError, match="DUPLICATED_RAW_E2"):
        validate_trial_contract(broken)


def test_normalize_frozen_plans_keeps_four_ordinal_events() -> None:
    plans = []
    for query in _queries():
        plan = {
            "query_id": query["query_id"],
            "task": query["task"],
            "raw_text": query["query"],
            "team_query": {key: value for key, value in query.items() if key != "raw_event_labels"},
        }
        if query["task"] == "TRAKE":
            plan["events"] = [
                {
                    "description": description,
                    "raw_event_label": query["raw_event_labels"][index],
                }
                for index, description in enumerate(query["event_descriptions"])
            ]
        plans.append(plan)
    normalized = normalize_trial_plans(plans)
    p18 = next(row for row in normalized if row["query_id"] == "query-p1-18-trake")
    assert p18["event_count"] == 4
    assert p18["raw_event_labels"] == ["E1", "E2", "E2", "E4"]


def test_qa_sufficient_rows_are_ranked_before_unsupported_and_generic_is_rejected() -> None:
    baseline = _rows([_queries()[18]])
    qwen = [
        {
            "query_id": "query-p1-15-qa",
            "video_id": "L30_V072",
            "frame_id": 20,
            "answer": "ghế",
            "evidence_sufficient": True,
            "rank": 1,
        },
        {
            "query_id": "query-p1-15-qa",
            "video_id": "L25_V001",
            "frame_id": 21,
            "answer": "FANA",
            "evidence_sufficient": True,
            "rank": 2,
        },
    ]
    ranked = prioritize_qa_sufficient(baseline, qwen, baseline)
    assert ranked[0]["answer"] == "FANA"
    assert ranked[0]["evidence_sufficient"] is True
    assert all(row.get("answer") != "ghế" for row in ranked)


def test_qa_hard_gate_blocks_zero_sufficient_candidates() -> None:
    queries = _queries()
    rows = _rows(queries)
    candidates = {name: rows for name in ("TRIAGEEG_M0_FULL", "TRIAGEEG_M1_FULL", "TRIAGEEG_SAFE")}
    assert qa_evidence_summary(queries, candidates)["hard_gate"] == "QA_BLOCK_SUBMISSION_2"


def test_comparison_has_all_24_queries() -> None:
    queries = _queries()
    baseline = _rows(queries)
    candidates = {
        name: _rows(queries, name)
        for name in ("TRIAGEEG_M0_FULL", "TRIAGEEG_M1_FULL", "TRIAGEEG_SAFE")
    }
    comparison = candidate_comparison(queries, baseline, candidates)
    assert len(comparison["rows"]) == 24
    assert comparison["top1_changed_vs_bcf1"] == 0


def test_blocked_artifacts_are_explicit_and_do_not_fabricate_candidate_zips(tmp_path: Path) -> None:
    output = write_blocked_artifacts(
        tmp_path,
        ["XCLIP_EVIDENCE_MISSING", "QWEN_ASSET_MISSING"],
        {"gt_opened": False},
    )
    decision = (output / "SUBMISSION_2_DECISION.md").read_text(encoding="utf-8")
    assert "DO_NOT_SUBMIT_2_YET" in decision
    assert "XCLIP_EVIDENCE_MISSING" in decision
    assert not list(output.glob("TRIAGEEG_*.zip"))
    assert (
        json.loads((output / "qa_evidence_summary.json").read_text())["status"]
        == "NOT_RUN_FAIL_CLOSED"
    )


def test_asr_evidence_can_introduce_new_video_through_injected_canonical_mapper() -> None:
    query = _queries()[0]

    class Loader:
        @staticmethod
        def retrieve_spans(text: str, max_spans: int) -> list[dict]:
            assert text and max_spans == 200
            return [
                {
                    "video_id": "L99_V999",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "text": "strong topic",
                }
            ]

        @staticmethod
        def map_span_to_frame(span, mapper):
            return {**span, "frame_id": mapper(span["video_id"], 1.5)}

    evidence = build_asr_candidate_evidence(
        [query], Loader(), canonical_mapper=lambda video_id, seconds: 777
    )
    row = evidence[query["query_id"]][0]
    assert row["video_id"] == "L99_V999"
    assert row["frame_id"] == 777
    assert row["asr_span"]["frame_id"] == 777


def test_canonical_btc_mapper_uses_nearest_declared_pts_time(tmp_path: Path) -> None:
    mapping = tmp_path / "L01_V001.csv"
    mapping.write_text(
        "n,pts_time,fps,frame_idx\n1,1.0,25,25\n2,2.0,25,50\n", encoding="utf-8"
    )
    mapper = CanonicalBTCFrameMapper(
        [
            {
                "video_id": "L01_V001",
                "mapping_available": True,
                "mapping_path": str(mapping),
                "total_frames": 100,
            }
        ]
    )
    assert mapper("L01_V001", 1.8) == 50


def test_runtime_asset_identity_gate_hashes_all_frozen_weights(
    tmp_path: Path, monkeypatch
) -> None:
    bcf1 = tmp_path / "bcf1.jsonl"
    bcf1.write_bytes(b"bcf1\n")
    xclip = tmp_path / "xasset" / "xclip"
    xclip.mkdir(parents=True)
    (xclip / "model.safetensors").write_bytes(b"xclip")
    manifests = xclip.parent / "manifests"
    manifests.mkdir()
    (manifests / "asset_manifest.json").write_text(
        json.dumps({"model_id": dryrun.XCLIP_MODEL_ID, "exact_revision": dryrun.XCLIP_REVISION})
    )
    e5 = tmp_path / "e5"
    e5.mkdir()
    (e5 / "model.onnx").write_bytes(b"e5")
    (e5 / "asset_manifest.json").write_text(
        json.dumps(
            {
                "model_id": dryrun.E5_MODEL_ID,
                "exact_revision": dryrun.E5_REVISION,
                "source_index_exact_encoder_revision_known": False,
            }
        )
    )
    qwen = tmp_path / "qwen"
    (qwen / ".cache/huggingface/download").mkdir(parents=True)
    (qwen / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    qwen_hashes = {}
    for index in (1, 2):
        name = f"model-{index:05d}-of-00002.safetensors"
        (qwen / name).write_bytes(f"qwen-{index}".encode())
        digest = dryrun.sha256_file(qwen / name)
        qwen_hashes[name] = digest
        (qwen / ".cache/huggingface/download" / f"{name}.metadata").write_text(
            f"{dryrun.QWEN_REVISION}\n{digest}\n"
        )
    monkeypatch.setattr(dryrun, "TRIAL_BCF1_F1_SHA256", dryrun.sha256_file(bcf1))
    monkeypatch.setattr(
        dryrun,
        "XCLIP_WEIGHT_SHA256",
        dryrun.sha256_file(xclip / "model.safetensors"),
    )
    monkeypatch.setattr(dryrun, "E5_MODEL_SHA256", dryrun.sha256_file(e5 / "model.onnx"))
    monkeypatch.setattr(dryrun, "QWEN_WEIGHT_SHA256", qwen_hashes)
    report = validate_trial_runtime_assets(
        bcf1_predictions=bcf1, xclip_root=xclip, e5_root=e5, qwen_root=qwen
    )
    assert report["status"] == "PASS"
    assert len(report["qwen"]["weights"]) == 2


def test_qwen_grounding_round_robin_prevents_ocr_starvation() -> None:
    ocr = [
        {"video_id": f"L01_V{rank:03d}", "frame_id": rank, "rank": rank}
        for rank in range(1, 101)
    ]
    asr = [{"video_id": "L02_V001", "frame_id": 20, "rank": 1}]
    baseline = [{"video_id": "L03_V001", "frame_id": 30, "rank": 1}]
    selected = select_qwen_grounding_rows(
        ocr_rows=ocr, asr_rows=asr, baseline_rows=baseline
    )
    assert len(selected) == 20
    assert {row["grounding_source"] for row in selected} == {"ocr", "asr", "bcf1"}


def test_qwen_context_is_nearest_first_and_persists_provenance() -> None:
    text, rows = build_qwen_context(
        {"video_id": "L01_V001", "frame_id": 100},
        ocr_rows=[
            {
                "video_id": "L01_V001",
                "frame_id": 105,
                "rank": 2,
                "text": "near title",
                "source": "ocr",
                "source_confidence": 0.9,
            },
            {"video_id": "L01_V001", "frame_id": 500, "rank": 1, "text": "far title"},
        ],
        asr_rows=[],
    )
    assert rows[0]["text"] == "near title"
    assert rows[0]["confidence"] == 0.9
    assert text.startswith("[ocr] near title")


def test_external_ocr_evidence_is_evidence_only(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "ocr.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video_id": "L21_V001",
                    "frame_idx": 10,
                    "corrected_text": "Japan Fest",
                    "combined_text": "Japan Fest",
                    "mean_confidence": 0.9,
                }
            ]
        ),
        path,
    )
    query = {**_queries()[0], "query": "Japan Fest"}
    rows = build_external_parquet_evidence([query], path, "ocr")[query["query_id"]]
    assert rows[0]["text"] == "Japan Fest"
    assert rows[0]["evidence_only"] is True
    assert "answer" not in rows[0]


def test_xclip_is_scored_for_every_ordinal_event() -> None:
    query = next(row for row in _queries() if row["query_id"] == "query-p1-18-trake")
    baseline = _rows([query])
    for rank, row in enumerate(baseline, 1):
        row["frame_ids"] = [100 + rank, 200 + rank, 300 + rank, 400 + rank]

    def score(text: str, video_id: str, frame_id: int) -> dict:
        return {
            "score": float(frame_id),
            "finite": True,
            "center_frame_id": frame_id,
            "text": text,
            "video_id": video_id,
        }

    rows = build_xclip_event_evidence([query], baseline, score, candidates_per_event=3)[
        query["query_id"]
    ]
    assert {row["event_index"] for row in rows} == {0, 1, 2, 3}
    assert len(rows) == 36
    assert {row["neighbor_offset"] for row in rows} == {-48, 0, 48}


def test_graph_revision_requires_coordinate_novel_to_complete_m0_pool() -> None:
    query = next(row for row in _queries() if row["task"] == "TRAKE")
    baseline = _rows([query])
    duplicate = {
        "query_id": query["query_id"],
        "event_index": 0,
        "video_id": baseline[0]["video_id"],
        "frame_id": baseline[0]["frame_ids"][0],
        "rank": 1,
    }
    initial_action = [
        {**duplicate, "frame_id": duplicate["frame_id"] + rank, "rank": rank}
        for rank in range(1, 21)
    ]
    novel = {**duplicate, "frame_id": duplicate["frame_id"] + 48, "rank": 21}
    selected = select_novel_graph_revision(
        query, 0, baseline_rows=baseline, action_rows=[*initial_action, novel]
    )
    assert selected[0]["frame_id"] == novel["frame_id"]
    assert selected[0]["source"] == "xclip_graph_revision"


def test_causal_graph_fixture_is_executed_and_changes_prediction_content() -> None:
    fixture = run_causal_graph_fixture()
    assert fixture["status"] == "PASS"
    assert fixture["content_changed"] is True
    assert fixture["graph"]["revision"]["evidence_added"] > 0
