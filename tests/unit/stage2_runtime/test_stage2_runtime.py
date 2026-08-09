from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.numpy_index import NumPyFlatCosineIndex
from triage_eg.retrieval.stage1b.contracts import CandidateContract
from triage_eg.retrieval.stage1b.writers import write_json, write_jsonl
from triage_eg.retrieval.stage2 import (
    OperationalRetrievalRuntime,
    QueryRequest,
    Stage2RuntimeConfig,
    Stage2RuntimeError,
    create_stage2_report_bundle,
    grouped_video_view,
    resolve_language,
)


class Catalog:
    rows = (
        ("V1", 1, 100),
        ("V1", 2, 100),
        ("V2", 3, 300),
        ("V3", 4, 400),
    )

    def map_row(self, row: int) -> dict[str, object]:
        video, n, frame = self.rows[row]
        return {
            "global_row": row,
            "video_id": video,
            "n": n,
            "original_frame_idx": frame,
            "keyframe_relative_path": f"{video}/{n:03d}.jpg",
        }


class Encoder:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = 0

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        output = np.zeros((len(texts), 512), dtype=np.float32)
        for index, text in enumerate(texts):
            output[index, 1 if "car" in text.lower() else 0] = 1.0
        return output

    def close(self) -> None:
        self.closed += 1


class Translator:
    def __init__(self) -> None:
        self.loads = 0
        self.calls = 0
        self.closed = 0

    def load(self) -> None:
        self.loads += 1

    def translate(self, texts: list[str]) -> list[dict[str, object]]:
        self.calls += 1
        return [
            {
                "translated_text_for_clip": "A red car." if "ô tô" in text else "A person cooking.",
                "translation_latency_ms": 1.0,
            }
            for text in texts
        ]

    def close(self) -> None:
        self.closed += 1


def _roots(tmp_path: Path) -> Stage2RuntimeConfig:
    stage1, stage1b, stage1e = tmp_path / "s1", tmp_path / "s1b", tmp_path / "s1e"
    write_json(
        stage1 / "stage1_summary.json",
        {"index_status": "COMPLETE", "index_fingerprint": "fingerprint"},
    )
    write_json(stage1 / "index/index_manifest.json", {"status": "COMPLETE"})
    write_json(
        stage1b / "stage1b_summary.json",
        {
            "evaluation_status": "COMPLETE",
            "stage1_index_fingerprint": "fingerprint",
        },
    )
    selected = {
        "candidate_id": "openai_clip_vit_b32_openai_official",
        "selected_candidate_id": "openai_clip_vit_b32_openai_official",
        "enabled": True,
        "implementation": "openai_clip",
        "architecture": "ViT-B/32",
        "pretrained": "openai",
        "checkpoint_path": "mock.pt",
        "source_root": "mock",
        "compatibility_status": "VERIFIED",
        "checkpoint_sha256": "clip-sha",
    }
    write_json(stage1b / "encoder/selected_encoder_contract.json", selected)
    write_json(
        stage1b / "encoder/runtime_adapter_manifest.json",
        {"model_space_status": "MODEL_SPACE_VERIFIED"},
    )
    contract = {
        "english_path": {
            "mode": "DIRECT",
            "text_encoder": "openai_clip_vit_b32_openai_official",
        },
        "vietnamese_path": {
            "mode": "TRANSLATE_TO_ENGLISH_THEN_CLIP",
            "translator": {
                "model_id": "Helsinki-NLP/opus-mt-vi-en",
                "exact_revision": "c8d2853e77f5fae31124d993e0b35176b1c8914e",
            },
            "text_encoder": "openai_clip_vit_b32_openai_official",
        },
        "stage1_index_fingerprint": "fingerprint",
        "clip_compatibility": "VERIFIED",
        "model_space_status": "MODEL_SPACE_VERIFIED",
        "language_path_status": "FROZEN_FOR_INTERNAL_BASELINE",
    }
    write_json(stage1e / "language_path_contract.json", contract)
    write_json(
        stage1e / "stage1e_summary.json",
        {"stage1e_execution": "COMPLETE", "stage2_readiness": "READY"},
    )
    write_jsonl(
        stage1e / "issues.jsonl",
        [
            {"code": "SEMANTIC_RETRIEVAL_FAILURE_AFTER_TRANSLATION", "pair_id": "difficult_01"},
            {
                "code": "LANGUAGE_BRIDGE_INSUFFICIENT_FOR_SEMANTIC_FAILURE",
                "pair_id": "obj_01",
            },
        ],
    )
    return Stage2RuntimeConfig(
        stage1,
        stage1b,
        stage1e,
        tmp_path / "clip",
        tmp_path / "opus",
        tmp_path / "output",
        Path("configs/retrieval/stage1d_translation_ablation.yaml"),
    )


def _runtime(tmp_path: Path) -> tuple[OperationalRetrievalRuntime, Encoder, Translator]:
    config = _roots(tmp_path)
    vectors = np.zeros((4, 512), dtype=np.float32)
    vectors[0, 0], vectors[1, 0], vectors[2, 1], vectors[3, 2] = 1, 1, 1, 1
    index = NumPyFlatCosineIndex()
    index.build(vectors, ["0", "1", "2", "3"])
    encoder, translator = Encoder(), Translator()
    selected = json.loads(
        (config.stage1b_root / "encoder/selected_encoder_contract.json").read_text()
    )
    candidate = CandidateContract.from_dict(selected)
    runtime = OperationalRetrievalRuntime(
        config,
        backend_loader=lambda _: (index, Catalog()),
        clip_preparer=lambda *_: (candidate, {"checkpoint_sha256": "clip-sha"}),
        translator_validator=lambda _: {
            "status": "VALID",
            "model_root": tmp_path / "model",
            "model_id": "Helsinki-NLP/opus-mt-vi-en",
            "exact_revision": "c8d2853e77f5fae31124d993e0b35176b1c8914e",
            "runtime_files": [],
        },
        encoder_factory=lambda _: encoder,
        translator_factory=lambda *_: translator,
        dependency_probe=lambda: {"offline": True},
    )
    return runtime, encoder, translator


@pytest.mark.parametrize(
    ("language", "text", "resolved", "path"),
    [
        ("en", "một người", "en", "DIRECT_CLIP"),
        ("vi", "an English sentence", "vi", "VI_TO_EN_THEN_CLIP"),
        ("auto", "một người đang nấu ăn", "vi", "VI_TO_EN_THEN_CLIP"),
        ("auto", "a red car", "en", "DIRECT_CLIP"),
    ],
)
def test_language_routes(language: str, text: str, resolved: str, path: str) -> None:
    value = resolve_language(QueryRequest("q", text, language, 5))
    assert value.resolved_language == resolved
    assert value.language_path == path
    if language != "auto":
        assert value.resolution_basis == "EXPLICIT"


@pytest.mark.parametrize("text", ["mot nguoi dang nau an", "hello", "123"])
def test_auto_ambiguous_fails(text: str) -> None:
    with pytest.raises(Stage2RuntimeError, match="LANGUAGE_AMBIGUOUS"):
        resolve_language(QueryRequest("q", text, "auto", 5))


def test_invalid_requests_fail() -> None:
    with pytest.raises(ValueError):
        QueryRequest("q", "   ", "en", 5)
    with pytest.raises(Stage2RuntimeError, match="LANGUAGE_UNSUPPORTED"):
        QueryRequest("q", "text", "fr", 5)
    with pytest.raises(ValueError):
        QueryRequest("q", "text", "en", 101)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("not_frozen", "STAGE1E_LANGUAGE_PATH_NOT_FROZEN"),
        ("fingerprint", "STAGE1_INDEX_FINGERPRINT_MISMATCH"),
        ("translator", "TRANSLATOR_REVISION_MISMATCH"),
        ("candidate", "STAGE1E_CONTRACT_INVALID"),
        ("clip", "STAGE1B_ENCODER_NOT_VERIFIED"),
        ("space", "STAGE1B_MODEL_SPACE_NOT_VERIFIED"),
    ],
)
def test_contract_preflight_rejects(tmp_path: Path, mutation: str, code: str) -> None:
    config = _roots(tmp_path)
    if mutation in {"not_frozen", "fingerprint", "translator", "candidate"}:
        path = config.stage1e_root / "language_path_contract.json"
    elif mutation == "clip":
        path = config.stage1b_root / "encoder/selected_encoder_contract.json"
    else:
        path = config.stage1b_root / "encoder/runtime_adapter_manifest.json"
    value = json.loads(path.read_text())
    if mutation == "not_frozen":
        value["language_path_status"] = "DRAFT"
    elif mutation == "fingerprint":
        value["stage1_index_fingerprint"] = "wrong"
    elif mutation == "translator":
        value["vietnamese_path"]["translator"]["exact_revision"] = "wrong"
    elif mutation == "candidate":
        value["english_path"]["text_encoder"] = "wrong"
    elif mutation == "clip":
        value["compatibility_status"] = "BLOCKED"
    else:
        value["model_space_status"] = "UNVERIFIED"
    write_json(path, value)
    runtime, _, _ = _runtime_from_config(config, tmp_path)
    with pytest.raises(Stage2RuntimeError, match=code):
        runtime.load()


def _runtime_from_config(
    config: Stage2RuntimeConfig, tmp_path: Path
) -> tuple[OperationalRetrievalRuntime, Encoder, Translator]:
    encoder, translator = Encoder(), Translator()
    candidate = CandidateContract.from_dict(
        json.loads((config.stage1b_root / "encoder/selected_encoder_contract.json").read_text())
    )
    index = NumPyFlatCosineIndex()
    matrix = np.eye(4, 512, dtype=np.float32)
    index.build(matrix, ["0", "1", "2", "3"])
    runtime = OperationalRetrievalRuntime(
        config,
        backend_loader=lambda _: (index, Catalog()),
        clip_preparer=lambda *_: (candidate, {}),
        translator_validator=lambda _: {
            "status": "VALID",
            "model_root": tmp_path / "model",
            "model_id": "Helsinki-NLP/opus-mt-vi-en",
            "exact_revision": "c8d2853e77f5fae31124d993e0b35176b1c8914e",
            "runtime_files": [],
        },
        encoder_factory=lambda _: encoder,
        translator_factory=lambda *_: translator,
        dependency_probe=lambda: {},
    )
    return runtime, encoder, translator


def test_runtime_en_vi_batch_reuses_assets_and_preserves_mapping(tmp_path: Path) -> None:
    runtime, encoder, translator = _runtime(tmp_path)
    runtime.load()
    assert translator.loads == 0
    en = runtime.search_one(QueryRequest("en", "a person cooking in a kitchen", "en", 4))
    assert translator.calls == 0
    assert len(en.ranked_frames) == 4
    assert [item["global_row"] for item in en.ranked_frames[:2]] == [0, 1]
    assert [item["rank"] for item in en.ranked_frames] == [1, 2, 3, 4]
    assert en.ranked_videos[0]["best_frame_rank"] == 1
    assert en.ranked_videos[0]["frames_in_raw_results"] == 2
    batch = runtime.search_many(
        [
            QueryRequest("vi", "một chiếc ô tô màu đỏ", "vi", 3),
            QueryRequest("en2", "a red car", "en", 2),
        ]
    )
    assert [item.query_id for item in batch] == ["vi", "en2"]
    assert translator.loads == 1 and translator.calls == 1
    assert encoder.calls == 2
    assert batch[0].encoding["translated_text"] == "A red car."
    assert batch[1].encoding["translation_applied"] is False
    with (en.output_root / "kis_candidates.csv").open(newline="") as stream:
        kis = list(csv.DictReader(stream))
    assert kis[0] == {"video_id": "V1", "frame_id": "100"}
    assert len(en.ranked_frames) == 4
    manifest = json.loads((runtime.config.output_root / "runtime_manifest.json").read_text())
    assert manifest["stage1_index_fingerprint"] == "fingerprint"
    assert len(manifest["known_failure_modes"]) == 2
    runtime.close()
    assert encoder.closed == 1 and translator.closed == 1


def test_grouped_view_does_not_mutate_raw() -> None:
    frames = [
        {
            "rank": 1,
            "video_id": "B",
            "global_row": 1,
            "n": 2,
            "original_frame_idx": 8,
            "score": 1.0,
        },
        {
            "rank": 2,
            "video_id": "A",
            "global_row": 2,
            "n": 3,
            "original_frame_idx": 9,
            "score": 1.0,
        },
        {
            "rank": 3,
            "video_id": "B",
            "global_row": 3,
            "n": 4,
            "original_frame_idx": 10,
            "score": 0.5,
        },
    ]
    original = json.loads(json.dumps(frames))
    videos = grouped_video_view(frames)
    assert frames == original
    assert [item["video_id"] for item in videos] == ["B", "A"]
    assert videos[0]["frames_in_raw_results"] == 2


def test_report_bundle_is_small_and_offline(tmp_path: Path) -> None:
    runtime, _, _ = _runtime(tmp_path)
    runtime.load()
    runtime.search_one(QueryRequest("q", "a red car", "en", 2))
    archive = create_stage2_report_bundle(runtime.config.output_root, tmp_path / "reports.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "runtime_manifest.json" in names and "q/ranked_frames.jsonl" in names
    assert not any(name.endswith((".npy", ".pt", ".jpg", ".mp4")) for name in names)


def test_stage2_source_has_no_new_models_or_network() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage2").glob("*.py")
    )
    for forbidden in ("requests.", "snapshot_download(", "faiss", "NLLB", "rerank("):
        assert forbidden not in source
