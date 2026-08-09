from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from tests.integration.test_stage1c_qualitative_eval import (
    MockVerifiedTextEncoder,
)
from tests.integration.test_stage1c_qualitative_eval import (
    config as stage1c_config,
)
from triage_eg.retrieval.stage1c import run_stage1c
from triage_eg.retrieval.stage1d import (
    ReviewConfig,
    Stage1DConfig,
    create_stage1d_bundle,
    patch_blinded_review_visuals,
    run_stage1d,
)
from triage_eg.retrieval.stage1d.contracts import TRANSLATOR_MODEL_ID, TRANSLATOR_REVISION
from triage_eg.retrieval.stage1d.inputs import TRANSLATOR_REQUIRED


def translator_asset(root: Path) -> Path:
    for relative in TRANSLATOR_REQUIRED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "manifests/MODEL_REVISION.txt":
            path.write_text(TRANSLATOR_REVISION)
        elif not relative.startswith("manifests/"):
            path.write_bytes(relative.encode())
    hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in TRANSLATOR_REQUIRED
        if relative.startswith("model/")
    }
    (root / "manifests/asset_manifest.json").write_text(
        json.dumps(
            {
                "model_id": TRANSLATOR_MODEL_ID,
                "exact_revision": TRANSLATOR_REVISION,
                "architecture": "MarianMT / transformer-align",
                "runtime_model_path": "model",
                "internet_required_at_runtime": False,
                "file_hashes": hashes,
            }
        )
    )
    (root / "manifests/file_inventory.jsonl").write_text(
        "".join(
            json.dumps({"path": path, "sha256": digest}) + "\n" for path, digest in hashes.items()
        )
    )
    return root


class MockTranslator:
    def __init__(self) -> None:
        self.device = "cpu"
        self.calls: list[list[str]] = []
        self.closed = False

    def load(self):
        return self

    def translate(self, texts: list[str]):
        self.calls.append(list(texts))
        return [
            {
                "translated_text_raw": "object english",
                "translated_text_for_clip": "object english",
                "translation_latency_ms": 1.0,
            }
            for _ in texts
        ]

    def runtime_manifest(self):
        return {
            "status": "LOADED",
            "device": "cpu",
            "local_files_only": True,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        }

    def close(self):
        self.closed = True


def test_synthetic_stage1d_preserves_frozen_baselines_and_builds_audit_bundle(
    tmp_path: Path,
) -> None:
    stage1c_settings = stage1c_config(tmp_path)
    stage1c_result = run_stage1c(
        stage1c_settings,
        adapter_factory=lambda _: MockVerifiedTextEncoder(),
    )
    frozen_en = stage1c_result.output_root / "queries/obj_en/ranked_frames.jsonl"
    frozen_vi = stage1c_result.output_root / "queries/obj_vi/ranked_frames.jsonl"
    frozen_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (frozen_en, frozen_vi)
    }
    translator = MockTranslator()
    settings = Stage1DConfig(
        repo_root=Path.cwd(),
        dataset_root=stage1c_settings.dataset_root,
        stage0_root=stage1c_settings.stage0_root,
        stage1_root=stage1c_settings.stage1_root,
        stage1b_root=stage1c_settings.stage1b_root,
        stage1c_root=stage1c_result.output_root,
        clip_asset_root=stage1c_settings.encoder_asset_root,
        translator_asset_root=translator_asset(tmp_path / "translator"),
        output_root=tmp_path / "stage1d",
        review=ReviewConfig(top_k=5, seed=2026),
        overwrite=True,
        expected_query_count=2,
        expected_pair_count=1,
        build_git_commit="stage1d-synthetic",
    )
    result = run_stage1d(
        settings,
        translator_factory=lambda *_: translator,
        clip_adapter_factory=lambda _: MockVerifiedTextEncoder(),
    )
    assert translator.calls == [["đồ vật tiếng Việt"]]
    assert translator.closed
    assert result.summary["execution_status"] in {"COMPLETE", "COMPLETE_WITH_WARNINGS"}
    assert result.summary["stage1c_frozen_baseline"]["baseline_regenerated"] is False
    assert result.summary["retrieval"]["baseline_retrieval_source"] == "FROZEN_STAGE1C_ARTIFACTS"
    assert result.summary["retrieval"]["translated_queries_completed"] == 1
    assert result.summary["human_review"]["judgments_expected"] == 15
    assert result.summary["language_bridge_quality_status"] == "NOT_REVIEWED"
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in frozen_hashes.items()
    )

    translated = result.output_root / "translated_queries/obj"
    frames = [
        json.loads(line) for line in (translated / "ranked_frames.jsonl").read_text().splitlines()
    ]
    assert len(frames) == 50
    assert [item["rank"] for item in frames] == list(range(1, 51))
    with (translated / "kis_candidates.csv").open(newline="") as stream:
        kis = list(csv.DictReader(stream))
    assert len(kis) == 100
    assert kis[0]["frame_id"] == str(frames[0]["original_frame_idx"])
    assert kis[1]["frame_id"] != str(frames[1]["n"])
    assert (translated / "contact_sheet_top20.jpg").is_file()
    assert (result.output_root / "comparisons/obj/comparison_top5.jpg").is_file()

    comparison = json.loads(
        (result.output_root / "comparisons/pair_comparisons.jsonl").read_text().splitlines()[0]
    )
    assert set(comparison["structural_summaries"]) == {
        "EN_DIRECT",
        "VI_DIRECT",
        "VI_TRANSLATED_EN",
    }
    assert comparison["human_review_status"] == "NOT_REVIEWED"
    manifest = json.loads((result.output_root / "run_manifest.json").read_text())
    assert manifest["no_stage0_rerun"]
    assert manifest["no_stage1_rebuild"]
    assert manifest["stage1c_baseline_regenerated"] is False
    assert manifest["no_model_download"]
    assert manifest["no_reranking"]

    patch = patch_blinded_review_visuals(
        result.output_root, stage1c_settings.dataset_root
    )
    assert patch["blinded_sheet_count"] == 1
    assert patch["review_rows"] == 15
    assert (
        result.output_root / "review/blinded_sheets/obj_top5.jpg"
    ).is_file()
    archive = create_stage1d_bundle(result.output_root, tmp_path / "stage1d_bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "translated_queries/obj/ranked_frames.jsonl" in names
    assert "comparisons/obj/comparison_top5.jpg" in names
    assert "review/review_template_blinded.csv" in names
    assert "review/blinded_sheet_index.csv" in names
    assert "review/blinded_sheets/obj_top5.jpg" in names
    assert not any(name.endswith((".bin", ".npy", ".mp4")) for name in names)


def test_stage1d_source_contains_no_out_of_scope_pipeline_features() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage1d").glob("*.py")
    )
    for forbidden in (
        "build_index(",
        "run_stage1b(",
        "run_stage1c(",
        "VideoCapture(",
        "translate.googleapis.com",
        "requests.post(",
        "OpenClipMultimodalEncoder(",
    ):
        assert forbidden not in sources
