from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage_eg.retrieval.stage1.builder import Stage1BuildConfig, build_index
from triage_eg.retrieval.stage1.contracts import EncoderContract, SearchConfig
from triage_eg.retrieval.stage1.encoder import (
    load_text_encoder,
    validate_encoder_output,
    write_compatibility_report,
)
from triage_eg.retrieval.stage1.runner import load_query_vector
from triage_eg.retrieval.stage1.stage0_loader import REQUIRED_FILES, load_stage0_bundle
from triage_eg.retrieval.stage1.writers import (
    create_report_bundle,
    create_stage1b_input_bundle,
)


def base_build(tmp_path: Path, **kwargs) -> Stage1BuildConfig:
    values = {
        "stage0_root": tmp_path / "stage0",
        "dataset_root": tmp_path / "data",
        "output_root": tmp_path / "output",
        "expected_rows": 1,
        "expected_videos": 1,
        "self_queries": 1,
    }
    values.update(kwargs)
    return Stage1BuildConfig(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "bad"},
        {"backend": "faiss_flat_ip"},
        {"metric": "bad"},
        {"overwrite": True, "reuse_index": True},
        {"dimension": 0},
        {"search_chunk_rows": 0},
        {"expected_rows": 0},
        {"self_queries": 0},
    ],
)
def test_invalid_build_config(tmp_path: Path, kwargs) -> None:
    with pytest.raises(ValueError):
        base_build(tmp_path, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_id": ""},
        {"query_id": "../escape"},
        {"query_id": "nested/query"},
        {"top_k": 0},
        {"max_predictions": 0},
        {"search_chunk_rows": 0},
        {"metric": "bad"},
        {"video_grouping": "bad"},
    ],
)
def test_invalid_search_config(tmp_path: Path, kwargs) -> None:
    values = {"stage1_root": tmp_path, "query_id": "q"}
    values.update(kwargs)
    with pytest.raises(ValueError):
        SearchConfig(**values)


@pytest.mark.parametrize("missing", REQUIRED_FILES)
def test_missing_required_stage0_artifact(tmp_path: Path, missing: str) -> None:
    for name in REQUIRED_FILES:
        if name == missing:
            continue
        path = tmp_path / name
        path.write_text("{}\n" if path.suffix == ".json" else "", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_stage0_bundle(tmp_path, require_full=False)


def test_valid_encoder_output_is_float32() -> None:
    result = validate_encoder_output(np.ones((1, 512), dtype=np.float16), 1)
    assert result.dtype == np.float32


def test_missing_encoder_asset() -> None:
    with pytest.raises(FileNotFoundError, match="ENCODER_ASSET_NOT_FOUND"):
        load_text_encoder(EncoderContract(implementation=None))


def test_openclip_requires_local_checkpoint_before_dependency_load() -> None:
    contract = EncoderContract(
        implementation="open_clip",
        model_name="ViT-B-32",
        pretrained="openai",
        tokenizer="open_clip_simple",
    )
    with pytest.raises(FileNotFoundError, match="local checkpoint_path required"):
        load_text_encoder(contract)


def test_unverified_override_report_keeps_warning_and_updates_summary(tmp_path: Path) -> None:
    (tmp_path / "stage1_summary.json").write_text(
        json.dumps({"next_stage_readiness": {}}), encoding="utf-8"
    )
    (tmp_path / "stage1_report.md").write_text(
        "# Report\n\n- Text search: BLOCKED\n", encoding="utf-8"
    )
    contract = EncoderContract(compatibility_status="USER_ASSERTED")
    write_compatibility_report(tmp_path, contract, "UNVERIFIED_OVERRIDE", "explicit override")
    report = json.loads((tmp_path / "encoder/compatibility_report.json").read_text())
    summary = json.loads((tmp_path / "stage1_summary.json").read_text())
    assert report["results_trusted"] is False and "warning" in report
    assert summary["encoder_compatibility_status"] == "USER_ASSERTED"
    assert summary["next_stage_readiness"]["text_retrieval"] == (
        "AVAILABLE_WITH_UNVERIFIED_OVERRIDE"
    )
    assert "results are untrusted" in (tmp_path / "stage1_report.md").read_text()


@pytest.mark.parametrize("shape", [(511,), (2, 512), (1, 511)])
def test_query_vector_shape_rejected(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "query.npy"
    np.save(path, np.ones(shape, dtype=np.float32))
    with pytest.raises(ValueError):
        load_query_vector(path)


def test_query_vector_nonfinite_rejected(tmp_path: Path) -> None:
    path = tmp_path / "query.npy"
    value = np.ones(512, dtype=np.float32)
    value[0] = np.nan
    np.save(path, value)
    with pytest.raises(ValueError):
        load_query_vector(path)


def test_output_cannot_write_under_dataset(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError):
        build_index(base_build(tmp_path, dataset_root=data, output_root=data / "stage1"))


def test_report_zip_cannot_be_inside_stage1_root(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        create_report_bundle(tmp_path, tmp_path / "reports.zip")


def test_stage1b_input_zip_is_fail_closed(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError):
        create_stage1b_input_bundle(tmp_path, tmp_path / "stage1b.zip")
    with pytest.raises(FileNotFoundError, match="Missing Stage 1B input artifacts"):
        create_stage1b_input_bundle(tmp_path, tmp_path.parent / "stage1b.zip")
