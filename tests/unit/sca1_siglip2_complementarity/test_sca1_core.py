from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from aic2026_eval.io import sha256_file
from triage_eg.diagnostics.sca1_siglip2_complementarity import (
    SCA1Settings,
    Siglip2ExactBackend,
    Siglip2GroundingPipeline,
    classify_complementarity,
    create_bundle,
    l2_normalize,
    load_preparation_freeze,
    oracle_union_diagnostics,
    paired_unit_deltas,
    validate_offline_asset,
)
from triage_eg.diagnostics.sca1_siglip2_complementarity import assets as assets_module
from triage_eg.diagnostics.sca1_siglip2_complementarity import backend as backend_module
from triage_eg.diagnostics.sca1_siglip2_complementarity import preparation as prep_module
from triage_eg.diagnostics.sca1_siglip2_complementarity.contracts import (
    EXPECTED_A0_PREDICTION_SHA256,
    EXPECTED_OPENAI_CLIP_ID,
    EXPECTED_OPENAI_CLIP_SHA256,
    EXPECTED_STAGE1_FINGERPRINT,
    EXPECTED_TRANSLATOR_ID,
    EXPECTED_TRANSLATOR_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SAFETENSORS_SHA256,
    PREPARATION_MEMBER_HASHES,
    RUNTIME_MODEL_FILES,
    TCA1_ANCHOR_COMMIT,
)
from triage_eg.diagnostics.sca1_siglip2_complementarity.index import (
    catalog_row_fingerprint,
)
from triage_eg.diagnostics.sca1_siglip2_complementarity.runner import (
    REQUIRED_BUNDLE_MEMBERS,
    build_text_identity_rows,
    evaluate_post_gt,
)


def _write_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "asset"
    model, manifests = root / "model", root / "manifests"
    model.mkdir(parents=True)
    manifests.mkdir()
    for index, name in enumerate(RUNTIME_MODEL_FILES):
        (model / name).write_bytes(f"file-{index}".encode())
    expected_model_hash = sha256_file(model / "model.safetensors")
    monkeypatch.setattr(assets_module, "MODEL_SAFETENSORS_SHA256", expected_model_hash)
    inventory = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(model.iterdir())
    ]
    manifest = {
        "asset_manifest_version": assets_module.ASSET_MANIFEST_VERSION,
        "status": "COMPLETE",
        "model_id": MODEL_ID,
        "exact_revision": MODEL_REVISION,
        "runtime_model_path": "model",
        "internet_required_at_runtime": False,
        "image_size": 224,
        "patch_size": 16,
        "embedding_dimension": 768,
        "text_padding": "max_length",
        "text_truncation": True,
        "text_max_length": 64,
        "manual_l2_normalization": True,
        "processor_use_fast": False,
        "files": inventory,
    }
    (manifests / "MODEL_REVISION.txt").write_text(MODEL_REVISION + "\n")
    (manifests / "asset_manifest.json").write_text(json.dumps(manifest))
    return root


def _write_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    decision = {
        "source_tca1_git_commit": TCA1_ANCHOR_COMMIT,
        "frozen_production_baseline": {
            "grounding_policy": "G1_COVERAGE_COARSE",
            "stage1_index_fingerprint": EXPECTED_STAGE1_FINGERPRINT,
            "openai_clip_candidate_id": EXPECTED_OPENAI_CLIP_ID,
            "openai_clip_checkpoint_sha256": EXPECTED_OPENAI_CLIP_SHA256,
            "translator_model": EXPECTED_TRANSLATOR_ID,
            "translator_revision": EXPECTED_TRANSLATOR_REVISION,
            "a0_prediction_sha256": EXPECTED_A0_PREDICTION_SHA256,
        },
    }
    model = {
        "experiment_model": {
            "model_id": MODEL_ID,
            "hf_revision_pin": MODEL_REVISION,
            "known_model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
            "image_size": 224,
            "patch_size": 16,
            "embedding_dim": 768,
            "text_max_length": 64,
            "text_padding": "max_length",
            "text_truncation": True,
            "manual_l2_normalization_required_for_get_features": True,
        },
        "isolation": {
            "frame_bank": "EXACT_SAME_177321_BTC_KEYFRAME_JPG_ROWS",
            "direct_vietnamese_siglip2": False,
            "fusion": False,
            "raw_video_expansion": False,
        },
    }
    members = {
        name: json.dumps(decision if name.endswith("decision_context.json") else model).encode()
        if name.endswith(("decision_context.json", "model_selection.json"))
        else b"frozen\n"
        for name in PREPARATION_MEMBER_HASHES
    }
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in members.items()}
    monkeypatch.setattr(prep_module, "PREPARATION_MEMBER_HASHES", hashes)
    path = tmp_path / "freeze.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    monkeypatch.setattr(prep_module, "FROZEN_PREPARATION_ZIP_SHA256", sha256_file(path))
    return path


def _semantic_unit(unit_id: str, text: str, *, s1=False) -> SimpleNamespace:
    query_id, event_id = unit_id.split(":")
    encoding = {"clip_input_text": text}
    if s1:
        encoding["sca1_clip_input_text"] = text
    return SimpleNamespace(
        unit_id=unit_id,
        query_id=query_id,
        task="TRAKE" if "TR" in query_id else "KIS",
        event_id=event_id,
        source_language="vi",
        source_text=f"nguồn {unit_id}",
        encoding=encoding,
    )


def _audit_rows(*, s1_rescue=False) -> dict:
    rows = []
    for index in range(100):
        hit = index != 0 or s1_rescue
        rows.append(
            {
                "unit_id": f"Q{index:03d}:E1",
                "query_id": f"Q{index:03d}",
                "task": "KIS",
                "event_id": None,
                "t3_pool_has_target": hit,
                "nearest_t3_distance": 0 if hit else 2,
                "primary_failure_reason": "SUCCESS" if hit else "MISS",
                "target_within_video_rank": 1 if hit else None,
                "target_global_rank": 1 if hit else None,
                "correct_video_rank": 1 if hit else 2,
            }
        )
    return {"single_rows": rows, "trake_event_rows": [], "trake_query_rows": []}


def test_01_preparation_freeze_hash_and_contract(tmp_path, monkeypatch) -> None:
    value = load_preparation_freeze(_write_preparation(tmp_path, monkeypatch))
    assert value.validation["status"] == "PASS"
    assert value.validation["a0_prediction_sha256"] == EXPECTED_A0_PREDICTION_SHA256


def test_02_asset_manifest_valid_and_sha_mismatch_fails(tmp_path, monkeypatch) -> None:
    root = _write_asset(tmp_path, monkeypatch)
    assert validate_offline_asset(root)["status"] == "VALID"
    (root / "model/config.json").write_text("tampered")
    with pytest.raises(RuntimeError, match="HASH_OR_SIZE"):
        validate_offline_asset(root)


def test_03_l2_normalization_and_768_shapes() -> None:
    text = l2_normalize(np.ones((3, 768), dtype=np.float32))
    image = l2_normalize(np.eye(768, dtype=np.float32)[:4])
    assert text.shape == (3, 768) and image.shape == (4, 768)
    assert np.allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(image, axis=1), 1.0, atol=1e-6)


def test_04_exact_backend_stable_tie_order(monkeypatch) -> None:
    vectors = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float16)
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1).astype(np.float32)
    monkeypatch.setattr(
        backend_module,
        "validate_siglip2_index",
        lambda *args, **kwargs: {"manifest": {}, "vectors": vectors, "norms": norms},
    )
    backend = Siglip2ExactBackend("ignored")
    scores, rows = backend.search(np.asarray([[1, 0]], dtype=np.float32), 3)
    assert rows.tolist() == [[0, 1, 2]]
    assert scores[0, 0] == scores[0, 1]


def test_04b_catalog_order_and_duplicate_frame_rows_are_fingerprinted() -> None:
    rows = [
        {
            "global_row": 0,
            "video_id": "L01_V001",
            "n": 1,
            "original_frame_idx": 100,
            "keyframe_relative_path": "keyframes/L01_V001/001.jpg",
            "duplicate_frame_idx_group_size": 2,
        },
        {
            "global_row": 1,
            "video_id": "L01_V001",
            "n": 2,
            "original_frame_idx": 100,
            "keyframe_relative_path": "keyframes/L01_V001/002.jpg",
            "duplicate_frame_idx_group_size": 2,
        },
    ]
    catalog = SimpleNamespace(n=np.arange(2), map_row=lambda index: rows[index])
    first = catalog_row_fingerprint(catalog)
    reversed_catalog = SimpleNamespace(
        n=np.arange(2),
        map_row=lambda index: [{**rows[1], "global_row": 0}, {**rows[0], "global_row": 1}][index],
    )
    assert first != catalog_row_fingerprint(reversed_catalog)
    assert rows[0]["original_frame_idx"] == rows[1]["original_frame_idx"]


def test_05_text_identity_exact_100_units() -> None:
    unit_ids = [f"CB-KIS-{index:03d}:E1" for index in range(100)]
    a0 = SimpleNamespace(
        units={
            unit_id: _semantic_unit(unit_id, f"text {index}")
            for index, unit_id in enumerate(unit_ids)
        }
    )
    s1 = SimpleNamespace(
        units={
            unit_id: _semantic_unit(unit_id, f"text {index}", s1=True)
            for index, unit_id in enumerate(unit_ids)
        }
    )
    rows = build_text_identity_rows(a0, s1)
    assert len(rows) == 100 and all(row["a0_text_sha256"] == row["s1_text_sha256"] for row in rows)


def test_06_text_identity_one_byte_change_fails() -> None:
    unit_ids = [f"CB-KIS-{index:03d}:E1" for index in range(100)]
    a0 = SimpleNamespace(units={unit_id: _semantic_unit(unit_id, "same") for unit_id in unit_ids})
    s1 = SimpleNamespace(
        units={unit_id: _semantic_unit(unit_id, "same", s1=True) for unit_id in unit_ids}
    )
    s1.units[unit_ids[-1]].encoding["sca1_clip_input_text"] = "changed"
    with pytest.raises(RuntimeError, match="TEXT_IDENTITY"):
        build_text_identity_rows(a0, s1)


def test_07_s1_qa_answer_head_is_inherited_openai_clip() -> None:
    assert "_frame_embedding" not in Siglip2GroundingPipeline.__dict__
    assert "_answer_embeddings" not in Siglip2GroundingPipeline.__dict__
    source = inspect.getsource(Siglip2GroundingPipeline)
    assert "runtime.backend =" not in source and "runtime.encoder =" not in source


def test_08_s1_pipeline_uses_current_route_text_without_rewrite() -> None:
    pipeline = object.__new__(Siglip2GroundingPipeline)
    pipeline._encoded_text = {}
    pipeline.text_identity_records = {}
    pipeline.runtime = SimpleNamespace(
        encode_requests=lambda requests: SimpleNamespace(
            encodings=({"clip_input_text": "current opus english"},),
            embeddings=np.ones((1, 512), dtype=np.float32),
        )
    )
    pipeline.grounding_encoder = SimpleNamespace(
        encode_text=lambda texts: l2_normalize(np.ones((len(texts), 768), dtype=np.float32))
    )
    vector, provenance = pipeline._encode("nguồn", "vi", "CB-KIS-01__grounding")
    assert vector.shape == (768,)
    assert provenance["sca1_clip_input_text"] == "current opus english"
    assert provenance["sca1_direct_vietnamese"] is False


def test_09_a0_hash_is_frozen_and_checked_before_gt() -> None:
    source = inspect.getsource(
        __import__(
            "triage_eg.diagnostics.sca1_siglip2_complementarity.runner",
            fromlist=["validate_pre_gt_integrity"],
        ).validate_pre_gt_integrity
    )
    assert EXPECTED_A0_PREDICTION_SHA256 in source or "EXPECTED_A0_PREDICTION_SHA256" in source
    assert source.index("EXPECTED_A0_PREDICTION_SHA256") < source.index("build_text_identity_rows")


def test_10_oracle_gt_phase_rejects_unfinalized_integrity(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="BEFORE_FINALIZED_HASHES"):
        evaluate_post_gt(
            a0_pipeline=SimpleNamespace(),
            s1_pipeline=SimpleNamespace(),
            a0_run={"sha256": "a0"},
            s1_run={"sha256": "s1"},
            a0_snapshot=SimpleNamespace(),
            s1_snapshot=SimpleNamespace(),
            integrity={"status": "FAIL"},
            benchmark_root=tmp_path / "missing",
            output_root=tmp_path / "output",
            temporary_root=tmp_path / "temp",
        )


def test_11_unique_rescue_and_loss_flags() -> None:
    rows = paired_unit_deltas(_audit_rows(s1_rescue=False), _audit_rows(s1_rescue=True))
    assert sum(row["target_global_top100_rescue"] for row in rows) == 1
    assert sum(row["target_global_top100_loss"] for row in rows) == 0
    assert sum(row["t3_target_rescue"] for row in rows) == 1


def test_11b_oracle_union_can_form_complementary_trake_chain() -> None:
    def query_row() -> dict:
        return {
            "query_id": "CB-TR-01",
            "btc_target_chain_exists": True,
            "t3_target_chain_exists": False,
            "g1_top100_full_target_chain_exists": False,
        }

    def event(event_id: str, hit: bool, frame: int) -> dict:
        return {
            "query_id": "CB-TR-01",
            "event_id": event_id,
            "t3_pool_has_target": hit,
            "t3_pool": ([{"original_frame_idx": frame, "distance_to_gt": 0}] if hit else []),
        }

    a0 = {
        "trake_query_rows": [query_row()],
        "trake_event_rows": [event("E1", True, 10), event("E2", False, 20)],
    }
    s1 = {
        "trake_query_rows": [query_row()],
        "trake_event_rows": [event("E1", False, 10), event("E2", True, 20)],
    }
    units = [
        {
            "task": "TRAKE",
            "t3_target_hit_a0": True,
            "t3_target_hit_s1": False,
            "t3_target_hit_u1": True,
        },
        {
            "task": "TRAKE",
            "t3_target_hit_a0": False,
            "t3_target_hit_s1": True,
            "t3_target_hit_u1": True,
        },
    ]
    oracle = oracle_union_diagnostics(a0, s1, units)
    assert oracle["trake_target_chain_counts"] == {"a0": 0, "s1": 0, "u1": 1}


@pytest.mark.parametrize(
    ("rescues", "event_delta", "chain_delta", "expected"),
    [
        (5, 3, 0, "OPEN_BOUNDED_FUSION"),
        (0, 1, 1, "NO_USEFUL_COMPLEMENTARITY"),
        (5, 0, 0, "LIMITED_OR_MIXED_COMPLEMENTARITY"),
    ],
)
def test_12_predeclared_complementarity_rule(
    rescues: int, event_delta: int, chain_delta: int, expected: str
) -> None:
    units = [{"target_global_top100_rescue": index < rescues} for index in range(max(5, rescues))]
    oracle = {
        "trake_t3_event_hit_counts": {"a0": 10, "s1": 10, "u1": 10 + event_delta},
        "trake_target_chain_counts": {"a0": 2, "s1": 2, "u1": 2 + chain_delta},
    }
    assert classify_complementarity(units, oracle)[0] == expected


def test_13_settings_forbid_out_of_scope_paths() -> None:
    settings = SCA1Settings()
    assert settings.fusion is False and settings.direct_vietnamese is False
    with pytest.raises(ValueError, match="diagnostic-only"):
        SCA1Settings(fusion=True)


def test_14_bundle_required_members_and_forbidden_heavy_files(tmp_path) -> None:
    root = tmp_path / "output"
    for name in REQUIRED_BUNDLE_MEMBERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    for name in ("diagnostics/a0/summary.json", "diagnostics/s1/summary.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    assert create_bundle(root, tmp_path / "bundle.zip")["member_count"] == 19
    (root / "model.safetensors").write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="FORBIDDEN_MEMBER"):
        create_bundle(root, tmp_path / "bundle2.zip")
