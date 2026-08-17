"""Pinned SigLIP2 offline asset preparation and fail-closed validation."""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

from aic2026_eval.io import sha256_file, write_json, write_jsonl

from .contracts import (
    EMBEDDING_DIMENSION,
    IMAGE_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SAFETENSORS_SHA256,
    PATCH_SIZE,
    PROCESSOR_USE_FAST,
    RUNTIME_MODEL_FILES,
    TEXT_MAX_LENGTH,
    TEXT_PADDING,
    TEXT_TRUNCATION,
)

ASSET_MANIFEST_VERSION = "SCA1_SIGLIP2_OFFLINE_ASSET_V1"


def _inventory(model_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in model_root.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(model_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def prepare_offline_asset(
    output_root: str | Path,
    *,
    cache_root: str | Path,
    revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Download the exact snapshot into cache, then create a clean runtime bundle."""

    if revision != MODEL_REVISION:
        raise ValueError("SCA-1 refuses any non-frozen SigLIP2 revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required only for asset preparation") from error
    root = Path(output_root).expanduser().resolve(strict=False)
    cache = Path(cache_root).expanduser().resolve(strict=False)
    if root == Path(root.anchor) or len(root.parts) < 3:
        raise ValueError("unsafe SCA-1 asset output root")
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache),
            allow_patterns=list(RUNTIME_MODEL_FILES),
        )
    ).resolve(strict=True)
    missing = [name for name in RUNTIME_MODEL_FILES if not (downloaded / name).is_file()]
    if missing:
        raise RuntimeError(f"SCA1_SIGLIP2_SNAPSHOT_INCOMPLETE: {missing}")
    staging = root.with_name(f".{root.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    model_root = staging / "model"
    manifests = staging / "manifests"
    model_root.mkdir(parents=True)
    manifests.mkdir(parents=True)
    for name in RUNTIME_MODEL_FILES:
        shutil.copy2(downloaded / name, model_root / name)
    inventory = _inventory(model_root)
    model_row = next(row for row in inventory if row["relative_path"] == "model.safetensors")
    if model_row["sha256"] != MODEL_SAFETENSORS_SHA256:
        raise RuntimeError(f"SCA1_SIGLIP2_MODEL_SHA256_MISMATCH: {model_row['sha256']}")
    manifest = {
        "asset_manifest_version": ASSET_MANIFEST_VERSION,
        "status": "COMPLETE",
        "model_id": MODEL_ID,
        "exact_revision": MODEL_REVISION,
        "source": "huggingface",
        "architecture": "SigLIP2 ViT-B/16 224",
        "runtime_model_path": "model",
        "internet_required_at_runtime": False,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "text_padding": TEXT_PADDING,
        "text_truncation": TEXT_TRUNCATION,
        "text_max_length": TEXT_MAX_LENGTH,
        "manual_l2_normalization": True,
        "processor_use_fast": PROCESSOR_USE_FAST,
        "files": inventory,
    }
    (manifests / "MODEL_REVISION.txt").write_text(MODEL_REVISION + "\n", encoding="utf-8")
    write_json(manifests / "asset_manifest.json", manifest)
    write_jsonl(manifests / "file_inventory.jsonl", inventory)
    if root.exists():
        shutil.rmtree(root)
    os.replace(staging, root)
    return validate_offline_asset(root)


def validate_offline_asset(asset_root: str | Path) -> dict[str, Any]:
    """Verify every runtime byte before an offline model load."""

    root = Path(asset_root).expanduser().resolve(strict=True)
    manifest_path = root / "manifests/asset_manifest.json"
    revision_path = root / "manifests/MODEL_REVISION.txt"
    if not manifest_path.is_file() or not revision_path.is_file():
        raise RuntimeError("SCA1_SIGLIP2_ASSET_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("asset_manifest_version") != ASSET_MANIFEST_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("exact_revision") != MODEL_REVISION
        or manifest.get("internet_required_at_runtime") is not False
        or manifest.get("image_size") != IMAGE_SIZE
        or manifest.get("patch_size") != PATCH_SIZE
        or manifest.get("embedding_dimension") != EMBEDDING_DIMENSION
        or manifest.get("text_padding") != TEXT_PADDING
        or manifest.get("text_truncation") is not TEXT_TRUNCATION
        or manifest.get("text_max_length") != TEXT_MAX_LENGTH
        or manifest.get("manual_l2_normalization") is not True
        or manifest.get("processor_use_fast") is not PROCESSOR_USE_FAST
        or revision_path.read_text(encoding="utf-8").strip() != MODEL_REVISION
    ):
        raise RuntimeError("SCA1_SIGLIP2_ASSET_CONTRACT_MISMATCH")
    model_root = root / str(manifest.get("runtime_model_path", ""))
    declared = manifest.get("files", [])
    if not isinstance(declared, list) or len(declared) != len(RUNTIME_MODEL_FILES):
        raise RuntimeError("SCA1_SIGLIP2_ASSET_INVENTORY_MISMATCH")
    actual = _inventory(model_root)
    if actual != declared or {row["relative_path"] for row in actual} != set(RUNTIME_MODEL_FILES):
        raise RuntimeError("SCA1_SIGLIP2_ASSET_FILE_HASH_OR_SIZE_MISMATCH")
    model_row = next(row for row in actual if row["relative_path"] == "model.safetensors")
    if model_row["sha256"] != MODEL_SAFETENSORS_SHA256:
        raise RuntimeError("SCA1_SIGLIP2_MODEL_SHA256_MISMATCH")
    return {
        "status": "VALID",
        "asset_root": str(root),
        "model_root": str(model_root),
        "model_id": MODEL_ID,
        "exact_revision": MODEL_REVISION,
        "model_safetensors_sha256": model_row["sha256"],
        "file_count": len(actual),
        "total_size_bytes": sum(row["size_bytes"] for row in actual),
        "manifest_sha256": sha256_file(manifest_path),
        "files": actual,
    }


def extract_pooled_features(output: Any, *, modality: str) -> Any:
    """Return the 2-D projected feature tensor across Transformers API variants.

    Older Transformers releases returned a tensor directly from SigLIP2
    ``get_*_features`` methods. Newer releases return
    ``BaseModelOutputWithPooling``. A tuple is also possible when
    ``return_dict=False``. Keep this compatibility boundary in one place so
    smoke tests and the persisted index use exactly the same representation.
    """

    features = output if getattr(output, "shape", None) is not None else None
    if features is None:
        features = getattr(output, "pooler_output", None)
    if features is None and isinstance(output, tuple | list) and len(output) > 1:
        features = output[1]
    shape = getattr(features, "shape", None)
    if shape is None:
        raise RuntimeError(f"SCA1_SIGLIP2_{modality.upper()}_FEATURE_OUTPUT_UNSUPPORTED")
    if len(shape) != 2:
        raise RuntimeError(
            f"SCA1_SIGLIP2_{modality.upper()}_FEATURE_OUTPUT_SHAPE_INVALID: {tuple(shape)}"
        )
    return features


def local_only_load_smoke(asset_root: str | Path) -> dict[str, Any]:
    """Load the official processor/model without any network fallback."""

    validated = validate_offline_asset(asset_root)
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise RuntimeError("transformers is required for SigLIP2 local-only smoke") from error
    model_root = validated["model_root"]
    processor = AutoProcessor.from_pretrained(
        model_root, local_files_only=True, use_fast=PROCESSOR_USE_FAST
    )
    model = AutoModel.from_pretrained(model_root, local_files_only=True).eval()
    config = model.config
    text_dim = int(getattr(config.text_config, "hidden_size", -1))
    vision_dim = int(getattr(config.vision_config, "hidden_size", -1))
    text_inputs = processor(
        text=["a red car"],
        padding=TEXT_PADDING,
        truncation=TEXT_TRUNCATION,
        max_length=TEXT_MAX_LENGTH,
        return_tensors="pt",
    )
    image_inputs = processor(
        images=[Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))], return_tensors="pt"
    )
    with torch.inference_mode():
        text_features = extract_pooled_features(
            model.get_text_features(**dict(text_inputs)), modality="text"
        )
        image_features = extract_pooled_features(
            model.get_image_features(**dict(image_inputs)), modality="image"
        )
    feature_shapes = {
        "text": list(text_features.shape),
        "image": list(image_features.shape),
    }
    del processor, model
    if (
        text_dim != EMBEDDING_DIMENSION
        or vision_dim != EMBEDDING_DIMENSION
        or feature_shapes
        != {
            "text": [1, EMBEDDING_DIMENSION],
            "image": [1, EMBEDDING_DIMENSION],
        }
    ):
        raise RuntimeError(f"SCA1_SIGLIP2_LOAD_DIMENSION_MISMATCH: {text_dim}/{vision_dim}")
    return {
        **validated,
        "local_only_load": "PASS",
        "text_dim": text_dim,
        "vision_dim": vision_dim,
        "feature_shapes": feature_shapes,
    }


def create_asset_zip(asset_root: str | Path, output_zip: str | Path) -> dict[str, Any]:
    validated = validate_offline_asset(asset_root)
    root = Path(asset_root).resolve(strict=True)
    target = Path(output_zip).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_STORED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(root).as_posix())
    return {
        **validated,
        "zip_path": str(target),
        "zip_sha256": sha256_file(target),
        "zip_size_bytes": target.stat().st_size,
    }


__all__ = [
    "ASSET_MANIFEST_VERSION",
    "create_asset_zip",
    "extract_pooled_features",
    "local_only_load_smoke",
    "prepare_offline_asset",
    "validate_offline_asset",
]
