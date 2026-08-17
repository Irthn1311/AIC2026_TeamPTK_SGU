"""Canonical BTC-row SigLIP2 diagnostic index builder and validator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from PIL import Image

from aic2026_eval.io import sha256_file, write_json
from triage_eg.retrieval.stage1.search import CompactCatalog

from .contracts import (
    EMBEDDING_DIMENSION,
    EXPECTED_ROWS,
    EXPECTED_STAGE1_FINGERPRINT,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SAFETENSORS_SHA256,
    PROCESSOR_USE_FAST,
)
from .encoder import Siglip2OfflineEncoder

INDEX_VERSION = "SCA1_SIGLIP2_BTC_INDEX_V1"
VECTOR_FILE = "siglip2_vectors.f16.npy"
NORM_FILE = "vector_norms.f32.npy"


def _stage1_index_root(stage1_root: Path) -> Path:
    root = stage1_root.expanduser().resolve(strict=True)
    return root / "index" if (root / "index").is_dir() else root


def catalog_row_fingerprint(catalog: CompactCatalog) -> str:
    digest = hashlib.sha256()
    for global_row in range(len(catalog.n)):
        row = catalog.map_row(global_row)
        payload = {
            key: row[key]
            for key in (
                "global_row",
                "video_id",
                "n",
                "original_frame_idx",
                "keyframe_relative_path",
                "duplicate_frame_idx_group_size",
            )
        }
        digest.update(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        )
    return digest.hexdigest()


def _validate_stage1(stage1_root: Path) -> tuple[Path, CompactCatalog, dict[str, Any]]:
    root = stage1_root.expanduser().resolve(strict=True)
    index_root = _stage1_index_root(root)
    summary_path = root / "stage1_summary.json"
    if not summary_path.is_file() and index_root.parent != root:
        summary_path = index_root.parent / "stage1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    catalog = CompactCatalog(index_root)
    if (
        summary.get("index_status") != "COMPLETE"
        or summary.get("index_fingerprint") != EXPECTED_STAGE1_FINGERPRINT
        or len(catalog.n) != EXPECTED_ROWS
    ):
        raise RuntimeError("SCA1_STAGE1_CATALOG_CONTRACT_MISMATCH")
    return index_root, catalog, summary


def keyframe_path(dataset_root: Path, catalog: CompactCatalog, global_row: int) -> Path:
    relative = Path(catalog.map_row(global_row)["keyframe_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("SCA1_KEYFRAME_PATH_TRAVERSAL")
    return dataset_root / relative


def smoke_sample_rows(catalog: CompactCatalog) -> tuple[int, ...]:
    count = len(catalog.n)
    base = {0, count // 4, count // 2, (3 * count) // 4, count - 1}
    video_first: dict[int, int] = {}
    for row, video_index in enumerate(np.asarray(catalog.video_index, dtype=np.int32)):
        video_first.setdefault(int(video_index), row)
        if len(video_first) >= 5:
            break
    return tuple(sorted(base | set(video_first.values())))


def run_index_smoke(
    encoder: Siglip2OfflineEncoder,
    dataset_root: str | Path,
    stage1_root: str | Path,
) -> dict[str, Any]:
    _, catalog, _ = _validate_stage1(Path(stage1_root))
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    rows = smoke_sample_rows(catalog)
    paths = [keyframe_path(dataset, catalog, row) for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"SCA1_SMOKE_KEYFRAMES_MISSING: {missing}")
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        first = encoder.encode_images(images)
        second = encoder.encode_images(images)
    finally:
        for image in images:
            image.close()
    if not np.allclose(first, second, rtol=0.0, atol=1e-5):
        raise RuntimeError("SCA1_SIGLIP2_REPEAT_ENCODING_UNSTABLE")
    return {
        "status": "PASS",
        "sample_rows": list(rows),
        "sample_paths": [str(path) for path in paths],
        "shape": list(first.shape),
        "finite": bool(np.isfinite(first).all()),
        "max_repeat_abs_delta": float(np.max(np.abs(first - second))),
        "max_norm_deviation": float(np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0))),
    }


def build_siglip2_index(
    *,
    encoder: Siglip2OfflineEncoder,
    dataset_root: str | Path,
    stage1_root: str | Path,
    output_root: str | Path,
    batch_size: int | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Encode every canonical BTC JPG once, preserving exact catalog order."""

    started = monotonic()
    dataset = Path(dataset_root).expanduser().resolve(strict=True)
    _, catalog, stage1_summary = _validate_stage1(Path(stage1_root))
    smoke = run_index_smoke(encoder, dataset, stage1_root)
    output = Path(output_root).expanduser().resolve(strict=False)
    if output == Path(output.anchor) or len(output.parts) < 3:
        raise ValueError("unsafe SCA-1 index output root")
    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    index_root = staging / "index"
    index_root.mkdir(parents=True)
    vectors = np.lib.format.open_memmap(
        index_root / VECTOR_FILE,
        mode="w+",
        dtype=np.float16,
        shape=(EXPECTED_ROWS, EMBEDDING_DIMENSION),
    )
    encode_batch = batch_size or encoder.batch_size
    missing: list[str] = []
    for start in range(0, EXPECTED_ROWS, encode_batch):
        stop = min(start + encode_batch, EXPECTED_ROWS)
        paths = [keyframe_path(dataset, catalog, row) for row in range(start, stop)]
        missing.extend(str(path) for path in paths if not path.is_file())
        if missing:
            raise RuntimeError(f"SCA1_INDEX_KEYFRAME_MISSING: {missing[:10]}")
        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            encoded = encoder.encode_images(images)
        finally:
            for image in images:
                image.close()
        vectors[start:stop] = encoded.astype(np.float16)
        vectors.flush()
    del vectors
    persisted = np.load(index_root / VECTOR_FILE, mmap_mode="r", allow_pickle=False)
    norms = np.empty(EXPECTED_ROWS, dtype=np.float32)
    for start in range(0, EXPECTED_ROWS, 16_384):
        stop = min(start + 16_384, EXPECTED_ROWS)
        chunk = np.asarray(persisted[start:stop], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise RuntimeError(f"SCA1_INDEX_NONFINITE_VECTOR_ROWS: {start}:{stop}")
        norms[start:stop] = np.linalg.norm(chunk, axis=1)
    if np.any(norms <= 0) or float(np.max(np.abs(norms - 1.0))) > 2e-3:
        raise RuntimeError("SCA1_INDEX_PERSISTED_NORMALIZATION_GATE_FAILED")
    np.save(index_root / NORM_FILE, norms, allow_pickle=False)
    duplicate_rows = int(np.count_nonzero(np.asarray(catalog.duplicate_size) > 1))
    row_fingerprint = catalog_row_fingerprint(catalog)
    vector_sha = sha256_file(index_root / VECTOR_FILE)
    norm_sha = sha256_file(index_root / NORM_FILE)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "stage1": EXPECTED_STAGE1_FINGERPRINT,
                "catalog_rows": row_fingerprint,
                "vectors": vector_sha,
                "norms": norm_sha,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "index_version": INDEX_VERSION,
        "status": "COMPLETE",
        "created_at": datetime.now(UTC).isoformat(),
        "build_git_commit": git_commit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "encoder_provenance": encoder.provenance(),
        "stage1_index_fingerprint": stage1_summary["index_fingerprint"],
        "catalog_row_fingerprint": row_fingerprint,
        "catalog_order": "EXACT_FROZEN_STAGE1_GLOBAL_ROW_ORDER",
        "rows": EXPECTED_ROWS,
        "shape": [EXPECTED_ROWS, EMBEDDING_DIMENSION],
        "dtype": "float16",
        "scoring_dtype": "float32",
        "metric": "exact_cosine",
        "stable_tie_policy": "score_desc_global_row_asc",
        "duplicate_frame_rows_preserved": True,
        "duplicate_group_member_rows": duplicate_rows,
        "vector_file": f"index/{VECTOR_FILE}",
        "vector_sha256": vector_sha,
        "norm_file": f"index/{NORM_FILE}",
        "norm_sha256": norm_sha,
        "max_persisted_norm_deviation": float(np.max(np.abs(norms - 1.0))),
        "smoke": smoke,
        "index_fingerprint": fingerprint,
        "build_seconds": monotonic() - started,
    }
    write_json(staging / "index_manifest.json", manifest)
    if output.exists():
        shutil.rmtree(output)
    os.replace(staging, output)
    return validate_siglip2_index(output, stage1_root=stage1_root)


def validate_siglip2_index(
    index_root: str | Path, *, stage1_root: str | Path | None = None
) -> dict[str, Any]:
    root = Path(index_root).expanduser().resolve(strict=True)
    manifest = json.loads((root / "index_manifest.json").read_text(encoding="utf-8"))
    provenance = manifest.get("encoder_provenance", {})
    vector_path, norm_path = (
        root / manifest.get("vector_file", ""),
        root / manifest.get("norm_file", ""),
    )
    if (
        manifest.get("index_version") != INDEX_VERSION
        or manifest.get("status") != "COMPLETE"
        or manifest.get("rows") != EXPECTED_ROWS
        or manifest.get("shape") != [EXPECTED_ROWS, EMBEDDING_DIMENSION]
        or manifest.get("stage1_index_fingerprint") != EXPECTED_STAGE1_FINGERPRINT
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("duplicate_frame_rows_preserved") is not True
        or manifest.get("dtype") != "float16"
        or manifest.get("scoring_dtype") != "float32"
        or manifest.get("metric") != "exact_cosine"
        or manifest.get("stable_tie_policy") != "score_desc_global_row_asc"
        or manifest.get("catalog_order") != "EXACT_FROZEN_STAGE1_GLOBAL_ROW_ORDER"
        or manifest.get("smoke", {}).get("status") != "PASS"
        or provenance.get("model_id") != MODEL_ID
        or provenance.get("exact_revision") != MODEL_REVISION
        or provenance.get("model_safetensors_sha256") != MODEL_SAFETENSORS_SHA256
        or provenance.get("embedding_dimension") != EMBEDDING_DIMENSION
        or provenance.get("processor_use_fast") is not PROCESSOR_USE_FAST
        or provenance.get("manual_l2_normalization") is not True
    ):
        raise RuntimeError("SCA1_INDEX_MANIFEST_CONTRACT_MISMATCH")
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    norms = np.load(norm_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (EXPECTED_ROWS, EMBEDDING_DIMENSION) or vectors.dtype != np.float16:
        raise RuntimeError("SCA1_INDEX_VECTOR_SHAPE_OR_DTYPE_MISMATCH")
    if norms.shape != (EXPECTED_ROWS,) or norms.dtype != np.float32:
        raise RuntimeError("SCA1_INDEX_NORM_SHAPE_OR_DTYPE_MISMATCH")
    if sha256_file(vector_path) != manifest.get("vector_sha256") or sha256_file(
        norm_path
    ) != manifest.get("norm_sha256"):
        raise RuntimeError("SCA1_INDEX_FILE_SHA256_MISMATCH")
    if stage1_root is not None:
        _, catalog, _ = _validate_stage1(Path(stage1_root))
        if catalog_row_fingerprint(catalog) != manifest.get("catalog_row_fingerprint"):
            raise RuntimeError("SCA1_INDEX_CATALOG_ROW_ORDER_MISMATCH")
    return {
        "status": "VALID",
        "root": str(root),
        "manifest": manifest,
        "vectors": vectors,
        "norms": norms,
    }


__all__ = [
    "INDEX_VERSION",
    "NORM_FILE",
    "VECTOR_FILE",
    "build_siglip2_index",
    "catalog_row_fingerprint",
    "keyframe_path",
    "run_index_smoke",
    "smoke_sample_rows",
    "validate_siglip2_index",
]
