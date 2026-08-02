"""NumPy feature matrix and JSONL manifest persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from triage_eg.common.schemas import FeatureRecord, load_jsonl, save_jsonl


def validate_feature_store(vectors: np.ndarray, records: list[FeatureRecord]) -> None:
    """Validate matrix shape, dimensions, and contiguous row metadata."""

    if vectors.ndim != 2:
        raise ValueError("Feature vectors must be a two-dimensional array")
    if len(records) != vectors.shape[0]:
        raise ValueError("Feature vector count does not match the manifest")
    for expected_row, record in enumerate(records):
        if record.feature_row != expected_row:
            raise ValueError("feature_row values must be contiguous from zero")
        if record.dimension != vectors.shape[1]:
            raise ValueError("FeatureRecord dimension does not match the matrix")


def save_feature_store(
    directory: str | Path, vectors: np.ndarray, records: list[FeatureRecord]
) -> tuple[Path, Path]:
    """Persist ``vectors.npy`` and ``feature_manifest.jsonl``."""

    validate_feature_store(vectors, records)
    store_dir = Path(directory)
    store_dir.mkdir(parents=True, exist_ok=True)
    vector_path = store_dir / "vectors.npy"
    manifest_path = store_dir / "feature_manifest.jsonl"
    np.save(vector_path, vectors.astype(np.float32, copy=False))
    save_jsonl(records, manifest_path)
    return vector_path, manifest_path


def load_feature_store(directory: str | Path) -> tuple[np.ndarray, list[FeatureRecord]]:
    """Load and validate a persisted feature store."""

    store_dir = Path(directory)
    vector_path = store_dir / "vectors.npy"
    manifest_path = store_dir / "feature_manifest.jsonl"
    if not vector_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete feature store: {store_dir}")
    vectors = np.load(vector_path, allow_pickle=False)
    records = load_jsonl(manifest_path, FeatureRecord)
    validate_feature_store(vectors, records)
    return vectors, records
