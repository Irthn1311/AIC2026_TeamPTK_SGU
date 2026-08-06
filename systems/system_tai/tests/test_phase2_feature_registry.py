from __future__ import annotations

import json

import numpy as np
import pytest

from system_tai.features.btc_clip_store import FeatureStoreRegistry, VideoFeatureStoreLoader
from tests.phase2_helpers import make_store, write_mapping


def _valid_paths(tmp_path, dimension: int = 512):
    mapping = tmp_path / "map.csv"
    features = tmp_path / "clip.npy"
    write_mapping(mapping, [(7, 0.0, 30.0, 0), (9, 1.0, 30.0, 30)])
    matrix = np.zeros((2, dimension), dtype=np.float16)
    matrix[0, 0] = 1
    matrix[1, 1] = 1
    np.save(features, matrix)
    return mapping, features


def test_loader_preserves_physical_row_and_frame_idx(tmp_path) -> None:
    mapping, features = _valid_paths(tmp_path)
    store = VideoFeatureStoreLoader().load(
        video_id="L21_V001", mapping_csv_path=mapping, clip_npy_path=features
    )
    assert [row.clip_row for row in store.mappings] == [0, 1]
    assert [row.keyframe_order for row in store.mappings] == [7, 9]
    assert [row.frame_id for row in store.mappings] == [0, 30]
    assert isinstance(store.matrix, np.memmap)
    assert not store.matrix.flags.writeable


def test_loader_rejects_row_mismatch_invalid_dimension_and_bad_values(tmp_path) -> None:
    mapping, features = _valid_paths(tmp_path)
    np.save(features, np.ones((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="row-count mismatch"):
        VideoFeatureStoreLoader().load(
            video_id="v", mapping_csv_path=mapping, clip_npy_path=features
        )

    np.save(features, np.ones((2, 511), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension mismatch"):
        VideoFeatureStoreLoader().load(
            video_id="v", mapping_csv_path=mapping, clip_npy_path=features
        )

    bad = np.ones((2, 512), dtype=np.float32)
    bad[0, 0] = np.nan
    np.save(features, bad)
    with pytest.raises(ValueError, match="NaN or Infinity"):
        VideoFeatureStoreLoader().load(
            video_id="v", mapping_csv_path=mapping, clip_npy_path=features
        )

    bad[0, 0] = 1
    bad[1] = 0
    np.save(features, bad)
    with pytest.raises(ValueError, match="zero-norm"):
        VideoFeatureStoreLoader().load(
            video_id="v", mapping_csv_path=mapping, clip_npy_path=features
        )


def test_mapping_validation_and_registry_failures(tmp_path) -> None:
    mapping, features = _valid_paths(tmp_path)
    write_mapping(mapping, [(2, 0, 30, 0), (1, 1, 30, 30)])
    with pytest.raises(ValueError, match="strictly increasing"):
        VideoFeatureStoreLoader().load(
            video_id="v", mapping_csv_path=mapping, clip_npy_path=features
        )
    write_mapping(mapping, [(1, 0, 30, 4), (2, 1, 30, 4)])
    duplicate_store = VideoFeatureStoreLoader().load(
        video_id="v", mapping_csv_path=mapping, clip_npy_path=features
    )
    assert duplicate_store.rows_for_frame(4) == (0, 1)
    assert duplicate_store.contains_frame(4)
    assert duplicate_store.unique_frame_count == 1
    assert duplicate_store.duplicate_frame_id_count == 1

    a = make_store("v", np.ones((1, 3), dtype=np.float32), [1])
    with pytest.raises(ValueError, match="duplicate video_id"):
        FeatureStoreRegistry([a, a])
    b = make_store("b", np.ones((1, 4), dtype=np.float32), [2])
    with pytest.raises(ValueError, match="inconsistent feature dimensions"):
        FeatureStoreRegistry([a, b])


def test_registry_loads_explicit_relative_manifest(tmp_path) -> None:
    mapping, features = _valid_paths(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "L21_V001",
                        "mapping_csv_path": mapping.name,
                        "clip_npy_path": features.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = FeatureStoreRegistry.from_manifest(manifest)
    assert registry.total_rows == 2
    assert registry.contains("L21_V001", 30)
    assert not registry.contains("L21_V001", 31)
