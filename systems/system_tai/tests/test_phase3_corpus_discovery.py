from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    discover_corpus,
    load_corpus_manifest,
    resolve_dataset_root,
)
from system_tai.kis.build_manifest import main as build_manifest_main
from tests.phase3_helpers import create_corpus, feature_matrix


def _videos() -> dict[str, tuple[list[int], np.ndarray]]:
    return {
        "L21_V002": ([20, 21], feature_matrix([(0, 1.0), (1, 1.0)])),
        "L21_V001": ([10, 11, 12], feature_matrix([(0, 1.0), (1, 1.0), (2, 1.0)])),
    }


def test_discovers_multiple_videos_deterministically_without_copying(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset = create_corpus(input_root, _videos())
    before = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    first = discover_corpus(input_root)
    second = discover_corpus(input_root)
    assert [video.video_id for video in first.videos] == ["L21_V001", "L21_V002"]
    assert first.fingerprint == second.fingerprint
    assert first.total_rows == 5
    assert resolve_dataset_root(input_root) == dataset.resolve()
    after = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_incomplete_pair_and_ambiguous_mapping_fail_clearly(tmp_path: Path) -> None:
    input_root = tmp_path / "missing"
    dataset = create_corpus(input_root, _videos())
    (dataset / "clip-features-32-aic25-b1" / "clip-features-32" / "L21_V002.npy").unlink()
    with pytest.raises(CorpusDiscoveryError) as missing:
        discover_corpus(input_root)
    assert any("L21_V002" in issue and "CLIP NPY" in issue for issue in missing.value.issues)

    ambiguous_root = tmp_path / "ambiguous"
    ambiguous_dataset = create_corpus(ambiguous_root, _videos())
    duplicate = (
        ambiguous_dataset
        / "map-keyframes-aic25-b1"
        / "duplicate"
        / "L21_V001.csv"
    )
    duplicate.parent.mkdir()
    original = (
        ambiguous_dataset
        / "map-keyframes-aic25-b1"
        / "map-keyframes"
        / "L21_V001.csv"
    )
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(CorpusDiscoveryError) as ambiguous:
        discover_corpus(ambiguous_root)
    assert any("expected one mapping CSV, found 2" in issue for issue in ambiguous.value.issues)


def test_mapping_feature_row_mismatch_fails(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset = create_corpus(input_root, _videos())
    np.save(
        dataset / "clip-features-32-aic25-b1" / "clip-features-32" / "L21_V001.npy",
        feature_matrix([(0, 1.0)]),
    )
    with pytest.raises(CorpusDiscoveryError) as exc:
        discover_corpus(input_root)
    assert any("row mismatch" in issue for issue in exc.value.issues)


def test_manifest_reuse_fingerprint_and_cli_are_deterministic(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _videos())
    manifest = discover_corpus(input_root)
    original_path = manifest.write(tmp_path / "original.json")
    loaded = load_corpus_manifest(original_path)
    assert loaded == manifest
    reused_path = tmp_path / "reused.json"
    assert (
        build_manifest_main(
            [
                "--output",
                str(reused_path),
                "--reuse-manifest",
                str(original_path),
            ]
        )
        == 0
    )
    original = json.loads(original_path.read_text(encoding="utf-8"))
    reused = json.loads(reused_path.read_text(encoding="utf-8"))
    assert reused == original

    payload = reused
    payload["videos"][0]["row_count"] += 1
    reused_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_corpus_manifest(reused_path)
