from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from system_tai.data.corpus_discovery import (
    CorpusDiscoveryError,
    DeterministicTreeWalker,
    DiscoveryValidation,
    WalkDirectory,
    _family_roots,
    _index_families,
    discover_corpus,
    load_corpus_manifest,
    load_or_build_manifest_cache,
)
from system_tai.kis.build_manifest import main as build_manifest_main
from system_tai.kis.contest import build_parser as contest_parser
from system_tai.refinement.video import RawVideoRegistry
from tests.phase3_helpers import create_corpus, feature_matrix


def _videos(count: int = 2) -> dict[str, tuple[list[int], np.ndarray]]:
    return {
        f"L21_V{index:03d}": ([0, index], feature_matrix([(0, 1), (1, 1)]))
        for index in range(1, count + 1)
    }


class CountingWalker:
    def __init__(self) -> None:
        self.calls: dict[Path, int] = {}
        self.delegate = DeterministicTreeWalker()

    def walk(self, root: Path) -> Iterator[WalkDirectory]:
        resolved = root.resolve(strict=False)
        self.calls[resolved] = self.calls.get(resolved, 0) + 1
        yield from self.delegate.walk(root)


def test_each_family_root_and_keyframe_tree_are_walked_once(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset = create_corpus(input_root, _videos())
    walker = CountingWalker()
    manifest = discover_corpus(input_root, walker=walker)
    roots = {
        root.resolve(strict=False) for family in _family_roots(dataset).values() for root in family
    }
    assert set(walker.calls) == roots
    assert all(count == 1 for count in walker.calls.values())
    assert all(count == 1 for count in manifest.discovery_metrics.family_root_traversals.values())
    assert manifest.discovery_metrics.keyframe_images_seen == 4
    for field in (
        "dataset_root_resolution_seconds",
        "family_index_seconds",
        "mapping_validation_seconds",
        "clip_shape_validation_seconds",
        "keyframe_stats_seconds",
        "raw_video_index_seconds",
        "manifest_fingerprint_seconds",
        "manifest_write_seconds",
        "total_discovery_seconds",
        "filesystem_directories_visited",
        "filesystem_files_visited",
        "mapping_files_validated",
        "clip_files_validated",
        "raw_video_files_seen",
    ):
        assert field in manifest.discovery_metrics.to_payload()


class Synthetic873Walker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def walk(self, root: Path) -> Iterator[WalkDirectory]:
        assert root == self.root
        self.calls += 1
        yield WalkDirectory(root, ("keyframes",), ())
        for index in range(1, 874):
            video_id = f"L21_V{index:03d}"
            directory = root / "keyframes" / video_id
            yield WalkDirectory(directory, (), ("001.jpg",))


def test_873_keyframe_directories_do_not_create_873_traversals(tmp_path: Path) -> None:
    root = tmp_path / "keyframes-family"
    walker = Synthetic873Walker(root)
    index = _index_families(
        {"mapping": (), "clip": (), "keyframes": (root,), "videos": ()},
        walker=walker,
        clock=lambda: 0.0,
    )
    assert walker.calls == 1
    assert len(index.keyframes) == 873
    assert index.keyframe_images_seen == 873


def test_strict_and_fast_build_same_logical_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _videos())
    strict = discover_corpus(input_root, validation_mode=DiscoveryValidation.STRICT)
    fast = discover_corpus(input_root, validation_mode=DiscoveryValidation.FAST)
    assert strict.fingerprint == fast.fingerprint
    assert strict.videos == fast.videos
    assert strict.discovery_metrics.mapping_files_validated == 2
    assert fast.discovery_metrics.clip_files_validated == 2


def test_schema_v1_load_and_portable_round_trip_rebase(tmp_path: Path) -> None:
    first_input = tmp_path / "runtime-a"
    first_dataset = create_corpus(first_input, _videos())
    manifest = discover_corpus(first_input)
    v1_path = manifest.write(tmp_path / "v1.json")
    assert load_corpus_manifest(v1_path) == manifest

    portable_path = manifest.write(tmp_path / "portable.json", portable=True)
    portable_text = portable_path.read_text(encoding="utf-8")
    payload = json.loads(portable_text)
    assert payload["schema_version"] == 2
    assert payload["path_mode"] == "dataset_root_relative_posix"
    assert str(first_input.resolve()) not in portable_text
    assert all("\\" not in item["mapping_csv_path"] for item in payload["videos"])
    assert not any(Path(item["mapping_csv_path"]).is_absolute() for item in payload["videos"])
    assert all(
        not Path(root).is_absolute()
        for root in payload["discovery_metrics"]["family_root_traversals"]
    )

    second_input = tmp_path / "runtime-b"
    target_dataset = second_input / "datasets" / "new-owner" / first_dataset.name
    target_dataset.parent.mkdir(parents=True)
    shutil.copytree(first_dataset, target_dataset)
    loaded = load_corpus_manifest(portable_path, input_root=second_input)
    assert loaded.portable is True
    assert loaded.fingerprint == payload["manifest_fingerprint"]
    assert [video.video_id for video in loaded.videos] == ["L21_V001", "L21_V002"]
    assert all(video.mapping_csv_path.is_relative_to(target_dataset) for video in loaded.videos)
    assert all(
        Path(root).is_relative_to(target_dataset)
        for root in loaded.discovery_metrics.family_root_traversals
    )
    raw = RawVideoRegistry.from_manifest(loaded)
    assert raw.get("L21_V001").raw_video_path.is_relative_to(target_dataset)


def test_portable_load_rejects_dataset_mismatch_and_missing_artifact(tmp_path: Path) -> None:
    first = tmp_path / "first"
    dataset = create_corpus(first, _videos(1))
    portable = discover_corpus(first, portable=True).write(
        tmp_path / "portable.json", portable=True
    )
    second = tmp_path / "second"
    target = second / "datasets" / "owner" / dataset.name
    target.parent.mkdir(parents=True)
    shutil.copytree(dataset, target)
    mapping = target / "map-keyframes-aic25-b1" / "map-keyframes" / "L21_V001.csv"
    mapping.write_text(mapping.read_text(encoding="utf-8-sig") + "3,1,30,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_corpus_manifest(portable, input_root=second)
    mapping.unlink()
    with pytest.raises(FileNotFoundError, match="mapping CSV"):
        load_corpus_manifest(portable, input_root=second)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("mapping", "expected one mapping CSV, found 2"),
        ("clip", "expected one CLIP NPY, found 2"),
        ("keyframe", "expected one keyframe directory, found 2"),
        ("raw", "expected at most one raw video, found 2"),
    ],
)
def test_ambiguous_artifacts_are_rejected(tmp_path: Path, kind: str, expected: str) -> None:
    input_root = tmp_path / kind
    dataset = create_corpus(input_root, _videos(1))
    if kind == "mapping":
        original = dataset / "map-keyframes-aic25-b1" / "map-keyframes" / "L21_V001.csv"
        duplicate = dataset / "map-keyframes-aic25-b1" / "duplicate" / original.name
        duplicate.parent.mkdir()
        duplicate.write_bytes(original.read_bytes())
    elif kind == "clip":
        original = dataset / "clip-features-32-aic25-b1" / "clip-features-32" / "L21_V001.npy"
        duplicate = dataset / "clip-features-32-aic25-b1" / "duplicate" / original.name
        duplicate.parent.mkdir()
        duplicate.write_bytes(original.read_bytes())
    elif kind == "keyframe":
        duplicate = dataset / "keyframes" / "duplicate" / "L21_V001"
        duplicate.mkdir(parents=True)
        (duplicate / "001.jpg").touch()
    else:
        duplicate = dataset / "Videos_L21_a" / "L21_V001.mkv"
        duplicate.touch()
    with pytest.raises(CorpusDiscoveryError) as exc:
        discover_corpus(input_root)
    assert any(expected in issue for issue in exc.value.issues)


def test_optional_raw_exact_stem_and_empty_keyframe_gate(tmp_path: Path) -> None:
    no_raw_root = tmp_path / "no-raw"
    create_corpus(no_raw_root, _videos(1), include_raw_video=False)
    assert discover_corpus(no_raw_root).videos[0].raw_video_path is None

    exact_root = tmp_path / "exact"
    dataset = create_corpus(exact_root, _videos(1))
    (dataset / "Videos_L21_a" / "L21_V001-extra.mp4").touch()
    video = discover_corpus(exact_root).videos[0]
    assert video.raw_video_path.name == "L21_V001.mp4"

    empty_root = tmp_path / "empty"
    empty_dataset = create_corpus(empty_root, _videos(1))
    keyframes = (
        empty_dataset / "keyframes" / "keyframes" / "Keyframes_L21" / "keyframes" / "L21_V001"
    )
    for image in keyframes.glob("*.jpg"):
        image.unlink()
    (keyframes / "001.txt").touch()
    with pytest.raises(CorpusDiscoveryError) as exc:
        discover_corpus(empty_root)
    assert any("no supported images" in issue for issue in exc.value.issues)


def test_fast_still_rejects_row_and_dimension_mismatch_and_accepts_duplicate_frame(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate-frame"
    create_corpus(
        duplicate_root,
        {"L21_V001": ([0, 0], feature_matrix([(0, 1), (1, 1)]))},
    )
    assert discover_corpus(duplicate_root, validation_mode=DiscoveryValidation.FAST).total_rows == 2

    row_root = tmp_path / "row"
    row_dataset = create_corpus(row_root, _videos(1))
    np.save(
        row_dataset / "clip-features-32-aic25-b1" / "clip-features-32" / "L21_V001.npy",
        feature_matrix([(0, 1)]),
    )
    with pytest.raises(CorpusDiscoveryError) as row_error:
        discover_corpus(row_root, validation_mode=DiscoveryValidation.FAST)
    assert any("row mismatch" in issue for issue in row_error.value.issues)

    dimension_root = tmp_path / "dimension"
    dimension_dataset = create_corpus(dimension_root, _videos(1))
    np.save(
        dimension_dataset / "clip-features-32-aic25-b1" / "clip-features-32" / "L21_V001.npy",
        np.ones((2, 3), dtype=np.float32),
    )
    with pytest.raises(CorpusDiscoveryError) as dimension_error:
        discover_corpus(dimension_root, validation_mode=DiscoveryValidation.FAST)
    assert any("dimension mismatch" in issue for issue in dimension_error.value.issues)


def test_cache_hit_bypasses_discovery_and_invalid_cache_needs_explicit_rebuild(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _videos(1))
    cache = tmp_path / "cache.json"
    discover_corpus(input_root, portable=True).write(cache, portable=True)

    def forbidden_discovery(*_args, **_kwargs):
        raise AssertionError("cache hit called discover_corpus")

    hit = load_or_build_manifest_cache(
        cache,
        input_root=input_root,
        discoverer=forbidden_discovery,
    )
    assert hit.status == "CACHE_HIT"

    cache.write_text("{invalid", encoding="utf-8")
    with pytest.raises(CorpusDiscoveryError, match="explicit rebuild required"):
        load_or_build_manifest_cache(cache, input_root=input_root)
    rebuilt = load_or_build_manifest_cache(
        cache,
        input_root=input_root,
        rebuild_invalid=True,
    )
    assert rebuilt.status == "CACHE_REBUILT"
    assert json.loads(cache.read_text(encoding="utf-8"))["schema_version"] == 2


def test_missing_cache_builds_strict_portable_and_cli_conflicts_are_explicit(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _videos(1))
    cache = tmp_path / "missing-cache.json"
    output = tmp_path / "runtime.json"
    assert (
        build_manifest_main(
            [
                "--input-root",
                str(input_root),
                "--manifest-cache",
                str(cache),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(cache.read_text(encoding="utf-8"))["schema_version"] == 2
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert build_manifest_main(["--output", str(output), "--rebuild-invalid-manifest-cache"]) == 2
    with pytest.raises(SystemExit):
        contest_parser().parse_args(
            [
                "--reuse-manifest",
                "one.json",
                "--manifest-cache",
                "two.json",
                "--query-id",
                "Q",
                "--query-vi",
                "text",
                "--output-directory",
                "out",
            ]
        )


def test_portable_load_never_invokes_full_family_walker(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _videos(1))
    portable = discover_corpus(input_root, portable=True).write(
        tmp_path / "portable.json", portable=True
    )

    def forbidden_walk(*_args, **_kwargs):
        raise AssertionError("portable load invoked full family traversal")

    monkeypatch.setattr(DeterministicTreeWalker, "walk", forbidden_walk)
    loaded = load_corpus_manifest(portable, input_root=input_root)
    assert loaded.total_rows == 2


def test_deterministic_order_fingerprint_and_cross_platform_path_contract(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    create_corpus(input_root, _videos())
    first = discover_corpus(input_root, portable=True)
    second = discover_corpus(input_root, portable=True)
    assert [item.video_id for item in first.videos] == ["L21_V001", "L21_V002"]
    assert first.fingerprint == second.fingerprint
    portable_path = first.to_payload(portable=True)["videos"][0]["mapping_csv_path"]
    assert PurePosixPath(portable_path).parts
    assert "\\" not in portable_path
    assert PureWindowsPath(r"C:\runtime\dataset-aic").name == "dataset-aic"
