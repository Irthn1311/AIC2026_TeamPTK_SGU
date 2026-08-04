from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from triage_eg.retrieval.stage1.stage0_loader import REQUIRED_FILES, resolve_stage0_root


def write_required(root: Path) -> Path:
    root.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (root / name).write_text("{}\n", encoding="utf-8")
    return root


def write_bundle(path: Path, *, missing: str | None = None) -> Path:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in REQUIRED_FILES:
            if name != missing:
                archive.writestr(f"triage_eg_stage0_audit_bundle/{name}", "{}\n")
        archive.writestr("logs/not_in_runtime.txt", "excluded")
    return path


def test_resolve_existing_requested_root(tmp_path: Path) -> None:
    expected = write_required(tmp_path / "stage0")
    assert resolve_stage0_root(expected) == expected.resolve()


def test_resolve_explicit_nested_directory(tmp_path: Path) -> None:
    expected = write_required(tmp_path / "mounted" / "triage_eg_stage0_audit_bundle")
    assert (
        resolve_stage0_root(tmp_path / "working-stage0", bundle_path=tmp_path / "mounted")
        == expected.resolve()
    )


def test_bounded_kaggle_input_discovery_excludes_dataset_root(tmp_path: Path) -> None:
    kaggle_input = tmp_path / "kaggle-input"
    dataset_root = kaggle_input / "datasets" / "owner" / "raw-data"
    dataset_root.mkdir(parents=True)
    # A misleading nested artifact under the raw dataset must never be discovered.
    write_required(dataset_root / "do-not-scan" / "stage0")
    expected = write_required(kaggle_input / "datasets" / "owner" / "stage0-artifacts")
    assert (
        resolve_stage0_root(
            tmp_path / "working-stage0",
            search_root=kaggle_input,
            excluded_roots=(dataset_root,),
        )
        == expected.resolve()
    )


def test_discover_attached_kaggle_notebook_output(tmp_path: Path) -> None:
    search_root = tmp_path / "input"
    expected = write_required(search_root / "notebooks" / "team" / "stage0-audit-run")
    assert (
        resolve_stage0_root(tmp_path / "working-stage0", search_root=search_root)
        == expected.resolve()
    )


def test_resolve_and_materialize_explicit_zip(tmp_path: Path) -> None:
    bundle = write_bundle(tmp_path / "triage_eg_stage0_audit_bundle.zip")
    target = tmp_path / "working" / "triage_eg_stage0_audit"
    resolved = resolve_stage0_root(target, bundle_path=bundle)
    assert resolved == target.resolve()
    assert sorted(path.name for path in target.iterdir()) == sorted(REQUIRED_FILES)


def test_explicit_mounted_dataset_directory_can_contain_zip(tmp_path: Path) -> None:
    mounted = tmp_path / "input" / "datasets" / "irthn1311" / "stage0-bundle"
    mounted.mkdir(parents=True)
    write_bundle(mounted / "triage_eg_stage0_audit_bundle.zip")
    target = tmp_path / "working" / "stage0"
    assert resolve_stage0_root(target, bundle_path=mounted) == target.resolve()


def test_auto_discover_single_zip(tmp_path: Path) -> None:
    mounted = tmp_path / "input" / "datasets" / "owner" / "audit-artifacts"
    mounted.mkdir(parents=True)
    write_bundle(mounted / "triage_eg_stage0_audit_bundle.zip")
    target = tmp_path / "working" / "stage0"
    assert resolve_stage0_root(target, search_root=tmp_path / "input") == target.resolve()


def test_missing_or_invalid_bundle_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Add Input"):
        resolve_stage0_root(tmp_path / "stage0", search_root=tmp_path / "empty-input")
    broken = write_bundle(tmp_path / "broken-stage0-audit.zip", missing="contract_notes.json")
    with pytest.raises(ValueError, match="contract_notes.json"):
        resolve_stage0_root(tmp_path / "stage0", bundle_path=broken)


def test_ambiguous_discovery_requires_explicit_bundle(tmp_path: Path) -> None:
    search_root = tmp_path / "input"
    write_required(search_root / "first")
    write_required(search_root / "second")
    with pytest.raises(ValueError, match="AIC_STAGE0_BUNDLE"):
        resolve_stage0_root(tmp_path / "stage0", search_root=search_root)
