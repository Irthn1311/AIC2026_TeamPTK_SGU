from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "build_quality_q1b_sampling_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("system_tai_quality_q1b_sampling", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SAMPLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLER)


def _inventory() -> list[str]:
    return ["L22_V002", "L21_V003", "L21_V001", "L22_V001"]


def test_repeated_runs_are_deterministic() -> None:
    assert SAMPLER.build_sampling_records(_inventory()) == SAMPLER.build_sampling_records(
        _inventory()
    )


def test_order_uses_hash_then_video_id_tie_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SAMPLER, "selection_hash", lambda _video_id: "0" * 64)
    rows = SAMPLER.build_sampling_records(reversed(_inventory()))
    assert [row.video_id for row in rows] == sorted(_inventory())


def test_known_hash_matches_independent_sha256() -> None:
    expected = hashlib.sha256(b"system_tai_q1b_v1|L21_V001").hexdigest()
    assert SAMPLER.selection_hash("L21_V001") == expected


def test_ranks_are_one_based_and_contiguous() -> None:
    rows = SAMPLER.build_sampling_records(_inventory())
    assert [row.sample_rank for row in rows] == list(range(1, len(rows) + 1))


def test_duplicate_video_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate video_id"):
        SAMPLER.build_sampling_records(["L21_V001", "L21_V001"])


def test_empty_inventory_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SAMPLER.build_sampling_records([])


@pytest.mark.parametrize("video_id", ["", " L21_V001", "L21-V001", "L1_V001", 7])
def test_malformed_or_noncanonical_inventory_is_rejected(video_id: object) -> None:
    with pytest.raises(ValueError, match="canonical|noncanonical"):
        SAMPLER.build_sampling_records([video_id])


def test_csv_contract_contains_no_source_path(tmp_path: Path) -> None:
    destination = tmp_path / "candidate_video_manifest.csv"
    SAMPLER.write_sampling_manifest(SAMPLER.build_sampling_records(_inventory()), destination)
    with destination.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["sample_rank", "video_id", "selection_hash"]
        rows = list(reader)
    assert rows
    assert all(set(row) == set(SAMPLER.CSV_COLUMNS) for row in rows)
    text = destination.read_text(encoding="utf-8")
    assert "source" not in text.casefold()
    assert "kaggle" not in text.casefold()
    assert ":\\" not in text


def test_csv_bytes_are_deterministic(tmp_path: Path) -> None:
    rows = SAMPLER.build_sampling_records(_inventory())
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    SAMPLER.write_sampling_manifest(rows, first)
    SAMPLER.write_sampling_manifest(rows, second)
    assert first.read_bytes() == second.read_bytes()


def test_changing_input_order_does_not_change_output(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    SAMPLER.write_sampling_manifest(SAMPLER.build_sampling_records(_inventory()), first)
    SAMPLER.write_sampling_manifest(
        SAMPLER.build_sampling_records(list(reversed(_inventory()))), second
    )
    assert first.read_bytes() == second.read_bytes()


def test_sampling_seed_is_frozen() -> None:
    assert SAMPLER.SAMPLING_SEED == "system_tai_q1b_v1"


def test_sampler_does_not_mutate_annotation_plan(tmp_path: Path) -> None:
    annotation_plan = (
        Path(__file__).parents[1] / "benchmarks" / "quality_q1b" / "annotation_plan.csv"
    )
    before = annotation_plan.read_bytes()
    SAMPLER.write_sampling_manifest(
        SAMPLER.build_sampling_records(_inventory()), tmp_path / "candidate.csv"
    )
    assert annotation_plan.read_bytes() == before


def test_current_corpus_identity_gate_checks_all_accepted_fields() -> None:
    accepted = SimpleNamespace(
        videos=tuple(range(873)),
        total_rows=177_321,
        fingerprint=SAMPLER.CURRENT_Q1B_CORPUS_FINGERPRINT,
    )
    SAMPLER.validate_current_q1b_corpus(accepted)
    for field, value in (
        ("videos", tuple(range(872))),
        ("total_rows", 177_320),
        ("fingerprint", "0" * 64),
    ):
        invalid = SimpleNamespace(**accepted.__dict__)
        setattr(invalid, field, value)
        with pytest.raises(ValueError, match="identity mismatch"):
            SAMPLER.validate_current_q1b_corpus(invalid)


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "candidate.csv"
    rows = SAMPLER.build_sampling_records(_inventory())
    SAMPLER.write_sampling_manifest(rows, destination)
    with pytest.raises(FileExistsError, match="already exists"):
        SAMPLER.write_sampling_manifest(rows, destination)
    SAMPLER.write_sampling_manifest(rows, destination, overwrite=True)
