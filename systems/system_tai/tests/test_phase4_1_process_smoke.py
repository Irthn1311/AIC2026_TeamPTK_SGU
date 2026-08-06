from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from system_tai.data.corpus_discovery import load_corpus_manifest
from system_tai.refinement.video import RawVideoRegistry
from tests.phase3_helpers import create_corpus, feature_matrix
from tests.test_phase4_process_smoke import FAKE_CLIP, FAKE_TORCH


def _run(command: list[str], *, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )
    print("PROCESS_COMMAND:", subprocess.list2cmdline(command))
    print("PROCESS_STDOUT:", completed.stdout.strip())
    print("PROCESS_STDERR:", completed.stderr.strip())
    assert completed.returncode == 0, completed.stderr
    return completed


def test_portable_build_rebase_cache_hit_and_phase4_raw_path(tmp_path: Path) -> None:
    first_input = tmp_path / "runtime-a" / "input"
    first_dataset = create_corpus(
        first_input,
        {
            "L21_V001": (
                [10, 20],
                feature_matrix([(0, 1.0), (1, 1.0)]),
            ),
            "L21_V002": (
                [30, 40],
                feature_matrix([(1, 1.0), (0, 1.0)]),
            ),
        },
    )
    source_root = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root), environment.get("PYTHONPATH", ""))
    )
    portable = tmp_path / "persistent" / "feature_manifest.json"
    strict_build = [
        sys.executable,
        "-m",
        "system_tai.kis.build_manifest",
        "--input-root",
        str(first_input),
        "--output",
        str(portable),
        "--portable",
        "--discovery-validation",
        "strict",
    ]
    first = _run(strict_build, environment=environment)
    first_summary = json.loads(first.stdout.strip().splitlines()[-1])
    assert first_summary["status"] == "BUILT"
    assert first_summary["portable"] is True
    assert first_summary["video_count"] == 2
    assert first_summary["feature_row_count"] == 4
    portable_payload = json.loads(portable.read_text(encoding="utf-8"))
    assert str(first_input.resolve()) not in portable.read_text(encoding="utf-8")

    second_input = tmp_path / "runtime-b" / "input"
    second_dataset = second_input / "datasets" / "new-owner" / first_dataset.name
    second_dataset.parent.mkdir(parents=True)
    shutil.copytree(first_dataset, second_dataset)
    shutil.rmtree(first_input.parent)

    runtime_manifest = tmp_path / "runtime_manifest.json"
    rebase = [
        sys.executable,
        "-m",
        "system_tai.kis.build_manifest",
        "--input-root",
        str(second_input),
        "--reuse-manifest",
        str(portable),
        "--output",
        str(runtime_manifest),
    ]
    _run(rebase, environment=environment)
    rebased = load_corpus_manifest(runtime_manifest)
    assert [video.video_id for video in rebased.videos] == ["L21_V001", "L21_V002"]
    assert rebased.total_rows == 4
    assert all(video.mapping_csv_path.is_relative_to(second_dataset) for video in rebased.videos)
    assert [
        item["mapping_csv_path"] for item in portable_payload["videos"]
    ] == [
        video.mapping_csv_path.relative_to(second_dataset).as_posix()
        for video in rebased.videos
    ]
    raw_registry = RawVideoRegistry.from_manifest(rebased)
    assert raw_registry.get("L21_V001").raw_video_path.is_relative_to(second_dataset)

    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
    (fake_modules / "clip.py").write_text(FAKE_CLIP, encoding="utf-8")
    cache_dir = tmp_path / "clip-cache"
    cache_dir.mkdir()
    (cache_dir / "ViT-B-32.pt").touch()
    marker = tmp_path / "model-loads.txt"
    contest_environment = dict(environment)
    contest_environment["PYTHONPATH"] = os.pathsep.join(
        (str(fake_modules), str(source_root), environment.get("PYTHONPATH", ""))
    )
    contest_environment["FAKE_MODEL_LOAD_MARKER"] = str(marker)
    output = tmp_path / "contest"
    contest = [
        sys.executable,
        "-m",
        "system_tai.kis.contest",
        "--input-root",
        str(second_input),
        "--manifest-cache",
        str(portable),
        "--query-id",
        "Q1",
        "--query-vi",
        "target",
        "--output-directory",
        str(output),
        "--device",
        "cpu",
        "--clip-cache-dir",
        str(cache_dir),
        "--fast-contest-mode",
    ]
    contest_run = _run(contest, environment=contest_environment)
    assert "manifest cache: CACHE_HIT" in contest_run.stdout
    validation = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["valid"] is True
    timing = json.loads((output / "timings.json").read_text(encoding="utf-8"))
    assert timing["manifest_source_status"] == "CACHE_HIT"
    assert timing["family_index_seconds"] == 0
    assert marker.read_text(encoding="utf-8").splitlines() == ["load"]
    assert not any(
        path.suffix.casefold() in {".mp4", ".avi", ".mkv", ".mov", ".webm", ".npy", ".jpg", ".png"}
        for path in output.rglob("*")
        if path.is_file()
    )
