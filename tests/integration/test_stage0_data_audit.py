from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from triage_eg.data.stage0_audit.contracts import AUDIT_VERSION, AuditConfig
from triage_eg.data.stage0_audit.runner import run_audit
from triage_eg.data.stage0_audit.writers import FINAL_ARTIFACTS, create_bundle


def payload(count: int = 2):
    return {
        "detection_boxes": [["0.1", "0.2", "0.3", "0.4"] for _ in range(count)],
        "detection_class_entities": ["Person"] * count,
        "detection_class_labels": ["84"] * count,
        "detection_class_names": ["/m/person"] * count,
        "detection_scores": ["0.9"] * count,
    }


def metadata():
    return {
        name: ([] if name == "keywords" else "x")
        for name in (
            "author",
            "channel_id",
            "channel_url",
            "description",
            "keywords",
            "length",
            "publish_date",
            "thumbnail_url",
            "title",
            "watch_url",
        )
    }


def fixture(root: Path, ids=("L01_V002", "L01_V001")) -> None:
    for video_id in ids:
        video = root / "Videos_L01/video" / f"{video_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fake")
        mapping = root / "map-keyframes-aic25-b1/map-keyframes" / f"{video_id}.csv"
        mapping.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_text("n,pts_time,fps,frame_idx\n1,0.0,25,0\n2,0.04,25,1\n", encoding="utf-8")
        key = root / "keyframes/keyframes/Keyframes_L01/keyframes" / video_id
        obj = root / "objects-aic25-b1/objects" / video_id
        key.mkdir(parents=True, exist_ok=True)
        obj.mkdir(parents=True, exist_ok=True)
        for ordinal in (1, 2):
            (key / f"{ordinal:03d}.jpg").write_bytes(b"\xff\xd8x\xff\xd9")
            (obj / f"{ordinal:03d}.json").write_text(json.dumps(payload()), encoding="utf-8")
        clip = root / "clip-features-32-aic25-b1/clip-features-32" / f"{video_id}.npy"
        clip.parent.mkdir(parents=True, exist_ok=True)
        np.save(clip, np.ones((2, 4), dtype=np.float32))
        meta = root / "media-info-aic25-b1/media-info" / f"{video_id}.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(metadata()), encoding="utf-8")


def fake_probe(root, paths, *, timeout):
    return (
        {
            "audit_version": AUDIT_VERSION,
            "dataset_version": "aic25-b1",
            "video_id": paths.video_id,
            "video_partition": paths.video_partition,
            "relative_video_path": paths.video.relative_to(root).as_posix(),
            "file_size_bytes": paths.video.stat().st_size,
            "probe_status": "SUCCESS",
            "container_format": "mp4",
            "video_codec": "h264",
            "width": 640,
            "height": 480,
            "pixel_format": "yuv420p",
            "avg_frame_rate_raw": "25/1",
            "r_frame_rate_raw": "25/1",
            "avg_fps": 25.0,
            "r_fps": 25.0,
            "time_base": "1/1000",
            "start_time_seconds": 0.0,
            "duration_seconds": 1.0,
            "nb_frames": 10,
            "has_video_stream": True,
            "has_audio_stream": True,
            "audio_codec": "aac",
            "cfr_vfr_indicator": "CFR_LIKE",
            "issues": [],
        },
        [],
    )


def config(data: Path, output: Path, **kwargs) -> AuditConfig:
    defaults = {
        "mode": "sample",
        "sample_size": 2,
        "clip_validation": "full",
        "object_validation": "full",
        "expected_clip_dimension": 4,
    }
    defaults.update(kwargs)
    return AuditConfig(dataset_root=data, output_root=output, **defaults)


def test_sample_audit_artifacts_gates_resume_and_bundle(tmp_path: Path, monkeypatch) -> None:
    data, output = tmp_path / "data", tmp_path / "out"
    fixture(data)
    monkeypatch.setattr("triage_eg.data.stage0_audit.runner.probe_video", fake_probe)
    first = run_audit(config(data, output, overwrite=True))
    assert first.summary["gates"] == {"btc_baseline": "PASS", "raw_video": "PASS"}
    assert first.summary["videos_completed"] == 2
    assert all((output / name).is_file() for name in FINAL_ARTIFACTS)
    frames = [
        json.loads(line)
        for line in (output / "btc_frame_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["video_id"], item["n"]) for item in frames] == sorted(
        (item["video_id"], item["n"]) for item in frames
    )
    assert all(item["original_frame_idx"] == item["n"] - 1 for item in frames)
    resumed = run_audit(config(data, output, resume=True))
    assert resumed.summary["videos_resumed"] == 2
    bundle = create_bundle(output, tmp_path / "triage_eg_stage0_audit_bundle.zip")
    with ZipFile(bundle) as archive:
        assert archive.namelist() == list(FINAL_ARTIFACTS)
        assert not any(name.startswith(("checkpoints/", "logs/")) for name in archive.namelist())
        assert bundle.name not in archive.namelist()


def test_resume_rejects_asset_size_change(tmp_path: Path, monkeypatch) -> None:
    data, output = tmp_path / "data", tmp_path / "out"
    fixture(data, ids=("L01_V001",))
    monkeypatch.setattr("triage_eg.data.stage0_audit.runner.probe_video", fake_probe)
    run_audit(config(data, output, sample_size=1, overwrite=True))
    (data / "Videos_L01/video/L01_V001.mp4").write_bytes(b"changed")
    result = run_audit(config(data, output, sample_size=1, resume=True))
    assert result.summary["videos_resumed"] == 0
    assert result.summary["issues"]["by_code"]["CHECKPOINT_INVALID"] == 1


def test_btc_gate_passes_when_raw_probe_fails(tmp_path: Path, monkeypatch) -> None:
    data, output = tmp_path / "data", tmp_path / "out"
    fixture(data, ids=("L01_V001",))
    from triage_eg.data.stage0_audit.contracts import issue

    def failed(root, paths, *, timeout):
        record, _ = fake_probe(root, paths, timeout=timeout)
        record["probe_status"] = "FAILED"
        return record, [
            issue(
                "FFPROBE_NOT_AVAILABLE",
                "ERROR",
                video_id=paths.video_id,
                asset_type="VIDEO",
                raw=True,
            )
        ]

    monkeypatch.setattr("triage_eg.data.stage0_audit.runner.probe_video", failed)
    result = run_audit(config(data, output, sample_size=1, overwrite=True))
    assert result.summary["gates"]["btc_baseline"] == "PASS"
    assert result.summary["gates"]["raw_video"] == "FAIL"


def test_checkpoint_config_mismatch_is_explicit(tmp_path: Path, monkeypatch) -> None:
    data, output = tmp_path / "data", tmp_path / "out"
    fixture(data, ids=("L01_V001",))
    monkeypatch.setattr("triage_eg.data.stage0_audit.runner.probe_video", fake_probe)
    run_audit(config(data, output, sample_size=1, overwrite=True))
    result = run_audit(config(data, output, sample_size=1, resume=True, seed=7))
    assert result.summary["issues"]["by_code"]["CHECKPOINT_CONFIG_MISMATCH"] == 1


def test_cli_sample_resume_and_intentional_issue(tmp_path: Path) -> None:
    data, output = tmp_path / "data", tmp_path / "out"
    fixture(data, ids=("L01_V001",))
    command = [
        sys.executable,
        "scripts/run_stage0_data_audit.py",
        "--dataset-root",
        str(data),
        "--output-root",
        str(output),
        "--mode",
        "sample",
        "--sample-size",
        "1",
        "--clip-validation",
        "shape",
        "--object-validation",
        "filenames",
    ]
    first = subprocess.run([*command, "--overwrite"], capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert (output / "audit_summary.json").is_file()
    summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["issues"]["total"] > 0  # fake MP4 and dimension 4 are intentional
    second = subprocess.run([*command, "--resume"], capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    resumed = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert resumed["videos_resumed"] == 1
