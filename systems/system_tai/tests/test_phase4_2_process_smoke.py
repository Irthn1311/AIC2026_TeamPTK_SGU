"""Process-level synthetic smoke test for Phase 4.2 operational session CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from system_tai.data.corpus_discovery import CorpusManifest, DiscoveredVideo, _fingerprint
from system_tai.features.btc_clip_store import FeatureStoreRegistry
from system_tai.validation.checkpoint_validator import CheckpointValidator
from tests.test_phase4_process_smoke import FAKE_CLIP, FAKE_CV2, FAKE_TORCH


def make_synthetic_corpus(tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "dataset"
    input_root.mkdir(parents=True, exist_ok=True)
    mapping = input_root / "L21_V001.csv"
    mapping.write_text("n,pts_time,fps,frame_idx\n1,5,10,50\n2,6,10,60\n", encoding="utf-8")
    clip = input_root / "L21_V001.npy"
    import numpy as np
    arr = np.zeros((2, 512), dtype=np.float32)
    arr[0, 0] = 1.0
    arr[1, 0] = 0.8
    np.save(str(clip), arr)
    keyframes = input_root / "keyframes"
    keyframes.mkdir(exist_ok=True)
    (keyframes / "1.jpg").touch()
    video = input_root / "L21_V001.mp4"
    video.touch()

    discovered = DiscoveredVideo(
        "L21_V001",
        mapping,
        clip,
        keyframes,
        video,
        1,
        2,
        mapping.stat().st_size,
        clip.stat().st_size,
        2,
    )
    manifest = CorpusManifest(input_root, input_root, _fingerprint((discovered,)), (discovered,))
    manifest_path = tmp_path / "feature_manifest.json"
    manifest.write(manifest_path)
    return input_root, manifest_path


def test_operational_session_subprocess_smoke(tmp_path: Path) -> None:
    fake_modules = tmp_path / "fake_modules"
    fake_modules.mkdir(parents=True, exist_ok=True)
    (fake_modules / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
    (fake_modules / "clip.py").write_text(FAKE_CLIP, encoding="utf-8")
    (fake_modules / "cv2.py").write_text(FAKE_CV2, encoding="utf-8")
    marker = tmp_path / "model_loads.txt"

    source_root = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(fake_modules), str(source_root), environment.get("PYTHONPATH", ""))
    )
    environment["FAKE_MODEL_LOAD_MARKER"] = str(marker)

    cache_dir = tmp_path / "clip-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "ViT-B-32.pt").touch()

    input_root, manifest_path = make_synthetic_corpus(tmp_path)
    output_root = tmp_path / "session_output"

    cmd = [
        sys.executable,
        "-m",
        "system_tai.kis.session",
        "--reuse-manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--clip-cache-dir",
        str(cache_dir),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )

    requests = [
        # 1. Health request
        json.dumps({"type": "health", "request_id": "req-health-1"}),
        # 2. Retrieval-only query
        json.dumps({
            "type": "query",
            "request_id": "req-q1",
            "query_id": "Q1",
            "query_vi": "tim kiem canh quay test 1",
            "refine_top_n": 0,
        }),
        # 3. Refinement query (refine_top_n=2)
        json.dumps({
            "type": "query",
            "request_id": "req-q2",
            "query_id": "Q2",
            "query_vi": "tim kiem canh quay test 2",
            "refine_top_n": 2,
        }),
        # 4. Malformed JSON line
        "INVALID_JSON_LINE_TEST_MALFORMED",
        # 5. Next valid query
        json.dumps({
            "type": "query",
            "request_id": "req-q3",
            "query_id": "Q3",
            "query_vi": "tim kiem canh quay test 3",
            "refine_top_n": 0,
        }),
        # 6. Shutdown request
        json.dumps({"type": "shutdown", "request_id": "req-shutdown-1"}),
    ]

    stdin_payload = "\n".join(requests) + "\n"
    stdout_data, stderr_data = proc.communicate(input=stdin_payload, timeout=30)

    assert proc.returncode == 0, f"Process exited non-zero ({proc.returncode}): {stderr_data}"

    stdout_lines = [line.strip() for line in stdout_data.splitlines() if line.strip()]
    assert len(stdout_lines) == 6, (
        f"Expected 6 output lines, got {len(stdout_lines)}: {stdout_lines}"
    )

    responses = [json.loads(line) for line in stdout_lines]

    # Assert line 1: health
    assert responses[0]["type"] == "health"
    assert responses[0]["request_id"] == "req-health-1"
    assert responses[0]["status"] == "READY"

    # Assert line 2: query retrieval-only
    assert responses[1]["type"] == "query_result"
    assert responses[1]["request_id"] == "req-q1"
    assert responses[1]["status"] == "SUCCESS"
    assert responses[1]["refinement_requested"] is False

    # Assert line 3: query refinement
    assert responses[2]["type"] == "query_result"
    assert responses[2]["request_id"] == "req-q2"
    assert responses[2]["status"] == "SUCCESS"
    assert responses[2]["refinement_requested"] is True

    # Assert line 4: malformed json
    assert responses[3]["type"] == "error"
    assert responses[3]["error_code"] == "MALFORMED_JSON"
    assert responses[3]["session_continues"] is True

    # Assert line 5: query post error
    assert responses[4]["type"] == "query_result"
    assert responses[4]["request_id"] == "req-q3"
    assert responses[4]["status"] == "SUCCESS"

    # Assert line 6: shutdown
    assert responses[5]["type"] == "shutdown"
    assert responses[5]["request_id"] == "req-shutdown-1"
    assert responses[5]["status"] == "STOPPING"
    assert responses[5]["processed_requests"] == 6

    # Verify session_manifest.json
    manifest_path_session = output_root / "session_manifest.json"
    assert manifest_path_session.is_file()
    session_manifest = json.loads(manifest_path_session.read_text(encoding="utf-8"))
    assert session_manifest["model_load_count"] == 1
    assert session_manifest["registry_load_count"] == 1
    assert session_manifest["request_count"] == 6
    assert session_manifest["successful_query_count"] == 3
    assert session_manifest["health_request_count"] == 1
    assert session_manifest["malformed_request_count"] == 1

    # Verify checkpoint outputs with CheckpointValidator
    validator = CheckpointValidator()
    registry = FeatureStoreRegistry.from_manifest(manifest_path, expected_dimension=512)

    top100_q1 = output_root / responses[1]["artifacts"]["top100_jsonl"]
    assert top100_q1.is_file()
    assert validator.validate(top100_q1, registry=registry).valid

    refined_q2 = output_root / responses[2]["artifacts"]["refined_top100_jsonl"]
    assert refined_q2.is_file()
    assert validator.validate(refined_q2).valid
