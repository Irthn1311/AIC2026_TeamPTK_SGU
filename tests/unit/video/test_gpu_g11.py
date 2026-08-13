from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.experiments.mb1_v021.signals import COARSE_SAMPLES_PER_SECOND
from triage_eg.video import HardwareConfig, resolve_hardware
from triage_eg.video.g11_audit import (
    G11_BUNDLE_FILES,
    benchmark_m1_local_workload,
    clip_gpu_verdict,
    consumer_specific_nvdec_verdicts,
    load_frozen_query_suite,
    retrieval_agreement,
    write_g11_bundle,
)


class Cuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        return "T4"


def torch(available: bool) -> SimpleNamespace:
    return SimpleNamespace(cuda=Cuda(available))


def nvdec(available: bool) -> dict[str, object]:
    return {"available": available, "reason": "missing" if not available else None}


def test_auto_respects_independent_initial_promotions() -> None:
    selected = resolve_hardware(
        HardwareConfig(), torch_module=torch(True), nvdec_probe=nvdec(True)
    )
    assert selected.clip_device == "cpu"
    assert selected.translator_device == "cuda:0"
    assert selected.video_backend == "opencv"


def test_auto_clip_can_be_promoted_independently() -> None:
    selected = resolve_hardware(
        HardwareConfig(auto_clip_promoted=True, auto_translator_promoted=False),
        torch_module=torch(True),
        nvdec_probe=nvdec(False),
    )
    assert selected.clip_device == "cuda:0"
    assert selected.translator_device == "cpu"


def test_cpu_mode_remains_all_cpu_opencv() -> None:
    selected = resolve_hardware(
        HardwareConfig(
            mode="cpu",
            auto_clip_promoted=True,
            auto_translator_promoted=True,
            auto_nvdec_promoted=True,
        ),
        torch_module=torch(True),
        nvdec_probe=nvdec(True),
    )
    assert (selected.video_backend, selected.clip_device, selected.translator_device) == (
        "opencv",
        "cpu",
        "cpu",
    )


def test_explicit_unavailable_cuda_and_nvdec_fail() -> None:
    with pytest.raises(RuntimeError, match="CLIP_CUDA_REQUESTED_BUT_UNAVAILABLE"):
        resolve_hardware(
            HardwareConfig(clip_device="cuda"),
            torch_module=torch(False),
            nvdec_probe=nvdec(False),
        )
    with pytest.raises(RuntimeError, match="NVDEC_REQUESTED"):
        resolve_hardware(
            HardwareConfig(video_backend="nvdec"),
            torch_module=torch(True),
            nvdec_probe=nvdec(False),
        )


def test_retrieval_agreement_does_not_require_exact_top50_order() -> None:
    cpu = np.eye(2, dtype=np.float32)
    gpu = cpu.copy()
    index = np.asarray([[1.0, 0], [0.99, 0], [0, 1.0], [0, 0.99]], dtype=np.float32)
    result = retrieval_agreement(cpu, gpu, index, top_k=4)
    assert result["status"] == "PASS"
    assert result["exact_top50_order_is_hard_gate"] is False


def test_rank_overlap_and_displacement_metrics(monkeypatch) -> None:
    from triage_eg.video import g11_audit

    rankings = iter([[[1, 2, 3, 4]], [[1, 3, 2, 4]]])
    monkeypatch.setattr(g11_audit, "_exact_rankings", lambda *args, **kwargs: next(rankings))
    result = retrieval_agreement(
        np.ones((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.ones((4, 2), dtype=np.float32),
        top_k=4,
    )
    assert result["top1_changes"] == 0
    assert result["mean_rank_displacement"] == 0.5
    assert result["maximum_rank_displacement"] == 1


def test_clip_verdict_accepts_set_parity_without_order_identity() -> None:
    assert clip_gpu_verdict({"status": "PASS"}, {"status": "PASS"}) == "KEEP"
    assert clip_gpu_verdict({"status": "PASS"}, {"status": "CONDITIONAL"}) == "CONDITIONAL"


def test_consumer_specific_nvdec_verdicts_do_not_force_mb1_promotion() -> None:
    result = consumer_specific_nvdec_verdicts(
        {"status": "NOT_PROMOTED"},
        {"status": "PASS", "rows": [{"frame_identity": True, "combined_speedup": 2.0}]},
        {"status": "PASS"},
        nvdec_available=True,
    )
    assert result == {"NVDEC_MB1": "NOT_PROMOTED", "NVDEC_NEURAL": "KEEP"}
    incomplete = consumer_specific_nvdec_verdicts(
        {"status": "NOT_PROMOTED"},
        {
            "status": "PASS",
            "failed_video_count": 1,
            "rows": [{"frame_identity": True, "combined_speedup": 2.0}],
        },
        {"status": "PASS"},
        nvdec_available=True,
    )
    assert incomplete["NVDEC_NEURAL"] == "OPTIONAL"


def test_mb1_sample_rate_is_the_frozen_current_constant() -> None:
    assert COARSE_SAMPLES_PER_SECOND == 10.0


def test_m1_benchmark_requests_are_local(monkeypatch, tmp_path: Path) -> None:
    from triage_eg.video import g11_audit

    requested: list[list[int]] = []

    class Decoder:
        info = SimpleNamespace(total_frames=1001, fps=25.0)

        def __init__(self, video_id: str, path: Path) -> None:
            pass

        def decode_indices(self, indices: list[int]):
            requested.append(indices)
            return [
                SimpleNamespace(
                    actual_frame_idx=value,
                    image=np.zeros((2, 2, 3), np.uint8),
                )
                for value in indices
            ]

        def close(self):
            pass

    monkeypatch.setattr(g11_audit, "OpenCVRawVideoDecoder", Decoder)
    path = tmp_path / "V.mp4"
    path.write_bytes(b"x")
    benchmark_m1_local_workload([path], nvdec_available=False)
    assert min(requested[0] + requested[1]) >= 350
    assert max(requested[0] + requested[1]) <= 650
    assert len(requested[1]) == 31
    monkeypatch.setattr(
        g11_audit,
        "create_raw_video_decoder",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("NVDEC_INDEXED_DECODE_FAILED")
        ),
    )
    benchmark, parity, cpu_images, gpu_images = benchmark_m1_local_workload(
        [path], nvdec_available=True
    )
    assert parity["failed_video_count"] == 1
    assert benchmark["issues"][0]["code"] == "NVDEC_M1_LOCAL_DECODE_FAILED"
    assert cpu_images and not gpu_images


def test_frozen_query_loader_reaches_minimum_without_inventing_labels(tmp_path: Path) -> None:
    stage1c = tmp_path / "stage1c.jsonl"
    stage1c.write_text(
        "".join(
            json.dumps(
                {"query_id": f"q{i}", "text": f"text {i}", "language": "en"}
            )
            + "\n"
            for i in range(28)
        ),
        encoding="utf-8",
    )
    rt2 = tmp_path / "rt2.jsonl"
    rt2.write_text(
        json.dumps(
            {
                "query_id": "r",
                "language": "en",
                "events": [
                    {"event_id": f"E{i}", "text": f"event {i}"}
                    for i in range(4)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_frozen_query_suite(stage1c, rt2)
    assert len(rows) == 32 and {row["source"] for row in rows} == {"stage1c", "rt2"}


def test_bundle_is_allowlisted_and_excludes_cache(tmp_path: Path) -> None:
    root = tmp_path / "g11"
    (root / "runtime_cache/__pycache__").mkdir(parents=True)
    (root / "runtime_cache/__pycache__/bad.pyc").write_bytes(b"bad")
    artifacts = {name: ([] if name.endswith(".jsonl") else {}) for name in G11_BUNDLE_FILES}
    archive = write_g11_bundle(root, artifacts)
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert set(names) == set(G11_BUNDLE_FILES) | {"README.md"}
    assert not any("runtime_cache" in name or "__pycache__" in name for name in names)
