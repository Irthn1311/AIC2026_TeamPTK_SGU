from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from triage_eg.retrieval.stage2.contracts import config_from_yaml, load_stage2_settings
from triage_eg.video import (
    HardwareConfig,
    OpenCVRawVideoDecoder,
    resolve_hardware,
    sampled_frame_indices,
)
from triage_eg.video.decoder import NvdecRawVideoDecoder, nvdec_preflight
from triage_eg.video.g1_audit import audit_indices, vector_parity


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake T4"


def fake_torch(cuda: bool) -> SimpleNamespace:
    return SimpleNamespace(cuda=FakeCuda(cuda), __version__="test")


def probe(available: bool) -> dict[str, object]:
    return {"available": available, "reason": None if available else "missing"}


def test_sampled_indices_include_final_without_duplicates() -> None:
    assert sampled_frame_indices(11, stride=4) == [0, 4, 8, 10]
    assert sampled_frame_indices(9, stride=4) == [0, 4, 8]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"total_frames": 0, "stride": 1}, ValueError),
        ({"total_frames": 5, "stride": 0}, ValueError),
        ({"total_frames": 5, "stride": 1, "start": -1}, IndexError),
    ],
)
def test_sampled_indices_reject_invalid_ranges(
    kwargs: dict[str, int], error: type[Exception]
) -> None:
    with pytest.raises(error):
        sampled_frame_indices(**kwargs)


def test_hardware_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        HardwareConfig(mode="magic")
    with pytest.raises(ValueError):
        HardwareConfig(video_backend="ffmpeg")


def test_cpu_mode_forces_cpu_and_opencv() -> None:
    result = resolve_hardware(
        HardwareConfig(mode="cpu", clip_device="cuda"),
        torch_module=fake_torch(True),
        nvdec_probe=probe(True),
    )
    assert (result.video_backend, result.clip_device, result.translator_device) == (
        "opencv",
        "cpu",
        "cpu",
    )
    assert result.cpu_fallback_ready is True


def test_auto_is_conservative_even_when_nvdec_exists() -> None:
    result = resolve_hardware(
        HardwareConfig(), torch_module=fake_torch(True), nvdec_probe=probe(True)
    )
    assert result.video_backend == "opencv"
    assert result.clip_device == result.translator_device == "cuda:0"
    assert "AUTO_NVDEC_NOT_PROMOTED" in result.reasons[0]


def test_auto_can_promote_nvdec_after_static_gate() -> None:
    result = resolve_hardware(
        HardwareConfig(auto_nvdec_promoted=True),
        torch_module=fake_torch(True),
        nvdec_probe=probe(True),
    )
    assert result.video_backend == "nvdec"


def test_auto_falls_back_fully_without_cuda_or_nvdec() -> None:
    result = resolve_hardware(
        HardwareConfig(), torch_module=fake_torch(False), nvdec_probe=probe(False)
    )
    assert result.video_backend == "opencv"
    assert result.clip_device == result.translator_device == "cpu"


def test_explicit_gpu_requires_cuda_but_not_nvdec() -> None:
    with pytest.raises(RuntimeError, match="CUDA_UNAVAILABLE"):
        resolve_hardware(
            HardwareConfig(mode="gpu"), torch_module=fake_torch(False), nvdec_probe=probe(False)
        )
    result = resolve_hardware(
        HardwareConfig(mode="gpu"), torch_module=fake_torch(True), nvdec_probe=probe(False)
    )
    assert result.video_backend == "opencv" and result.clip_device == "cuda:0"


def test_gpu_mode_requests_both_neural_components() -> None:
    result = resolve_hardware(
        HardwareConfig(mode="gpu", clip_device="cpu", translator_device="cpu"),
        torch_module=fake_torch(True),
        nvdec_probe=probe(False),
    )
    assert result.clip_device == result.translator_device == "cuda:0"


def test_explicit_nvdec_fails_clearly_when_missing() -> None:
    with pytest.raises(RuntimeError, match="NVDEC_REQUESTED_BUT_UNAVAILABLE"):
        resolve_hardware(
            HardwareConfig(video_backend="nvdec"),
            torch_module=fake_torch(True),
            nvdec_probe=probe(False),
        )


def test_nvdec_import_failure_is_safe() -> None:
    def loader(name: str):
        raise ImportError(name)

    result = nvdec_preflight(loader)
    assert result["available"] is False
    assert str(result["reason"]).startswith("PYNVVIDEOCODEC_UNAVAILABLE")


def test_nvdec_mock_sorts_deduplicates_and_preserves_integer_ids(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    class Metadata:
        average_fps = 25.0
        codec = "h264"

    class Decoder:
        def __init__(self, *args, **kwargs):
            self.requested = []

        def __len__(self):
            return 10

        def get_stream_metadata(self):
            return Metadata()

        def get_batch_frames_by_index(self, values):
            self.requested = values
            return [np.full((2, 2, 3), value, dtype=np.uint8) for value in values]

    nvc = SimpleNamespace(
        SimpleDecoder=Decoder, OutputColorType=SimpleNamespace(RGB="rgb"), __version__="test"
    )
    torch = fake_torch(True)
    torch.from_dlpack = lambda value: SimpleNamespace(
        cpu=lambda: SimpleNamespace(numpy=lambda: value)
    )
    modules = {"PyNvVideoCodec": nvc, "torch": torch}
    decoder = NvdecRawVideoDecoder("V", video, module_loader=modules.__getitem__)
    frames = decoder.decode_indices([4, 2, 4])
    assert [frame.actual_frame_idx for frame in frames] == [2, 4]
    assert decoder._decoder.requested == [2, 4]
    with pytest.raises(IndexError):
        decoder.decode_indices([10])


def test_opencv_sequential_scan_opens_and_seeks_once(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    class Capture:
        def __init__(self, path: str) -> None:
            self.position = 0
            self.set_calls: list[tuple[int, int]] = []

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            values = {1: 5.0, 2: 10.0, 3: 0.0, 4: float(self.position)}
            return values[prop]

        def set(self, prop: int, value: int) -> None:
            self.position = int(value)
            self.set_calls.append((prop, int(value)))

        def read(self):
            value = np.full((2, 2, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, value

        def release(self) -> None:
            return None

    capture = Capture(str(video))
    cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FOURCC=3,
        CAP_PROP_POS_FRAMES=4,
        COLOR_BGR2RGB=5,
        VideoCapture=lambda path: capture,
        cvtColor=lambda value, code: value[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    decoder = OpenCVRawVideoDecoder("V", video)
    frames = list(decoder.iter_sampled_frames(stride=3))
    assert [frame.actual_frame_idx for frame in frames] == [0, 3, 6, 9]
    assert capture.set_calls == [(cv2.CAP_PROP_POS_FRAMES, 0)]
    assert decoder.metrics.decoded_frame_count == 10


def test_audit_indices_cover_beginning_middle_and_end() -> None:
    values = audit_indices(101)
    assert values[:3] == [0, 1, 2]
    assert 50 in values and values[-1] == 100


def test_vector_parity_reports_ranking_and_numeric_difference() -> None:
    values = np.eye(3, 512, dtype=np.float32)
    result = vector_parity(values, values.copy(), top_k=3)
    assert result["status"] == "PASS"
    assert result["top_k_ranking_equal"] is True


def test_gpu_stage2_yaml_keeps_exact_numpy_backend(tmp_path: Path) -> None:
    source = Path("configs/retrieval/stage2_operational_runtime_gpu.yaml")
    settings = load_stage2_settings(source)
    assert settings["hardware"]["mode"] == "auto"
    assert settings["video"]["auto_nvdec_promoted"] is False
    config = config_from_yaml(
        source,
        stage1_root=tmp_path / "s1",
        stage1b_root=tmp_path / "s1b",
        stage1e_root=tmp_path / "s1e",
        clip_asset_root=tmp_path / "clip",
        translator_asset_root=tmp_path / "opus",
        output_root=tmp_path / "out",
        stage1d_config=tmp_path / "stage1d.yaml",
    )
    assert config.search_backend == "existing_stage1_exact"
    assert config.clip_device == config.translator_device == "auto"


def test_pynvvideocodec_is_not_a_core_dependency() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "PyNvVideoCodec" not in pyproject
    assert importlib.util.find_spec("triage_eg.video") is not None
