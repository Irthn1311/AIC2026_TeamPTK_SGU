"""Centralized conservative hardware selection for TRIAGE-EG."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

from .decoder import nvdec_preflight


@dataclass(frozen=True)
class HardwareConfig:
    mode: str = "auto"
    video_backend: str = "auto"
    clip_device: str = "auto"
    translator_device: str = "auto"
    auto_clip_promoted: bool = False
    auto_translator_promoted: bool = True
    auto_nvdec_promoted: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "cpu", "gpu"}:
            raise ValueError("hardware mode must be auto, cpu, or gpu")
        if self.video_backend not in {"auto", "opencv", "nvdec"}:
            raise ValueError("video backend must be auto, opencv, or nvdec")
        for value in (self.clip_device, self.translator_device):
            if value not in {"auto", "cpu", "cuda", "cuda:0"}:
                raise ValueError("neural device must be auto, cpu, cuda, or cuda:0")


@dataclass(frozen=True)
class EffectiveHardware:
    mode: str
    video_backend: str
    clip_device: str
    translator_device: str
    cuda_available: bool
    nvdec_available: bool
    cpu_fallback_ready: bool
    gpu_name: str | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_hardware(
    config: HardwareConfig,
    *,
    torch_module: Any | None = None,
    nvdec_probe: dict[str, Any] | None = None,
) -> EffectiveHardware:
    reasons: list[str] = []
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except (ImportError, OSError):
            torch_module = None
    cuda = bool(torch_module is not None and torch_module.cuda.is_available())
    gpu_name = torch_module.cuda.get_device_name(0) if cuda else None
    nvdec = nvdec_probe if nvdec_probe is not None else nvdec_preflight()
    nvdec_available = bool(nvdec.get("available"))

    if config.mode == "cpu":
        return EffectiveHardware(
            "cpu", "opencv", "cpu", "cpu", cuda, nvdec_available, True, gpu_name, ()
        )
    if config.mode == "gpu" and not cuda:
        raise RuntimeError("GPU_MODE_REQUESTED_BUT_CUDA_UNAVAILABLE")

    def neural(requested: str, component: str, promoted: bool) -> str:
        if config.mode == "gpu":
            return "cuda:0"
        if requested == "cpu":
            return "cpu"
        if requested.startswith("cuda"):
            if not cuda:
                raise RuntimeError(f"{component.upper()}_CUDA_REQUESTED_BUT_UNAVAILABLE")
            return "cuda:0"
        if cuda and promoted:
            return "cuda:0"
        if cuda and not promoted:
            reasons.append(f"AUTO_{component.upper()}_NOT_PROMOTED")
        return "cpu"

    clip = neural(config.clip_device, "clip", config.auto_clip_promoted)
    translator = neural(
        config.translator_device, "translator", config.auto_translator_promoted
    )
    if config.video_backend == "opencv":
        video = "opencv"
    elif config.video_backend == "nvdec":
        if not nvdec_available:
            raise RuntimeError(f"NVDEC_REQUESTED_BUT_UNAVAILABLE: {nvdec.get('reason')}")
        video = "nvdec"
    elif config.mode == "gpu":
        video = "opencv"
        reasons.append("GPU_MODE_VIDEO_AUTO_REMAINS_OPENCV_UNLESS_NVDEC_EXPLICIT")
    elif config.auto_nvdec_promoted and nvdec_available:
        video = "nvdec"
    else:
        video = "opencv"
        reasons.append("AUTO_NVDEC_NOT_PROMOTED_WITHOUT_REAL_PARITY_AND_SPEED_GATE")
    if not cuda:
        reasons.append("CUDA_UNAVAILABLE_CPU_NEURAL_FALLBACK")
    return EffectiveHardware(
        config.mode,
        video,
        clip,
        translator,
        cuda,
        nvdec_available,
        True,
        gpu_name,
        tuple(reasons),
    )


__all__ = ["EffectiveHardware", "HardwareConfig", "resolve_hardware"]
