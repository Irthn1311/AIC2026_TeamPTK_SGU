"""Shared raw-video and hardware runtime contracts."""

from .decoder import (
    DecodedFrame,
    DecoderMetrics,
    NvdecRawVideoDecoder,
    OpenCVRawVideoDecoder,
    RawVideoDecoder,
    VideoInfo,
    create_raw_video_decoder,
    nvdec_preflight,
    sampled_frame_indices,
)
from .hardware import EffectiveHardware, HardwareConfig, resolve_hardware

__all__ = [
    "DecodedFrame",
    "DecoderMetrics",
    "EffectiveHardware",
    "HardwareConfig",
    "NvdecRawVideoDecoder",
    "OpenCVRawVideoDecoder",
    "RawVideoDecoder",
    "VideoInfo",
    "create_raw_video_decoder",
    "nvdec_preflight",
    "resolve_hardware",
    "sampled_frame_indices",
]
