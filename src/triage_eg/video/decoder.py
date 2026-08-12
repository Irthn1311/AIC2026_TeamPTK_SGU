"""Hardware-aware raw-video decoders with exact integer frame identities."""

from __future__ import annotations

import importlib
import math
import platform
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    total_frames: int
    codec: str | None = None


@dataclass(frozen=True)
class DecodedFrame:
    video_id: str
    actual_frame_idx: int
    image: np.ndarray


@dataclass
class DecoderMetrics:
    backend: str
    init_ms: float = 0.0
    sequential_decode_ms: float = 0.0
    indexed_decode_ms: float = 0.0
    decoded_frame_count: int = 0
    retained_frame_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        elapsed = self.sequential_decode_ms + self.indexed_decode_ms
        value["effective_decoded_fps"] = (
            1000.0 * self.decoded_frame_count / elapsed if elapsed > 0 else None
        )
        return value


class RawVideoDecoder(Protocol):
    info: VideoInfo

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]: ...

    def iter_sampled_frames(
        self, *, stride: int, start: int = 0, end: int | None = None, include_final: bool = True
    ) -> Iterator[DecodedFrame]: ...

    def runtime_manifest(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def sampled_frame_indices(
    total_frames: int,
    *,
    stride: int,
    start: int = 0,
    end: int | None = None,
    include_final: bool = True,
) -> list[int]:
    if total_frames <= 0 or stride <= 0:
        raise ValueError("video length and sample stride must be positive")
    final = total_frames - 1 if end is None else int(end)
    if not 0 <= start <= final < total_frames:
        raise IndexError("sample range is outside the video")
    values = list(range(int(start), final + 1, int(stride)))
    if include_final and values[-1] != final:
        values.append(final)
    return values


class OpenCVRawVideoDecoder:
    """Canonical CPU decoder; sequential scans open once and never seek per sample batch."""

    def __init__(self, video_id: str, video_path: Path) -> None:
        started = monotonic()
        try:
            import cv2
        except ImportError as error:
            raise ImportError("OpenCV raw-video decoding requires cv2") from error
        self.video_id = str(video_id)
        self.video_path = Path(video_path).expanduser().resolve(strict=True)
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"RAW_VIDEO_OPEN_FAILED: {self.video_path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(round(float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        codec_value = int(round(float(self._capture.get(cv2.CAP_PROP_FOURCC))))
        codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4)).strip()
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            self.close()
            raise RuntimeError(f"RAW_VIDEO_METADATA_INVALID: {self.video_path}")
        self.info = VideoInfo(fps=fps, total_frames=total_frames, codec=codec or None)
        self.metrics = DecoderMetrics("opencv", init_ms=(monotonic() - started) * 1000)

    def _read_expected(self, expected: int) -> np.ndarray:
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise RuntimeError(f"RAW_FRAME_DECODE_FAILED: {self.video_id} frame={expected}")
        actual = int(round(float(self._capture.get(self._cv2.CAP_PROP_POS_FRAMES)))) - 1
        if actual != expected:
            raise RuntimeError(
                f"RAW_FRAME_COORDINATE_MISMATCH: requested={expected} actual={actual}"
            )
        return bgr

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        requested = sorted(set(int(value) for value in frame_indices))
        if not requested:
            return []
        if requested[0] < 0 or requested[-1] >= self.info.total_frames:
            raise IndexError("raw frame index is outside the video")
        requested_set = set(requested)
        started = monotonic()
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, requested[0])
        frames = []
        for expected in range(requested[0], requested[-1] + 1):
            bgr = self._read_expected(expected)
            self.metrics.decoded_frame_count += 1
            if expected in requested_set:
                rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                frames.append(DecodedFrame(self.video_id, expected, rgb))
        self.metrics.indexed_decode_ms += (monotonic() - started) * 1000
        self.metrics.retained_frame_count += len(frames)
        if [frame.actual_frame_idx for frame in frames] != requested:
            raise RuntimeError("RAW_FRAME_DECODE_INCOMPLETE")
        return frames

    def iter_sampled_frames(
        self, *, stride: int, start: int = 0, end: int | None = None, include_final: bool = True
    ) -> Iterator[DecodedFrame]:
        requested = sampled_frame_indices(
            self.info.total_frames,
            stride=stride,
            start=start,
            end=end,
            include_final=include_final,
        )
        requested_set = set(requested)
        final = requested[-1]
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, start)
        returned = []
        for expected in range(start, final + 1):
            started = monotonic()
            try:
                bgr = self._read_expected(expected)
                self.metrics.decoded_frame_count += 1
                if expected in requested_set:
                    returned.append(expected)
                    self.metrics.retained_frame_count += 1
                    rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                else:
                    rgb = None
            finally:
                self.metrics.sequential_decode_ms += (monotonic() - started) * 1000
            if rgb is not None:
                yield DecodedFrame(
                    self.video_id,
                    expected,
                    rgb,
                )
        if returned != requested:
            raise RuntimeError("RAW_SEQUENTIAL_FRAME_DECODE_INCOMPLETE")

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            **self.metrics.as_dict(),
            "video_id": self.video_id,
            "video_path": str(self.video_path),
            "fps": self.info.fps,
            "total_frames": self.info.total_frames,
            "codec": self.info.codec,
            "frame_identity_policy": "CAP_PROP_POS_FRAMES_EXACT_INTEGER",
        }

    def close(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()


ModuleLoader = Callable[[str], Any]


def nvdec_preflight(module_loader: ModuleLoader = importlib.import_module) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "python_version": platform.python_version(),
        "selected_decoder": None,
        "reason": None,
    }
    try:
        nvc = module_loader("PyNvVideoCodec")
    except (ImportError, OSError) as error:
        result["reason"] = f"PYNVVIDEOCODEC_UNAVAILABLE: {error}"
        return result
    try:
        torch = module_loader("torch")
    except (ImportError, OSError) as error:
        result["reason"] = f"TORCH_UNAVAILABLE: {error}"
        return result
    cuda_available = bool(torch.cuda.is_available())
    result.update(
        {
            "pynvvideocodec_version": str(getattr(nvc, "__version__", "UNKNOWN")),
            "pynvvideocodec_cuda_version": str(getattr(nvc, "__cuda_version__", "UNKNOWN")),
            "video_codec_sdk_version": str(getattr(nvc, "__video_codec_sdk_version__", "UNKNOWN")),
            "torch_version": str(getattr(torch, "__version__", "UNKNOWN")),
            "cuda_available": cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "driver_version": (
                str(torch.cuda.driver_version())
                if cuda_available and hasattr(torch.cuda, "driver_version")
                else None
            ),
        }
    )
    if not cuda_available:
        result["reason"] = "CUDA_UNAVAILABLE"
        return result
    if not hasattr(nvc, "SimpleDecoder") or not hasattr(nvc, "OutputColorType"):
        result["reason"] = "PYNVVIDEOCODEC_SIMPLE_DECODER_API_UNAVAILABLE"
        return result
    result.update({"available": True, "selected_decoder": "PyNvVideoCodec.SimpleDecoder"})
    return result


class NvdecRawVideoDecoder:
    """Optional PyNvVideoCodec index-exact decoder; imported only when requested."""

    def __init__(
        self,
        video_id: str,
        video_path: Path,
        *,
        module_loader: ModuleLoader = importlib.import_module,
    ) -> None:
        started = monotonic()
        preflight = nvdec_preflight(module_loader)
        if not preflight["available"]:
            raise RuntimeError(f"NVDEC_UNAVAILABLE: {preflight['reason']}")
        self.video_id = str(video_id)
        self.video_path = Path(video_path).expanduser().resolve(strict=True)
        self._nvc = module_loader("PyNvVideoCodec")
        self._torch = module_loader("torch")
        try:
            self._decoder = self._nvc.SimpleDecoder(
                str(self.video_path),
                gpu_id=0,
                use_device_memory=True,
                need_scanned_stream_metadata=True,
                output_color_type=self._nvc.OutputColorType.RGB,
            )
            metadata = self._decoder.get_stream_metadata()
            fps = _metadata_float(metadata, "average_fps", "avg_frame_rate", "frame_rate")
            total_frames = int(len(self._decoder))
            codec = str(getattr(metadata, "codec", "") or "") or None
        except Exception as error:
            raise RuntimeError(f"NVDEC_INITIALIZATION_FAILED: {error}") from error
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            raise RuntimeError("NVDEC_METADATA_INVALID")
        self.info = VideoInfo(fps=fps, total_frames=total_frames, codec=codec)
        self.preflight = preflight
        self.metrics = DecoderMetrics("nvdec", init_ms=(monotonic() - started) * 1000)

    def _to_rgb(self, frame: Any) -> np.ndarray:
        tensor = self._torch.from_dlpack(frame)
        array = tensor.cpu().numpy()
        if array.ndim != 3:
            raise RuntimeError(f"NVDEC_FRAME_SHAPE_INVALID: {array.shape}")
        if array.shape[-1] != 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] != 3:
            raise RuntimeError(f"NVDEC_FRAME_LAYOUT_INVALID: {array.shape}")
        return np.ascontiguousarray(array, dtype=np.uint8)

    def decode_indices(self, frame_indices: list[int]) -> list[DecodedFrame]:
        requested = sorted(set(int(value) for value in frame_indices))
        if not requested:
            return []
        if requested[0] < 0 or requested[-1] >= self.info.total_frames:
            raise IndexError("raw frame index is outside the video")
        started = monotonic()
        try:
            raw = list(self._decoder.get_batch_frames_by_index(requested))
        except Exception as error:
            raise RuntimeError(f"NVDEC_INDEXED_DECODE_FAILED: {error}") from error
        if len(raw) != len(requested):
            raise RuntimeError("NVDEC_FRAME_COUNT_MISMATCH")
        frames = [
            DecodedFrame(self.video_id, frame_index, self._to_rgb(value))
            for frame_index, value in zip(requested, raw, strict=True)
        ]
        self.metrics.indexed_decode_ms += (monotonic() - started) * 1000
        self.metrics.decoded_frame_count += len(frames)
        self.metrics.retained_frame_count += len(frames)
        return frames

    def iter_sampled_frames(
        self, *, stride: int, start: int = 0, end: int | None = None, include_final: bool = True
    ) -> Iterator[DecodedFrame]:
        requested = sampled_frame_indices(
            self.info.total_frames,
            stride=stride,
            start=start,
            end=end,
            include_final=include_final,
        )
        for start_index in range(0, len(requested), 128):
            batch = requested[start_index : start_index + 128]
            started = monotonic()
            indexed_before = self.metrics.indexed_decode_ms
            frames = self.decode_indices(batch)
            elapsed_ms = (monotonic() - started) * 1000
            self.metrics.indexed_decode_ms = indexed_before
            self.metrics.sequential_decode_ms += elapsed_ms
            yield from frames

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            **self.metrics.as_dict(),
            **self.preflight,
            "video_id": self.video_id,
            "video_path": str(self.video_path),
            "fps": self.info.fps,
            "total_frames": self.info.total_frames,
            "codec": self.info.codec,
            "frame_identity_policy": "PYNVVIDEOCODEC_INTEGER_INDEX_API_REQUIRES_PARITY_GATE",
        }

    def close(self) -> None:
        self._decoder = None


def _metadata_float(metadata: Any, *names: str) -> float:
    for name in names:
        value = getattr(metadata, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def create_raw_video_decoder(
    video_id: str,
    video_path: Path,
    *,
    backend: str = "auto",
    auto_nvdec_promoted: bool = False,
    module_loader: ModuleLoader = importlib.import_module,
) -> RawVideoDecoder:
    if backend not in {"auto", "opencv", "nvdec"}:
        raise ValueError(f"unsupported video backend: {backend}")
    selected = backend
    if backend == "auto" and auto_nvdec_promoted:
        selected = "nvdec" if nvdec_preflight(module_loader)["available"] else "opencv"
    if selected in {"auto", "opencv"}:
        return OpenCVRawVideoDecoder(video_id, video_path)
    return NvdecRawVideoDecoder(video_id, video_path, module_loader=module_loader)


__all__ = [
    "DecodedFrame",
    "DecoderMetrics",
    "NvdecRawVideoDecoder",
    "OpenCVRawVideoDecoder",
    "RawVideoDecoder",
    "VideoInfo",
    "create_raw_video_decoder",
    "nvdec_preflight",
    "sampled_frame_indices",
]
