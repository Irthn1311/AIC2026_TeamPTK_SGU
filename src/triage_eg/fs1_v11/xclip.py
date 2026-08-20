"""Official Transformers X-CLIP video-text adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class XClipAdapter:
    def __init__(self, asset_root: Path, device: str = "cuda") -> None:
        self.root = Path(asset_root).resolve(strict=True)
        self.device = device
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        from transformers import XCLIPModel, XCLIPProcessor

        # X-CLIP stores a VideoMAE image processor, not a ProcessorMixin
        # ``video_processor``.  Recent Transformers releases therefore ignore
        # the generic ``videos=`` route for this legacy processor contract.
        # Keep the frozen slow processor and feed the eight-frame clip through
        # its official ``images=`` boundary.
        self.processor = XCLIPProcessor.from_pretrained(
            self.root, local_files_only=True, use_fast=False
        )
        self.model = (
            XCLIPModel.from_pretrained(self.root, local_files_only=True).to(self.device).eval()
        )

    def score(self, text: str, frames: list[Any]) -> dict[str, Any]:
        import torch

        if self.model is None or self.processor is None or len(frames) != 8:
            raise RuntimeError("XCLIP_REQUIRES_LOADED_MODEL_AND_EXACTLY_8_FRAMES")
        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        text_limit = int(getattr(text_config, "max_position_embeddings", 77))
        if text_limit <= 0:
            raise RuntimeError(f"XCLIP_TEXT_POSITION_LIMIT_INVALID:{text_limit}")
        inputs = self.processor(
            text=[text],
            images=frames,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_limit,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[-1] > text_limit:
            shape = None if input_ids is None else list(input_ids.shape)
            raise RuntimeError(f"XCLIP_PROCESSOR_TEXT_TENSOR_INVALID:{shape}:{text_limit}")
        pixel_values = inputs.get("pixel_values")
        if (
            pixel_values is None
            or pixel_values.ndim != 5
            or tuple(pixel_values.shape[:2]) != (1, 8)
        ):
            shape = None if pixel_values is None else list(pixel_values.shape)
            raise RuntimeError(f"XCLIP_PROCESSOR_VIDEO_TENSOR_INVALID:{shape}")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
        logits = output.logits_per_video
        if logits is None or not torch.isfinite(logits).all():
            raise RuntimeError("XCLIP_NONFINITE_OR_MISSING_LOGITS")
        return {
            "score": float(logits.reshape(-1)[0].item()),
            "logits_shape": list(logits.shape),
            "text_token_count": int(input_ids.shape[-1]),
            "text_max_position_embeddings": text_limit,
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "finite": True,
        }

    def unload(self) -> None:
        self.processor = self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def uniform_indices(start: int, end: int, count: int = 8) -> list[int]:
    if start < 0 or end < start or count != 8:
        raise ValueError("XCLIP window requires valid bounds and exactly eight frames")
    return [int(round(value)) for value in np.linspace(start, end, count)]
