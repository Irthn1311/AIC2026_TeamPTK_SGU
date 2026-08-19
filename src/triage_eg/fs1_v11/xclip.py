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

        self.processor = XCLIPProcessor.from_pretrained(self.root, local_files_only=True)
        self.model = (
            XCLIPModel.from_pretrained(self.root, local_files_only=True).to(self.device).eval()
        )

    def score(self, text: str, frames: list[Any]) -> dict[str, Any]:
        import torch

        if self.model is None or self.processor is None or len(frames) != 8:
            raise RuntimeError("XCLIP_REQUIRES_LOADED_MODEL_AND_EXACTLY_8_FRAMES")
        inputs = self.processor(text=[text], videos=frames, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
        logits = output.logits_per_video
        if logits is None or not torch.isfinite(logits).all():
            raise RuntimeError("XCLIP_NONFINITE_OR_MISSING_LOGITS")
        return {
            "score": float(logits.reshape(-1)[0].item()),
            "logits_shape": list(logits.shape),
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
