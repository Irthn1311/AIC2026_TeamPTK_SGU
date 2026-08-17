"""Offline-only official SigLIP2 text/image feature adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .assets import extract_pooled_features, validate_offline_asset
from .contracts import (
    EMBEDDING_DIMENSION,
    PROCESSOR_USE_FAST,
    TEXT_MAX_LENGTH,
    TEXT_PADDING,
    TEXT_TRUNCATION,
)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("SCA1_SIGLIP2_FEATURE_MATRIX_INVALID")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("SCA1_SIGLIP2_ZERO_OR_INVALID_FEATURE_NORM")
    output = values / norms
    if not np.allclose(np.linalg.norm(output, axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("SCA1_SIGLIP2_L2_NORMALIZATION_FAILED")
    return output.astype(np.float32, copy=False)


class Siglip2OfflineEncoder:
    """Load the pinned asset locally and expose manually normalized features."""

    def __init__(
        self,
        asset_root: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        if device not in {"auto", "cpu", "cuda", "cuda:0"} or batch_size <= 0:
            raise ValueError("invalid SigLIP2 device or batch size")
        self.asset = validate_offline_asset(asset_root)
        self.requested_device = device
        self.batch_size = batch_size
        self.device = "cpu"
        self.compute_dtype = "float32"
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> Siglip2OfflineEncoder:
        if self.loaded:
            return self
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("torch and transformers are required for SCA-1 runtime") from error
        cuda = bool(torch.cuda.is_available())
        selected = "cuda:0" if self.requested_device == "auto" and cuda else self.requested_device
        if selected == "auto":
            selected = "cpu"
        if selected.startswith("cuda") and not cuda:
            raise RuntimeError("SCA1_SIGLIP2_CUDA_REQUESTED_BUT_UNAVAILABLE")
        model_root = self.asset["model_root"]
        self.processor = AutoProcessor.from_pretrained(
            model_root, local_files_only=True, use_fast=PROCESSOR_USE_FAST
        )
        self.model = AutoModel.from_pretrained(model_root, local_files_only=True)
        self.device = selected
        self.compute_dtype = "float16" if selected.startswith("cuda") else "float32"
        dtype = torch.float16 if self.compute_dtype == "float16" else torch.float32
        self.model = self.model.to(device=selected, dtype=dtype).eval()
        self.torch = torch
        return self

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            name: tensor.to(self.device) for name, tensor in inputs.items() if hasattr(tensor, "to")
        }

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Siglip2OfflineEncoder.load() must be called first")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("SigLIP2 text batch must contain non-empty strings")
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.processor(
                text=batch,
                padding=TEXT_PADDING,
                truncation=TEXT_TRUNCATION,
                max_length=TEXT_MAX_LENGTH,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                features = extract_pooled_features(
                    self.model.get_text_features(**self._to_device(dict(inputs))),
                    modality="text",
                )
            outputs.append(features.detach().float().cpu().numpy())
        matrix = l2_normalize(np.concatenate(outputs, axis=0))
        if matrix.shape != (len(texts), EMBEDDING_DIMENSION):
            raise RuntimeError(f"SCA1_SIGLIP2_TEXT_SHAPE_INVALID: {matrix.shape}")
        return matrix

    def encode_images(self, images: list[Any]) -> np.ndarray:
        if not self.loaded:
            raise RuntimeError("Siglip2OfflineEncoder.load() must be called first")
        if not images:
            raise ValueError("SigLIP2 image batch must not be empty")
        outputs = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            with self.torch.inference_mode():
                features = extract_pooled_features(
                    self.model.get_image_features(**self._to_device(dict(inputs))),
                    modality="image",
                )
            outputs.append(features.detach().float().cpu().numpy())
        matrix = l2_normalize(np.concatenate(outputs, axis=0))
        if matrix.shape != (len(images), EMBEDDING_DIMENSION):
            raise RuntimeError(f"SCA1_SIGLIP2_IMAGE_SHAPE_INVALID: {matrix.shape}")
        return matrix

    def provenance(self) -> dict[str, Any]:
        return {
            **{
                key: self.asset[key]
                for key in (
                    "model_id",
                    "exact_revision",
                    "model_safetensors_sha256",
                    "manifest_sha256",
                )
            },
            "device": self.device,
            "compute_dtype": self.compute_dtype,
            "batch_size": self.batch_size,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "text_padding": TEXT_PADDING,
            "text_truncation": TEXT_TRUNCATION,
            "text_max_length": TEXT_MAX_LENGTH,
            "manual_l2_normalization": True,
            "processor_use_fast": PROCESSOR_USE_FAST,
            "network_required": False,
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        if self.torch is not None and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


__all__ = ["Siglip2OfflineEncoder", "l2_normalize"]
