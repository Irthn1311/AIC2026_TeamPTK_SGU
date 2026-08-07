"""Canonical optional OpenAI CLIP text encoder for exact KIS retrieval."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class TextEncoder(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def identifiers(self) -> Mapping[str, Any]: ...

    def encode(self, text: str) -> NDArray[np.float32]: ...


class TextEncoderUnavailable(RuntimeError):
    """The configured canonical text encoder cannot be loaded safely."""


class OpenAIClipTextEncoder:
    MODEL_NAME = "ViT-B/32"
    CACHE_FILENAME = "ViT-B-32.pt"

    def __init__(
        self,
        *,
        device: str = "cpu",
        allow_model_download: bool = False,
        cache_dir: Path | None = None,
        expected_dimension: int = 512,
        clip_module: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")
        try:
            clip = clip_module or importlib.import_module("clip")
            torch = torch_module or importlib.import_module("torch")
        except ImportError as exc:
            raise TextEncoderUnavailable(
                f"official OpenAI CLIP dependency unavailable: {exc}"
            ) from exc
        if (
            not callable(getattr(clip, "available_models", None))
            or not callable(getattr(clip, "load", None))
            or not callable(getattr(clip, "tokenize", None))
        ):
            raise TextEncoderUnavailable(
                "installed clip package does not expose official load, tokenize, and "
                "available_models APIs"
            )
        available_models = tuple(str(name) for name in clip.available_models())
        if self.MODEL_NAME not in available_models:
            raise TextEncoderUnavailable(f"official OpenAI CLIP does not provide {self.MODEL_NAME}")
        if device == "cuda" and not torch.cuda.is_available():
            raise TextEncoderUnavailable("CUDA was requested but is unavailable")

        resolved_cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "clip"
        checkpoint = resolved_cache / self.CACHE_FILENAME
        if not allow_model_download and not checkpoint.is_file():
            raise TextEncoderUnavailable(
                f"OpenAI CLIP weights are not cached at {checkpoint}; "
                "enable allow_model_download explicitly to download them"
            )
        try:
            model, preprocess = clip.load(
                self.MODEL_NAME,
                device=device,
                jit=False,
                download_root=str(resolved_cache),
            )
            model.eval()
        except Exception as exc:
            raise TextEncoderUnavailable(f"OpenAI CLIP load failed: {exc}") from exc

        self._clip = clip
        self._torch = torch
        self._model = model
        self._device = device
        self._dimension = expected_dimension
        self._identifiers: Mapping[str, Any] = MappingProxyType(
            {
                "library": "openai-clip",
                "package_path": getattr(clip, "__file__", "unknown"),
                "library_version": getattr(clip, "__version__", "unknown"),
                "model": self.MODEL_NAME,
                "device": device,
                "checkpoint_cache_path": str(checkpoint),
                "model_download_allowed": allow_model_download,
                "available_models": available_models,
                "image_preprocessing": repr(preprocess),
                "tokenization": repr(clip.tokenize),
                "context_length": getattr(model, "context_length", "unknown"),
            }
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def identifiers(self) -> Mapping[str, Any]:
        return self._identifiers

    def encode(self, text: str) -> NDArray[np.float32]:
        if not text.strip():
            raise ValueError("query text must not be empty")
        try:
            tokens = self._clip.tokenize([text], truncate=True).to(self._device)
            with self._torch.no_grad():
                output = self._model.encode_text(tokens)
                array = output.float().cpu().numpy()
        except Exception as exc:
            raise RuntimeError(f"OpenAI CLIP text encoding failed: {exc}") from exc
        vector = np.asarray(array, dtype=np.float32)
        if vector.shape == (1, self._dimension):
            vector = vector[0]
        if vector.shape != (self._dimension,):
            raise ValueError(
                f"text embedding dimension mismatch: shape={vector.shape}, "
                f"expected=({self._dimension},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError("text embedding contains NaN or Infinity")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("text embedding must have a finite non-zero norm")
        return np.asarray(vector / norm, dtype=np.float32)


class SharedOpenAIClipEncoder:
    MODEL_NAME = "ViT-B/32"
    CACHE_FILENAME = "ViT-B-32.pt"

    def __init__(
        self,
        *,
        device: str = "cpu",
        allow_model_download: bool = False,
        cache_dir: Path | None = None,
        expected_dimension: int = 512,
        clip_module: Any | None = None,
        torch_module: Any | None = None,
        image_module: Any | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")
        try:
            clip = clip_module or importlib.import_module("clip")
            torch = torch_module or importlib.import_module("torch")
            image_api = image_module or importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise TextEncoderUnavailable(
                f"official OpenAI CLIP dependency unavailable: {exc}"
            ) from exc

        if not all(
            callable(getattr(clip, name, None)) for name in ("available_models", "load", "tokenize")
        ):
            raise TextEncoderUnavailable(
                "installed clip package does not expose official load, tokenize, and available_models APIs"
            )
        available_models = tuple(str(name) for name in clip.available_models())
        if self.MODEL_NAME not in available_models:
            raise TextEncoderUnavailable(f"official OpenAI CLIP does not provide {self.MODEL_NAME}")
        if device == "cuda" and not torch.cuda.is_available():
            raise TextEncoderUnavailable("CUDA was requested but is unavailable")

        resolved_cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "clip"
        checkpoint = resolved_cache / self.CACHE_FILENAME
        if not allow_model_download and not checkpoint.is_file():
            raise TextEncoderUnavailable(
                f"OpenAI CLIP weights are not cached at {checkpoint}; "
                "enable allow_model_download explicitly to download them"
            )
        try:
            model, preprocess = clip.load(
                self.MODEL_NAME,
                device=device,
                jit=False,
                download_root=str(resolved_cache),
            )
            model.eval()
        except Exception as exc:
            raise TextEncoderUnavailable(f"OpenAI CLIP load failed: {exc}") from exc

        self._clip = clip
        self._torch = torch
        self._image_api = image_api
        self._model = model
        self._preprocess = preprocess
        self._device = device
        self._dimension = expected_dimension
        self._identifiers: Mapping[str, Any] = MappingProxyType(
            {
                "library": "openai-clip",
                "package_path": getattr(clip, "__file__", "unknown"),
                "library_version": getattr(clip, "__version__", "unknown"),
                "model": self.MODEL_NAME,
                "device": device,
                "checkpoint_cache_path": str(checkpoint),
                "model_download_allowed": allow_model_download,
                "available_models": available_models,
                "image_preprocessing": repr(preprocess),
                "tokenization": repr(clip.tokenize),
                "context_length": getattr(model, "context_length", "unknown"),
            }
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def identifiers(self) -> Mapping[str, Any]:
        return self._identifiers

    def encode(self, text: str) -> NDArray[np.float32]:
        if not text.strip():
            raise ValueError("query text must not be empty")
        matrix = self.encode_texts([text])
        return matrix[0]

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        resolved = tuple(texts)
        if not resolved or any(not text.strip() for text in resolved):
            raise ValueError("text encoding requires non-empty texts")
        try:
            tokens = self._clip.tokenize(list(resolved), truncate=True).to(self._device)
            with self._torch.no_grad():
                output = self._model.encode_text(tokens)
                matrix = output.float().cpu().numpy()
        except Exception as exc:
            raise RuntimeError(f"OpenAI CLIP text encoding failed: {exc}") from exc
        return self._normalize_rows(
            matrix,
            expected_rows=len(resolved),
            label="text",
        )

    def encode_images(
        self,
        images: Sequence[Any],
        *,
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        resolved = tuple(images)
        if not resolved:
            raise ValueError("image encoding requires at least one image")
        if batch_size <= 0:
            raise ValueError("image batch_size must be positive")
        encoded_batches: list[NDArray[np.float32]] = []
        for start in range(0, len(resolved), batch_size):
            tensors = [
                self._preprocess(self._to_rgb_image(image))
                for image in resolved[start : start + batch_size]
            ]
            try:
                batch = self._torch.stack(tensors).to(self._device)
                with self._torch.no_grad():
                    output = self._model.encode_image(batch)
                encoded_batches.append(np.asarray(output.float().cpu().numpy(), dtype=np.float32))
            except Exception as exc:
                raise RuntimeError(f"OpenAI CLIP image encoding failed: {exc}") from exc
        matrix = np.concatenate(encoded_batches, axis=0)
        return self._normalize_rows(
            matrix,
            expected_rows=len(resolved),
            label="image",
        )

    def _normalize_rows(
        self,
        matrix: Any,
        *,
        expected_rows: int,
        label: str,
    ) -> NDArray[np.float32]:
        array = np.asarray(matrix, dtype=np.float32)
        if array.shape != (expected_rows, self._dimension):
            raise ValueError(
                f"{label} embedding shape mismatch: observed={array.shape}, "
                f"expected=({expected_rows}, {self._dimension})"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label} embeddings contain NaN or Infinity")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError(f"{label} embeddings contain a zero or invalid norm")
        return np.asarray(array / norms, dtype=np.float32)

    def _to_rgb_image(self, image: Any) -> Any:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            array = array[:, :, ::-1]
        return self._image_api.fromarray(np.asarray(array, dtype=np.uint8))

