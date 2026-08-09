"""Offline-only deterministic OPUS-MT Vietnamese-to-English adapter."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from types import ModuleType
from typing import Any

from .contracts import GenerationConfig, TranslatorConfig

ModuleLoader = Callable[[str], ModuleType]


def translator_dependency_versions(
    module_loader: ModuleLoader = importlib.import_module,
) -> dict[str, Any]:
    modules: dict[str, ModuleType] = {}
    missing = []
    for name in ("torch", "transformers", "sentencepiece"):
        try:
            modules[name] = module_loader(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            "TRANSLATOR_DEPENDENCY_NOT_AVAILABLE: " + ", ".join(sorted(missing))
        )
    try:
        sacremoses = module_loader("sacremoses")
    except ImportError:
        sacremoses = None
    torch = modules["torch"]
    return {
        "torch_version": str(getattr(torch, "__version__", "UNKNOWN")),
        "transformers_version": str(
            getattr(modules["transformers"], "__version__", "UNKNOWN")
        ),
        "sentencepiece_version": str(
            getattr(modules["sentencepiece"], "__version__", "UNKNOWN")
        ),
        "sacremoses_available": sacremoses is not None,
        "sacremoses_version": (
            str(getattr(sacremoses, "__version__", "UNKNOWN"))
            if sacremoses is not None
            else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
    }


class OfflineViEnTranslator:
    """Small lifecycle wrapper around a local MarianMT model directory."""

    def __init__(
        self,
        model_path: str | Path,
        config: TranslatorConfig,
        generation: GenerationConfig,
        *,
        module_loader: ModuleLoader = importlib.import_module,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        self.config = config
        self.generation = generation
        self.module_loader = module_loader
        self.tokenizer: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.device = "cpu"
        self.load_latency_ms: float | None = None
        self.dependencies: dict[str, Any] = {}
        self.model_generation_defaults: dict[str, Any] = {}

    def load(self) -> OfflineViEnTranslator:
        if self.tokenizer is not None and self.model is not None and self.torch is not None:
            return self
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.dependencies = translator_dependency_versions(self.module_loader)
        transformers = self.module_loader("transformers")
        self.torch = self.module_loader("torch")
        requested = self.config.device
        if requested == "auto":
            self.device = "cuda:0" if self.torch.cuda.is_available() else "cpu"
        elif requested.startswith("cuda"):
            if not self.torch.cuda.is_available():
                raise RuntimeError("TRANSLATOR_LOAD_FAILED: CUDA requested but unavailable")
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        started = monotonic()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as error:
            raise RuntimeError(f"TRANSLATOR_LOAD_FAILED: {error}") from error
        self.load_latency_ms = (monotonic() - started) * 1000
        generation_config = getattr(self.model, "generation_config", None)
        for name in (
            "do_sample",
            "num_beams",
            "max_new_tokens",
            "length_penalty",
            "early_stopping",
        ):
            self.model_generation_defaults[name] = getattr(generation_config, name, None)
        return self

    def translate(self, texts: list[str]) -> list[dict[str, Any]]:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("TRANSLATOR_LOAD_FAILED: translator is not loaded")
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            batch_started = monotonic()
            try:
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        **self.generation.as_generate_kwargs(),
                    )
                decoded = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            except Exception as error:
                raise RuntimeError(f"TRANSLATION_FAILED: {error}") from error
            elapsed = (monotonic() - batch_started) * 1000
            if len(decoded) != len(batch):
                raise RuntimeError("TRANSLATION_FAILED: output count mismatch")
            for raw in decoded:
                text = str(raw)
                normalized = text.strip()
                if not normalized:
                    raise ValueError("TRANSLATION_EMPTY")
                outputs.append(
                    {
                        "translated_text_raw": text,
                        "translated_text_for_clip": normalized,
                        "translation_latency_ms": elapsed / len(batch),
                    }
                )
        return outputs

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "exact_revision": self.config.exact_revision,
            "architecture": "MarianMT",
            "model_path": str(self.model_path),
            "local_files_only": True,
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
            "device": self.device,
            "batch_size": self.config.batch_size,
            "dependencies": self.dependencies,
            "load_latency_ms": self.load_latency_ms,
            "model_generation_defaults": self.model_generation_defaults,
            "punctuation_normalizer": (
                "SACREMOSES_MOSES_PUNCT_NORMALIZER"
                if self.dependencies.get("sacremoses_available")
                else "TRANSFORMERS_IDENTITY_FALLBACK"
            ),
            "effective_generation_config": self.generation.as_generate_kwargs(),
        }

    def close(self) -> None:
        self.tokenizer = None
        self.model = None
        if self.torch is not None and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
