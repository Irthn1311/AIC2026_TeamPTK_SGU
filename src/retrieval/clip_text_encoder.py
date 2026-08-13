from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np


@dataclass
class ClipTextEncoderInfo:
    model_name: str
    pretrained: str
    embedding_dim: int
    model_assumption: bool
    assumption_reason: str


class ClipTextEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        prefer_backend: str = "auto",
        open_clip_extra_site_packages: str | None = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.prefer_backend = prefer_backend
        self.open_clip_extra_site_packages = open_clip_extra_site_packages
        self._model = None
        self._tokenizer = None
        self._embedding_dim = 512
        self._backend = "open_clip"

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch

        if self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda:0"

        if self.prefer_backend == "open_clip":
            self._load_open_clip()
            return

        try:
            from transformers import CLIPModel, CLIPTokenizerFast

            model_name = "openai/clip-vit-base-patch32"
            try:
                self._model = CLIPModel.from_pretrained(model_name).to(self.device)
                self._tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
            except Exception:
                self._model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(self.device)
                self._tokenizer = CLIPTokenizerFast.from_pretrained(model_name, local_files_only=True)

            self._model.eval()
            self._embedding_dim = int(self._model.config.projection_dim)
            self._backend = "transformers"
            return
        except Exception as e:
            try:
                self._load_open_clip()
                return
            except Exception:
                raise RuntimeError(f"Failed to load CLIP text encoder: {e}")

    def _load_open_clip(self) -> None:
        if self.open_clip_extra_site_packages:
            extra_path = str(Path(self.open_clip_extra_site_packages).resolve())
            if extra_path not in sys.path:
                sys.path.append(extra_path)

        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self._model = model.to(self.device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        if hasattr(self._model, "text_projection"):
            self._embedding_dim = int(self._model.text_projection.shape[1])
        else:
            self._embedding_dim = 512
        self._backend = "open_clip"

    @property
    def embedding_dim(self) -> int:
        self._lazy_load()
        return int(self._embedding_dim)

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        self._lazy_load()
        import torch

        if isinstance(texts, str):
            texts = [texts]
        if self._backend == "transformers":
            tokens = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        else:
            tokens = self._tokenizer(list(texts))
            if hasattr(tokens, "to"):
                tokens = tokens.to(self.device)
        with torch.no_grad():
            if self._backend == "transformers":
                feats = self._model.get_text_features(**tokens)
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
            else:
                feats = self._model.encode_text(tokens)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype(np.float32)

    def info(self, model_assumption: bool = True, assumption_reason: str = "No official metadata found; inferred from feature name and embedding dimension.") -> ClipTextEncoderInfo:
        return ClipTextEncoderInfo(
            model_name=self.model_name,
            pretrained=self.pretrained,
            embedding_dim=512,
            model_assumption=model_assumption,
            assumption_reason=assumption_reason,
        )
