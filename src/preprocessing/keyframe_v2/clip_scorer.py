from __future__ import annotations

import glob
from pathlib import Path
import sys

import cv2
import numpy as np

from src.preprocessing.model_assets import ensure_open_clip_weights


class ImageEmbeddingScorer:
    """CLIP scorer when local weights exist, otherwise deterministic image embedding fallback."""

    def __init__(self, project_root: Path, cfg: dict):
        self.project_root = project_root
        self.cfg = cfg
        self.backend = "image_embedding_fallback"
        self.warning: str | None = None
        self._model = None
        self._preprocess = None
        self._device = "cpu"
        self.info = {
            "backend": self.backend,
            "clip_model": "",
            "clip_weights_path": "",
            "embedding_dim": 0,
            "device": self._device,
            "dtype": "",
        }
        self._try_load_clip()

    def _try_load_clip(self) -> None:
        require_real = bool(self.cfg.get("require_real_clip", False))
        try:
            import torch

            device_cfg = str(self.cfg.get("device", "auto"))
            self._device = "cuda" if device_cfg == "auto" and torch.cuda.is_available() else ("cpu" if device_cfg == "auto" else device_cfg)
            backend = str(self.cfg.get("backend", "open_clip"))
            if backend == "open_clip":
                model, preprocess, weights_path = self._load_open_clip(torch)
            else:
                model, preprocess, weights_path = self._load_openai_clip(torch)
            model.eval()
            self._model = model
            self._preprocess = preprocess
            dim = self._detect_embedding_dim()
            self.backend = "clip"
            self.info = {
                "backend": "clip",
                "clip_model": str(self.cfg.get("display_model_name", self.cfg.get("model_name", ""))),
                "clip_weights_path": str(weights_path),
                "embedding_dim": int(dim),
                "device": self._device,
                "dtype": str(next(self._model.parameters()).dtype),
            }
            self.warning = None
        except Exception as exc:
            if require_real:
                raise RuntimeError(f"Real CLIP image encoder required but unavailable: {exc}") from exc
            self.warning = f"CLIP image load failed: {exc}; using image_embedding_fallback"

    def _load_open_clip(self, torch_module):
        try:
            import open_clip
        except ImportError:
            extra = self.cfg.get("open_clip_extra_site_packages")
            if extra:
                extra_path = str((self.project_root / str(extra)).resolve())
                if extra_path not in sys.path:
                    sys.path.append(extra_path)
            import open_clip

        model_name = str(self.cfg.get("model_name", "ViT-B-32"))
        precision = str(self.cfg.get("precision", "fp32"))
        pretrained_alias = str(self.cfg.get("pretrained", "") or "").strip()

        weights_path = self._resolve_weights_path(str(self.cfg.get("open_clip_weights", "")))
        if weights_path is None and bool(self.cfg.get("allow_download", False)):
            weights_path = ensure_open_clip_weights(
                repo_id=str(self.cfg.get("open_clip_repo_id", "timm/vit_base_patch32_clip_224.openai")),
                filename=str(self.cfg.get("open_clip_filename", "open_clip_model.safetensors")),
                cache_root=self.project_root / str(self.cfg.get("download_root", ".model_cache/clip")),
            )
        if weights_path is not None and self._should_load_as_safetensors(weights_path):
            return self._load_open_clip_safetensors(open_clip, model_name, precision, weights_path)

        if pretrained_alias:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_alias,
                precision=precision,
                device=self._device,
            )
            return model, preprocess, pretrained_alias

        if weights_path is None:
            raise FileNotFoundError(f"open_clip_weights not found: {self.cfg.get('open_clip_weights')}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(weights_path),
            precision=precision,
            device=self._device,
        )
        return model, preprocess, weights_path

    def _should_load_as_safetensors(self, weights_path: Path) -> bool:
        filename = str(self.cfg.get("open_clip_filename", ""))
        pattern = str(self.cfg.get("open_clip_weights", ""))
        return (
            weights_path.suffix.lower() == ".safetensors"
            or filename.endswith(".safetensors")
            or pattern.endswith(".safetensors")
        )

    def _load_open_clip_safetensors(self, open_clip, model_name: str, precision: str, weights_path: Path):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,
            precision=precision,
            device=self._device,
        )
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "OpenCLIP safetensors checkpoint does not match model "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        return model, preprocess, weights_path

    def _load_openai_clip(self, torch_module):
        import clip

        root = self.project_root / str(self.cfg.get("download_root", ".model_cache/clip"))
        if not bool(self.cfg.get("allow_download", False)) and not any(root.glob("*.pt")):
            raise FileNotFoundError(f"CLIP image weights not found in {root}")
        model, preprocess = clip.load(str(self.cfg.get("model_name", "ViT-B/32")), device=self._device, download_root=str(root))
        weights = next(root.glob("*.pt"), root)
        return model, preprocess, weights

    def _resolve_weights_path(self, pattern: str) -> Path | None:
        if not pattern:
            return None
        if any(ch in pattern for ch in "*?[]"):
            if Path(pattern).is_absolute():
                matches = [Path(match) for match in sorted(glob.glob(pattern))]
            else:
                matches = sorted(self.project_root.glob(pattern))
        else:
            path = Path(pattern)
            matches = [path if path.is_absolute() else self.project_root / path]
        for match in matches:
            if match.exists():
                return match.resolve()
        return None

    def _detect_embedding_dim(self) -> int:
        if hasattr(self._model, "visual") and hasattr(self._model.visual, "output_dim"):
            return int(self._model.visual.output_dim)
        return 0

    def embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        if not images_bgr:
            return np.zeros((0, 1), dtype=np.float32)
        if self._model is not None:
            return self._embed_clip(images_bgr)
        return np.vstack([self._embed_fallback(img) for img in images_bgr]).astype(np.float32)

    def _embed_clip(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        feats_all = []
        batch_size = max(1, int(self.cfg.get("batch_size", 16)))
        for start in range(0, len(images_bgr), batch_size):
            tensors = []
            for img in images_bgr[start : start + batch_size]:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensors.append(self._preprocess(Image.fromarray(rgb)))
            batch = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                feats_all.append(self._model.encode_image(batch).float().cpu().numpy())
        return _normalize(np.vstack(feats_all))

    def _embed_fallback(self, image_bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(image_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten() / 255.0
        vec = np.concatenate([hist.astype(np.float32), gray])
        return _normalize(vec.reshape(1, -1))[0]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
