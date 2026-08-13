from __future__ import annotations

import glob
import logging
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
        pretrained_alias = str(self.cfg.get("pretrained", "") or "openai").strip()

        # 1. Try standard open_clip pretrained model load on GPU
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_alias,
                precision=precision,
                device=self._device,
            )
            return model, preprocess, pretrained_alias
        except Exception:
            pass

        # 2. Try loading local safetensors if available
        weights_path = self._resolve_weights_path(str(self.cfg.get("open_clip_weights", "")))
        if weights_path is not None and self._should_load_as_safetensors(weights_path):
            return self._load_open_clip_safetensors(open_clip, model_name, precision, weights_path)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=str(weights_path) if weights_path else "openai",
            precision=precision,
            device=self._device,
        )
        return model, preprocess, weights_path or "openai"

    def _should_load_as_safetensors(self, weights_path: Path) -> bool:
        filename = str(self.cfg.get("open_clip_filename", ""))
        pattern = str(self.cfg.get("open_clip_weights", ""))
        return (
            weights_path.suffix.lower() == ".safetensors"
            or filename.endswith(".safetensors")
            or pattern.endswith(".safetensors")
        )

    def _load_open_clip_safetensors(self, open_clip, model_name: str, precision: str, weights_path: Path):
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(max(previous_level, logging.ERROR))
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=None,
                precision=precision,
                device=self._device,
            )
        finally:
            root_logger.setLevel(previous_level)
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self._device)
        return model, preprocess, weights_path

    def _load_openai_clip(self, torch_module):
        import clip

        root = self.project_root / str(self.cfg.get("download_root", ".model_cache/clip"))
        if not bool(self.cfg.get("allow_download", False)) and not any(root.glob("*.pt")):
            raise FileNotFoundError(f"CLIP image weights not found in {root}")
        model, preprocess = clip.load(str(self.cfg.get("model_name", "ViT-B/32")), device=self._device, download_root=str(root))
        weights = next(root.glob("*.pt"), root)
        model = model.to(self._device)
        return model, preprocess, weights

    def _resolve_weights_path(self, pattern: str) -> Path | None:
        if not pattern:
            return None
        candidates: list[Path] = []
        if any(ch in pattern for ch in "*?[]"):
            if Path(pattern).is_absolute():
                candidates.extend([Path(match) for match in sorted(glob.glob(pattern))])
            else:
                candidates.extend(sorted(self.project_root.glob(pattern)))
        else:
            path = Path(pattern)
            candidates.append(path if path.is_absolute() else self.project_root / path)
        for match in candidates:
            if match.exists() and match.is_file():
                return match.resolve()
        return None

    def _detect_embedding_dim(self) -> int:
        if hasattr(self._model, "visual") and hasattr(self._model.visual, "output_dim"):
            return int(self._model.visual.output_dim)
        return 0

    def embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        if not images_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        if self._model is not None:
            return self._embed_clip(images_bgr)
        return np.vstack([self._embed_fallback(img) for img in images_bgr]).astype(np.float32)

    def _embed_clip(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        import torch

        if not images_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        batch_size = max(1, int(self.cfg.get("batch_size", 64)))
        feats_all = []

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self._device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self._device, dtype=torch.float32).view(1, 3, 1, 1)

        for start in range(0, len(images_bgr), batch_size):
            batch_imgs = images_bgr[start : start + batch_size]
            resized_list = []
            for img in batch_imgs:
                r = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                resized_list.append(rgb)

            arr = np.stack(resized_list, axis=0)
            t_batch = torch.from_numpy(arr).to(self._device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            t_batch = (t_batch - mean) / std

            with torch.no_grad():
                feats = self._model.encode_image(t_batch)
                feats_all.append(feats.float().cpu().numpy())

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
