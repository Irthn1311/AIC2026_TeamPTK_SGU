"""Offline-only model asset provenance and lazy encoder adapters."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1b.contracts import CandidateContract, MultimodalEncoder

AdapterFactory = Callable[[CandidateContract], MultimodalEncoder]


def _image_tensor(path: Path, contract: CandidateContract, image_module: Any, torch: Any):
    settings = contract.image_preprocessing
    interpolation_name = str(settings["interpolation"]).lower()
    interpolation = {
        "bicubic": image_module.Resampling.BICUBIC,
        "bilinear": image_module.Resampling.BILINEAR,
        "nearest": image_module.Resampling.NEAREST,
    }.get(interpolation_name)
    if interpolation is None:
        raise ValueError(f"Unsupported declared interpolation: {interpolation_name}")
    image = image_module.open(path)
    if settings["convert_rgb"]:
        image = image.convert("RGB")
    resize = int(settings["resize"])
    width, height = image.size
    scale = resize / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), interpolation)
    crop = int(settings["crop"])
    left = (image.width - crop) // 2
    top = (image.height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (crop, crop, 3):
        raise ValueError(f"IMAGE_PREPROCESS_FAILED: unexpected image shape {array.shape}")
    mean = np.asarray(settings["mean"], dtype=np.float32)
    std = np.asarray(settings["std"], dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std == 0):
        raise ValueError("IMAGE_PREPROCESS_FAILED: invalid declared mean/std")
    array = (array - mean) / std
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    candidate = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_fingerprint(path: str | Path, max_files: int = 256, max_depth: int = 4) -> str:
    root = Path(path).expanduser().resolve(strict=True)
    if root.is_file():
        return sha256_file(root)
    records = []
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        for item in sorted(current.iterdir(), key=lambda value: value.name):
            if item.is_file():
                if len(records) >= max_files:
                    raise ValueError("Model asset inventory exceeds bounded file limit")
                stat = item.stat()
                records.append((item.relative_to(root).as_posix(), stat.st_size, sha256_file(item)))
            elif item.is_dir() and depth < max_depth:
                frontier.append((item, depth + 1))
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def asset_source(path: Path, repo_root: Path, dataset_root: Path) -> str:
    resolved = path.resolve(strict=False)
    repository = repo_root.resolve(strict=False)
    dataset = dataset_root.resolve(strict=False)
    if resolved == repository or repository in resolved.parents:
        return "REPOSITORY"
    if (
        resolved == dataset
        or dataset in resolved.parents
        or str(resolved).startswith("/kaggle/input/")
    ):
        return "KAGGLE_INPUT"
    if path.exists():
        return "LOCAL_CACHE"
    return "UNKNOWN"


def dependency_name(implementation: str) -> str | None:
    return {"open_clip": "open_clip", "openai_clip": "clip"}.get(implementation)


def preflight_candidate(
    candidate: CandidateContract, repo_root: Path, dataset_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    dependency = dependency_name(candidate.implementation)
    dependency_ok = dependency is None or importlib.util.find_spec(dependency) is not None
    checkpoint = (
        Path(candidate.checkpoint_path).expanduser().resolve(strict=False)
        if candidate.checkpoint_path
        else None
    )
    asset_ok = checkpoint is not None and checkpoint.exists()
    if candidate.output_dimension != 512:
        issues.append(issue("ERROR", "ENCODER_OUTPUT_DIMENSION_MISMATCH", candidate, None))
    if not dependency_ok:
        issues.append(issue("ERROR", "ENCODER_DEPENDENCY_NOT_AVAILABLE", candidate, None))
    if not asset_ok:
        issues.append(issue("ERROR", "ENCODER_ASSET_NOT_FOUND", candidate, checkpoint))
    fingerprint = inventory_fingerprint(checkpoint) if asset_ok and checkpoint else None
    if (
        asset_ok
        and checkpoint
        and checkpoint.is_file()
        and candidate.checkpoint_sha256
        and candidate.checkpoint_sha256.lower() != fingerprint
    ):
        issues.append(
            issue(
                "ERROR",
                "ENCODER_CHECKPOINT_HASH_MISMATCH",
                candidate,
                checkpoint,
                expected=candidate.checkpoint_sha256.lower(),
                actual=fingerprint,
            )
        )
    return {
        "dependency": dependency,
        "dependency_available": dependency_ok,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_fingerprint": fingerprint,
        "asset_kind": ("FILE" if asset_ok and checkpoint and checkpoint.is_file() else "DIRECTORY")
        if asset_ok
        else None,
        "asset_available": asset_ok,
        "asset_source": (
            asset_source(checkpoint, repo_root, dataset_root) if checkpoint else "UNKNOWN"
        ),
    }, issues


def issue(
    severity: str,
    code: str,
    candidate: CandidateContract | None,
    path: Path | None,
    message: str | None = None,
    **evidence: Any,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "candidate_id": candidate.candidate_id if candidate else None,
        "global_row": None,
        "video_id": None,
        "path": str(path) if path else None,
        "message": message or code.replace("_", " ").title(),
        "evidence": evidence,
    }


class OpenClipMultimodalEncoder:
    def __init__(self, contract: CandidateContract) -> None:
        checkpoint = Path(contract.checkpoint_path or "").expanduser().resolve(strict=False)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ENCODER_ASSET_NOT_FOUND: {checkpoint}")
        if contract.architecture.startswith("hf-hub:"):
            raise ValueError("hf-hub model names are blocked")
        if contract.tokenizer != "open_clip_simple":
            raise ValueError("Unsupported declared OpenCLIP tokenizer")
        import open_clip
        import torch
        from PIL import Image

        self._torch, self._image = torch, Image
        self._model, _, _ = open_clip.create_model_and_transforms(
            contract.architecture, pretrained=None, device="cpu"
        )
        open_clip.load_checkpoint(self._model, str(checkpoint))
        self._tokenizer = open_clip.tokenizer.SimpleTokenizer()
        self._contract = contract

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        tensors = [_image_tensor(path, self._contract, self._image, self._torch) for path in paths]
        with self._torch.no_grad():
            return self._model.encode_image(self._torch.stack(tensors)).cpu().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            return self._model.encode_text(self._tokenizer(texts)).cpu().numpy()


class OpenAIClipMultimodalEncoder:
    def __init__(self, contract: CandidateContract) -> None:
        checkpoint = Path(contract.checkpoint_path or "").expanduser().resolve(strict=False)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ENCODER_ASSET_NOT_FOUND: {checkpoint}")
        if contract.tokenizer != "openai_clip_simple":
            raise ValueError("Unsupported declared OpenAI CLIP tokenizer")
        import clip
        import torch
        from PIL import Image

        self._torch, self._clip, self._image = torch, clip, Image
        self._model, _ = clip.load(str(checkpoint), device="cpu", jit=False)
        self._contract = contract

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        tensors = [_image_tensor(path, self._contract, self._image, self._torch) for path in paths]
        with self._torch.no_grad():
            return self._model.encode_image(self._torch.stack(tensors)).cpu().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self._clip.tokenize(texts)
        with self._torch.no_grad():
            return self._model.encode_text(tokens).cpu().numpy()


def load_multimodal_encoder(
    candidate: CandidateContract, adapter_factory: AdapterFactory | None = None
) -> MultimodalEncoder:
    if adapter_factory is not None:
        return adapter_factory(candidate)
    if candidate.implementation == "open_clip":
        return OpenClipMultimodalEncoder(candidate)
    if candidate.implementation == "openai_clip":
        return OpenAIClipMultimodalEncoder(candidate)
    raise FileNotFoundError(
        f"ENCODER_ASSET_NOT_FOUND: unsupported implementation {candidate.implementation!r}"
    )
