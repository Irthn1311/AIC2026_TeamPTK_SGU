"""Controlled, offline-only loader for an official OpenAI CLIP source checkout."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from types import ModuleType
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1b.contracts import CandidateContract

DEFAULT_ASSET_ROOT = Path("/kaggle/input/aic2026-openai-clip-vit-b32")
TOKENIZER_ASSET = Path("clip/bpe_simple_vocab_16e6.txt.gz")
KAGGLE_EXPANDED_TOKENIZER_ASSET = Path("clip/bpe_simple_vocab_16e6.txt")
OFFLINE_DEPENDENCY_ROOT = Path("source/dependencies")
REQUIRED_APIS = ("load", "tokenize")


class NetworkDownloadAttempted(RuntimeError):
    """Raised when the vendored package tries to enter its download helper."""


@dataclass(frozen=True)
class OfficialAssetPaths:
    asset_root: Path
    source_root: Path
    checkpoint_path: Path
    asset_manifest_path: Path


def _absolute(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def resolve_official_asset_paths(
    asset_root: str | Path | None = None,
    source_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    asset_manifest_path: str | Path | None = None,
) -> OfficialAssetPaths:
    configured_root = asset_root or os.environ.get("AIC_OPENAI_CLIP_ASSET_ROOT")
    if configured_root:
        root = _absolute(configured_root)
    elif asset_manifest_path:
        root = _absolute(asset_manifest_path).parent.parent
    elif checkpoint_path:
        root = _absolute(checkpoint_path).parent.parent
    else:
        root = _absolute(DEFAULT_ASSET_ROOT)
    source = _absolute(
        source_root or os.environ.get("AIC_OPENAI_CLIP_SOURCE_ROOT", root / "source/openai_clip")
    )
    checkpoint = _absolute(
        checkpoint_path
        or os.environ.get("AIC_OPENAI_CLIP_CHECKPOINT", root / "checkpoint/ViT-B-32.pt")
    )
    manifest = _absolute(asset_manifest_path or root / "manifests/asset_manifest.json")
    return OfficialAssetPaths(root, source, checkpoint, manifest)


def _is_within(path: Path, root: Path) -> bool:
    resolved, parent = path.resolve(strict=False), root.resolve(strict=False)
    return resolved == parent or parent in resolved.parents


def materialize_kaggle_expanded_tokenizer(
    source_root: str | Path,
    runtime_root: str | Path,
) -> tuple[Path, bool]:
    """Restore the tokenizer gzip Kaggle expands, without mutating read-only input."""
    source = _absolute(source_root)
    expected = source / TOKENIZER_ASSET
    if expected.is_file():
        return source, False
    expanded = source / KAGGLE_EXPANDED_TOKENIZER_ASSET
    if not expanded.is_file():
        return source, False
    runtime = _absolute(runtime_root)
    if runtime == Path(runtime.anchor) or len(runtime.parts) < 3 or _is_within(runtime, source):
        raise ValueError("OpenAI CLIP runtime source path is unsafe")
    staging = runtime.with_name(f".{runtime.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(source, staging)
        runtime_tokenizer = staging / TOKENIZER_ASSET
        runtime_tokenizer.parent.mkdir(parents=True, exist_ok=True)
        with (
            expanded.open("rb") as source_file,
            runtime_tokenizer.open("wb") as target_file,
            gzip.GzipFile(fileobj=target_file, mode="wb", mtime=0) as compressed,
        ):
            shutil.copyfileobj(source_file, compressed)
        if runtime.exists():
            shutil.rmtree(runtime)
        os.replace(staging, runtime)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return runtime, True


def _offline_dependency_wheels(asset_root: Path) -> tuple[Path, ...]:
    root = asset_root / OFFLINE_DEPENDENCY_ROOT
    return tuple(sorted(path.resolve() for path in root.glob("*.whl") if path.is_file()))


@contextmanager
def _temporary_dependency_paths(paths: Sequence[Path]) -> Iterator[None]:
    inserted: list[str] = []
    try:
        for path in reversed(paths):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
                inserted.append(value)
        yield
    finally:
        for value in inserted:
            if value in sys.path:
                sys.path.remove(value)


def _read_declared_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip().split()
    return value[0].lower() if value else None


def _read_manifest(path: Path, asset_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ["ASSET_MANIFEST_MALFORMED"]
    if not isinstance(value, dict):
        return None, ["ASSET_MANIFEST_MALFORMED"]
    required = {
        "asset_bundle_version",
        "implementation",
        "architecture",
        "checkpoint_relative_path",
        "internet_required_at_runtime",
    }
    if not required <= set(value):
        return value, ["ASSET_MANIFEST_MALFORMED"]
    relative = Path(str(value["checkpoint_relative_path"]))
    if relative.is_absolute() or not _is_within(asset_root / relative, asset_root):
        return value, ["ASSET_MANIFEST_PATH_TRAVERSAL"]
    issues = []
    if (
        value.get("implementation") != "openai_clip"
        or value.get("architecture") != "ViT-B/32"
        or value.get("internet_required_at_runtime") is not False
    ):
        issues.append("ASSET_MANIFEST_MALFORMED")
    return value, issues


def _module_file(module: ModuleType) -> Path | None:
    value = getattr(module, "__file__", None)
    return Path(value).resolve(strict=False) if value else None


def controlled_import_clip(
    source_root: Path,
) -> tuple[ModuleType | None, dict[str, Any], list[str]]:
    source = source_root.resolve(strict=False)
    package_root = source / "clip"
    init_file = package_root / "__init__.py"
    details: dict[str, Any] = {
        "module_name": "clip",
        "module_file": None,
        "configured_source_root": str(source),
        "module_origin_valid": False,
        "required_api_present": False,
        "owned_import": False,
    }
    if not source.is_dir():
        return None, details, ["OPENAI_CLIP_SOURCE_ROOT_MISSING"]
    if not init_file.is_file():
        return None, details, ["OPENAI_CLIP_PACKAGE_INVALID"]
    existing = sys.modules.get("clip")
    if existing is not None:
        origin = _module_file(existing)
        details["module_file"] = str(origin) if origin else None
        if origin is None or not _is_within(origin, source):
            return None, details, ["OPENAI_CLIP_MODULE_ORIGIN_MISMATCH"]
        missing = [name for name in REQUIRED_APIS if not callable(getattr(existing, name, None))]
        details["module_origin_valid"] = True
        details["required_api_present"] = not missing
        return (
            existing,
            details,
            ["OPENAI_CLIP_REQUIRED_API_MISSING"] if missing else [],
        )
    before = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "clip", init_file, submodule_search_locations=[str(package_root)]
        )
        if spec is None or spec.loader is None:
            return None, details, ["OPENAI_CLIP_PACKAGE_INVALID"]
        module = importlib.util.module_from_spec(spec)
        sys.modules["clip"] = module
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        for name in set(sys.modules) - before:
            if name == "clip" or name.startswith("clip."):
                sys.modules.pop(name, None)
        details["missing_dependency"] = error.name
        return None, details, ["ENCODER_DEPENDENCY_NOT_AVAILABLE"]
    except Exception:
        for name in set(sys.modules) - before:
            if name == "clip" or name.startswith("clip."):
                sys.modules.pop(name, None)
        return None, details, ["OPENAI_CLIP_PACKAGE_INVALID"]
    origin = _module_file(module)
    details["module_file"] = str(origin) if origin else None
    details["module_origin_valid"] = origin is not None and _is_within(origin, source)
    details["owned_import"] = True
    missing = [name for name in REQUIRED_APIS if not callable(getattr(module, name, None))]
    details["required_api_present"] = not missing
    issues = []
    if not details["module_origin_valid"]:
        issues.append("OPENAI_CLIP_MODULE_ORIGIN_MISMATCH")
    if missing:
        issues.append("OPENAI_CLIP_REQUIRED_API_MISSING")
    return module, details, issues


def _device_details(torch: Any, requested: str) -> tuple[str, dict[str, Any], list[str]]:
    if requested not in {"auto", "cpu", "cuda", "cuda:0"}:
        return requested, {}, ["ENCODER_DEVICE_INVALID"]
    cuda_available = bool(torch.cuda.is_available())
    selected = "cuda:0" if requested == "auto" and cuda_available else requested
    if selected == "auto":
        selected = "cpu"
    if selected.startswith("cuda") and not cuda_available:
        return selected, {}, ["ENCODER_DEVICE_UNAVAILABLE"]
    gpu_name = torch.cuda.get_device_name(0) if selected.startswith("cuda") else None
    return (
        selected,
        {
            "torch_version": str(getattr(torch, "__version__", "UNKNOWN")),
            "cuda_available": cuda_available,
            "selected_device": selected,
            "gpu_name": gpu_name,
        },
        [],
    )


def preflight_official_openai_clip(
    paths: OfficialAssetPaths,
    *,
    requested_device: str = "auto",
) -> tuple[dict[str, Any], list[dict[str, Any]], ModuleType | None]:
    from triage_eg.retrieval.stage1b.assets import issue, sha256_file

    issues: list[dict[str, Any]] = []
    dependency_wheels = _offline_dependency_wheels(paths.asset_root)
    with _temporary_dependency_paths(dependency_wheels):
        module, module_details, module_codes = controlled_import_clip(paths.source_root)
    module_details["offline_dependency_wheels"] = [str(path) for path in dependency_wheels]
    for code in module_codes:
        issues.append(
            issue(
                "ERROR",
                code,
                None,
                paths.source_root,
                dependency=module_details.get("missing_dependency"),
            )
        )
    tokenizer_path = paths.source_root / TOKENIZER_ASSET
    if not tokenizer_path.is_file():
        issues.append(
            issue(
                "ERROR",
                "OPENAI_CLIP_TOKENIZER_ASSET_MISSING",
                None,
                tokenizer_path,
            )
        )
    checkpoint_ok = paths.checkpoint_path.is_file() and paths.checkpoint_path.stat().st_size > 0
    if not checkpoint_ok:
        code = (
            "ENCODER_CHECKPOINT_INVALID"
            if paths.checkpoint_path.exists()
            else "ENCODER_ASSET_NOT_FOUND"
        )
        issues.append(issue("ERROR", code, None, paths.checkpoint_path))
    checkpoint_hash = sha256_file(paths.checkpoint_path) if checkpoint_ok else None
    checkpoint_size = paths.checkpoint_path.stat().st_size if checkpoint_ok else 0
    manifest, manifest_codes = _read_manifest(paths.asset_manifest_path, paths.asset_root)
    for code in manifest_codes:
        issues.append(issue("ERROR", code, None, paths.asset_manifest_path))
    sha_file = paths.asset_root / "manifests/checkpoint.sha256"
    declared_hash = _read_declared_hash(sha_file)
    manifest_hash = str(manifest.get("checkpoint_sha256", "")).lower() if manifest else None
    expected_hashes = {value for value in (declared_hash, manifest_hash) if value}
    if not expected_hashes:
        issues.append(issue("WARNING", "CHECKPOINT_DECLARED_HASH_MISSING", None, sha_file))
    elif checkpoint_hash and (len(expected_hashes) != 1 or checkpoint_hash not in expected_hashes):
        issues.append(
            issue(
                "ERROR",
                "CHECKPOINT_HASH_MISMATCH",
                None,
                paths.checkpoint_path,
                expected=sorted(expected_hashes),
                actual=checkpoint_hash,
            )
        )
    source_commit_path = paths.asset_root / "manifests/SOURCE_COMMIT.txt"
    source_commit = (
        source_commit_path.read_text(encoding="utf-8").strip()
        if source_commit_path.is_file()
        else str(manifest.get("source_commit", "")).strip()
        if manifest
        else ""
    )
    if not source_commit or source_commit.upper() == "UNKNOWN":
        issues.append(issue("WARNING", "SOURCE_COMMIT_UNKNOWN", None, source_commit_path))
    device_missing_dependency = None
    try:
        import torch

        device, device_details, device_codes = _device_details(torch, requested_device)
    except ImportError as error:
        device_missing_dependency = error.name or "torch"
        device = requested_device
        device_details = {
            "torch_version": None,
            "cuda_available": False,
            "selected_device": device,
            "gpu_name": None,
        }
        device_codes = ["ENCODER_DEPENDENCY_NOT_AVAILABLE"]
    for code in device_codes:
        issues.append(
            issue(
                "ERROR",
                code,
                None,
                None,
                dependency=device_missing_dependency,
            )
        )
    try:
        from PIL import Image  # noqa: F401

        image_library_available = True
    except ImportError:
        image_library_available = False
        issues.append(
            issue(
                "ERROR",
                "ENCODER_DEPENDENCY_NOT_AVAILABLE",
                None,
                None,
                dependency="PIL",
            )
        )
    manifest_checkpoint_valid = True
    if manifest:
        manifest_checkpoint = paths.asset_root / str(manifest.get("checkpoint_relative_path", ""))
        manifest_checkpoint_valid = (
            manifest_checkpoint.resolve(strict=False) == paths.checkpoint_path
        )
        if not manifest_checkpoint_valid:
            issues.append(
                issue(
                    "ERROR",
                    "ASSET_MANIFEST_MALFORMED",
                    None,
                    paths.asset_manifest_path,
                )
            )
    blockers = [item for item in issues if item["severity"] == "ERROR"]
    normalized_checkpoint = str(paths.checkpoint_path).replace("\\", "/")
    provenance = {
        **module_details,
        **device_details,
        "asset_root": str(paths.asset_root),
        "source_root": str(paths.source_root),
        "checkpoint_path": str(paths.checkpoint_path),
        "checkpoint_filename": paths.checkpoint_path.name,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_fingerprint": checkpoint_hash,
        "declared_checkpoint_sha256": declared_hash or manifest_hash,
        "declared_hash_match": bool(checkpoint_hash and expected_hashes == {checkpoint_hash}),
        "asset_manifest_path": str(paths.asset_manifest_path),
        "asset_manifest": manifest,
        "manifest_checkpoint_valid": manifest_checkpoint_valid,
        "source_repository": (manifest.get("source_repository") if manifest else None),
        "source_commit": (
            source_commit if source_commit and source_commit.upper() != "UNKNOWN" else None
        ),
        "tokenizer_asset_path": str(tokenizer_path),
        "tokenizer_asset_available": tokenizer_path.is_file(),
        "image_library_available": image_library_available,
        "asset_kind": "FILE" if checkpoint_ok else None,
        "asset_available": checkpoint_ok,
        "asset_source": (
            "KAGGLE_INPUT"
            if normalized_checkpoint.startswith("/kaggle/input/")
            else "LOCAL"
            if checkpoint_ok
            else "UNKNOWN"
        ),
        "selected_device": device,
        "reproducible": not blockers,
        "offline_runtime_required": True,
    }
    return provenance, issues, module


@contextmanager
def _blocked_download_helpers(module: ModuleType) -> Iterator[None]:
    patched: list[tuple[ModuleType, str, Any]] = []

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise NetworkDownloadAttempted("NETWORK_DOWNLOAD_ATTEMPTED")

    for candidate in (module, sys.modules.get("clip.clip")):
        if candidate is not None and hasattr(candidate, "_download"):
            patched.append((candidate, "_download", candidate._download))
            candidate._download = blocked
    try:
        yield
    finally:
        for target, _, original in patched:
            target._download = original


def _numpy_output(values: Any, rows: int) -> tuple[np.ndarray, str, np.ndarray]:
    detached = values.detach() if hasattr(values, "detach") else values
    original_dtype = str(getattr(detached, "dtype", np.asarray(detached).dtype))
    if hasattr(detached, "float"):
        detached = detached.float()
    if hasattr(detached, "cpu"):
        detached = detached.cpu()
    if hasattr(detached, "numpy"):
        detached = detached.numpy()
    matrix = np.asarray(detached, dtype=np.float32)
    if matrix.shape != (rows, 512):
        raise ValueError("ENCODER_OUTPUT_DIMENSION_MISMATCH")
    if not np.isfinite(matrix).all():
        raise ValueError("ENCODER_OUTPUT_NON_FINITE")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("ENCODER_OUTPUT_ZERO_NORM")
    return matrix, original_dtype, norms


class OfficialOpenAIClipAdapter:
    """One-device official OpenAI CLIP adapter using only explicit local assets."""

    def __init__(
        self,
        contract: CandidateContract,
        paths: OfficialAssetPaths,
        module: ModuleType,
        provenance: dict[str, Any],
    ) -> None:
        import torch
        from PIL import Image

        self.contract = contract
        self.paths = paths
        self._clip = module
        self._torch = torch
        self._image = Image
        self.device = str(provenance["selected_device"])
        checkpoint = paths.checkpoint_path.resolve(strict=True)
        if not checkpoint.is_absolute() or not checkpoint.is_file():
            raise FileNotFoundError(f"ENCODER_ASSET_NOT_FOUND: {checkpoint}")
        try:
            with _blocked_download_helpers(module):
                self._model, self._preprocess = module.load(
                    str(checkpoint), device=self.device, jit=False
                )
        except NetworkDownloadAttempted:
            raise
        except Exception as error:
            raise RuntimeError(f"ENCODER_CHECKPOINT_LOAD_FAILED: {error}") from error
        if hasattr(self._model, "eval"):
            self._model.eval()
        first_parameter = (
            next(iter(self._model.parameters()), None)
            if hasattr(self._model, "parameters")
            else None
        )
        model_dtype = str(first_parameter.dtype) if first_parameter is not None else "UNKNOWN"
        self._runtime = {
            **provenance,
            "preprocess_source": "official_clip_load_return_value",
            "preprocess_repr": repr(self._preprocess),
            "manual_preprocess_override": False,
            "model_parameter_dtype": model_dtype,
            "batch_size": contract.batch_size,
            "text_truncate": contract.text_truncate,
        }
        self.last_image_metrics: dict[str, Any] = {}
        self.last_text_metrics: dict[str, Any] = {}

    @classmethod
    def load(cls, contract: CandidateContract) -> OfficialOpenAIClipAdapter:
        paths = resolve_official_asset_paths(
            source_root=contract.source_root,
            checkpoint_path=contract.checkpoint_path,
            asset_manifest_path=contract.asset_manifest_path,
        )
        provenance, issues, module = preflight_official_openai_clip(
            paths, requested_device=contract.device
        )
        blockers = [item for item in issues if item["severity"] == "ERROR"]
        if blockers or module is None:
            code = blockers[0]["code"] if blockers else "OPENAI_CLIP_PACKAGE_INVALID"
            raise RuntimeError(code)
        return cls(contract, paths, module, provenance)

    def _normalize(self, matrix: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
        norms = np.linalg.norm(matrix, axis=1)
        result = matrix / norms[:, None] if enabled else matrix
        return (
            result.astype(np.float32, copy=False),
            np.linalg.norm(result, axis=1),
        )

    def _synchronize(self) -> None:
        cuda = getattr(self._torch, "cuda", None)
        if self.device.startswith("cuda") and cuda is not None and hasattr(cuda, "synchronize"):
            cuda.synchronize()

    def _encode_image_inputs(self, inputs: Sequence[Any], *, kind: str) -> np.ndarray:
        outputs, raw_norms, normalized_norms, latencies = [], [], [], []
        input_dtype = None
        output_dtype = None
        preprocessing_seconds = 0.0
        inference_seconds = 0.0
        batch_size = max(1, self.contract.batch_size)
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start : start + batch_size]
            tensors = []
            preprocess_started = monotonic()
            for value in batch_inputs:
                try:
                    if kind == "path":
                        with self._image.open(value) as image:
                            tensors.append(self._preprocess(image))
                    else:
                        tensors.append(self._preprocess(value))
                except Exception as error:
                    raise ValueError(f"IMAGE_PREPROCESS_FAILED: {value}: {error}") from error
            preprocessing_seconds += monotonic() - preprocess_started
            batch_started = monotonic()
            stacked = self._torch.stack(tensors).to(self.device)
            input_dtype = str(stacked.dtype)
            self._synchronize()
            with self._torch.no_grad():
                raw = self._model.encode_image(stacked)
            self._synchronize()
            elapsed = monotonic() - batch_started
            inference_seconds += elapsed
            matrix, output_dtype, norms = _numpy_output(raw, len(batch_inputs))
            normalized, final_norms = self._normalize(
                matrix, self.contract.image_embedding_normalization
            )
            outputs.append(normalized)
            raw_norms.extend(norms.tolist())
            normalized_norms.extend(final_norms.tolist())
            latencies.extend([elapsed / len(batch_inputs)] * len(batch_inputs))
        result = np.concatenate(outputs) if outputs else np.empty((0, 512), dtype=np.float32)
        total_seconds = preprocessing_seconds + inference_seconds
        self.last_image_metrics = {
            "raw_norms": raw_norms,
            "normalized_norms": normalized_norms,
            "latency_seconds": latencies,
            "image_input_dtype": input_dtype,
            "model_output_dtype": output_dtype,
            "output_dtype": str(result.dtype),
            "normalized_output": self.contract.image_embedding_normalization,
            "input_kind": kind,
            "image_count": len(inputs),
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
            "images_per_second": len(inputs) / total_seconds if total_seconds > 0 else None,
        }
        return result

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        return self._encode_image_inputs(paths, kind="path")

    def encode_pil_images(self, images: Sequence[Any]) -> np.ndarray:
        return self._encode_image_inputs(images, kind="pil")

    def encode_rgb_arrays(self, images: Sequence[np.ndarray]) -> np.ndarray:
        pil_images = []
        for index, image in enumerate(images):
            value = np.asarray(image)
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(
                    f"RGB_ARRAY_INVALID: index={index} shape={value.shape} dtype={value.dtype}"
                )
            pil_images.append(self._image.fromarray(value, mode="RGB"))
        try:
            return self._encode_image_inputs(pil_images, kind="rgb_array")
        finally:
            for image in pil_images:
                image.close()

    def _tokenize(self, texts: Sequence[str]) -> tuple[Any, list[bool]]:
        truncated = [False] * len(texts)
        if self.contract.text_truncate:
            for index, text in enumerate(texts):
                try:
                    self._clip.tokenize([text], truncate=False)
                except RuntimeError:
                    truncated[index] = True
        try:
            tokens = self._clip.tokenize(list(texts), truncate=self.contract.text_truncate)
            return tokens, truncated
        except RuntimeError as error:
            code = (
                "TEXT_CONTEXT_LENGTH_EXCEEDED"
                if not self.contract.text_truncate
                else "TEXT_TOKENIZATION_FAILED"
            )
            raise ValueError(f"{code}: {error}") from error
        except Exception as error:
            raise ValueError(f"TEXT_TOKENIZATION_FAILED: {error}") from error

    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens, truncated = self._tokenize(texts)
        tokens = tokens.to(self.device) if hasattr(tokens, "to") else tokens
        started = monotonic()
        with self._torch.no_grad():
            raw = self._model.encode_text(tokens)
        matrix, output_dtype, raw_norms = _numpy_output(raw, len(texts))
        normalized, normalized_norms = self._normalize(
            matrix, self.contract.text_embedding_normalization
        )
        latency = (monotonic() - started) / max(1, len(texts))
        self.last_text_metrics = {
            "tokenization_status": ["SUCCESS"] * len(texts),
            "text_was_truncated": truncated,
            "raw_norms": raw_norms.tolist(),
            "normalized_norms": normalized_norms.tolist(),
            "latency_seconds": [latency] * len(texts),
            "model_output_dtype": output_dtype,
            "output_dtype": str(normalized.dtype),
            "normalized_output": self.contract.text_embedding_normalization,
        }
        return normalized

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_text(list(texts))

    def runtime_manifest(self) -> dict[str, Any]:
        return {
            **self._runtime,
            "image_execution": dict(self.last_image_metrics),
            "text_execution": dict(self.last_text_metrics),
        }

    def close(self) -> None:
        self._model = None
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
