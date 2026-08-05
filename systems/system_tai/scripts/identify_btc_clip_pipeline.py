"""Identify an optional BTC ViT-B/32 image pipeline without implementing retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CANDIDATE_NAMES = (
    "openai_clip",
    "open_clip_vit_b32_openai",
    "open_clip_vit_b32_quickgelu_openai",
    "huggingface_clip",
)
CANDIDATE_DESCRIPTORS: dict[str, dict[str, str]] = {
    "openai_clip": {
        "implementation": "official_openai_clip",
        "model_name": "ViT-B/32",
    },
    "open_clip_vit_b32_openai": {
        "implementation": "open_clip",
        "model_name": "ViT-B-32",
        "pretrained_tag": "openai",
    },
    "open_clip_vit_b32_quickgelu_openai": {
        "implementation": "open_clip",
        "model_name": "ViT-B-32-quickgelu",
        "pretrained_tag": "openai",
    },
    "huggingface_clip": {
        "implementation": "transformers",
        "model_name": "openai/clip-vit-base-patch32",
    },
}
OPENAI_CLIP_MODEL_NAME = "ViT-B/32"
OPENAI_CLIP_CACHE_FILENAME = "ViT-B-32.pt"


class BackendUnavailable(RuntimeError):
    """An optional image-encoder backend cannot run in the current environment."""

    def __init__(self, message: str, *, identifiers: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.identifiers = identifiers or {}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return repr(value)
    return value


def _load_mapping(path: Path) -> tuple[dict[str, Any], ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mapping CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    seen_frames: set[int] = set()
    previous_order: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "pts_time", "fps", "frame_idx"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"mapping CSV missing columns: {', '.join(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            try:
                order = int((raw.get("n") or "").strip())
                frame_idx = int((raw.get("frame_idx") or "").strip())
                pts_time = float((raw.get("pts_time") or "").strip())
                fps = float((raw.get("fps") or "").strip())
            except ValueError as exc:
                raise ValueError(f"invalid mapping value at line {line_number}") from exc
            if order < 0 or frame_idx < 0 or pts_time < 0 or fps <= 0:
                raise ValueError(f"invalid mapping value at line {line_number}")
            if not np.isfinite(pts_time) or not np.isfinite(fps):
                raise ValueError(f"non-finite mapping value at line {line_number}")
            if order in seen_orders:
                raise ValueError(f"duplicate keyframe order: {order}")
            if previous_order is not None and order <= previous_order:
                raise ValueError(
                    "keyframe order must be strictly increasing for validated "
                    f"feature-row alignment: previous={previous_order}, current={order}"
                )
            if frame_idx in seen_frames:
                raise ValueError(f"ambiguous duplicate frame_idx: {frame_idx}")
            seen_orders.add(order)
            seen_frames.add(frame_idx)
            previous_order = order
            rows.append(
                {
                    "keyframe_order": order,
                    "actual_frame_id": frame_idx,
                    "clip_row": len(rows),
                    "pts_time": pts_time,
                    "fps": fps,
                }
            )
    if not rows:
        raise ValueError("mapping CSV contains no records")
    return tuple(rows)


def _select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    sample_count: int,
    explicit_orders: Sequence[int] | None,
) -> tuple[dict[str, Any], ...]:
    if explicit_orders:
        by_order = {row["keyframe_order"]: row for row in rows}
        missing = [order for order in explicit_orders if order not in by_order]
        if missing:
            raise ValueError(f"unknown keyframe orders: {missing}")
        return tuple(by_order[order] for order in explicit_orders)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(sample_count, len(rows))
    indices = (
        [0]
        if count == 1
        else [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    )
    return tuple(rows[index] for index in indices)


def _trailing_number(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _resolve_keyframes(source: Path, orders: Sequence[int]) -> list[Path]:
    source = Path(source)
    if source.is_file():
        if len(orders) != 1:
            raise ValueError("a single keyframe file requires one sampled order")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"keyframe path not found: {source}")
    indexed: dict[int, list[Path]] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            order = _trailing_number(path)
            if order is not None:
                indexed.setdefault(order, []).append(path)
    resolved: list[Path] = []
    for order in orders:
        matches = sorted(indexed.get(order, []), key=lambda path: str(path).lower())
        if not matches:
            raise FileNotFoundError(f"no keyframe image found for order {order}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous keyframe images for order {order}: {matches}")
        resolved.append(matches[0])
    return resolved


def _norm_stats(matrix: NDArray[np.floating[Any]]) -> dict[str, float]:
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "minimum": float(np.min(norms)),
        "maximum": float(np.max(norms)),
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
    }


def compute_alignment_metrics(
    original: NDArray[np.number], candidate: NDArray[np.number]
) -> dict[str, Any]:
    if original.ndim != 2 or candidate.ndim != 2:
        raise ValueError("original and candidate embeddings must be two-dimensional")
    if original.shape != candidate.shape:
        raise ValueError(
            f"embedding shape mismatch: original={original.shape}, candidate={candidate.shape}"
        )
    if original.shape[0] == 0 or original.shape[1] == 0:
        raise ValueError("embedding matrices must be non-empty")
    original_float = original.astype(np.float64, copy=False)
    candidate_float = candidate.astype(np.float64, copy=False)
    if not np.isfinite(original_float).all() or not np.isfinite(candidate_float).all():
        raise ValueError("embedding matrices must contain only finite values")
    original_norms = np.linalg.norm(original_float, axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_float, axis=1, keepdims=True)
    if np.any(original_norms == 0) or np.any(candidate_norms == 0):
        raise ValueError("embedding matrices must not contain zero-norm rows")
    original_unit = original_float / original_norms
    candidate_unit = candidate_float / candidate_norms
    row_cosine = np.sum(original_unit * candidate_unit, axis=1)
    normalized_l2 = np.linalg.norm(original_unit - candidate_unit, axis=1)
    similarity = original_unit @ candidate_unit.T
    ranks: list[int] = []
    top1_matches = 0
    for row_index, scores in enumerate(similarity):
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == row_index)[0]) + 1
        ranks.append(rank)
        top1_matches += int(order[0] == row_index)
    return {
        "row_count": original.shape[0],
        "dimension": original.shape[1],
        "row_wise_cosine": {
            "mean": float(np.mean(row_cosine)),
            "median": float(np.median(row_cosine)),
            "minimum": float(np.min(row_cosine)),
            "p05": float(np.percentile(row_cosine, 5)),
        },
        "normalized_l2_distance": {
            "mean": float(np.mean(normalized_l2)),
            "median": float(np.median(normalized_l2)),
            "maximum": float(np.max(normalized_l2)),
        },
        "maximum_absolute_difference": float(np.max(np.abs(original_float - candidate_float))),
        "normalized_maximum_absolute_difference": float(
            np.max(np.abs(original_unit - candidate_unit))
        ),
        "self_match_top1_accuracy": top1_matches / original.shape[0],
        "mean_self_match_rank": float(np.mean(ranks)),
        "self_match_ranks": ranks,
        "original_norm_statistics": _norm_stats(original_float),
        "candidate_norm_statistics": _norm_stats(candidate_float),
    }


def _torch_device(torch: Any) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_openai_clip_public_api(clip_module: Any) -> tuple[str, ...]:
    if not callable(getattr(clip_module, "load", None)) or not callable(
        getattr(clip_module, "available_models", None)
    ):
        raise BackendUnavailable(
            "installed 'clip' package does not expose the official public CLIP APIs",
            identifiers={"package_path": getattr(clip_module, "__file__", "unknown")},
        )
    try:
        available_models = tuple(str(name) for name in clip_module.available_models())
    except Exception as exc:
        raise BackendUnavailable(
            f"OpenAI CLIP available_models() failed: {exc}",
            identifiers={"package_path": getattr(clip_module, "__file__", "unknown")},
        ) from exc
    if OPENAI_CLIP_MODEL_NAME not in available_models:
        raise BackendUnavailable(
            f"official OpenAI CLIP does not list {OPENAI_CLIP_MODEL_NAME}",
            identifiers={
                "package_path": getattr(clip_module, "__file__", "unknown"),
                "available_models": list(available_models),
            },
        )
    return available_models


def _encode_openai_clip(
    image_paths: Sequence[Path], *, allow_download: bool
) -> tuple[NDArray[np.number], dict[str, Any]]:
    try:
        import clip  # type: ignore[import-not-found]
        import torch
        from PIL import Image
    except ImportError as exc:
        raise BackendUnavailable(f"optional package unavailable: {exc}") from exc
    available_models = _validate_openai_clip_public_api(clip)
    cache_root = Path.home() / ".cache" / "clip"
    checkpoint = cache_root / OPENAI_CLIP_CACHE_FILENAME
    identifiers = {
        "library": "openai-clip",
        "library_version": getattr(clip, "__version__", "unknown"),
        "package_path": getattr(clip, "__file__", "unknown"),
        "available_models": list(available_models),
        "model_name": OPENAI_CLIP_MODEL_NAME,
        "checkpoint_cache_path": str(checkpoint),
        "checkpoint_cached_before_load": checkpoint.is_file(),
        "model_download_allowed": allow_download,
    }
    if not checkpoint.is_file() and not allow_download:
        raise BackendUnavailable(
            f"official OpenAI CLIP validated, but weights are not cached at {checkpoint}",
            identifiers=identifiers,
        )
    device = _torch_device(torch)
    try:
        model, preprocess = clip.load(
            OPENAI_CLIP_MODEL_NAME,
            device=device,
            jit=False,
            download_root=str(cache_root),
        )
        batch = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
        ).to(device)
        with torch.no_grad():
            embeddings = model.encode_image(batch).float().cpu().numpy()
    except Exception as exc:
        raise BackendUnavailable(
            f"OpenAI CLIP load/encode failed: {exc}", identifiers=identifiers
        ) from exc
    identifiers.update(
        {
            "checkpoint_cached_after_load": checkpoint.is_file(),
            "preprocessing": repr(preprocess),
            "device": device,
        }
    )
    return embeddings, identifiers


def _encode_open_clip_variant(
    image_paths: Sequence[Path],
    *,
    allow_download: bool,
    model_name: str,
    pretrained: str,
) -> tuple[NDArray[np.number], dict[str, Any]]:
    try:
        import open_clip  # type: ignore[import-not-found]
        import torch
        from PIL import Image
    except ImportError as exc:
        raise BackendUnavailable(f"optional package unavailable: {exc}") from exc
    if not allow_download:
        raise BackendUnavailable(
            "OpenCLIP pretrained resolution is disabled without --allow-model-download"
        )
    device = _torch_device(torch)
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        batch = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
        ).to(device)
        with torch.no_grad():
            embeddings = model.encode_image(batch).float().cpu().numpy()
    except Exception as exc:
        raise BackendUnavailable(f"OpenCLIP load/encode failed: {exc}") from exc
    get_model_config = getattr(open_clip, "get_model_config", None)
    resolved_config = get_model_config(model_name) if callable(get_model_config) else None
    return embeddings, {
        "library": "open_clip",
        "library_version": getattr(open_clip, "__version__", "unknown"),
        "package_path": getattr(open_clip, "__file__", "unknown"),
        "model_name": model_name,
        "activation_or_model_variant": (
            "QuickGELU" if "quickgelu" in model_name.lower() else "model_default"
        ),
        "pretrained_tag": pretrained,
        "checkpoint": pretrained,
        "resolved_model_configuration": _json_safe(resolved_config),
        "preprocessing": repr(preprocess),
        "device": device,
    }


def _encode_open_clip_vit_b32_openai(
    image_paths: Sequence[Path], *, allow_download: bool
) -> tuple[NDArray[np.number], dict[str, Any]]:
    return _encode_open_clip_variant(
        image_paths,
        allow_download=allow_download,
        model_name="ViT-B-32",
        pretrained="openai",
    )


def _encode_open_clip_vit_b32_quickgelu_openai(
    image_paths: Sequence[Path], *, allow_download: bool
) -> tuple[NDArray[np.number], dict[str, Any]]:
    return _encode_open_clip_variant(
        image_paths,
        allow_download=allow_download,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
    )


def _type_identifier(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _extract_huggingface_image_embeddings(
    output: Any, *, torch: Any
) -> tuple[Any, str]:
    output_type = _type_identifier(output)
    if torch.is_tensor(output):
        embeddings = output
    elif hasattr(output, "pooler_output"):
        embeddings = output.pooler_output
    elif isinstance(output, tuple):
        if len(output) == 1 and torch.is_tensor(output[0]):
            embeddings = output[0]
        elif len(output) >= 2 and torch.is_tensor(output[1]):
            embeddings = output[1]
        else:
            raise BackendUnavailable(
                "unsupported Hugging Face tuple output: expected projected pooled "
                "features at index 1 or a single Tensor",
                identifiers={"output_type": output_type},
            )
    else:
        raise BackendUnavailable(
            f"unsupported Hugging Face image-feature output type: {output_type}",
            identifiers={"output_type": output_type},
        )
    if not torch.is_tensor(embeddings):
        raise BackendUnavailable(
            f"Hugging Face pooled image feature is not a Tensor: {_type_identifier(embeddings)}",
            identifiers={"output_type": output_type},
        )
    return embeddings, output_type


def _encode_huggingface_clip(
    image_paths: Sequence[Path], *, allow_download: bool
) -> tuple[NDArray[np.number], dict[str, Any]]:
    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise BackendUnavailable(f"optional package unavailable: {exc}") from exc
    checkpoint = "openai/clip-vit-base-patch32"
    device = _torch_device(torch)
    output_type = "not_produced"
    base_identifiers = {
        "library": "transformers",
        "library_version": getattr(transformers, "__version__", "unknown"),
        "package_path": getattr(transformers, "__file__", "unknown"),
        "model_name": "CLIPModel",
        "model_checkpoint": checkpoint,
        "device": device,
    }
    try:
        model = CLIPModel.from_pretrained(checkpoint, local_files_only=not allow_download)
        processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=not allow_download)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        model = model.to(device)
        with torch.no_grad():
            output = model.get_image_features(pixel_values=pixel_values)
            embeddings, output_type = _extract_huggingface_image_embeddings(
                output, torch=torch
            )
            embeddings = embeddings.float().cpu().numpy()
    except BackendUnavailable as exc:
        raise BackendUnavailable(
            f"Hugging Face CLIP output compatibility failed: {exc}",
            identifiers={**base_identifiers, **exc.identifiers},
        ) from exc
    except Exception as exc:
        raise BackendUnavailable(
            f"Hugging Face CLIP load/encode failed: {exc}",
            identifiers={**base_identifiers, "output_type": output_type},
        ) from exc
    return embeddings, {
        **base_identifiers,
        "output_type": output_type,
        "preprocessing": processor.to_json_string(),
        "processor_type": _type_identifier(processor),
        "image_processor_metadata": _json_safe(
            getattr(getattr(processor, "image_processor", None), "to_dict", lambda: {})()
        ),
        "device": device,
    }


def run_candidate_backend(
    backend: str,
    image_paths: Sequence[Path],
    original: NDArray[np.number],
    *,
    allow_download: bool,
    encoder: Callable[..., tuple[NDArray[np.number], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    implementations: dict[str, Callable[..., tuple[NDArray[np.number], dict[str, Any]]]] = {
        "openai_clip": _encode_openai_clip,
        "open_clip_vit_b32_openai": _encode_open_clip_vit_b32_openai,
        "open_clip_vit_b32_quickgelu_openai": (
            _encode_open_clip_vit_b32_quickgelu_openai
        ),
        "huggingface_clip": _encode_huggingface_clip,
    }
    selected = encoder or implementations.get(backend)
    if selected is None:
        return {
            "status": "SKIPPED",
            "reason": f"unknown backend: {backend}",
            "candidate": CANDIDATE_DESCRIPTORS.get(backend, {"name": backend}),
        }
    try:
        candidate, metadata = selected(image_paths, allow_download=allow_download)
        metrics = compute_alignment_metrics(original, candidate)
    except BackendUnavailable as exc:
        return {
            "status": "SKIPPED",
            "reason": str(exc),
            "candidate": CANDIDATE_DESCRIPTORS.get(backend, {"name": backend}),
            "identifiers": exc.identifiers,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "UNVERIFIED",
            "reason": str(exc),
            "candidate": CANDIDATE_DESCRIPTORS.get(backend, {"name": backend}),
        }
    return {
        "status": "MEASURED",
        "candidate": CANDIDATE_DESCRIPTORS.get(backend, {"name": backend}),
        "identifiers": metadata,
        "preprocessing": metadata.get("preprocessing", "unreported"),
        "metrics": metrics,
    }


def prepare_case(
    *,
    video_id: str,
    mapping_csv: Path,
    clip_npy: Path,
    keyframes: Path,
    sample_count: int,
    keyframe_orders: Sequence[int] | None,
    expected_dimension: int | None = None,
) -> tuple[dict[str, Any], NDArray[np.number], list[Path]]:
    rows = _load_mapping(mapping_csv)
    matrix = np.load(Path(clip_npy), allow_pickle=False)
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("BTC CLIP artifact must contain a two-dimensional NumPy array")
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        raise ValueError(f"BTC CLIP matrix must have a real numeric dtype: {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise ValueError("BTC CLIP matrix contains NaN or Infinity")
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"BTC CLIP dimension mismatch: observed={matrix.shape[1]}, "
            f"expected={expected_dimension}"
        )
    if len(rows) != matrix.shape[0]:
        raise ValueError(
            f"mapping/feature row-count mismatch: mapping={len(rows)}, features={matrix.shape[0]}"
        )
    selected = _select_rows(rows, sample_count=sample_count, explicit_orders=keyframe_orders)
    image_paths = _resolve_keyframes(keyframes, [int(row["keyframe_order"]) for row in selected])
    clip_rows = [int(row["clip_row"]) for row in selected]
    original = matrix[clip_rows]
    case_info = {
        "video_id": video_id,
        "mapping_csv": str(Path(mapping_csv).resolve(strict=False)),
        "clip_npy": str(Path(clip_npy).resolve(strict=False)),
        "keyframes": str(Path(keyframes).resolve(strict=False)),
        "mapping_row_count": len(rows),
        "feature_shape": list(matrix.shape),
        "sampled_keyframe_orders": [int(row["keyframe_order"]) for row in selected],
        "sampled_actual_frame_ids": [int(row["actual_frame_id"]) for row in selected],
        "sampled_clip_rows": clip_rows,
        "keyframe_order_is_strictly_increasing": True,
        "feature_row_mapping_validated": True,
    }
    return case_info, original, image_paths


def classify_backends(
    cases: Sequence[dict[str, Any]],
    *,
    minimum_videos: int = 3,
    minimum_top1: float = 0.99,
    near_exact_mean_cosine: float = 0.999,
    near_exact_p05_cosine: float = 0.995,
    superior_mean_cosine: float = 0.95,
    superiority_margin: float = 0.02,
) -> dict[str, Any]:
    if minimum_videos < 2:
        raise ValueError("minimum_videos must be at least two for reproduction")
    discovered_names = {
        str(name)
        for case in cases
        for name in case.get("backends", {})
    }
    candidate_names = [
        name for name in DEFAULT_CANDIDATE_NAMES if name in discovered_names
    ]
    candidate_names.extend(sorted(discovered_names - set(candidate_names)))
    measured_means: dict[str, float] = {}
    for backend in candidate_names:
        results = [
            case["backends"][backend]
            for case in cases
            if backend in case.get("backends", {})
            and case["backends"][backend]["status"] == "MEASURED"
        ]
        if results:
            measured_means[backend] = float(
                np.mean([result["metrics"]["row_wise_cosine"]["mean"] for result in results])
            )

    summary: dict[str, Any] = {}
    measured_ranking = sorted(
        measured_means, key=lambda name: measured_means[name], reverse=True
    )
    for backend in candidate_names:
        backend_results = [
            case.get("backends", {}).get(backend, {"status": "SKIPPED", "reason": "not run"})
            for case in cases
        ]
        measured_pairs = [
            (case, case["backends"][backend])
            for case in cases
            if backend in case.get("backends", {})
            and case["backends"][backend]["status"] == "MEASURED"
        ]
        measured = [result for _case, result in measured_pairs]
        if not measured:
            reasons = sorted({result.get("reason", "unavailable") for result in backend_results})
            summary[backend] = {
                "status": "SKIPPED",
                "candidate": CANDIDATE_DESCRIPTORS.get(backend, {"name": backend}),
                "measured_unique_video_count": 0,
                "reasons": reasons,
                "identifiers": [
                    result["identifiers"]
                    for result in backend_results
                    if result.get("identifiers")
                ],
            }
            continue
        mean_cosines = [item["metrics"]["row_wise_cosine"]["mean"] for item in measured]
        p05_cosines = [item["metrics"]["row_wise_cosine"]["p05"] for item in measured]
        top1_values = [item["metrics"]["self_match_top1_accuracy"] for item in measured]
        mean_ranks = [item["metrics"]["mean_self_match_rank"] for item in measured]
        normalized_l2_values = [
            item["metrics"]["normalized_l2_distance"]["mean"] for item in measured
        ]
        backend_mean = float(np.mean(mean_cosines))
        ranking_index = measured_ranking.index(backend)
        next_candidate = (
            measured_ranking[ranking_index + 1]
            if ranking_index + 1 < len(measured_ranking)
            else None
        )
        margin = (
            backend_mean - measured_means[next_candidate]
            if next_candidate is not None
            else None
        )
        unique_video_ids = {str(case.get("video_id", "")) for case, _result in measured_pairs}
        mappings_validated = all(
            case.get("feature_row_mapping_validated") is True
            for case, _result in measured_pairs
        )
        reproduced = len(unique_video_ids) >= minimum_videos
        correct_self_match = min(top1_values) >= minimum_top1 and max(mean_ranks) <= 1.05
        near_exact = (
            min(mean_cosines) >= near_exact_mean_cosine
            and min(p05_cosines) >= near_exact_p05_cosine
        )
        clearly_superior = (
            min(mean_cosines) >= superior_mean_cosine
            and ranking_index == 0
            and margin is not None
            and margin >= superiority_margin
        )
        identified = (
            reproduced
            and mappings_validated
            and correct_self_match
            and (near_exact or clearly_superior)
        )
        identifier_map = {
            json.dumps(_json_safe(item.get("identifiers", {})), sort_keys=True): _json_safe(
                item.get("identifiers", {})
            )
            for item in measured
        }
        summary[backend] = {
            "status": "IDENTIFIED" if identified else "UNVERIFIED",
            "candidate": CANDIDATE_DESCRIPTORS.get(backend, {"name": backend}),
            "measured_video_count": len(measured),
            "unique_measured_video_count": len(unique_video_ids),
            "measured_unique_video_count": len(unique_video_ids),
            "all_feature_row_mappings_validated": mappings_validated,
            "reproduction_requirement_met": reproduced,
            "correct_self_match": correct_self_match,
            "near_exact": near_exact,
            "clearly_superior": clearly_superior,
            "mean_row_wise_cosine_across_videos": float(np.mean(mean_cosines)),
            "row_wise_cosine_mean": float(np.mean(mean_cosines)),
            "minimum_p05_cosine_across_videos": float(min(p05_cosines)),
            "row_wise_cosine_p05": float(min(p05_cosines)),
            "minimum_self_match_top1_across_videos": float(min(top1_values)),
            "self_match_top1": float(min(top1_values)),
            "maximum_mean_self_match_rank_across_videos": float(max(mean_ranks)),
            "mean_self_match_rank": float(np.mean(mean_ranks)),
            "normalized_l2": float(np.mean(normalized_l2_values)),
            "mean_cosine_margin_over_next_backend": margin,
            "margin_over_next_measured_candidate": margin,
            "next_measured_candidate": next_candidate,
            "identifiers": list(identifier_map.values()),
        }
    return summary


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("batch manifest must contain a non-empty cases list")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id")
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--clip-npy", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--keyframe-orders", type=int, nargs="+")
    parser.add_argument("--expected-dimension", type=int)
    parser.add_argument("--backend", action="append", choices=DEFAULT_CANDIDATE_NAMES)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--minimum-identification-videos", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _single_case(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "video_id": args.video_id,
        "mapping_csv": args.mapping_csv,
        "clip_npy": args.clip_npy,
        "keyframes": args.keyframes,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"single-case execution missing arguments: {', '.join(missing)}")
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    enabled = args.backend or list(DEFAULT_CANDIDATE_NAMES)
    try:
        raw_cases = (
            _load_cases(args.batch_manifest) if args.batch_manifest else [_single_case(args)]
        )
        cases: list[dict[str, Any]] = []
        for raw in raw_cases:
            configured_dimension = raw.get("expected_dimension", args.expected_dimension)
            case_info, original, image_paths = prepare_case(
                video_id=str(raw["video_id"]),
                mapping_csv=Path(raw["mapping_csv"]),
                clip_npy=Path(raw["clip_npy"]),
                keyframes=Path(raw["keyframes"]),
                sample_count=int(raw.get("sample_count", args.sample_count)),
                keyframe_orders=raw.get("keyframe_orders", args.keyframe_orders),
                expected_dimension=(
                    int(configured_dimension) if configured_dimension is not None else None
                ),
            )
            case_info["backends"] = {
                backend: run_candidate_backend(
                    backend,
                    image_paths,
                    original,
                    allow_download=args.allow_model_download,
                )
                for backend in enabled
            }
            cases.append(case_info)
        summary = classify_backends(cases, minimum_videos=args.minimum_identification_videos)
        identified = [name for name, result in summary.items() if result["status"] == "IDENTIFIED"]
        output = {
            "status": "IDENTIFIED" if identified else "UNVERIFIED",
            "identified_backends": identified,
            "text_query_encoder_implemented": False,
            "dimension_alone_is_sufficient": False,
            "identification_criteria": {
                "minimum_unique_videos": args.minimum_identification_videos,
                "validated_feature_row_mapping_required": True,
                "correct_self_match_required": True,
                "near_exact_or_clearly_superior_required": True,
            },
            "case_count": len(cases),
            "cases": cases,
            "backend_summary": summary,
        }
        exit_code = 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        output = {
            "status": "ERROR",
            "error": str(exc),
            "text_query_encoder_implemented": False,
        }
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
