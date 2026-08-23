"""Bounded structured visual verification for KIS timeline candidates.

This module is deliberately isolated from canonical CLIP retrieval.  It verifies a
small, automatically selected set of raw-video frames and never accepts target video,
timestamp, frame, benchmark label, or ground truth inputs.
"""

from __future__ import annotations

import importlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


class VisualVerificationError(RuntimeError):
    """The optional visual verifier could not produce a trustworthy result."""


@dataclass(frozen=True, slots=True)
class VisualVerificationInput:
    video_id: str
    absolute_frame_id: int
    timestamp_seconds: float
    images: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("visual verification video_id must not be empty")
        if self.absolute_frame_id < 0:
            raise ValueError("visual verification frame ID must be non-negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("visual verification timestamp must be finite and non-negative")
        if not self.images:
            raise ValueError("visual verification requires at least one image")


@dataclass(frozen=True, slots=True)
class VisualPredicateScore:
    requirement: str
    score: float
    visible: bool
    evidence: str

    def __post_init__(self) -> None:
        if not self.requirement.strip():
            raise ValueError("visual predicate requirement must not be empty")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("visual predicate score must be in [0, 1]")
        if type(self.visible) is not bool:
            raise ValueError("visual predicate visible must be boolean")


@dataclass(frozen=True, slots=True)
class VisualVerificationResult:
    video_id: str
    absolute_frame_id: int
    match_score: float
    requirement_coverage: float
    all_visible_requirements_satisfied: bool
    predicates: tuple[VisualPredicateScore, ...]
    summary: str

    def __post_init__(self) -> None:
        if not self.video_id.strip() or self.absolute_frame_id < 0:
            raise ValueError("invalid visual verification identity")
        for field_name, value in (
            ("match_score", self.match_score),
            ("requirement_coverage", self.requirement_coverage),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if type(self.all_visible_requirements_satisfied) is not bool:
            raise ValueError("all_visible_requirements_satisfied must be boolean")
        if not self.predicates:
            raise ValueError("visual verification must contain predicate scores")

    def to_trace(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "video_id": self.video_id,
                "absolute_frame_id": self.absolute_frame_id,
                "match_score": self.match_score,
                "requirement_coverage": self.requirement_coverage,
                "all_visible_requirements_satisfied": (
                    self.all_visible_requirements_satisfied
                ),
                "predicates": [
                    {
                        "requirement": predicate.requirement,
                        "score": predicate.score,
                        "visible": predicate.visible,
                        "evidence": predicate.evidence,
                    }
                    for predicate in self.predicates
                ],
                "summary": self.summary,
            }
        )


class StructuredVisualVerifier(Protocol):
    identifiers: Mapping[str, Any]

    def verify(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidates: Sequence[VisualVerificationInput],
    ) -> tuple[VisualVerificationResult, ...]: ...


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VisualVerificationError(f"duplicate JSON key from visual verifier: {key}")
        result[key] = value
    return result


def parse_visual_verification_json(
    text: str,
    *,
    video_id: str,
    absolute_frame_id: int,
) -> VisualVerificationResult:
    """Parse one bounded JSON object; markdown fences and surrounding prose are tolerated."""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise VisualVerificationError("visual verifier response did not contain a JSON object")
    try:
        payload = json.loads(
            stripped[start : end + 1],
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise VisualVerificationError(f"invalid visual verifier JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VisualVerificationError("visual verifier response must be an object")
    raw_predicates = payload.get("predicates")
    if not isinstance(raw_predicates, list) or not raw_predicates:
        raise VisualVerificationError("visual verifier predicates must be a non-empty list")
    predicates: list[VisualPredicateScore] = []
    for index, item in enumerate(raw_predicates, start=1):
        if not isinstance(item, dict):
            raise VisualVerificationError(f"predicate {index} must be an object")
        try:
            predicates.append(
                VisualPredicateScore(
                    requirement=str(item["requirement"]),
                    score=float(item["score"]),
                    visible=item["visible"],
                    evidence=str(item.get("evidence", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VisualVerificationError(f"invalid predicate {index}: {exc}") from exc
    try:
        return VisualVerificationResult(
            video_id=video_id,
            absolute_frame_id=absolute_frame_id,
            match_score=float(payload["match_score"]),
            requirement_coverage=float(payload["requirement_coverage"]),
            all_visible_requirements_satisfied=payload[
                "all_visible_requirements_satisfied"
            ],
            predicates=tuple(predicates),
            summary=str(payload.get("summary", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualVerificationError(f"invalid visual verifier result: {exc}") from exc


class HuggingFaceStructuredVisualVerifier:
    """Optional local Hugging Face VLM adapter loaded once per operational session."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: str,
        allow_model_download: bool,
        cache_dir: Path | None,
        max_new_tokens: int,
        max_image_pixels: int | None = None,
        execution_profile: str = "full",
        progress_callback: Callable[[str], None] | None = None,
        transformers_module: Any | None = None,
        torch_module: Any | None = None,
        image_module: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("visual verifier model_name must not be empty")
        if device not in {"cpu", "cuda"}:
            raise ValueError("visual verifier device must be cpu or cuda")
        if max_new_tokens <= 0:
            raise ValueError("visual verifier max_new_tokens must be positive")
        if max_image_pixels is not None and max_image_pixels <= 0:
            raise ValueError("visual verifier max_image_pixels must be positive")
        try:
            transformers = transformers_module or importlib.import_module("transformers")
            torch = torch_module or importlib.import_module("torch")
            image_api = image_module or importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise VisualVerificationError(
                f"visual verifier optional dependency unavailable: {exc}"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise VisualVerificationError("CUDA visual verification requested but unavailable")
        processor_class = getattr(transformers, "AutoProcessor", None)
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if processor_class is None or model_class is None:
            raise VisualVerificationError(
                "installed transformers lacks an image-text generation model adapter"
            )
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": not allow_model_download,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        processor_kwargs = dict(load_kwargs)
        if max_image_pixels is not None:
            processor_kwargs["max_pixels"] = max_image_pixels
        try:
            self._processor = processor_class.from_pretrained(
                model_name,
                **processor_kwargs,
            )
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._model = model_class.from_pretrained(
                model_name,
                torch_dtype=dtype,
                **load_kwargs,
            ).to(device)
            self._model.eval()
        except Exception as exc:
            raise VisualVerificationError(f"visual verifier model load failed: {exc}") from exc
        self._torch = torch
        self._image_api = image_api
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._progress_callback = progress_callback
        self.identifiers: Mapping[str, Any] = MappingProxyType(
            {
                "provider": "huggingface-structured-visual-verifier",
                "model": model_name,
                "revision": revision,
                "device": device,
                "model_download_allowed": allow_model_download,
                "candidate_batching": "one-temporal-candidate-per-generation",
                "execution_profile": execution_profile,
                "max_new_tokens": max_new_tokens,
                "max_image_pixels": max_image_pixels,
            }
        )

    def verify(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidates: Sequence[VisualVerificationInput],
    ) -> tuple[VisualVerificationResult, ...]:
        if not query_vi.strip() or not query_en.strip():
            raise ValueError("visual verification requires Vietnamese and English query text")
        results: list[VisualVerificationResult] = []
        total = len(candidates)
        for index, candidate in enumerate(candidates, start=1):
            started = time.perf_counter()
            self._progress(
                f"visual verifier candidate {index}/{total} started: "
                f"{candidate.video_id}/{candidate.absolute_frame_id} "
                f"images={len(candidate.images)}"
            )
            prompt = self._build_prompt(query_vi=query_vi, query_en=query_en)
            images = [self._to_rgb_image(image) for image in candidate.images]
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            try:
                rendered = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self._processor(
                    text=[rendered],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {
                    key: value.to(self._device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                inference_context = getattr(
                    self._torch,
                    "inference_mode",
                    self._torch.no_grad,
                )
                with inference_context():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                    )
                input_length = int(inputs["input_ids"].shape[1])
                decoded = self._processor.batch_decode(
                    generated[:, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception as exc:
                raise VisualVerificationError(
                    f"visual verification generation failed for "
                    f"{candidate.video_id}/{candidate.absolute_frame_id}: {exc}"
                ) from exc
            results.append(
                parse_visual_verification_json(
                    decoded,
                    video_id=candidate.video_id,
                    absolute_frame_id=candidate.absolute_frame_id,
                )
            )
            self._progress(
                f"visual verifier candidate {index}/{total} completed in "
                f"{time.perf_counter() - started:.2f}s"
            )
        return tuple(results)

    @staticmethod
    def _build_prompt(*, query_vi: str, query_en: str) -> str:
        return (
            "You are a strict visual evidence verifier for video known-item search. "
            "The images are neighboring frames from one automatically retrieved temporal "
            "candidate, ordered by time. Decompose the query into every independently "
            "visible requirement, including actions, counts, colors, objects, people, and "
            "relations. Score only what is visibly supported. Never infer a hidden person, "
            "attribute, count, or action. Exact counts and conjunctions matter. Return one "
            "JSON object only with keys: match_score (0..1), requirement_coverage (0..1), "
            "all_visible_requirements_satisfied (boolean), predicates (non-empty array of "
            "objects with requirement, score 0..1, visible boolean, and optional evidence), "
            "and summary. Keep each requirement under six words, each evidence under eight "
            "words, and summary under ten words. "
            f"Vietnamese query: {query_vi}\nEnglish translation: {query_en}"
        )

    def _progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _to_rgb_image(self, image: Any) -> Any:
        import numpy as np

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            array = array[:, :, ::-1]
        return self._image_api.fromarray(np.asarray(array, dtype=np.uint8))
