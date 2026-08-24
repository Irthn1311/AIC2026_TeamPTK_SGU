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

    @property
    def predicate_bottleneck_score(self) -> float:
        """Return the weakest independently verified requirement score.

        An invisible requirement is a hard zero.  This keeps conjunction ranking from
        rewarding a high broad-scene score when one required count, attribute, action,
        or relation is absent from the frame.
        """

        return min(
            predicate.score if predicate.visible else 0.0
            for predicate in self.predicates
        )

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
                "predicate_bottleneck_score": self.predicate_bottleneck_score,
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


@dataclass(frozen=True, slots=True)
class VisualVerificationFailure:
    """Bounded diagnostic for one candidate that failed primary and retry attempts."""

    video_id: str
    absolute_frame_id: int
    primary_error: str
    retry_error: str

    def __post_init__(self) -> None:
        if not self.video_id.strip() or self.absolute_frame_id < 0:
            raise ValueError("invalid visual verification failure identity")

    def to_trace(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "video_id": self.video_id,
                "absolute_frame_id": self.absolute_frame_id,
                "attempt_count": 2,
                "primary_error": self.primary_error,
                "retry_error": self.retry_error,
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


def _bounded_error(exc: BaseException, *, limit: int = 500) -> str:
    rendered = f"{type(exc).__name__}: {exc}"
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


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
    def aliased_value(canonical: str, compact: str, *, required: bool = True) -> Any:
        if canonical in payload and compact in payload:
            raise VisualVerificationError(
                f"visual verifier response contains both {canonical!r} and {compact!r}"
            )
        if canonical in payload:
            return payload[canonical]
        if compact in payload:
            return payload[compact]
        if required:
            raise VisualVerificationError(
                f"visual verifier response is missing {canonical!r}"
            )
        return ""

    raw_predicates = aliased_value("predicates", "p")
    if not isinstance(raw_predicates, list) or not raw_predicates:
        raise VisualVerificationError("visual verifier predicates must be a non-empty list")
    predicates: list[VisualPredicateScore] = []
    for index, item in enumerate(raw_predicates, start=1):
        try:
            if isinstance(item, dict):
                requirement = item["requirement"]
                score = item["score"]
                visible = item["visible"]
                evidence = item.get("evidence", "")
            elif isinstance(item, list) and len(item) in {3, 4}:
                requirement, score, visible = item[:3]
                evidence = item[3] if len(item) == 4 else ""
            else:
                raise TypeError(
                    "must be an object or a compact [requirement, score, visible] array"
                )
            predicates.append(
                VisualPredicateScore(
                    requirement=str(requirement),
                    score=float(score),
                    visible=visible,
                    evidence=str(evidence),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VisualVerificationError(f"invalid predicate {index}: {exc}") from exc
    try:
        return VisualVerificationResult(
            video_id=video_id,
            absolute_frame_id=absolute_frame_id,
            match_score=float(aliased_value("match_score", "m")),
            requirement_coverage=float(aliased_value("requirement_coverage", "c")),
            all_visible_requirements_satisfied=aliased_value(
                "all_visible_requirements_satisfied", "a"
            ),
            predicates=tuple(predicates),
            summary=str(aliased_value("summary", "s", required=False)),
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
        self._last_failures: tuple[VisualVerificationFailure, ...] = ()
        self._last_recovered_retries: tuple[Mapping[str, Any], ...] = ()
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
                "wire_format": "compact-json-v1",
            }
        )

    @property
    def last_failures(self) -> tuple[VisualVerificationFailure, ...]:
        """Candidate-local failures from the most recent verify call."""

        return self._last_failures

    @property
    def last_recovered_retries(self) -> tuple[Mapping[str, Any], ...]:
        """Successful bounded retries from the most recent verify call."""

        return self._last_recovered_retries

    def verify(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidates: Sequence[VisualVerificationInput],
    ) -> tuple[VisualVerificationResult, ...]:
        """Verify frames independently so one malformed result is not batch-fatal.

        Inputs retain absolute original-video frame IDs. The return value contains only
        successful results; bounded candidate failures and recovered retries are exposed
        through diagnostic properties for the caller's explicit policy handling.
        """

        self._last_failures = ()
        self._last_recovered_retries = ()
        if not query_vi.strip() or not query_en.strip():
            raise ValueError("visual verification requires Vietnamese and English query text")
        results: list[VisualVerificationResult] = []
        failures: list[VisualVerificationFailure] = []
        recovered_retries: list[Mapping[str, Any]] = []
        total = len(candidates)
        for index, candidate in enumerate(candidates, start=1):
            started = time.perf_counter()
            self._progress(
                f"visual verifier candidate {index}/{total} started: "
                f"{candidate.video_id}/{candidate.absolute_frame_id} "
                f"images={len(candidate.images)}"
            )
            try:
                result = self._verify_candidate(
                    query_vi=query_vi,
                    query_en=query_en,
                    candidate=candidate,
                    images=candidate.images,
                    max_new_tokens=self._max_new_tokens,
                )
            except Exception as primary_exc:
                retry_images = (candidate.images[len(candidate.images) // 2],)
                # Keep the configured generation budget.  The former 192-token retry
                # cap could truncate otherwise valid JSON for conjunction-heavy queries.
                # The retry is bounded by one center image and the same caller-approved
                # token budget; the compact wire schema keeps its output small.
                retry_max_tokens = self._max_new_tokens
                self._progress(
                    f"visual verifier candidate {index}/{total} primary failed: "
                    f"{type(primary_exc).__name__}: {primary_exc}; retrying with "
                    f"images=1 max_new_tokens={retry_max_tokens}"
                )
                try:
                    result = self._verify_candidate(
                        query_vi=query_vi,
                        query_en=query_en,
                        candidate=candidate,
                        images=retry_images,
                        max_new_tokens=retry_max_tokens,
                    )
                except Exception as retry_exc:
                    failures.append(
                        VisualVerificationFailure(
                            video_id=candidate.video_id,
                            absolute_frame_id=candidate.absolute_frame_id,
                            primary_error=_bounded_error(primary_exc),
                            retry_error=_bounded_error(retry_exc),
                        )
                    )
                    self._progress(
                        f"visual verifier candidate {index}/{total} failed after "
                        f"retry in {time.perf_counter() - started:.2f}s: "
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    )
                    continue
                recovered_retries.append(
                    MappingProxyType(
                        {
                            "video_id": candidate.video_id,
                            "absolute_frame_id": candidate.absolute_frame_id,
                            "primary_error": _bounded_error(primary_exc),
                            "retry_image_count": 1,
                            "retry_max_new_tokens": retry_max_tokens,
                        }
                    )
                )
                self._progress(
                    f"visual verifier candidate {index}/{total} recovered on retry"
                )
            results.append(result)
            self._progress(
                f"visual verifier candidate {index}/{total} completed in "
                f"{time.perf_counter() - started:.2f}s"
            )
        self._last_failures = tuple(failures)
        self._last_recovered_retries = tuple(recovered_retries)
        return tuple(results)

    def _verify_candidate(
        self,
        *,
        query_vi: str,
        query_en: str,
        candidate: VisualVerificationInput,
        images: Sequence[Any],
        max_new_tokens: int,
    ) -> VisualVerificationResult:
        prompt = self._build_prompt(query_vi=query_vi, query_en=query_en)
        rgb_images = [self._to_rgb_image(image) for image in images]
        content = [{"type": "image"} for _ in rgb_images]
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
                images=rgb_images,
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
                    max_new_tokens=max_new_tokens,
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
        return parse_visual_verification_json(
            decoded,
            video_id=candidate.video_id,
            absolute_frame_id=candidate.absolute_frame_id,
        )

    @staticmethod
    def _build_prompt(*, query_vi: str, query_en: str) -> str:
        return (
            "You are a strict visual evidence verifier for video known-item search. "
            "The images are neighboring frames from one automatically retrieved temporal "
            "candidate, ordered by time. Decompose the query into every independently "
            "visible requirement, including actions, counts, colors, objects, people, and "
            "relations. Score only what is visibly supported. Never infer a hidden person, "
            "attribute, count, or action. Exact counts and conjunctions matter. Return "
            "exactly one minified JSON object with no prose or markdown, using only this "
            "compact schema: {\"m\":0.0,\"c\":0.0,\"a\":false,\"p\":["
            "[\"requirement\",0.0,false]],\"s\":\"\"}. m is whole-frame match "
            "score, c is visible-requirement coverage, a is true only when every required "
            "predicate is both visible and satisfied, and p contains exactly one array for every "
            "independently checkable action, count, color, object, person attribute, and "
            "relation. Do not merge or omit requirements. Each p array is requirement, "
            "score, visible. Keep requirement labels under four words and s under six "
            "words. Do not emit evidence text. "
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
