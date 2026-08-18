"""Bounded, evidence-grounded OCR answer provider using Tesseract CLI."""

from __future__ import annotations

import csv
import io
import math
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from system_tai.preliminary.schemas import QAPrediction

from .models import QAEvidenceCandidate, QAResult
from .question_types import QuestionType


class OCRBackendUnavailableError(RuntimeError):
    """The configured OCR executable or language data is unavailable."""


class OCRInferenceError(RuntimeError):
    """A bounded OCR inference call failed."""


@dataclass(frozen=True, slots=True)
class OCRAnswerProviderConfig:
    enabled: bool = False
    evidence_frame_budget: int = 10
    executable: str = "tesseract"
    languages: tuple[str, ...] = ("eng",)
    page_segmentation_mode: int = 6
    inference_timeout_seconds: float = 30.0
    minimum_confidence: float = 0.0
    telemetry_candidate_limit: int = 10

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if (
            type(self.evidence_frame_budget) is not int
            or not 1 <= self.evidence_frame_budget <= 100
        ):
            raise ValueError("evidence_frame_budget must be in [1, 100]")
        if not isinstance(self.executable, str) or not self.executable.strip():
            raise ValueError("executable must be a non-empty string")
        if type(self.languages) is not tuple or not self.languages:
            raise ValueError("languages must be a non-empty tuple")
        if any(
            not isinstance(language, str)
            or not language.strip()
            or not language.replace("_", "").isalnum()
            for language in self.languages
        ):
            raise ValueError("languages must contain safe non-empty language identifiers")
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("languages must be unique")
        if (
            type(self.page_segmentation_mode) is not int
            or not 0 <= self.page_segmentation_mode <= 13
        ):
            raise ValueError("page_segmentation_mode must be in [0, 13]")
        if (
            type(self.inference_timeout_seconds) is bool
            or not isinstance(self.inference_timeout_seconds, (int, float))
            or not math.isfinite(float(self.inference_timeout_seconds))
            or self.inference_timeout_seconds <= 0
        ):
            raise ValueError("inference_timeout_seconds must be positive and finite")
        if (
            type(self.minimum_confidence) is bool
            or not isinstance(self.minimum_confidence, (int, float))
            or not math.isfinite(float(self.minimum_confidence))
            or not 0 <= self.minimum_confidence <= 100
        ):
            raise ValueError("minimum_confidence must be in [0, 100]")
        if (
            type(self.telemetry_candidate_limit) is not int
            or not 1 <= self.telemetry_candidate_limit <= 100
        ):
            raise ValueError("telemetry_candidate_limit must be in [1, 100]")


@dataclass(frozen=True, slots=True)
class OCRDetection:
    text: str
    confidence: float | None
    bounding_box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("OCR detection text must be non-empty")
        if self.confidence is not None and (
            type(self.confidence) is bool
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
        ):
            raise ValueError("OCR confidence must be finite when present")
        if self.bounding_box is not None and (
            len(self.bounding_box) != 4
            or any(type(value) is not int or value < 0 for value in self.bounding_box)
        ):
            raise ValueError("OCR bounding_box must contain four non-negative integers")


@dataclass(frozen=True, slots=True)
class OCRObservation:
    video_id: str
    frame_id: int
    evidence_rank: int
    text: str
    normalized_text: str
    confidence: float
    bounding_box: tuple[int, int, int, int] | None
    backend_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_identity",
            MappingProxyType(dict(self.backend_identity)),
        )


class OCRBackend(Protocol):
    @property
    def identifiers(self) -> Mapping[str, Any]: ...

    def recognize(self, image: NDArray[np.generic]) -> tuple[OCRDetection, ...]: ...


_TSV_FIELDS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)


def normalize_ocr_text(text: str) -> str:
    """Return a conservative comparison key without inventing text."""

    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def parse_tesseract_tsv(payload: bytes) -> tuple[OCRDetection, ...]:
    """Parse Tesseract word TSV into deterministic line-level detections."""

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OCRInferenceError("Tesseract TSV is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    if tuple(reader.fieldnames or ()) != _TSV_FIELDS:
        raise OCRInferenceError("unexpected Tesseract TSV header")
    grouped: dict[
        tuple[int, int, int, int],
        list[tuple[int, str, float, tuple[int, int, int, int]]],
    ] = {}
    for row_number, row in enumerate(reader, start=2):
        try:
            raw_text = row["text"]
            if raw_text is None or not raw_text.strip():
                continue
            confidence = float(row["conf"])
            if not math.isfinite(confidence) or confidence < 0:
                continue
            key = tuple(
                int(row[field])
                for field in ("page_num", "block_num", "par_num", "line_num")
            )
            word_number = int(row["word_num"])
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OCRInferenceError(f"malformed Tesseract TSV row {row_number}") from exc
        if any(value < 0 for value in (left, top, width, height)):
            raise OCRInferenceError(f"negative Tesseract bounding box at row {row_number}")
        grouped.setdefault(key, []).append(
            (word_number, raw_text.strip(), confidence, (left, top, width, height))
        )

    detections: list[OCRDetection] = []
    for key in sorted(grouped):
        words = sorted(grouped[key], key=lambda item: item[0])
        line_text = " ".join(word[1] for word in words)
        confidence = sum(word[2] for word in words) / len(words)
        left = min(word[3][0] for word in words)
        top = min(word[3][1] for word in words)
        right = max(word[3][0] + word[3][2] for word in words)
        bottom = max(word[3][1] + word[3][3] for word in words)
        detections.append(
            OCRDetection(
                text=line_text,
                confidence=confidence,
                bounding_box=(left, top, right - left, bottom - top),
            )
        )
    return tuple(detections)


def _portable_pixmap(image: NDArray[np.generic]) -> bytes:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise OCRInferenceError("OCR image must use uint8 pixels")
    if array.ndim == 2:
        height, width = array.shape
        return f"P5\n{width} {height}\n255\n".encode() + array.tobytes(order="C")
    if array.ndim == 3 and array.shape[2] == 3:
        height, width, _channels = array.shape
        rgb = np.ascontiguousarray(array[:, :, ::-1])
        return f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes(order="C")
    raise OCRInferenceError("OCR image must have shape (H,W) or OpenCV BGR (H,W,3)")


class TesseractCLIBackend:
    """Explicit CPU Tesseract backend with no runtime model download."""

    def __init__(
        self,
        config: OCRAnswerProviderConfig,
        *,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        self.config = config
        self._runner = runner
        resolved = shutil.which(config.executable)
        if resolved is None:
            raise OCRBackendUnavailableError(
                f"Tesseract executable not found: {config.executable!r}"
            )
        self.executable = resolved
        version = self._invoke(("--version",)).stdout.decode(
            "utf-8", errors="replace"
        ).splitlines()
        language_lines = self._invoke(("--list-langs",)).stdout.decode(
            "utf-8", errors="strict"
        ).splitlines()
        available_languages = tuple(
            sorted(line.strip() for line in language_lines[1:] if line.strip())
        )
        missing = sorted(set(config.languages) - set(available_languages))
        if missing:
            raise OCRBackendUnavailableError(
                f"Tesseract language data unavailable: {missing}"
            )
        self._identifiers = MappingProxyType(
            {
                "backend": "tesseract_cli",
                "executable": self.executable,
                "version": version[0] if version else "UNKNOWN",
                "languages": list(config.languages),
                "available_languages": list(available_languages),
                "page_segmentation_mode": config.page_segmentation_mode,
                "device": "cpu",
                "model_download": False,
            }
        )

    @property
    def identifiers(self) -> Mapping[str, Any]:
        return self._identifiers

    def _invoke(
        self,
        arguments: tuple[str, ...],
        *,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            return self._runner(
                [self.executable, *arguments],
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=float(self.config.inference_timeout_seconds),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OCRBackendUnavailableError(
                f"Tesseract command failed: {type(exc).__name__}: {exc}"
            ) from exc

    def recognize(self, image: NDArray[np.generic]) -> tuple[OCRDetection, ...]:
        payload = _portable_pixmap(image)
        completed = self._invoke(
            (
                "stdin",
                "stdout",
                "-l",
                "+".join(self.config.languages),
                "--psm",
                str(self.config.page_segmentation_mode),
                "tsv",
            ),
            input_bytes=payload,
        )
        return parse_tesseract_tsv(completed.stdout)


@dataclass(slots=True)
class _OCRAggregate:
    normalized_text: str
    observations: list[OCRObservation] = field(default_factory=list)

    @property
    def distinct_supports(self) -> tuple[OCRObservation, ...]:
        by_frame: dict[tuple[str, int], OCRObservation] = {}
        for observation in self.observations:
            identity = (observation.video_id, observation.frame_id)
            previous = by_frame.get(identity)
            if previous is None or (
                observation.evidence_rank,
                -observation.confidence,
                observation.text.casefold(),
            ) < (
                previous.evidence_rank,
                -previous.confidence,
                previous.text.casefold(),
            ):
                by_frame[identity] = observation
        return tuple(
            sorted(
                by_frame.values(),
                key=lambda item: (
                    item.evidence_rank,
                    -item.confidence,
                    item.video_id,
                    item.frame_id,
                ),
            )
        )

    @property
    def best_support(self) -> OCRObservation:
        return self.distinct_supports[0]

    @property
    def best_confidence(self) -> float:
        return max(item.confidence for item in self.observations)


class OCRAnswerProvider:
    """Recognize and aggregate text only from bounded QA-A1 evidence frames."""

    def __init__(
        self,
        *,
        backend: OCRBackend,
        config: OCRAnswerProviderConfig,
        clock: Callable[[], float],
    ) -> None:
        self.backend = backend
        self.config = config
        self.clock = clock

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def identifiers(self) -> Mapping[str, Any]:
        return self.backend.identifiers

    def supports(self, question_type: QuestionType) -> bool:
        return self.enabled and question_type is QuestionType.OCR

    def answer(
        self,
        *,
        query_id: str,
        question_type: QuestionType,
        evidence: Sequence[tuple[QAEvidenceCandidate, NDArray[np.generic]]],
        output_top_k: int,
        warnings: list[str] | None = None,
    ) -> tuple[QAResult, dict[str, Any]]:
        if not self.supports(question_type):
            raise ValueError(f"OCR provider does not support {question_type.value}")
        if not 1 <= output_top_k <= 100:
            raise ValueError("output_top_k must be in [1, 100]")
        emitted_warnings = warnings if warnings is not None else []
        selected = tuple(sorted(evidence, key=lambda item: item[0].rank))[
            : self.config.evidence_frame_budget
        ]
        observations: list[OCRObservation] = []
        processed_frames = 0
        inference_seconds = 0.0
        for candidate, image in selected:
            started = self.clock()
            try:
                detections = self.backend.recognize(image)
                processed_frames += 1
            except (OCRBackendUnavailableError, OCRInferenceError, ValueError) as exc:
                emitted_warnings.append(
                    f"OCR failed for {candidate.video_id} frame {candidate.frame_id}: {exc}"
                )
                inference_seconds += self.clock() - started
                continue
            inference_seconds += self.clock() - started
            for detection in detections:
                if detection.confidence is None:
                    continue
                confidence = float(detection.confidence)
                if confidence < self.config.minimum_confidence:
                    continue
                normalized = normalize_ocr_text(detection.text)
                if not normalized:
                    continue
                observations.append(
                    OCRObservation(
                        video_id=candidate.video_id,
                        frame_id=candidate.frame_id,
                        evidence_rank=candidate.rank,
                        text=" ".join(unicodedata.normalize("NFKC", detection.text).split()),
                        normalized_text=normalized,
                        confidence=confidence,
                        bounding_box=detection.bounding_box,
                        backend_identity=self.backend.identifiers,
                    )
                )

        aggregates: dict[str, _OCRAggregate] = {}
        for observation in observations:
            aggregates.setdefault(
                observation.normalized_text,
                _OCRAggregate(observation.normalized_text),
            ).observations.append(observation)
        ranked = sorted(
            aggregates.values(),
            key=lambda item: (
                -len(item.distinct_supports),
                item.best_support.evidence_rank,
                -item.best_confidence,
                item.normalized_text,
            ),
        )
        predictions = [
            QAPrediction(
                query_id=query_id,
                rank=rank,
                video_id=aggregate.best_support.video_id,
                frame_id=aggregate.best_support.frame_id,
                answer=aggregate.best_support.text,
            )
            for rank, aggregate in enumerate(ranked[:output_top_k], start=1)
        ]
        unsupported_reason = None
        if not predictions:
            unsupported_reason = "NO_OCR_TEXT_EVIDENCE"
            emitted_warnings.append(
                "No confidence-bearing OCR text was recognized in bounded QA evidence."
            )
        top_candidates = []
        for aggregate in ranked[: self.config.telemetry_candidate_limit]:
            supports = aggregate.distinct_supports
            top_candidates.append(
                {
                    "normalized_text": aggregate.normalized_text,
                    "answer_text": aggregate.best_support.text,
                    "supporting_frame_count": len(supports),
                    "supporting_frame_ids": [item.frame_id for item in supports],
                    "supporting_video_ids": sorted({item.video_id for item in supports}),
                    "best_evidence_rank": aggregate.best_support.evidence_rank,
                    "best_ocr_confidence": aggregate.best_confidence,
                    "source_text_observations": [
                        {
                            "video_id": item.video_id,
                            "frame_id": item.frame_id,
                            "evidence_rank": item.evidence_rank,
                            "text": item.text,
                            "confidence": item.confidence,
                            "bounding_box": (
                                list(item.bounding_box)
                                if item.bounding_box is not None
                                else None
                            ),
                        }
                        for item in aggregate.observations
                    ],
                }
            )
        telemetry = {
            "ocr_backend_identity": dict(self.backend.identifiers),
            "ocr_frames_requested": len(selected),
            "ocr_frames_processed": processed_frames,
            "ocr_observation_count": len(observations),
            "ocr_nonempty_observation_count": len(observations),
            "ocr_unique_candidate_count": len(aggregates),
            "ocr_inference_seconds": inference_seconds,
            "top_ocr_candidates": top_candidates,
        }
        return (
            QAResult(
                query_id=query_id,
                question_type=question_type,
                predictions=predictions,
                unsupported_reason=unsupported_reason,
                warnings=emitted_warnings,
                diagnostics={"confidence_level": "OCR_EVIDENCE", **telemetry},
            ),
            telemetry,
        )
