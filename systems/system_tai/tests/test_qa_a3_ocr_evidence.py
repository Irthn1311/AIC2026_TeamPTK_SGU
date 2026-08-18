from __future__ import annotations

import inspect
import subprocess
import time
from collections.abc import Mapping

import numpy as np
import pytest

from system_tai.kis.session_schema import SessionConfig
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.models import QAEvidenceCandidate
from system_tai.qa.ocr_provider import (
    OCRAnswerProvider,
    OCRAnswerProviderConfig,
    OCRBackendUnavailableError,
    OCRDetection,
    TesseractCLIBackend,
    normalize_ocr_text,
    parse_tesseract_tsv,
)
from system_tai.qa.question_types import QuestionType
from system_tai.qa.runtime import (
    LEGACY_PHASE_P0,
    QA_A2_CAPABILITY_AWARE,
    QA_A3_CAPABILITY_AWARE,
    classify_runtime_question,
)


class _FakeOCRBackend:
    def __init__(self, outputs: Mapping[int, tuple[OCRDetection, ...]]) -> None:
        self.outputs = dict(outputs)
        self.calls: list[int] = []
        self.identifiers = {
            "backend": "fake_ocr",
            "device": "cpu",
            "model_download": False,
        }

    def recognize(self, image: np.ndarray) -> tuple[OCRDetection, ...]:
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        return self.outputs.get(marker, ())


def _candidate(rank: int, frame_id: int) -> QAEvidenceCandidate:
    return QAEvidenceCandidate(
        query_id="q",
        rank=rank,
        video_id=f"V{rank}",
        frame_id=frame_id,
        retrieval_score=1.0 / rank,
    )


def _image(marker: int) -> np.ndarray:
    return np.full((2, 2, 3), marker, dtype=np.uint8)


def test_qa_a3_disabled_preserves_ocr_unsupported_and_legacy_behavior() -> None:
    classification, policy = classify_runtime_question(
        "Biển số xe ghi gì?",
        None,
        qa_a2_enabled=True,
        qa_ocr_enabled=False,
    )
    assert classification.question_type is QuestionType.UNSUPPORTED
    assert classification.reason.startswith("OCR_PATTERN_PROVIDER_MISSING")
    assert policy == QA_A2_CAPABILITY_AWARE

    legacy, legacy_policy = classify_runtime_question(
        "Có bao nhiêu người?",
        None,
        qa_a2_enabled=False,
        qa_ocr_enabled=False,
    )
    assert legacy.question_type is QuestionType.COUNT
    assert legacy_policy == LEGACY_PHASE_P0


@pytest.mark.parametrize(
    ("question", "expected", "expected_policy"),
    [
        ("Biển số xe ghi gì?", QuestionType.OCR, QA_A3_CAPABILITY_AWARE),
        ("Giá là bao nhiêu?", QuestionType.OCR, QA_A3_CAPABILITY_AWARE),
        (
            "Người đó đang cầm gì?",
            QuestionType.OBJECT_ENTITY,
            QA_A2_CAPABILITY_AWARE,
        ),
        ("Có bao nhiêu người?", QuestionType.OBJECT_COUNT, QA_A2_CAPABILITY_AWARE),
        ("Chiếc xe màu gì?", QuestionType.COLOR, QA_A2_CAPABILITY_AWARE),
    ],
)
def test_combined_capability_routing_is_precise(
    question: str,
    expected: QuestionType,
    expected_policy: str,
) -> None:
    classification, policy = classify_runtime_question(
        question,
        None,
        qa_a2_enabled=True,
        qa_ocr_enabled=True,
    )
    assert classification.question_type is expected
    assert policy == expected_policy


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Biển số xe ghi gì?", QuestionType.OCR),
        ("Giá là bao nhiêu?", QuestionType.OCR),
    ],
)
def test_qa_a3_only_routes_high_precision_ocr_intent(
    question: str,
    expected: QuestionType,
) -> None:
    classification, policy = classify_runtime_question(
        question,
        None,
        qa_a2_enabled=False,
        qa_ocr_enabled=True,
    )
    assert classification.question_type is expected
    assert policy == QA_A3_CAPABILITY_AWARE


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Có bao nhiêu người?", QuestionType.COUNT),
        ("Chiếc xe màu gì?", QuestionType.COLOR),
        ("Có chiếc xe nào không?", QuestionType.YES_NO),
        ("Người đó đi bên trái hay bên phải?", QuestionType.DIRECTION),
        ("Người đó đang làm gì?", QuestionType.UNSUPPORTED),
    ],
)
def test_qa_a3_only_preserves_exact_non_ocr_legacy_classification(
    question: str,
    expected: QuestionType,
) -> None:
    legacy, legacy_policy = classify_runtime_question(
        question,
        None,
        qa_a2_enabled=False,
        qa_ocr_enabled=False,
    )
    with_ocr, with_ocr_policy = classify_runtime_question(
        question,
        None,
        qa_a2_enabled=False,
        qa_ocr_enabled=True,
    )

    assert legacy.question_type is expected
    assert with_ocr == legacy
    assert legacy_policy == LEGACY_PHASE_P0
    assert with_ocr_policy == LEGACY_PHASE_P0


def test_ocr_aggregation_is_bounded_deterministic_and_preserves_frame_identity() -> None:
    backend = _FakeOCRBackend(
        {
            1: (
                OCRDetection("  ĐỒNG BẰNG SÔNG HỒNG ", 88.0, (1, 2, 3, 4)),
                OCRDetection("50.000", 92.0),
            ),
            2: (OCRDetection("Đồng bằng sông Hồng", 95.0),),
            3: (OCRDetection("MUST NOT RUN", 99.0),),
        }
    )
    provider = OCRAnswerProvider(
        backend=backend,
        config=OCRAnswerProviderConfig(enabled=True, evidence_frame_budget=2),
        clock=time.perf_counter,
    )
    evidence = (
        (_candidate(1, 101), _image(1)),
        (_candidate(2, 202), _image(2)),
        (_candidate(3, 303), _image(3)),
    )

    result, telemetry = provider.answer(
        query_id="q",
        question_type=QuestionType.OCR,
        evidence=evidence,
        output_top_k=10,
    )

    assert backend.calls == [1, 2]
    assert [(item.answer, item.video_id, item.frame_id) for item in result.predictions] == [
        ("ĐỒNG BẰNG SÔNG HỒNG", "V1", 101),
        ("50.000", "V1", 101),
    ]
    assert telemetry["ocr_frames_requested"] == 2
    assert telemetry["ocr_frames_processed"] == 2
    assert telemetry["ocr_observation_count"] == 3
    top = telemetry["top_ocr_candidates"][0]
    assert top["supporting_frame_count"] == 2
    assert top["supporting_frame_ids"] == [101, 202]
    assert top["best_evidence_rank"] == 1
    assert result.predictions[1].answer == "50.000"


def test_no_confidence_or_no_text_fails_closed() -> None:
    backend = _FakeOCRBackend({1: (OCRDetection("VISIBLE BUT UNTRUSTED", None),), 2: ()})
    provider = OCRAnswerProvider(
        backend=backend,
        config=OCRAnswerProviderConfig(enabled=True),
        clock=time.perf_counter,
    )
    result, telemetry = provider.answer(
        query_id="q",
        question_type=QuestionType.OCR,
        evidence=((_candidate(1, 10), _image(1)), (_candidate(2, 20), _image(2))),
        output_top_k=10,
    )
    assert result.predictions == []
    assert result.unsupported_reason == "NO_OCR_TEXT_EVIDENCE"
    assert telemetry["ocr_observation_count"] == 0
    assert telemetry["ocr_unique_candidate_count"] == 0


def test_tesseract_tsv_line_parsing_and_conservative_normalization() -> None:
    payload = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\t"
        "height\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t90.0\tGiá\n"
        "5\t1\t1\t1\t1\t2\t45\t20\t50\t40\t80.0\t50.000\n"
        "5\t1\t1\t1\t2\t1\t5\t80\t30\t20\t-1\tignored\n"
    ).encode()
    detections = parse_tesseract_tsv(payload)
    assert detections == (OCRDetection("Giá 50.000", 85.0, (10, 20, 85, 40)),)
    assert normalize_ocr_text("  Giá\u00a0 50.000 ") == "giá 50.000"


def test_parse_tesseract_tsv_handles_quotes_and_multiple_lines_without_leak() -> None:
    # Representative Tesseract TSV containing literal double quotes in text
    payload = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t89\t36\t85\t47\t70.0\t&\n"
        "5\t1\t1\t1\t1\t2\t189\t28\t87\t31\t92.0\tCHƯƠNG\n"
        '5\t1\t1\t1\t1\t3\t1247\t55\t5\t3\t27.0\t"\n'
        "5\t1\t1\t1\t2\t1\t140\t58\t25\t14\t48.0\tCD\n"
        "5\t1\t1\t1\t14\t1\t144\t587\t47\t56\t85.0\tA\n"
        "5\t1\t1\t1\t14\t2\t199\t585\t93\t35\t21.0\tTV\n"
        "5\t1\t1\t1\t14\t3\t401\t593\t60\t19\t92.0\tDIOR\n"
        "5\t1\t1\t1\t14\t4\t470\t593\t89\t19\t91.0\tTRƯNG\n"
        "5\t1\t1\t1\t14\t5\t569\t587\t49\t24\t89.0\tBÀY\n"
    ).encode()
    detections = parse_tesseract_tsv(payload)

    # Must produce separate detections per line, not swallow subsequent lines into quote
    assert len(detections) == 3
    # Line 1: '& CHƯƠNG "'
    assert detections[0].text == '& CHƯƠNG "'
    # Line 2: 'CD'
    assert detections[1].text == "CD"
    # Line 14: 'A TV DIOR TRƯNG BÀY'
    assert detections[2].text == "A TV DIOR TRƯNG BÀY"
    # Assert no TSV numeric metadata leaks into any detection text
    for d in detections:
        assert "\t" not in d.text
        assert "5\t1\t1" not in d.text
        assert "conf" not in d.text


def test_tesseract_backend_fails_closed_without_executable() -> None:
    with pytest.raises(OCRBackendUnavailableError, match="executable not found"):
        TesseractCLIBackend(
            OCRAnswerProviderConfig(
                enabled=True,
                executable="system_tai_definitely_missing_tesseract",
            )
        )


def test_tesseract_backend_preflights_identity_and_uses_bounded_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bytes | None]] = []
    tsv = (
        b"level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\t"
        b"height\tconf\ttext\n"
        b"5\t1\t1\t1\t1\t1\t0\t0\t2\t2\t91.0\tOPENAI\n"
    )

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        input_bytes = kwargs.get("input")
        assert input_bytes is None or isinstance(input_bytes, bytes)
        calls.append((command, input_bytes))
        if "--version" in command:
            stdout = b"tesseract 5.3.0\n"
        elif "--list-langs" in command:
            stdout = b"List of available languages in /data (2):\neng\nvie\n"
        else:
            stdout = tsv
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(
        "system_tai.qa.ocr_provider.shutil.which",
        lambda _name: "/opt/tesseract",
    )
    backend = TesseractCLIBackend(
        OCRAnswerProviderConfig(enabled=True, languages=("eng", "vie")),
        runner=runner,
    )
    detections = backend.recognize(np.zeros((2, 2, 3), dtype=np.uint8))

    assert backend.identifiers["version"] == "tesseract 5.3.0"
    assert backend.identifiers["languages"] == ["eng", "vie"]
    assert detections == (OCRDetection("OPENAI", 91.0, (0, 0, 2, 2)),)
    inference_command, input_bytes = calls[-1]
    assert inference_command == [
        "/opt/tesseract",
        "stdin",
        "stdout",
        "-l",
        "eng+vie",
        "--psm",
        "6",
        "tsv",
    ]
    assert input_bytes is not None and input_bytes.startswith(b"P6\n2 2\n255\n")


def test_ocr_provider_runtime_contract_has_no_ground_truth_input() -> None:
    parameters = inspect.signature(OCRAnswerProvider.answer).parameters
    assert "accepted_answers" not in parameters
    assert "ground_truth" not in parameters


def test_session_config_requires_qa_a1_for_ocr_and_allows_qa_a2_combination() -> None:
    with pytest.raises(ValueError, match="QA OCR evidence requires"):
        SessionConfig(
            qa_ocr_answer_provider_config=OCRAnswerProviderConfig(enabled=True)
        )

    config = SessionConfig(
        qa_video_conditioned_evidence_config=QAVideoConditionedEvidenceConfig(
            enabled=True
        ),
        qa_ocr_answer_provider_config=OCRAnswerProviderConfig(enabled=True),
    )
    assert config.qa_ocr_answer_provider_config.enabled is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"evidence_frame_budget": 0},
        {"languages": ()},
        {"languages": ("eng", "eng")},
        {"page_segmentation_mode": 14},
        {"inference_timeout_seconds": 0},
    ],
)
def test_ocr_config_rejects_unbounded_or_ambiguous_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        OCRAnswerProviderConfig(**kwargs)
