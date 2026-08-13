from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from system_tai.kis.session_schema import (
    InvalidRequestError,
    QAQueryRequest,
    parse_session_request,
)
from system_tai.qa.grounding import QAVideoConditionedEvidenceConfig
from system_tai.qa.runtime import classify_runtime_question
from system_tai.quality.l21_150_qa_translation import (
    load_qa_dev_translation_sidecar,
)
from system_tai.quality.l21_150_schema import (
    L21150QAQuery,
    load_l21_150_benchmark,
)
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariantType

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/benchmark.json"
SIDECAR_PATH = SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/qa_dev_translations_en.json"
SIDECAR_SHA256 = "45929059506de93aac574a6d2d5581691af81ae12405c18d57289485948c1f4d"
RUNNER_PATH = SYSTEM_ROOT / "scripts/l21_150_run_baseline.py"


def _load_runner():
    name = "l21_150_runner_qa_d1_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _qa_payload(**updates):
    payload = {
        "type": "qa_query",
        "request_id": "request-1",
        "query_id": "QA-1",
        "event_description": "Một chiếc xe dừng lại.",
        "question": "Chiếc xe có màu gì?",
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("include_vi", "event_en", "languages", "variant_types"),
    (
        (True, None, ["vi"], ["vietnamese_direct"]),
        (
            True,
            "A car stops.",
            ["vi", "en"],
            ["vietnamese_direct", "english_translation"],
        ),
        (False, "A car stops.", ["en"], ["english_translation"]),
    ),
)
def test_qa_localization_variant_matrix(
    include_vi: bool,
    event_en: str | None,
    languages: list[str],
    variant_types: list[str],
) -> None:
    request = QAQueryRequest(
        request_id="request-1",
        query_id="QA-1",
        event_description="Một chiếc xe dừng lại.",
        question="Chiếc xe có màu gì?",
        event_description_en=event_en,
        include_vi_variant=include_vi,
    )
    variants = request.variants()
    assert [variant.language.value for variant in variants] == languages
    assert [variant.variant_type.value for variant in variants] == variant_types


def test_qa_en_only_requires_english_localization() -> None:
    with pytest.raises(ValueError, match="event_description_en must be non-empty"):
        QAQueryRequest(
            request_id="request-1",
            query_id="QA-1",
            event_description="Một chiếc xe dừng lại.",
            question="Chiếc xe có màu gì?",
            include_vi_variant=False,
        )
    with pytest.raises(InvalidRequestError, match="event_description_en must be non-empty"):
        parse_session_request(
            json.dumps(_qa_payload(include_vi_variant=False), ensure_ascii=False)
        )


def test_qa_session_parser_defaults_true_and_rejects_non_boolean() -> None:
    legacy = parse_session_request(json.dumps(_qa_payload(), ensure_ascii=False))
    en_only = parse_session_request(
        json.dumps(
            _qa_payload(
                event_description_en="A car stops.",
                include_vi_variant=False,
            ),
            ensure_ascii=False,
        )
    )
    assert isinstance(legacy, QAQueryRequest)
    assert legacy.include_vi_variant is True
    assert [variant.language for variant in legacy.variants()] == [
        QueryLanguage.VIETNAMESE
    ]
    assert isinstance(en_only, QAQueryRequest)
    assert en_only.include_vi_variant is False
    assert len(en_only.variants()) == 1
    assert en_only.variants()[0].language is QueryLanguage.ENGLISH
    assert en_only.variants()[0].variant_type is QueryVariantType.ENGLISH_TRANSLATION

    with pytest.raises(InvalidRequestError, match="include_vi_variant must be a boolean"):
        parse_session_request(
            json.dumps(
                _qa_payload(
                    event_description_en="A car stops.",
                    include_vi_variant=1,
                ),
                ensure_ascii=False,
            )
        )


def test_l21_en_only_request_uses_sidecar_only_for_localization() -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_qa_dev_translation_sidecar(
        SIDECAR_PATH,
        benchmark,
        BENCHMARK_PATH,
    )
    query = next(
        item
        for item in benchmark.queries
        if isinstance(item, L21150QAQuery) and item.split == "DEV"
    )
    request = RUNNER._runtime_request(
        query,
        "request-1",
        100,
        0,
        qa_localization_language_policy="en_only",
        qa_translations=sidecar.translations,
    )

    assert request.event_description == query.question_vi
    assert request.event_description_en == sidecar.translations[query.query_id]
    assert request.include_vi_variant is False
    assert request.question == query.question_vi
    assert request.question_en is None
    assert request.refine_top_n == 1
    assert [variant.text for variant in request.variants()] == [
        sidecar.translations[query.query_id]
    ]


@pytest.mark.parametrize(
    ("qa_a2_enabled", "qa_ocr_enabled"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_english_localization_does_not_change_answer_classifier_inputs(
    qa_a2_enabled: bool,
    qa_ocr_enabled: bool,
) -> None:
    legacy = QAQueryRequest(
        "legacy",
        "QA-1",
        "Một người cầm túi.",
        "Có bao nhiêu người?",
    )
    en_only = QAQueryRequest(
        "en-only",
        "QA-1",
        "Một người cầm túi.",
        "Có bao nhiêu người?",
        event_description_en="A person holds a bag.",
        question_en=None,
        include_vi_variant=False,
    )
    legacy_result = classify_runtime_question(
        legacy.question,
        legacy.question_en,
        qa_a2_enabled=qa_a2_enabled,
        qa_ocr_enabled=qa_ocr_enabled,
    )
    en_only_result = classify_runtime_question(
        en_only.question,
        en_only.question_en,
        qa_a2_enabled=qa_a2_enabled,
        qa_ocr_enabled=qa_ocr_enabled,
    )
    assert en_only_result == legacy_result
    assert [variant.language for variant in en_only.variants()] == [
        QueryLanguage.ENGLISH
    ]


def test_l21_runner_en_only_guards_frozen_sha_and_dev_scope(tmp_path: Path) -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    sidecar = load_qa_dev_translation_sidecar(
        SIDECAR_PATH,
        benchmark,
        BENCHMARK_PATH,
    )
    common = {
        "experiment_id": "qa-d1-guard",
        "task": "qa",
        "top_k": 100,
        "refine_top_n": 3,
        "resume": False,
        "fail_fast": True,
        "benchmark_sha256": hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        "manifest_sha256": None,
        "gt_policy": "proposed",
        "qa_localization_language_policy": "en_only",
        "qa_translation_sidecar": sidecar,
        "qa_translation_sidecar_path": SIDECAR_PATH,
        "qa_video_conditioned_evidence_config": QAVideoConditionedEvidenceConfig(
            enabled=True
        ),
    }
    with pytest.raises(ValueError, match="frozen QA-D0 DEV artifact"):
        RUNNER.run_l21_150_baseline(
            benchmark,
            object(),
            tmp_path / "wrong-sha",
            split="dev",
            qa_translation_sidecar_sha256="0" * 64,
            **common,
        )
    with pytest.raises(ValueError, match="QA-A1 is restricted"):
        RUNNER.run_l21_150_baseline(
            benchmark,
            object(),
            tmp_path / "holdout",
            split="holdout",
            qa_translation_sidecar_sha256=SIDECAR_SHA256,
            **common,
        )


def test_l21_runner_cli_qa_policy_defaults_off_and_is_explicit() -> None:
    defaults = RUNNER.build_parser().parse_args(
        [
            "--benchmark",
            str(BENCHMARK_PATH),
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
        ]
    )
    assert defaults.qa_localization_language_policy == "legacy_vi"
    assert defaults.qa_dev_en_sidecar is None
    assert defaults.qa_keyframe_evidence_bank is False
    assert defaults.qa_keyframe_evidence_video_cap == 32
    assert defaults.qa_multi_seed_temporal_refinement is False
    assert defaults.qa_temporal_seeds_per_video == 3
    assert defaults.qa_temporal_refinement_video_cap == 32
    assert defaults.qa_temporal_refinement_total_seed_cap == 96

    enabled = RUNNER.build_parser().parse_args(
        [
            "--benchmark",
            str(BENCHMARK_PATH),
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--qa-localization-language-policy",
            "en_only",
            "--qa-dev-en-sidecar",
            str(SIDECAR_PATH),
            "--qa-video-conditioned-evidence",
            "--qa-keyframe-evidence-bank",
            "--qa-keyframe-evidence-video-cap",
            "32",
            "--qa-multi-seed-temporal-refinement",
            "--qa-temporal-seeds-per-video",
            "3",
            "--qa-temporal-refinement-video-cap",
            "16",
            "--qa-temporal-refinement-total-seed-cap",
            "24",
            "--refine-top-n",
            "1",
        ]
    )
    assert enabled.qa_localization_language_policy == "en_only"
    assert enabled.qa_dev_en_sidecar == SIDECAR_PATH
    assert enabled.qa_keyframe_evidence_bank is True
    assert enabled.qa_keyframe_evidence_video_cap == 32
    assert enabled.qa_multi_seed_temporal_refinement is True
    assert enabled.qa_temporal_seeds_per_video == 3
    assert enabled.qa_temporal_refinement_video_cap == 16
    assert enabled.qa_temporal_refinement_total_seed_cap == 24
    assert enabled.refine_top_n == 1


def test_l21_runner_cli_qa_d12_requires_dev_qa_grounding_bank_and_en_only() -> None:
    common = [
        "--benchmark",
        str(BENCHMARK_PATH),
        "--reuse-manifest",
        "manifest.json",
        "--output-dir",
        "out",
        "--qa-multi-seed-temporal-refinement",
    ]
    assert RUNNER.main([*common, "--task", "qa"]) == 2
    assert (
        RUNNER.main(
            [
                *common,
                "--task",
                "qa",
                "--qa-video-conditioned-evidence",
                "--qa-keyframe-evidence-bank",
            ]
        )
        == 2
    )
    assert (
        RUNNER.main(
            [
                *common,
                "--split",
                "holdout",
                "--task",
                "qa",
                "--qa-video-conditioned-evidence",
                "--qa-keyframe-evidence-bank",
                "--qa-localization-language-policy",
                "en_only",
            ]
        )
        == 2
    )
