from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from system_tai.kis.session_schema import TRAKEQueryRequest
from system_tai.quality.l21_150_schema import (
    L21150TRAKEQuery,
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_trake_translation import (
    TRAKEDevTranslationSidecar,
    load_trake_dev_translation_sidecar,
)
from system_tai.trake.video_first import TRAKEVideoFirstConfig

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = SYSTEM_ROOT / "benchmarks/l21_150_diagnostic/benchmark.json"
SIDECAR_PATH = (
    SYSTEM_ROOT
    / "benchmarks/l21_150_diagnostic/tr_a2_d0_trake_dev_en_translation.json"
)
RUNNER_PATH = SYSTEM_ROOT / "scripts/l21_150_run_baseline.py"
SIDECAR_SHA256 = "021980a96f8a59677b143df556abd407f7d70588bd98d8d413bc25741754fcf7"


def _load_runner():
    spec = importlib.util.spec_from_file_location("l21_150_runner_d1_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


@pytest.fixture(scope="module")
def benchmark():
    return load_l21_150_benchmark(BENCHMARK_PATH)


@pytest.fixture(scope="module")
def sidecar(benchmark) -> TRAKEDevTranslationSidecar:
    return load_trake_dev_translation_sidecar(
        SIDECAR_PATH,
        benchmark,
        BENCHMARK_PATH,
    )


def _first_dev_trake(benchmark) -> L21150TRAKEQuery:
    return next(
        query
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "DEV"
    )


def test_runtime_request_vi_only_is_exact_legacy_shape(benchmark) -> None:
    query = _first_dev_trake(benchmark)
    request = RUNNER._runtime_request(query, "d1-vi", 100, 0)
    explicit = RUNNER._runtime_request(
        query,
        "d1-vi",
        100,
        0,
        trake_language_policy="vi_only",
    )

    assert isinstance(request, TRAKEQueryRequest)
    assert request == explicit
    assert request.include_vi_variant is True
    assert request.events == tuple(
        {"description": event.description_vi} for event in query.events
    )
    assert all("description_en" not in event for event in request.events)


def test_runtime_request_en_only_retains_vi_and_uses_frozen_event_indexes(
    benchmark,
    sidecar,
) -> None:
    query = _first_dev_trake(benchmark)
    request = RUNNER._runtime_request(
        query,
        "d1-en",
        100,
        0,
        trake_language_policy="en_only",
        trake_translations=sidecar.translations,
    )

    assert request.include_vi_variant is False
    assert [event["description"] for event in request.events] == [
        event.description_vi for event in query.events
    ]
    assert [event["description_en"] for event in request.events] == [
        sidecar.translations[(query.query_id, event.event_index)]
        for event in query.events
    ]
    assert all(
        set(event) == {"description", "description_en"}
        for event in request.events
    )


class _RecordingRuntime:
    def __init__(self, root: Path, config: TRAKEVideoFirstConfig) -> None:
        self.output_root = root
        self.output_root.mkdir(parents=True)
        self.manifest = SimpleNamespace(fingerprint="fixture", schema_version=2)
        self.shared_encoder = SimpleNamespace(
            identifiers=MappingProxyType({"model": "ViT-B/32", "device": "cpu"})
        )
        self.config = SimpleNamespace(
            device="cpu",
            trake_video_first_config=config,
        )
        self.requests: list[TRAKEQueryRequest] = []

    def handle_query(self, request):  # pragma: no cover
        raise AssertionError(request)

    def handle_qa_query(self, request):  # pragma: no cover
        raise AssertionError(request)

    def handle_trake_query(self, request: TRAKEQueryRequest) -> dict[str, Any]:
        self.requests.append(request)
        return {
            "status": "SUCCESS",
            "predictions": [],
            "timings": {"total_seconds": 0.01},
        }


def _run_normal_d1(
    tmp_path: Path,
    benchmark,
    sidecar: TRAKEDevTranslationSidecar | None,
    *,
    language_policy: str,
    sidecar_sha256: str | None,
    enabled: bool = True,
):
    config = TRAKEVideoFirstConfig(enabled=enabled)
    runtime = _RecordingRuntime(tmp_path / "runtime", config)
    report = RUNNER.run_l21_150_baseline(
        benchmark,
        runtime,
        tmp_path / "run",
        experiment_id="tr-a2-d1-fixture",
        split="dev",
        task="trake",
        top_k=100,
        refine_top_n=0,
        resume=False,
        fail_fast=True,
        benchmark_sha256=hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest(),
        manifest_sha256=None,
        gt_policy="proposed",
        trake_video_first_config=config,
        trake_language_policy=language_policy,
        trake_query_sidecar=sidecar,
        trake_query_sidecar_path=SIDECAR_PATH if sidecar is not None else None,
        trake_query_sidecar_sha256=sidecar_sha256,
    )
    return runtime, report


def test_normal_d1_en_only_uses_all_dev_records_and_explicit_provenance(
    tmp_path: Path,
    benchmark,
    sidecar,
) -> None:
    runtime, report = _run_normal_d1(
        tmp_path,
        benchmark,
        sidecar,
        language_policy="en_only",
        sidecar_sha256=SIDECAR_SHA256,
    )

    assert len(runtime.requests) == sidecar.query_count == 38
    assert all(request.include_vi_variant is False for request in runtime.requests)
    assert all(
        event["description_en"]
        == sidecar.translations[(request.query_id, event_index)]
        for request in runtime.requests
        for event_index, event in enumerate(request.events, start=1)
    )
    assert report["successful_query_count"] == 38
    assert report["trake_query_policy"] == "EN_ONLY"
    assert report["query_policy_changed_from_e0"] is True
    assert report["trake_query_experiment"] == {
        "trake_query_policy": "EN_ONLY",
        "translation_sidecar_sha256": SIDECAR_SHA256,
        "translation_status": "MODEL_AUTHORED_FROZEN_NOT_HUMAN_REVIEWED",
        "source_vi_retained_for_provenance": True,
        "source_vi_used_for_retrieval": False,
        "translation_en_used_for_retrieval": True,
        "include_vi_variant": False,
        "variant_count_policy": "1_VARIANT_EN_ONLY_PER_EVENT",
        "retrieval_feedback_used": False,
    }
    holdout_ids = {
        query.query_id
        for query in benchmark.queries
        if isinstance(query, L21150TRAKEQuery) and query.split == "HOLDOUT"
    }
    assert holdout_ids.isdisjoint({request.query_id for request in runtime.requests})


@pytest.mark.parametrize(
    ("policy", "sidecar_present", "sha", "enabled", "message"),
    (
        ("en_only", False, None, True, "requires a validated frozen D0 sidecar"),
        ("en_only", True, "0" * 64, True, "SHA256 does not match"),
        ("en_only", True, SIDECAR_SHA256, False, "TR-A1 enabled"),
        ("vi_only", True, SIDECAR_SHA256, True, "must not receive"),
        (
            "vi_plus_en_weighted_rrf",
            True,
            SIDECAR_SHA256,
            True,
            "supports only vi_only or en_only",
        ),
    ),
)
def test_normal_e2e_policy_is_fail_closed(
    tmp_path: Path,
    benchmark,
    sidecar,
    policy: str,
    sidecar_present: bool,
    sha: str | None,
    enabled: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _run_normal_d1(
            tmp_path,
            benchmark,
            sidecar if sidecar_present else None,
            language_policy=policy,
            sidecar_sha256=sha,
            enabled=enabled,
        )


def test_parser_defaults_and_explicit_en_only_contract() -> None:
    args = RUNNER.build_parser().parse_args(
        [
            "--benchmark",
            str(BENCHMARK_PATH),
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
        ]
    )
    assert args.trake_language_policy == "vi_only"
    assert args.trake_dev_en_sidecar is None

    explicit = RUNNER.build_parser().parse_args(
        [
            "--benchmark",
            str(BENCHMARK_PATH),
            "--reuse-manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--split",
            "dev",
            "--task",
            "trake",
            "--trake-video-first-restricted-search",
            "--trake-language-policy",
            "en_only",
            "--trake-dev-en-sidecar",
            str(SIDECAR_PATH),
        ]
    )
    assert explicit.trake_language_policy == "en_only"
    assert explicit.trake_dev_en_sidecar == SIDECAR_PATH


def test_frozen_sidecar_sha_is_reused_unchanged() -> None:
    assert hashlib.sha256(SIDECAR_PATH.read_bytes()).hexdigest() == SIDECAR_SHA256
    assert RUNNER.FROZEN_TR_A2_D0_TRAKE_DEV_EN_SIDECAR_SHA256 == SIDECAR_SHA256
