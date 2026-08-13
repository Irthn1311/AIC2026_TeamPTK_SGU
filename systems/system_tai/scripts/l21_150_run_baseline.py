"""Run the frozen system_tai runtime against the L21-150 diagnostic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SYSTEM_ROOT.parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.kis.session_engine import OperationalKISRuntime  # noqa: E402
from system_tai.kis.session_schema import (  # noqa: E402
    QAQueryRequest,
    QueryRequest,
    SessionConfig,
    TRAKEQueryRequest,
)
from system_tai.qa.grounding import (  # noqa: E402
    QA_KEYFRAME_EVIDENCE_BANK_V1,
    QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1,
    QA_VIDEO_CONDITIONED_EVIDENCE_V1,
    QAVideoConditionedEvidenceConfig,
)
from system_tai.qa.object_provider import ObjectAnswerProviderConfig  # noqa: E402
from system_tai.qa.ocr_provider import OCRAnswerProviderConfig  # noqa: E402
from system_tai.quality.l21_150_qa_translation import (  # noqa: E402
    QADevTranslationSidecar,
    QATranslationSidecarError,
    load_qa_dev_translation_sidecar,
)
from system_tai.quality.l21_150_schema import (  # noqa: E402
    L21150Benchmark,
    L21150FormatError,
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEQuery,
    load_l21_150_benchmark,
)
from system_tai.quality.l21_150_trake_nomination import (  # noqa: E402
    TRAKELanguagePolicy,
    TRAKENominationError,
    build_nomination_inputs,
    run_nomination_only,
    write_json_document,
)
from system_tai.quality.l21_150_trake_translation import (  # noqa: E402
    TRAKEDevTranslationSidecar,
    TRAKETranslationSidecarError,
    load_trake_dev_translation_sidecar,
)
from system_tai.quality.l21_150_translation import (  # noqa: E402
    KISDevTranslationSidecar,
    KISTranslationSidecarError,
    load_kis_dev_translation_sidecar,
)
from system_tai.refinement.models import (  # noqa: E402
    Q3AnchorRefinementConfig,
    SharedRawRegionRefinementConfig,
)
from system_tai.retrieval.video_restricted import (  # noqa: E402
    VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
    VideoConditionedKeyframeConfig,
)
from system_tai.trake.video_first import (  # noqa: E402
    TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH,
    TRAKEVideoFirstConfig,
)

FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256 = (
    "fa48d7af2001d8d5eca178301736d1409916961f256b4ccb779490d78495ccea"
)
FROZEN_TR_A2_D0_TRAKE_DEV_EN_SIDECAR_SHA256 = (
    "021980a96f8a59677b143df556abd407f7d70588bd98d8d413bc25741754fcf7"
)
FROZEN_QA_D0_DEV_EN_SIDECAR_SHA256 = (
    "45929059506de93aac574a6d2d5581691af81ae12405c18d57289485948c1f4d"
)
QA_ARTIFACT_BACKED_OBJECT_EVIDENCE = "QA_ARTIFACT_BACKED_OBJECT_EVIDENCE"
QA_EVIDENCE_GROUNDED_OCR = "QA_EVIDENCE_GROUNDED_OCR"


class L21150Runtime(Protocol):
    output_root: Path

    def handle_query(self, request: QueryRequest) -> dict[str, Any]: ...

    def handle_qa_query(self, request: QAQueryRequest) -> dict[str, Any]: ...

    def handle_trake_query(self, request: TRAKEQueryRequest) -> dict[str, Any]: ...


def _json_safe(value: Any, *, context: str = "metadata") -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError(f"{context} contains a non-string mapping key")
            result[key] = _json_safe(nested, context=f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(nested, context=f"{context}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise TypeError(
        f"{context} contains unsupported type {type(value).__name__}"
    )


def _write_json_document(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    safe_payload = _json_safe(payload)
    serialized = json.dumps(
        safe_payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")
    return safe_payload


def _prediction_identity(prediction: Mapping[str, Any]) -> tuple[Any, ...]:
    task = prediction.get("task")
    common = (prediction.get("query_id"), task, prediction.get("video_id"))
    if task == "kis":
        return (*common, prediction.get("actual_frame_id"))
    if task == "qa":
        return (
            *common,
            prediction.get("actual_frame_id"),
            prediction.get("answer"),
        )
    if task == "trake":
        frame_ids = prediction.get("actual_frame_ids")
        return (*common, tuple(frame_ids) if type(frame_ids) is list else frame_ids)
    raise ValueError(f"unsupported prediction task for identity: {task!r}")


def _output_depth_diagnostics(
    predictions: list[dict[str, Any]],
    successful_query_ids: set[str],
    requested_top_k: int,
) -> dict[str, Any]:
    depth_by_query = Counter(str(record["query_id"]) for record in predictions)
    all_successful_ids = successful_query_ids | set(depth_by_query)
    over_depth = sorted(
        query_id
        for query_id in all_successful_ids
        if depth_by_query[query_id] > requested_top_k
    )
    if over_depth:
        raise ValueError(
            "runtime output exceeds requested_top_k for queries: "
            + ", ".join(over_depth)
        )
    depths = [depth_by_query[query_id] for query_id in sorted(all_successful_ids)]
    depth_distribution = Counter(depths)
    identities = [_prediction_identity(record) for record in predictions]
    return {
        "requested_top_k": requested_top_k,
        "prediction_record_count": len(predictions),
        "prediction_depth_min": min(depths) if depths else None,
        "prediction_depth_max": max(depths) if depths else None,
        "prediction_depth_distribution": {
            str(depth): count for depth, count in sorted(depth_distribution.items())
        },
        "queries_below_requested_depth": sorted(
            query_id
            for query_id in all_successful_ids
            if depth_by_query[query_id] < requested_top_k
        ),
        "duplicate_output_identity_count": len(identities) - len(set(identities)),
    }


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path} contains a non-object JSON line")
            records.append(value)
    return records


def _response_latency(response: dict[str, Any]) -> float | None:
    timings = response.get("timings")
    if not isinstance(timings, dict):
        return None
    for key in ("total_seconds", "total_time_seconds", "query_total_seconds"):
        if type(timings.get(key)) in {int, float}:
            return float(timings[key])
    return None


def _resolve_artifact(runtime: L21150Runtime, response: dict[str, Any], key: str) -> Path:
    artifacts = response.get("artifacts")
    if not isinstance(artifacts, dict) or type(artifacts.get(key)) is not str:
        raise ValueError(f"runtime response is missing artifact {key}")
    path = Path(runtime.output_root) / artifacts[key]
    if not path.is_file():
        raise FileNotFoundError(f"runtime artifact does not exist: {path}")
    return path


def _kis_predictions(
    runtime: L21150Runtime,
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts = response.get("artifacts", {})
    artifact_key = (
        "refined_top100_jsonl"
        if isinstance(artifacts, dict) and "refined_top100_jsonl" in artifacts
        else "top100_jsonl"
    )
    records = _load_jsonl(_resolve_artifact(runtime, response, artifact_key))
    return [
        {
            "query_id": record["query_id"],
            "rank": record["rank"],
            "video_id": record["video_id"],
            "actual_frame_id": record["frame_id"],
        }
        for record in records
    ]


def _response_predictions(response: dict[str, Any], task: str) -> list[dict[str, Any]]:
    predictions = response.get("predictions")
    if type(predictions) is not list:
        raise ValueError(f"{task} runtime response is missing predictions")
    converted: list[dict[str, Any]] = []
    for prediction in predictions:
        if type(prediction) is not dict:
            raise ValueError(f"{task} runtime response contains a non-object prediction")
        base = {
            "query_id": prediction["query_id"],
            "rank": prediction["rank"],
            "video_id": prediction["video_id"],
        }
        if task == "qa":
            base["actual_frame_id"] = prediction["frame_id"]
            base["answer"] = prediction["answer"]
        else:
            base["actual_frame_ids"] = list(prediction["frame_ids"])
        converted.append(base)
    return converted


def _runtime_request(
    query: Any,
    experiment_id: str,
    top_k: int,
    refine_top_n: int,
    *,
    kis_query_policy: str = "vi_only",
    kis_translations: Mapping[str, str] | None = None,
    qa_localization_language_policy: str = "legacy_vi",
    qa_translations: Mapping[str, str] | None = None,
    trake_language_policy: str = "vi_only",
    trake_translations: Mapping[tuple[str, int], str] | None = None,
):
    request_id = f"{experiment_id}:{query.query_id}"
    if isinstance(query, L21150KISQuery):
        query_en = None
        if kis_query_policy in {"translation_augmented_rrf", "en_only"}:
            if kis_translations is None or query.query_id not in kis_translations:
                raise ValueError(
                    f"missing frozen English translation for {query.query_id}"
                )
            query_en = kis_translations[query.query_id]
        elif kis_query_policy != "vi_only":
            raise ValueError(f"unsupported KIS query policy: {kis_query_policy}")
        return QueryRequest(
            request_id=request_id,
            query_id=query.query_id,
            query_vi=query.query_vi,
            query_en=query_en,
            include_vi_variant=kis_query_policy != "en_only",
            top_k_per_variant=top_k,
            output_top_k=top_k,
            refine_top_n=refine_top_n,
        )
    if isinstance(query, L21150QAQuery):
        if qa_localization_language_policy == "legacy_vi":
            if qa_translations is not None:
                raise ValueError("QA legacy_vi must not receive English translations")
            event_description_en = None
            include_vi_variant = True
        elif qa_localization_language_policy == "en_only":
            if qa_translations is None:
                raise ValueError("QA EN_ONLY requires frozen English translations")
            event_description_en = qa_translations.get(query.query_id)
            if event_description_en is None:
                raise ValueError(
                    f"missing frozen English QA translation for {query.query_id}"
                )
            include_vi_variant = False
        else:
            raise ValueError(
                "qa_localization_language_policy must be legacy_vi or en_only"
            )
        return QAQueryRequest(
            request_id=request_id,
            query_id=query.query_id,
            event_description=query.question_vi,
            question=query.question_vi,
            event_description_en=event_description_en,
            question_en=None,
            include_vi_variant=include_vi_variant,
            top_k_per_variant=top_k,
            output_top_k=top_k,
            refine_top_n=max(1, refine_top_n),
        )
    if isinstance(query, L21150TRAKEQuery):
        if trake_language_policy == "vi_only":
            if trake_translations is not None:
                raise ValueError("TRAKE VI_ONLY must not receive English translations")
            events = tuple(
                {"description": event.description_vi} for event in query.events
            )
            include_vi_variant = True
        elif trake_language_policy == "en_only":
            if trake_translations is None:
                raise ValueError("TRAKE EN_ONLY requires frozen English translations")
            events_list: list[dict[str, str]] = []
            for event in query.events:
                key = (query.query_id, event.event_index)
                translation = trake_translations.get(key)
                if translation is None:
                    raise ValueError(
                        "missing frozen English translation for "
                        f"{query.query_id}/event {event.event_index}"
                    )
                events_list.append(
                    {
                        "description": event.description_vi,
                        "description_en": translation,
                    }
                )
            events = tuple(events_list)
            include_vi_variant = False
        else:
            raise ValueError(
                "normal TRAKE E2E supports only vi_only or en_only language policy"
            )
        return TRAKEQueryRequest(
            request_id=request_id,
            query_id=query.query_id,
            events=events,
            include_vi_variant=include_vi_variant,
            top_k_per_variant=top_k,
            event_candidate_top_k=top_k,
            output_top_k=top_k,
            refine_top_n=refine_top_n,
        )
    raise TypeError(f"unsupported query type: {type(query).__name__}")


def _run_request(runtime: L21150Runtime, query: Any, request: Any) -> dict[str, Any]:
    if isinstance(query, L21150KISQuery):
        return runtime.handle_query(request)
    if isinstance(query, L21150QAQuery):
        return runtime.handle_qa_query(request)
    return runtime.handle_trake_query(request)


def run_trake_nomination_diagnostic(
    benchmark: L21150Benchmark,
    runtime: Any,
    output_dir: Path,
    *,
    experiment_id: str,
    language_policy: str,
    sidecar: TRAKEDevTranslationSidecar | None,
    sidecar_sha256: str | None,
    rrf_constant: float = 60.0,
    event_video_nomination_depth: int = 100,
) -> dict[str, Any]:
    """Run target-agnostic D0 nomination and stop before restricted search."""

    policy = TRAKELanguagePolicy(language_policy)
    safe_queries = build_nomination_inputs(
        benchmark,
        language_policy=policy,
        sidecar=sidecar,
    )
    searcher = getattr(runtime, "video_restricted_searcher", None)
    encoder = getattr(runtime, "shared_encoder", None)
    if searcher is None or encoder is None:
        raise ValueError("D0 runtime requires the shared encoder and video searcher")
    runtime_manifest = getattr(runtime, "manifest", None)
    artifact = run_nomination_only(
        safe_queries,
        benchmark_id=benchmark.benchmark_id,
        experiment_id=experiment_id,
        git_sha=_git_sha(),
        corpus_fingerprint=getattr(runtime_manifest, "fingerprint", None),
        model_identity=getattr(encoder, "identifiers", None),
        language_policy=policy,
        translation_sidecar_sha256=sidecar_sha256,
        translation_status=(sidecar.translation_status if sidecar else None),
        encoder=encoder,
        searcher=searcher,
        rrf_constant=rrf_constant,
        event_video_nomination_depth=event_video_nomination_depth,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ranking_path = output / "trake_nomination_rankings.json"
    write_json_document(ranking_path, artifact)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "git_sha": artifact["git_sha"],
        "corpus_fingerprint": artifact["corpus_fingerprint"],
        "diagnostic_mode": "TR_A2_D0_NOMINATION_ONLY",
        "language_policy": policy.value,
        "translation_sidecar_sha256": sidecar_sha256,
        "translation_status": sidecar.translation_status if sidecar else None,
        "retrieval_feedback_used": False,
        "GT_USED_IN_RUNTIME": False,
        "query_count": artifact["query_count"],
        "video_count": artifact["video_count"],
        "physical_rows_scored": artifact["physical_rows_scored"],
        "store_scan_count": artifact["store_scan_count"],
        "runtime_duration_seconds": artifact["runtime_duration_seconds"],
        "outputs": {"nomination_rankings": ranking_path.name},
    }
    write_json_document(output / "experiment_manifest.json", manifest)
    (output / "run_summary.md").write_text(
        "\n".join(
            [
                "# TR-A2-D0 Nomination-Only Run",
                "",
                f"- Language policy: `{policy.value}`",
                f"- Queries: {artifact['query_count']}",
                f"- Videos/query: {artifact['video_count']}",
                "- GT used in runtime: `false`",
                "- Restricted selected-video search: `not executed`",
                "- Planner: `not executed`",
                "- Raw refinement: `not executed`",
                "- Semantic quality claim: `false` until offline DEV join",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return artifact


def run_l21_150_baseline(
    benchmark: L21150Benchmark,
    runtime: L21150Runtime,
    output_dir: Path,
    *,
    experiment_id: str,
    split: str,
    task: str,
    top_k: int,
    refine_top_n: int,
    resume: bool,
    fail_fast: bool,
    benchmark_sha256: str,
    manifest_sha256: str | None,
    gt_policy: str,
    kis_query_policy: str = "vi_only",
    kis_query_sidecar: KISDevTranslationSidecar | None = None,
    kis_query_sidecar_path: Path | None = None,
    kis_query_sidecar_sha256: str | None = None,
    qa_localization_language_policy: str = "legacy_vi",
    qa_translation_sidecar: QADevTranslationSidecar | None = None,
    qa_translation_sidecar_path: Path | None = None,
    qa_translation_sidecar_sha256: str | None = None,
    q3_temporal_policy: str = "none",
    q3_config: VideoConditionedKeyframeConfig | None = None,
    q3_anchor_refinement_config: Q3AnchorRefinementConfig | None = None,
    trake_video_first_config: TRAKEVideoFirstConfig | None = None,
    trake_shared_raw_region_config: SharedRawRegionRefinementConfig | None = None,
    trake_language_policy: str = "vi_only",
    trake_query_sidecar: TRAKEDevTranslationSidecar | None = None,
    trake_query_sidecar_path: Path | None = None,
    trake_query_sidecar_sha256: str | None = None,
    qa_video_conditioned_evidence_config: QAVideoConditionedEvidenceConfig | None = None,
    qa_object_answer_provider_config: ObjectAnswerProviderConfig | None = None,
    qa_ocr_answer_provider_config: OCRAnswerProviderConfig | None = None,
) -> dict[str, Any]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be in [1, 100]")
    if split not in {"dev", "holdout", "all"}:
        raise ValueError("split must be dev, holdout, or all")
    if task not in {"kis", "qa", "trake", "all"}:
        raise ValueError("task must be kis, qa, trake, or all")
    if qa_localization_language_policy not in {"legacy_vi", "en_only"}:
        raise ValueError(
            "qa_localization_language_policy must be legacy_vi or en_only"
        )
    supported_kis_policies = {
        "vi_only",
        "translation_augmented_rrf",
        "en_only",
    }
    if kis_query_policy not in supported_kis_policies:
        raise ValueError(
            "kis_query_policy must be vi_only, translation_augmented_rrf, or en_only"
        )
    supported_q3_policies = {"none", "video_conditioned_keyframe_diversity"}
    if q3_temporal_policy not in supported_q3_policies:
        raise ValueError(
            "q3_temporal_policy must be none or video_conditioned_keyframe_diversity"
        )
    q3_enabled = q3_temporal_policy == "video_conditioned_keyframe_diversity"
    resolved_q3_config = q3_config or VideoConditionedKeyframeConfig(enabled=q3_enabled)
    if resolved_q3_config.enabled != q3_enabled:
        raise ValueError("q3_temporal_policy and q3_config.enabled disagree")
    if q3_enabled and (split != "dev" or task != "kis" or kis_query_policy != "en_only"):
        raise ValueError(
            "VIDEO_CONDITIONED_KEYFRAME_DIVERSITY is restricted to KIS DEV EN_ONLY"
        )
    resolved_q3_anchor_config = (
        q3_anchor_refinement_config or Q3AnchorRefinementConfig()
    )
    resolved_trake_video_first = (
        trake_video_first_config or TRAKEVideoFirstConfig()
    )
    resolved_trake_shared_raw = (
        trake_shared_raw_region_config or SharedRawRegionRefinementConfig()
    )
    resolved_qa_grounding = (
        qa_video_conditioned_evidence_config or QAVideoConditionedEvidenceConfig()
    )
    resolved_qa_object_provider = (
        qa_object_answer_provider_config or ObjectAnswerProviderConfig()
    )
    resolved_qa_ocr_provider = (
        qa_ocr_answer_provider_config or OCRAnswerProviderConfig()
    )
    if resolved_qa_grounding.enabled and (split != "dev" or task != "qa"):
        raise ValueError("QA-A1 is restricted to the QA DEV diagnostic experiment")
    if resolved_qa_object_provider.enabled and (
        split != "dev" or task != "qa" or not resolved_qa_grounding.enabled
    ):
        raise ValueError("QA-A2 requires QA DEV with QA-A1 evidence grounding enabled")
    if resolved_qa_ocr_provider.enabled and (
        split != "dev" or task != "qa" or not resolved_qa_grounding.enabled
    ):
        raise ValueError("QA-A3 requires QA DEV with QA-A1 evidence grounding enabled")
    if qa_localization_language_policy == "en_only":
        if split != "dev" or task != "qa" or not resolved_qa_grounding.enabled:
            raise ValueError(
                "QA EN_ONLY is restricted to DEV QA with QA-A1 evidence grounding"
            )
        if qa_translation_sidecar is None:
            raise ValueError("QA EN_ONLY requires a validated frozen QA-D0 sidecar")
        if (
            qa_translation_sidecar_path is None
            or qa_translation_sidecar_sha256 is None
        ):
            raise ValueError("QA EN_ONLY requires frozen sidecar path and SHA256")
        actual_qa_sidecar_sha256 = hashlib.sha256(
            Path(qa_translation_sidecar_path).read_bytes()
        ).hexdigest()
        if (
            qa_translation_sidecar_sha256 != actual_qa_sidecar_sha256
            or actual_qa_sidecar_sha256 != FROZEN_QA_D0_DEV_EN_SIDECAR_SHA256
        ):
            raise ValueError(
                "QA sidecar SHA256 does not match the frozen QA-D0 DEV artifact"
            )
    elif any(
        value is not None
        for value in (
            qa_translation_sidecar,
            qa_translation_sidecar_path,
            qa_translation_sidecar_sha256,
        )
    ):
        raise ValueError("QA legacy_vi must not receive a translation sidecar")
    if resolved_trake_video_first.enabled and (split != "dev" or task != "trake"):
        raise ValueError("TR-A1 is restricted to the TRAKE DEV diagnostic experiment")
    if resolved_trake_shared_raw.enabled and (
        split != "dev"
        or task != "trake"
        or not resolved_trake_video_first.enabled
        or refine_top_n <= 0
    ):
        raise ValueError(
            "TR-A3 shared raw-region refinement requires TRAKE DEV, TR-A1, "
            "and refine_top_n > 0"
        )
    if trake_language_policy not in {"vi_only", "en_only"}:
        raise ValueError("normal TRAKE E2E supports only vi_only or en_only policy")
    if trake_language_policy == "en_only":
        if not resolved_trake_video_first.enabled or split != "dev" or task != "trake":
            raise ValueError("TRAKE EN_ONLY is restricted to DEV TRAKE with TR-A1 enabled")
        if trake_query_sidecar is None:
            raise ValueError("TRAKE EN_ONLY requires a validated frozen D0 sidecar")
        if trake_query_sidecar_path is None or trake_query_sidecar_sha256 is None:
            raise ValueError("TRAKE EN_ONLY requires frozen sidecar path and SHA256")
        actual_trake_sidecar_sha256 = hashlib.sha256(
            Path(trake_query_sidecar_path).read_bytes()
        ).hexdigest()
        if (
            trake_query_sidecar_sha256 != actual_trake_sidecar_sha256
            or actual_trake_sidecar_sha256
            != FROZEN_TR_A2_D0_TRAKE_DEV_EN_SIDECAR_SHA256
        ):
            raise ValueError(
                "TRAKE sidecar SHA256 does not match the frozen D0 DEV artifact"
            )
    elif any(
        value is not None
        for value in (
            trake_query_sidecar,
            trake_query_sidecar_path,
            trake_query_sidecar_sha256,
        )
    ):
        raise ValueError("TRAKE VI_ONLY must not receive a translation sidecar")
    if resolved_q3_anchor_config.enabled and (
        not q3_enabled
        or split != "dev"
        or task != "kis"
        or kis_query_policy != "en_only"
        or refine_top_n <= 0
    ):
        raise ValueError(
            "Q3 anchor raw refinement requires KIS DEV EN_ONLY, enabled Q3, "
            "and refine_top_n > 0"
        )
    if kis_query_policy in {"translation_augmented_rrf", "en_only"}:
        if kis_query_sidecar is None:
            raise ValueError(
                f"{kis_query_policy} requires a validated KIS query sidecar"
            )
        if split != "dev" or task != "kis":
            raise ValueError(
                f"{kis_query_policy} is restricted to the KIS DEV experiment"
            )
        if kis_query_sidecar_path is None or kis_query_sidecar_sha256 is None:
            raise ValueError(
                f"{kis_query_policy} requires frozen sidecar path and SHA256 provenance"
            )
        actual_sidecar_sha256 = hashlib.sha256(
            Path(kis_query_sidecar_path).read_bytes()
        ).hexdigest()
        if (
            kis_query_sidecar_sha256 != actual_sidecar_sha256
            or actual_sidecar_sha256 != FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256
        ):
            raise ValueError(
                "KIS query sidecar SHA256 does not match the frozen Q2 DEV artifact"
            )
    elif kis_query_sidecar is not None:
        raise ValueError("a KIS query sidecar is not valid with vi_only policy")

    runtime_q3_config = getattr(
        getattr(runtime, "config", None),
        "video_conditioned_keyframe_config",
        None,
    )
    if q3_enabled and runtime_q3_config is None:
        raise ValueError("runtime is missing the enabled Q3 production configuration")
    if runtime_q3_config is not None and runtime_q3_config != resolved_q3_config:
        raise ValueError("runtime Q3 configuration does not match experiment Q3 configuration")
    runtime_q3_anchor_config = getattr(
        getattr(runtime, "config", None),
        "q3_anchor_refinement_config",
        None,
    )
    if runtime_q3_anchor_config is not None and (
        runtime_q3_anchor_config != resolved_q3_anchor_config
    ):
        raise ValueError(
            "runtime Q3 anchor configuration does not match experiment configuration"
        )
    runtime_trake_video_first = getattr(
        getattr(runtime, "config", None),
        "trake_video_first_config",
        None,
    )
    if resolved_trake_video_first.enabled and runtime_trake_video_first is None:
        raise ValueError("runtime is missing the enabled TR-A1 production configuration")
    if runtime_trake_video_first is not None and (
        runtime_trake_video_first != resolved_trake_video_first
    ):
        raise ValueError(
            "runtime TR-A1 configuration does not match experiment configuration"
        )
    runtime_qa_grounding = getattr(
        getattr(runtime, "config", None),
        "qa_video_conditioned_evidence_config",
        None,
    )
    if resolved_qa_grounding.enabled and runtime_qa_grounding is None:
        raise ValueError("runtime is missing the enabled QA-A1 production configuration")
    if runtime_qa_grounding is not None and runtime_qa_grounding != resolved_qa_grounding:
        raise ValueError(
            "runtime QA-A1 configuration does not match experiment configuration"
        )
    runtime_qa_object_provider = getattr(
        getattr(runtime, "config", None),
        "qa_object_answer_provider_config",
        None,
    )
    if resolved_qa_object_provider.enabled and runtime_qa_object_provider is None:
        raise ValueError("runtime is missing the enabled QA-A2 object provider")
    if (
        runtime_qa_object_provider is not None
        and runtime_qa_object_provider != resolved_qa_object_provider
    ):
        raise ValueError(
            "runtime QA-A2 configuration does not match experiment configuration"
        )
    runtime_qa_ocr_provider = getattr(
        getattr(runtime, "config", None),
        "qa_ocr_answer_provider_config",
        None,
    )
    if resolved_qa_ocr_provider.enabled and runtime_qa_ocr_provider is None:
        raise ValueError("runtime is missing the enabled QA-A3 OCR provider")
    if (
        runtime_qa_ocr_provider is not None
        and runtime_qa_ocr_provider != resolved_qa_ocr_provider
    ):
        raise ValueError(
            "runtime QA-A3 configuration does not match experiment configuration"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    failures_path = output / "failures.jsonl"
    existing_predictions = _load_jsonl(predictions_path) if resume else []
    existing_failures = _load_jsonl(failures_path) if resume else []
    completed_ids = {record["query_id"] for record in existing_predictions}
    completed_ids.update(record["query_id"] for record in existing_failures)

    selected = [
        query
        for query in benchmark.queries
        if (split == "all" or query.split.casefold() == split)
        and (task == "all" or query.task_type == task)
    ]
    predictions = list(existing_predictions)
    failures = list(existing_failures)
    query_summaries: list[dict[str, Any]] = []
    kis_translations = (
        kis_query_sidecar.translations if kis_query_sidecar is not None else None
    )
    qa_translations = (
        qa_translation_sidecar.translations
        if qa_translation_sidecar is not None
        else None
    )
    trake_translations = (
        trake_query_sidecar.translations
        if trake_query_sidecar is not None
        else None
    )
    for query in selected:
        if query.query_id in completed_ids:
            continue
        request = _runtime_request(
            query,
            experiment_id,
            top_k,
            refine_top_n,
            kis_query_policy=kis_query_policy,
            kis_translations=kis_translations,
            qa_localization_language_policy=qa_localization_language_policy,
            qa_translations=qa_translations,
            trake_language_policy=trake_language_policy,
            trake_translations=trake_translations,
        )
        try:
            response = _run_request(runtime, query, request)
            if response.get("status") != "SUCCESS":
                raise RuntimeError(f"runtime returned status {response.get('status')!r}")
            if query.task_type == "kis":
                query_predictions = _kis_predictions(runtime, response)
            else:
                query_predictions = _response_predictions(response, query.task_type)
            if len(query_predictions) > top_k:
                raise ValueError(
                    f"runtime returned {len(query_predictions)} predictions, "
                    f"exceeding requested_top_k={top_k}"
                )
            latency = _response_latency(response)
            for prediction in query_predictions:
                prediction.update(
                    {
                        "experiment_id": experiment_id,
                        "task": query.task_type,
                        "request_id": request.request_id,
                        "latency_seconds": latency,
                        "combined_score": None,
                        "branch_scores": None,
                        "retrieval_source": "existing_system_tai_runtime",
                        "query_variant_id": None,
                    }
                )
            predictions.extend(query_predictions)
            query_summaries.append(
                {
                    "query_id": query.query_id,
                    "task": query.task_type,
                    "status": "SUCCESS",
                    "prediction_count": len(query_predictions),
                    "latency_seconds": latency,
                }
            )
        except Exception as exc:
            failure = {
                "experiment_id": experiment_id,
                "query_id": query.query_id,
                "task": query.task_type,
                "request_id": request.request_id,
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
            }
            failures.append(failure)
            query_summaries.append({**failure, "status": "FAILED"})
            if fail_fast:
                break

    successful_query_ids = {
        str(summary["query_id"])
        for summary in query_summaries
        if summary["status"] == "SUCCESS"
    }
    output_diagnostics = _output_depth_diagnostics(
        predictions,
        successful_query_ids,
        top_k,
    )

    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with failures_path.open("w", encoding="utf-8", newline="") as stream:
        for failure in failures:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")

    task_counts = Counter(summary["task"] for summary in query_summaries)
    success_count = sum(summary["status"] == "SUCCESS" for summary in query_summaries)
    runtime_manifest = getattr(runtime, "manifest", None)
    runtime_encoder = getattr(runtime, "shared_encoder", None)
    resolved_kis_query_policy = {
        "vi_only": "VI_ONLY",
        "translation_augmented_rrf": "TRANSLATION_AUGMENTED_RRF",
        "en_only": "EN_ONLY",
    }[kis_query_policy]
    query_policy_changed_from_e0 = (
        kis_query_policy != "vi_only"
        or trake_language_policy != "vi_only"
        or qa_localization_language_policy != "legacy_vi"
    )
    resolved_trake_query_policy = trake_language_policy.upper()
    production_algorithm_modified = (
        q3_enabled
        or resolved_trake_video_first.enabled
        or resolved_qa_grounding.enabled
        or resolved_qa_object_provider.enabled
        or resolved_qa_ocr_provider.enabled
    )
    qa_capability_scope = [
        scope
        for enabled, scope in (
            (resolved_qa_grounding.enabled, QA_VIDEO_CONDITIONED_EVIDENCE_V1),
            (
                resolved_qa_grounding.preserve_keyframe_evidence,
                QA_KEYFRAME_EVIDENCE_BANK_V1,
            ),
            (
                resolved_qa_grounding.temporal_refinement_enabled,
                QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1,
            ),
            (resolved_qa_object_provider.enabled, QA_ARTIFACT_BACKED_OBJECT_EVIDENCE),
            (resolved_qa_ocr_provider.enabled, QA_EVIDENCE_GROUNDED_OCR),
        )
        if enabled
    ]
    metadata = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "git_sha": _git_sha(),
        "benchmark_sha256": benchmark_sha256,
        "manifest_sha256": manifest_sha256,
        "benchmark_id": benchmark.benchmark_id,
        "corpus_fingerprint": getattr(runtime_manifest, "fingerprint", None),
        "index_identity": getattr(runtime_manifest, "schema_version", None),
        "model_identity": getattr(runtime_encoder, "identifiers", None),
        "top_k": top_k,
        **output_diagnostics,
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "device_runtime": getattr(getattr(runtime, "config", None), "device", None),
        "gt_policy": gt_policy,
        "split": split,
        "task": task,
        "selected_query_count": len(selected),
        "executed_query_count": len(query_summaries),
        "successful_query_count": success_count,
        "failed_query_count": len(query_summaries) - success_count,
        "task_counts": dict(sorted(task_counts.items())),
        "production_algorithm_modified": production_algorithm_modified,
        "production_algorithm_modified_scope": (
            "+".join(qa_capability_scope)
            if resolved_qa_ocr_provider.enabled
            else QA_ARTIFACT_BACKED_OBJECT_EVIDENCE
            if resolved_qa_object_provider.enabled
            else QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1
            if resolved_qa_grounding.temporal_refinement_enabled
            else QA_KEYFRAME_EVIDENCE_BANK_V1
            if resolved_qa_grounding.preserve_keyframe_evidence
            else QA_VIDEO_CONDITIONED_EVIDENCE_V1
            if resolved_qa_grounding.enabled
            else TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH
            if resolved_trake_video_first.enabled
            else "KIS_Q3_ANCHOR_AWARE_RAW_REFINEMENT"
            if resolved_q3_anchor_config.enabled
            else "KIS_VIDEO_CONDITIONED_KEYFRAME_DIVERSITY"
            if q3_enabled
            else "CORE_PRODUCTION_IMPLEMENTATION"
        ),
        "core_production_algorithm_modified": production_algorithm_modified,
        "kis_query_policy": resolved_kis_query_policy,
        "trake_query_policy": resolved_trake_query_policy,
        "qa_localization_language_policy": (
            "EN_ONLY"
            if qa_localization_language_policy == "en_only"
            else "LEGACY_VI"
        ),
        "query_policy_changed_from_e0": query_policy_changed_from_e0,
        "runtime_contract": "OperationalKISRuntime public task handlers",
        "known_current_limitations": [
            "QA closed-set support centers on COLOR, COUNT, YES_NO, and DIRECTION",
            "TRAKE baseline does not implement all design-level gap constraints",
            "OCR is opt-in, bounded to QA-A1 evidence, and requires Tesseract language data",
            "ASR and BM25 are not integrated in runtime retrieval",
        ],
        "qa_capability_scope": qa_capability_scope,
        "outputs": {
            "predictions_jsonl": predictions_path.name,
            "failures_jsonl": failures_path.name,
        },
        "queries": query_summaries,
    }
    if q3_enabled:
        metadata.update(
            {
                "q3_temporal_policy": VIDEO_CONDITIONED_KEYFRAME_DIVERSITY,
                "q3_protected_prefix_rank": refine_top_n,
                "q3_anchor_refinement": {
                    "enabled": resolved_q3_anchor_config.enabled,
                    "max_extra_q3_anchors": (
                        resolved_q3_anchor_config.max_extra_q3_anchors
                    ),
                },
                "q3_config": {
                    "selected_video_global_rank_cap": (
                        resolved_q3_config.selected_video_global_rank_cap
                    ),
                    "max_selected_videos": resolved_q3_config.max_selected_videos,
                    "max_anchors_per_video": resolved_q3_config.max_anchors_per_video,
                    "minimum_anchor_gap_seconds": (
                        resolved_q3_config.minimum_anchor_gap_seconds
                    ),
                    "preserve_first_video_occurrence": (
                        resolved_q3_config.preserve_first_video_occurrence
                    ),
                },
            }
        )
    if resolved_trake_video_first.enabled:
        metadata.update(
            {
                "trake_video_first_policy": (
                    TRAKE_VIDEO_FIRST_RESTRICTED_EVENT_SEARCH
                ),
                "trake_video_first_config": {
                    "selected_video_cap": (
                        resolved_trake_video_first.selected_video_cap
                    ),
                    "event_video_nomination_depth": (
                        resolved_trake_video_first.event_video_nomination_depth
                    ),
                    "anchors_per_event_video": (
                        resolved_trake_video_first.anchors_per_event_video
                    ),
                },
                "trake_shared_raw_region_refinement": {
                    "enabled": resolved_trake_shared_raw.enabled,
                },
            }
        )
    if resolved_qa_grounding.enabled:
        metadata.update(
            {
                "qa_grounding_policy": QA_VIDEO_CONDITIONED_EVIDENCE_V1,
                "localization_input_contract": "QUESTION_AS_LOCALIZATION_FALLBACK",
                "official_ground_truth": False,
                "diagnostic_development_only": True,
                "qa_video_conditioned_evidence_config": {
                    "selected_video_cap": resolved_qa_grounding.selected_video_cap,
                    "anchors_per_video": resolved_qa_grounding.anchors_per_video,
                    "video_rrf_constant": resolved_qa_grounding.video_rrf_constant,
                    "preserve_keyframe_evidence": (
                        resolved_qa_grounding.preserve_keyframe_evidence
                    ),
                    "keyframe_evidence_video_cap": (
                        resolved_qa_grounding.keyframe_evidence_video_cap
                    ),
                    "temporal_refinement_enabled": (
                        resolved_qa_grounding.temporal_refinement_enabled
                    ),
                    "temporal_seed_anchors_per_video": (
                        resolved_qa_grounding.temporal_seed_anchors_per_video
                    ),
                    "temporal_refinement_video_cap": (
                        resolved_qa_grounding.temporal_refinement_video_cap
                    ),
                    "temporal_refinement_total_seed_cap": (
                        resolved_qa_grounding.temporal_refinement_total_seed_cap
                    ),
                },
            }
        )
        if resolved_qa_grounding.preserve_keyframe_evidence:
            metadata["qa_keyframe_evidence_bank"] = {
                "policy": QA_KEYFRAME_EVIDENCE_BANK_V1,
                "enabled": True,
                "selection": (
                    "BOUNDED_MULTI_SEED_LOCAL_ANCHORS_PER_NOMINATED_VIDEO"
                    if resolved_qa_grounding.temporal_refinement_enabled
                    else "ONE_PRIMARY_LOCAL_ANCHOR_PER_NOMINATED_VIDEO"
                ),
                "video_cap": resolved_qa_grounding.keyframe_evidence_video_cap,
                "raw_refinement_budget": (
                    resolved_qa_grounding.temporal_refinement_total_seed_cap
                    if resolved_qa_grounding.temporal_refinement_enabled
                    else refine_top_n
                ),
                "refinement_is_upgrade_not_admission_gate": True,
            }
        if resolved_qa_grounding.temporal_refinement_enabled:
            metadata["qa_multi_seed_temporal_refinement"] = {
                "policy": QA_MULTI_SEED_TEMPORAL_REFINEMENT_V1,
                "enabled": True,
                "seeds_per_video": (
                    resolved_qa_grounding.temporal_seed_anchors_per_video
                ),
                "video_cap": resolved_qa_grounding.temporal_refinement_video_cap,
                "total_seed_cap": (
                    resolved_qa_grounding.temporal_refinement_total_seed_cap
                ),
                "legacy_refine_top_n_executed": False,
                "bidirectional_window": True,
            }
    if qa_localization_language_policy == "en_only":
        assert qa_translation_sidecar is not None
        metadata["qa_localization_experiment"] = {
            "qa_localization_policy": "EN_ONLY",
            "include_vi_variant": False,
            "localization_variant_count": 1,
            "localization_language": "en",
            "localization_variant_type": "english_translation",
            "english_localization_provenance": "explicit_frozen_translation_input",
            "question_language_for_answer_routing": "vi",
            "question_en_populated": False,
            "translation_sidecar_path": str(qa_translation_sidecar_path),
            "translation_sidecar_sha256": qa_translation_sidecar_sha256,
            "translation_status": qa_translation_sidecar.translation_status,
            "localization_input_contract": "QUESTION_AS_LOCALIZATION_FALLBACK",
            "source_vi_retained_for_answer_routing": True,
            "source_vi_used_for_localization_retrieval": False,
            "retrieval_feedback_used": False,
        }
    if resolved_qa_object_provider.enabled:
        object_index = getattr(runtime, "object_artifact_index", None)
        metadata.update(
            {
                "qa_object_provider_enabled": True,
                "object_artifact_schema": (
                    object_index.schema_identity if object_index is not None else None
                ),
                "object_artifact_root_identity": (
                    object_index.source_root_identity if object_index is not None else None
                ),
                "frame_identity_contract": (
                    "object JSON filename ordinal -> mapping n -> original frame_idx"
                ),
                "official_ground_truth": False,
                "diagnostic_development_only": True,
            }
        )
    if resolved_qa_ocr_provider.enabled:
        ocr_provider = getattr(runtime, "ocr_answer_provider", None)
        metadata.update(
            {
                "qa_ocr_provider_enabled": True,
                "ocr_backend_identity": (
                    dict(ocr_provider.identifiers)
                    if ocr_provider is not None
                    else None
                ),
                "ocr_evidence_frame_budget": (
                    resolved_qa_ocr_provider.evidence_frame_budget
                ),
                "ocr_frame_identity_contract": (
                    "decoded QA-A1 evidence frame -> original-video absolute frame_id"
                ),
                "official_ground_truth": False,
                "diagnostic_development_only": True,
            }
        )
    if trake_language_policy == "en_only":
        assert trake_query_sidecar is not None
        metadata["trake_query_experiment"] = {
            "trake_query_policy": "EN_ONLY",
            "translation_sidecar_sha256": trake_query_sidecar_sha256,
            "translation_status": trake_query_sidecar.translation_status,
            "source_vi_retained_for_provenance": True,
            "source_vi_used_for_retrieval": False,
            "translation_en_used_for_retrieval": True,
            "include_vi_variant": False,
            "variant_count_policy": "1_VARIANT_EN_ONLY_PER_EVENT",
            "retrieval_feedback_used": False,
        }
    if kis_query_policy in {"translation_augmented_rrf", "en_only"}:
        assert kis_query_sidecar is not None
        experiment_metadata = {
            "query_policy": resolved_kis_query_policy,
            "sidecar_path": (
                str(kis_query_sidecar_path) if kis_query_sidecar_path is not None else None
            ),
            "sidecar_basename": (
                kis_query_sidecar_path.name
                if kis_query_sidecar_path is not None
                else None
            ),
            "sidecar_sha256": kis_query_sidecar_sha256,
            "sidecar_schema_version": kis_query_sidecar.schema_version,
            "translation_status": kis_query_sidecar.translation_status,
            "query_en_expansion_used": False,
        }
        if kis_query_policy == "translation_augmented_rrf":
            experiment_metadata.update(
                {
                    "variant_count_policy": "2_VARIANTS_VI_PLUS_EN",
                    "variant_weights": {"vi": 1.0, "en": 1.0},
                }
            )
        else:
            experiment_metadata.update(
                {
                    "variant_count_policy": "1_VARIANT_EN_ONLY",
                    "variant_weights": {"en": 1.0},
                    "source_vi_used_for_retrieval": False,
                    "translation_en_used_for_retrieval": True,
                }
            )
        metadata["kis_query_experiment"] = experiment_metadata
    metadata = _write_json_document(output / "experiment_manifest.json", metadata)
    (output / "run_summary.md").write_text(
        "\n".join(
            [
                "# L21-150 Experiment Run",
                "",
                f"- Experiment: `{experiment_id}`",
                f"- Selected queries: {len(selected)}",
                f"- Executed queries: {len(query_summaries)}",
                f"- Successful: {success_count}",
                f"- Failed: {len(query_summaries) - success_count}",
                f"- Prediction records: {metadata['prediction_record_count']}",
                (
                    "- Prediction depth range: "
                    f"{metadata['prediction_depth_min']}.."
                    f"{metadata['prediction_depth_max']}"
                ),
                (
                    "- Queries below requested depth: "
                    f"{len(metadata['queries_below_requested_depth'])}"
                ),
                (
                    "- Duplicate output identities: "
                    f"{metadata['duplicate_output_identity_count']}"
                ),
                "- Core production retrieval/ranking implementation changed: "
                f"`{str(production_algorithm_modified).lower()}`",
                f"- KIS query policy: `{resolved_kis_query_policy}`",
                f"- TRAKE query policy: `{resolved_trake_query_policy}`",
                "- Q3 temporal policy: "
                f"`{VIDEO_CONDITIONED_KEYFRAME_DIVERSITY if q3_enabled else 'NONE'}`",
                (
                    "- Query policy changed from E0: "
                    f"`{str(query_policy_changed_from_e0).lower()}`"
                ),
                *(
                    [
                        "- English translation input: `REVIEWED_FROZEN`",
                        (
                            "- Vietnamese source retained in benchmark provenance "
                            "but used for retrieval: `false`"
                        ),
                        "- Causal or official accuracy claim: `false`",
                    ]
                    if kis_query_policy == "en_only"
                    else []
                ),
                "- Semantic quality claim: `false` until evaluator output is reviewed",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    manifest = parser.add_mutually_exclusive_group(required=True)
    manifest.add_argument("--reuse-manifest", type=Path)
    manifest.add_argument("--manifest-cache", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--task", choices=("kis", "qa", "trake", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--refine-top-n", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--gt-policy", choices=("proposed", "validated-only"), default="proposed")
    parser.add_argument("--experiment-id")
    parser.add_argument("--manifest-sha256")
    parser.add_argument(
        "--kis-query-policy",
        choices=("vi_only", "translation_augmented_rrf", "en_only"),
        default="vi_only",
    )
    parser.add_argument("--kis-query-sidecar", type=Path)
    parser.add_argument(
        "--q3-temporal-policy",
        choices=("none", "video_conditioned_keyframe_diversity"),
        default="none",
    )
    parser.add_argument("--q3-selected-video-global-rank-cap", type=int, default=50)
    parser.add_argument("--q3-max-selected-videos", type=int, default=50)
    parser.add_argument("--q3-max-anchors-per-video", type=int, default=3)
    parser.add_argument("--q3-minimum-anchor-gap-seconds", type=float, default=5.0)
    parser.add_argument("--q3-anchor-raw-refinement", action="store_true")
    parser.add_argument("--q3-max-extra-raw-anchors", type=int, default=6)
    parser.add_argument(
        "--trake-video-first-restricted-search",
        action="store_true",
    )
    parser.add_argument("--trake-selected-video-cap", type=int, default=32)
    parser.add_argument(
        "--trake-event-video-nomination-depth",
        type=int,
        default=100,
    )
    parser.add_argument("--trake-anchors-per-event-video", type=int, default=5)
    parser.add_argument(
        "--trake-shared-raw-region-refinement",
        action="store_true",
    )
    parser.add_argument("--trake-nomination-only", action="store_true")
    parser.add_argument(
        "--trake-language-policy",
        choices=tuple(policy.value for policy in TRAKELanguagePolicy),
        default=TRAKELanguagePolicy.VI_ONLY.value,
    )
    parser.add_argument("--trake-dev-en-sidecar", type=Path)
    parser.add_argument("--trake-rrf-constant", type=float, default=60.0)
    parser.add_argument("--qa-video-conditioned-evidence", action="store_true")
    parser.add_argument(
        "--qa-localization-language-policy",
        choices=("legacy_vi", "en_only"),
        default="legacy_vi",
    )
    parser.add_argument("--qa-dev-en-sidecar", type=Path)
    parser.add_argument("--qa-keyframe-evidence-bank", action="store_true")
    parser.add_argument(
        "--qa-keyframe-evidence-video-cap",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--qa-multi-seed-temporal-refinement",
        action="store_true",
    )
    parser.add_argument("--qa-temporal-seeds-per-video", type=int, default=3)
    parser.add_argument("--qa-temporal-refinement-video-cap", type=int, default=32)
    parser.add_argument(
        "--qa-temporal-refinement-total-seed-cap",
        type=int,
        default=96,
    )
    parser.add_argument("--qa-object-evidence", action="store_true")
    parser.add_argument("--qa-ocr-evidence", action="store_true")
    parser.add_argument("--qa-ocr-evidence-frame-budget", type=int, default=10)
    parser.add_argument("--qa-ocr-executable", default="tesseract")
    parser.add_argument("--qa-ocr-languages", default="eng")
    parser.add_argument("--qa-ocr-page-segmentation-mode", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.qa_video_conditioned_evidence and (
            args.split != "dev" or args.task != "qa"
        ):
            raise ValueError(
                "--qa-video-conditioned-evidence is restricted to --split dev --task qa"
            )
        if args.qa_object_evidence and (
            args.split != "dev"
            or args.task != "qa"
            or not args.qa_video_conditioned_evidence
        ):
            raise ValueError(
                "--qa-object-evidence requires --split dev --task qa "
                "--qa-video-conditioned-evidence"
            )
        if args.qa_keyframe_evidence_bank and (
            args.split != "dev"
            or args.task != "qa"
            or not args.qa_video_conditioned_evidence
        ):
            raise ValueError(
                "--qa-keyframe-evidence-bank requires --split dev --task qa "
                "--qa-video-conditioned-evidence"
            )
        if args.qa_multi_seed_temporal_refinement and (
            args.split != "dev"
            or args.task != "qa"
            or not args.qa_video_conditioned_evidence
            or not args.qa_keyframe_evidence_bank
            or args.qa_localization_language_policy != "en_only"
        ):
            raise ValueError(
                "--qa-multi-seed-temporal-refinement requires --split dev --task qa "
                "--qa-video-conditioned-evidence --qa-keyframe-evidence-bank "
                "--qa-localization-language-policy en_only"
            )
        if args.qa_ocr_evidence and (
            args.split != "dev"
            or args.task != "qa"
            or not args.qa_video_conditioned_evidence
        ):
            raise ValueError(
                "--qa-ocr-evidence requires --split dev --task qa "
                "--qa-video-conditioned-evidence"
            )
        benchmark = load_l21_150_benchmark(args.benchmark)
        benchmark_sha = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
        qa_sidecar = None
        qa_sidecar_sha = None
        if args.qa_localization_language_policy == "en_only":
            if (
                args.split != "dev"
                or args.task != "qa"
                or not args.qa_video_conditioned_evidence
            ):
                raise ValueError(
                    "QA EN_ONLY requires --split dev --task qa "
                    "--qa-video-conditioned-evidence"
                )
            if args.qa_dev_en_sidecar is None:
                raise ValueError("--qa-dev-en-sidecar is required for QA EN_ONLY")
            qa_sidecar = load_qa_dev_translation_sidecar(
                args.qa_dev_en_sidecar,
                benchmark,
                args.benchmark,
            )
            qa_sidecar_sha = hashlib.sha256(
                args.qa_dev_en_sidecar.read_bytes()
            ).hexdigest()
            if qa_sidecar_sha != FROZEN_QA_D0_DEV_EN_SIDECAR_SHA256:
                raise ValueError(
                    "--qa-dev-en-sidecar does not match the frozen QA-D0 DEV SHA256"
                )
        elif args.qa_dev_en_sidecar is not None:
            raise ValueError(
                "--qa-dev-en-sidecar requires "
                "--qa-localization-language-policy en_only"
            )
        kis_sidecar = None
        kis_sidecar_sha = None
        if args.kis_query_policy in {"translation_augmented_rrf", "en_only"}:
            if args.kis_query_sidecar is None:
                raise ValueError(
                    f"--kis-query-sidecar is required for {args.kis_query_policy}"
                )
            kis_sidecar = load_kis_dev_translation_sidecar(
                args.kis_query_sidecar,
                benchmark,
                args.benchmark,
            )
            kis_sidecar_sha = hashlib.sha256(
                args.kis_query_sidecar.read_bytes()
            ).hexdigest()
            if kis_sidecar_sha != FROZEN_Q2_KIS_DEV_EN_SIDECAR_SHA256:
                raise ValueError(
                    "--kis-query-sidecar does not match the frozen Q2 DEV SHA256"
                )
        elif args.kis_query_sidecar is not None:
            raise ValueError(
                "--kis-query-sidecar requires --kis-query-policy "
                "translation_augmented_rrf or en_only"
            )
        trake_sidecar = None
        trake_sidecar_sha = None
        trake_policy = TRAKELanguagePolicy(args.trake_language_policy)
        if args.trake_nomination_only:
            if args.split != "dev" or args.task != "trake":
                raise ValueError(
                    "--trake-nomination-only is restricted to --split dev --task trake"
                )
            if args.trake_video_first_restricted_search:
                raise ValueError(
                    "--trake-nomination-only conflicts with "
                    "--trake-video-first-restricted-search"
                )
            if args.resume:
                raise ValueError("--resume is not supported in nomination-only mode")
            if trake_policy is TRAKELanguagePolicy.VI_ONLY:
                if args.trake_dev_en_sidecar is not None:
                    raise ValueError(
                        "--trake-dev-en-sidecar must not be supplied for VI_ONLY"
                    )
            else:
                if args.trake_dev_en_sidecar is None:
                    raise ValueError(
                        f"--trake-dev-en-sidecar is required for {trake_policy.value}"
                    )
                trake_sidecar = load_trake_dev_translation_sidecar(
                    args.trake_dev_en_sidecar,
                    benchmark,
                    args.benchmark,
                )
                trake_sidecar_sha = hashlib.sha256(
                    args.trake_dev_en_sidecar.read_bytes()
                ).hexdigest()
                if (
                    trake_sidecar_sha
                    != FROZEN_TR_A2_D0_TRAKE_DEV_EN_SIDECAR_SHA256
                ):
                    raise ValueError(
                        "--trake-dev-en-sidecar does not match the frozen D0 DEV SHA256"
                    )
        else:
            if trake_policy is TRAKELanguagePolicy.VI_ONLY:
                if args.trake_dev_en_sidecar is not None:
                    raise ValueError(
                        "--trake-dev-en-sidecar must not be supplied for VI_ONLY"
                    )
            elif trake_policy is TRAKELanguagePolicy.EN_ONLY:
                if (
                    args.split != "dev"
                    or args.task != "trake"
                    or not args.trake_video_first_restricted_search
                ):
                    raise ValueError(
                        "normal TRAKE EN_ONLY requires --split dev --task trake "
                        "--trake-video-first-restricted-search"
                    )
                if args.trake_dev_en_sidecar is None:
                    raise ValueError(
                        "--trake-dev-en-sidecar is required for normal TRAKE EN_ONLY"
                    )
                trake_sidecar = load_trake_dev_translation_sidecar(
                    args.trake_dev_en_sidecar,
                    benchmark,
                    args.benchmark,
                )
                trake_sidecar_sha = hashlib.sha256(
                    args.trake_dev_en_sidecar.read_bytes()
                ).hexdigest()
                if (
                    trake_sidecar_sha
                    != FROZEN_TR_A2_D0_TRAKE_DEV_EN_SIDECAR_SHA256
                ):
                    raise ValueError(
                        "--trake-dev-en-sidecar does not match the frozen D0 DEV SHA256"
                    )
            else:
                raise ValueError(
                    "normal TRAKE E2E supports only vi_only or en_only; "
                    "vi_plus_en_weighted_rrf remains nomination-only"
                )
        experiment_id = args.experiment_id or datetime.now(UTC).strftime(
            "l21-150-e0-%Y%m%dT%H%M%SZ"
        )
        q3_config = VideoConditionedKeyframeConfig(
            enabled=args.q3_temporal_policy == "video_conditioned_keyframe_diversity",
            selected_video_global_rank_cap=args.q3_selected_video_global_rank_cap,
            max_selected_videos=args.q3_max_selected_videos,
            max_anchors_per_video=args.q3_max_anchors_per_video,
            minimum_anchor_gap_seconds=args.q3_minimum_anchor_gap_seconds,
            preserve_first_video_occurrence=True,
        )
        q3_anchor_config = Q3AnchorRefinementConfig(
            enabled=args.q3_anchor_raw_refinement,
            max_extra_q3_anchors=args.q3_max_extra_raw_anchors,
        )
        trake_video_first_config = TRAKEVideoFirstConfig(
            enabled=args.trake_video_first_restricted_search,
            selected_video_cap=args.trake_selected_video_cap,
            event_video_nomination_depth=args.trake_event_video_nomination_depth,
            anchors_per_event_video=args.trake_anchors_per_event_video,
        )
        trake_shared_raw_region_config = SharedRawRegionRefinementConfig(
            enabled=args.trake_shared_raw_region_refinement,
        )
        qa_video_conditioned_evidence_config = QAVideoConditionedEvidenceConfig(
            enabled=args.qa_video_conditioned_evidence,
            preserve_keyframe_evidence=args.qa_keyframe_evidence_bank,
            keyframe_evidence_video_cap=args.qa_keyframe_evidence_video_cap,
            temporal_refinement_enabled=args.qa_multi_seed_temporal_refinement,
            temporal_seed_anchors_per_video=args.qa_temporal_seeds_per_video,
            temporal_refinement_video_cap=args.qa_temporal_refinement_video_cap,
            temporal_refinement_total_seed_cap=(
                args.qa_temporal_refinement_total_seed_cap
            ),
        )
        qa_object_answer_provider_config = ObjectAnswerProviderConfig(
            enabled=args.qa_object_evidence,
        )
        qa_ocr_answer_provider_config = OCRAnswerProviderConfig(
            enabled=args.qa_ocr_evidence,
            evidence_frame_budget=args.qa_ocr_evidence_frame_budget,
            executable=args.qa_ocr_executable,
            languages=tuple(
                language.strip()
                for language in args.qa_ocr_languages.split("+")
                if language.strip()
            ),
            page_segmentation_mode=args.qa_ocr_page_segmentation_mode,
        )
        session_output = args.output_dir / "runtime"
        config = SessionConfig(
            input_root=args.input_root,
            reuse_manifest=args.reuse_manifest,
            manifest_cache=args.manifest_cache,
            output_root=session_output,
            device=args.device,
            allow_model_download=args.allow_model_download,
            default_output_top_k=args.top_k,
            default_refine_top_n=args.refine_top_n,
            video_conditioned_keyframe_config=q3_config,
            q3_anchor_refinement_config=q3_anchor_config,
            trake_video_first_config=trake_video_first_config,
            trake_shared_raw_region_config=trake_shared_raw_region_config,
            qa_video_conditioned_evidence_config=(
                qa_video_conditioned_evidence_config
            ),
            qa_object_answer_provider_config=qa_object_answer_provider_config,
            qa_ocr_answer_provider_config=qa_ocr_answer_provider_config,
        )
        runtime = OperationalKISRuntime.bootstrap(config)
        try:
            if args.trake_nomination_only:
                report = run_trake_nomination_diagnostic(
                    benchmark,
                    runtime,
                    args.output_dir,
                    experiment_id=experiment_id,
                    language_policy=trake_policy.value,
                    sidecar=trake_sidecar,
                    sidecar_sha256=trake_sidecar_sha,
                    rrf_constant=args.trake_rrf_constant,
                    event_video_nomination_depth=(
                        args.trake_event_video_nomination_depth
                    ),
                )
            else:
                report = run_l21_150_baseline(
                    benchmark,
                    runtime,
                    args.output_dir,
                    experiment_id=experiment_id,
                    split=args.split,
                    task=args.task,
                    top_k=args.top_k,
                    refine_top_n=args.refine_top_n,
                    resume=args.resume,
                    fail_fast=args.fail_fast,
                    benchmark_sha256=benchmark_sha,
                    manifest_sha256=args.manifest_sha256,
                    gt_policy=args.gt_policy,
                    kis_query_policy=args.kis_query_policy,
                    kis_query_sidecar=kis_sidecar,
                    kis_query_sidecar_path=args.kis_query_sidecar,
                    kis_query_sidecar_sha256=kis_sidecar_sha,
                    qa_localization_language_policy=(
                        args.qa_localization_language_policy
                    ),
                    qa_translation_sidecar=qa_sidecar,
                    qa_translation_sidecar_path=args.qa_dev_en_sidecar,
                    qa_translation_sidecar_sha256=qa_sidecar_sha,
                    q3_temporal_policy=args.q3_temporal_policy,
                    q3_config=q3_config,
                    q3_anchor_refinement_config=q3_anchor_config,
                    trake_video_first_config=trake_video_first_config,
                    trake_shared_raw_region_config=(
                        trake_shared_raw_region_config
                    ),
                    trake_language_policy=trake_policy.value,
                    trake_query_sidecar=trake_sidecar,
                    trake_query_sidecar_path=args.trake_dev_en_sidecar,
                    trake_query_sidecar_sha256=trake_sidecar_sha,
                    qa_video_conditioned_evidence_config=(
                        qa_video_conditioned_evidence_config
                    ),
                    qa_object_answer_provider_config=(
                        qa_object_answer_provider_config
                    ),
                    qa_ocr_answer_provider_config=(
                        qa_ocr_answer_provider_config
                    ),
                )
        finally:
            runtime.close(shutdown_reason="l21_150_baseline_complete")
    except (
        FileNotFoundError,
        KISTranslationSidecarError,
        L21150FormatError,
        OSError,
        QATranslationSidecarError,
        RuntimeError,
        TRAKENominationError,
        TRAKETranslationSidecarError,
        ValueError,
    ) as exc:
        print(f"L21-150 baseline run failed: {exc}", file=sys.stderr)
        return 2
    if args.trake_nomination_only:
        print(
            "TR-A2-D0 nomination-only run complete: "
            f"queries={report['query_count']} videos={report['video_count']}"
        )
        return 0
    print(
        "L21-150 baseline run complete: "
        f"success={report['successful_query_count']} failed={report['failed_query_count']}"
    )
    return 0 if report["failed_query_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
