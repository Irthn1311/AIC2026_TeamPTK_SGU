"""Run the frozen system_tai runtime against the L21-150 diagnostic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
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
from system_tai.quality.l21_150_schema import (  # noqa: E402
    L21150Benchmark,
    L21150FormatError,
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEQuery,
    load_l21_150_benchmark,
)


class L21150Runtime(Protocol):
    output_root: Path

    def handle_query(self, request: QueryRequest) -> dict[str, Any]: ...

    def handle_qa_query(self, request: QAQueryRequest) -> dict[str, Any]: ...

    def handle_trake_query(self, request: TRAKEQueryRequest) -> dict[str, Any]: ...


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


def _runtime_request(query: Any, experiment_id: str, top_k: int, refine_top_n: int):
    request_id = f"{experiment_id}:{query.query_id}"
    if isinstance(query, L21150KISQuery):
        return QueryRequest(
            request_id=request_id,
            query_id=query.query_id,
            query_vi=query.query_vi,
            top_k_per_variant=top_k,
            output_top_k=top_k,
            refine_top_n=refine_top_n,
        )
    if isinstance(query, L21150QAQuery):
        return QAQueryRequest(
            request_id=request_id,
            query_id=query.query_id,
            event_description=query.question_vi,
            question=query.question_vi,
            top_k_per_variant=top_k,
            output_top_k=top_k,
            refine_top_n=max(1, refine_top_n),
        )
    if isinstance(query, L21150TRAKEQuery):
        return TRAKEQueryRequest(
            request_id=request_id,
            query_id=query.query_id,
            events=tuple({"description": event.description_vi} for event in query.events),
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
) -> dict[str, Any]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be in [1, 100]")
    if split not in {"dev", "holdout", "all"}:
        raise ValueError("split must be dev, holdout, or all")
    if task not in {"kis", "qa", "trake", "all"}:
        raise ValueError("task must be kis, qa, trake, or all")

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
    for query in selected:
        if query.query_id in completed_ids:
            continue
        request = _runtime_request(query, experiment_id, top_k, refine_top_n)
        try:
            response = _run_request(runtime, query, request)
            if response.get("status") != "SUCCESS":
                raise RuntimeError(f"runtime returned status {response.get('status')!r}")
            if query.task_type == "kis":
                query_predictions = _kis_predictions(runtime, response)
            else:
                query_predictions = _response_predictions(response, query.task_type)
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
    metadata = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "git_sha": _git_sha(),
        "benchmark_sha256": benchmark_sha256,
        "manifest_sha256": manifest_sha256,
        "benchmark_id": benchmark.benchmark_id,
        "corpus_fingerprint": getattr(runtime_manifest, "fingerprint", None),
        "index_identity": getattr(runtime_manifest, "schema_version", None),
        "model_identity": getattr(runtime_encoder, "identifiers", None),
        "top_k": top_k,
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
        "production_algorithm_modified": False,
        "runtime_contract": "OperationalKISRuntime public task handlers",
        "known_current_limitations": [
            "QA closed-set support centers on COLOR, COUNT, YES_NO, and DIRECTION",
            "TRAKE baseline does not implement all design-level gap constraints",
            "OCR/ASR/Object/BM25 are not fully integrated in runtime retrieval",
        ],
        "outputs": {
            "predictions_jsonl": predictions_path.name,
            "failures_jsonl": failures_path.name,
        },
        "queries": query_summaries,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "run_summary.md").write_text(
        "\n".join(
            [
                "# L21-150 E0 Baseline Run",
                "",
                f"- Experiment: `{experiment_id}`",
                f"- Selected queries: {len(selected)}",
                f"- Executed queries: {len(query_summaries)}",
                f"- Successful: {success_count}",
                f"- Failed: {len(query_summaries) - success_count}",
                "- Production retrieval/ranking policy changed: `false`",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark = load_l21_150_benchmark(args.benchmark)
        benchmark_sha = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
        experiment_id = args.experiment_id or datetime.now(UTC).strftime(
            "l21-150-e0-%Y%m%dT%H%M%SZ"
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
        )
        runtime = OperationalKISRuntime.bootstrap(config)
        try:
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
            )
        finally:
            runtime.close(shutdown_reason="l21_150_baseline_complete")
    except (FileNotFoundError, L21150FormatError, OSError, RuntimeError, ValueError) as exc:
        print(f"L21-150 baseline run failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 baseline run complete: "
        f"success={report['successful_query_count']} failed={report['failed_query_count']}"
    )
    return 0 if report["failed_query_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
