"""GT-isolated execution, evaluation, diagnostics, and packaging for E2E-1."""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import monotonic
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from aic2026_eval.io import read_jsonl, sha256_file, write_json, write_jsonl
from aic2026_eval.scoring import CUTOFFS, evaluate
from aic2026_eval.validation import validate_predictions

from .contracts import VARIANTS
from .pipeline import CanonicalTriagePipeline, PredictionResult
from .planning import plan_queries
from .qa import route_intent

BENCHMARKS = ("DEV_CROSS_60", "DEV_L21_150")
BENCHMARK_DIRECTORIES = {
    "DEV_CROSS_60": "dev_cross_60",
    "DEV_L21_150": "dev_l21_150",
}


def extract_development_bundle(archive_path: str | Path, output_root: str | Path) -> Path:
    source = Path(archive_path).expanduser().resolve(strict=True)
    target = Path(output_root).expanduser().resolve(strict=False)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with ZipFile(source) as archive:
        members = archive.namelist()
        if any("sealed" in name.casefold() for name in members):
            raise RuntimeError("SEALED_CONTENT_REJECTED")
        expected = {
            f"benchmarks/{directory}/{filename}"
            for directory in BENCHMARK_DIRECTORIES.values()
            for filename in ("queries.jsonl", "gt.jsonl", "manifest.json")
        }
        if not expected <= set(members):
            raise RuntimeError(
                f"TEAM_EVAL_DEVELOPMENT_BUNDLE_INCOMPLETE: {sorted(expected - set(members))}"
            )
        for info in archive.infolist():
            destination = (target / info.filename).resolve()
            if target not in destination.parents and destination != target:
                raise RuntimeError("TEAM_EVAL_ARCHIVE_PATH_TRAVERSAL")
            archive.extract(info, target)
    return target


def materialize_inference_only(benchmark_root: str | Path, output_root: str | Path) -> Path:
    source = Path(benchmark_root).resolve(strict=True) / "queries.jsonl"
    target = Path(output_root).resolve(strict=False)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(source, target / "queries.jsonl")
    if list(target.iterdir()) != [target / "queries.jsonl"]:
        raise RuntimeError("INFERENCE_ONLY_DIRECTORY_VIOLATION")
    return target


def _collect_results(
    results: list[PredictionResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = [row for result in results for row in result.predictions]
    diagnostics = [row for result in results for row in result.diagnostics]
    return predictions, diagnostics


def _diagnostic_target(row: dict[str, Any]) -> str:
    if "intent" in row:
        return "qa_answer_diagnostics.jsonl"
    if "t3_score" in row:
        return "trake_chain_diagnostics.jsonl"
    if "refined_frame_idx" in row:
        return "m1_refinement_diagnostics.jsonl"
    return "candidate_provenance.jsonl"


def run_prediction_variant(
    pipeline: CanonicalTriagePipeline,
    inference_root: str | Path,
    benchmark_id: str,
    variant: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Generate one immutable variant from a directory that physically contains no GT."""

    if benchmark_id not in BENCHMARKS:
        raise ValueError(f"unknown development benchmark: {benchmark_id}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown E2E-1 variant: {variant}")
    inference = Path(inference_root).resolve(strict=True)
    if {path.name for path in inference.iterdir()} != {"queries.jsonl"}:
        raise RuntimeError("GT_UNAVAILABLE_DURING_PREDICTION_GATE_FAILED")
    queries = read_jsonl(inference / "queries.jsonl")
    plans = plan_queries(queries)
    root = Path(output_root).resolve(strict=False)
    query_plan_path = root / "query_plans" / f"{benchmark_id.casefold()}_query_plans.jsonl"
    write_jsonl(query_plan_path, (plan.as_dict() for plan in plans))
    started = monotonic()
    results = pipeline.predict_queries(queries, variant)
    predictions, diagnostics = _collect_results(results)
    variant_slug = "p0_coarse" if variant == "P0_COARSE" else "p1_canonical"
    prediction_path = (
        root / "predictions" / f"{benchmark_id.casefold()}_{variant_slug}_predictions.jsonl"
    )
    write_jsonl(prediction_path, predictions)
    digest = sha256_file(prediction_path)
    validation, issues = validate_predictions(queries, predictions)
    if validation["status"] != "PASS":
        write_json(
            root / "diagnostics" / f"{benchmark_id.casefold()}_{variant_slug}_validation.json",
            {"summary": validation, "issues": issues},
        )
        raise RuntimeError(f"PREDICTION_CONTRACT_GATE_FAILED: {benchmark_id}/{variant}")
    diagnostic_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        diagnostic_groups[_diagnostic_target(row)].append({"benchmark_id": benchmark_id, **row})
    for filename, new_rows in diagnostic_groups.items():
        target = root / "diagnostics" / filename
        existing = read_jsonl(target) if target.is_file() else []
        write_jsonl(target, [*existing, *new_rows])
    return {
        "queries": queries,
        "plans": plans,
        "variant": variant,
        "results": results,
        "predictions": predictions,
        "prediction_path": prediction_path,
        "sha256": digest,
        "validation": validation,
        "seconds": monotonic() - started,
    }


def combine_prediction_variants(*runs: dict[str, Any]) -> dict[str, Any]:
    if {run["variant"] for run in runs} != set(VARIANTS):
        raise ValueError("exactly P0_COARSE and P1_CANONICAL are required")
    first = runs[0]
    if any(run["queries"] != first["queries"] for run in runs[1:]):
        raise ValueError("prediction variants used different query inputs")
    return {
        "queries": first["queries"],
        "plans": first["plans"],
        "variants": {
            run["variant"]: {
                key: value
                for key, value in run.items()
                if key not in {"queries", "plans", "variant"}
            }
            for run in runs
        },
    }


def run_predictions_only(
    pipeline: CanonicalTriagePipeline,
    inference_root: str | Path,
    benchmark_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper that preserves the required P0-then-P1 order."""

    return combine_prediction_variants(
        *(
            run_prediction_variant(pipeline, inference_root, benchmark_id, variant, output_root)
            for variant in VARIANTS
        )
    )


def _task_diagnostics(
    per_query: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    query_map = {row["query_id"]: row for row in queries}
    output: dict[str, Any] = {}
    for task in ("KIS", "QA", "TRAKE"):
        members = [row for row in per_query if row["task"] == task]
        task_output: dict[str, Any] = {"query_count": len(members), "cutoffs": {}}
        for cutoff in CUTOFFS:
            values = []
            position_hits: defaultdict[int, list[float]] = defaultdict(list)
            for row in members:
                scored = [item for item in row["prediction_diagnostics"] if item["rank"] <= cutoff]
                diagnostics = [item["diagnostics"] for item in scored]
                base = {
                    "correct_video": float(
                        any(item.get("video_correct", False) for item in diagnostics)
                    )
                }
                if task == "KIS":
                    base["grounded_interval"] = float(
                        any(item.get("grounding_correct", False) for item in diagnostics)
                    )
                elif task == "QA":
                    grounded = [
                        item for item in diagnostics if item.get("grounding_correct", False)
                    ]
                    base.update(
                        {
                            "grounding": float(bool(grounded)),
                            "answer_correct_conditional_on_grounding": float(
                                any(item.get("answer_alias_correct", False) for item in grounded)
                            ),
                            "full_tuple": float(
                                any(item.get("full_tuple_correct", False) for item in diagnostics)
                            ),
                        }
                    )
                else:
                    event_count = int(query_map[row["query_id"]]["event_count"])
                    fractions = [item.get("events_hit", 0) / event_count for item in diagnostics]
                    base.update(
                        {
                            "mean_event_fraction_hit": max(fractions, default=0.0),
                            "full_chain": float(
                                any(item.get("full_chain_correct", False) for item in diagnostics)
                            ),
                        }
                    )
                    for index in range(event_count):
                        position_hits[index].append(
                            float(
                                any(
                                    len(item.get("per_event_hit", [])) > index
                                    and item["per_event_hit"][index]
                                    for item in diagnostics
                                )
                            )
                        )
                values.append(base)
            keys = sorted({key for value in values for key in value})
            aggregate = {
                key: mean(value[key] for value in values) if values else 0.0 for key in keys
            }
            if position_hits:
                aggregate["per_event_position_hit"] = {
                    f"E{index + 1}": mean(items) for index, items in sorted(position_hits.items())
                }
            task_output["cutoffs"][str(cutoff)] = aggregate
        output[task] = task_output
    qa_intents: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        if row["task"] == "QA":
            intent = route_intent(str(query_map[row["query_id"]].get("question", "")))
            qa_intents[intent].append(row)
    output["QA_BY_INTENT"] = {
        intent: {
            "query_count": len(rows),
            **{f"R@{cutoff}": mean(row[f"R@{cutoff}"] for row in rows) for cutoff in (1, 5, 20)},
        }
        for intent, rows in sorted(qa_intents.items())
    }
    return output


def evaluate_finalized(
    prediction_run: dict[str, Any],
    benchmark_root: str | Path,
    benchmark_id: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Load GT only after both immutable prediction hashes exist, then evaluate."""

    variants = prediction_run["variants"]
    if set(variants) != set(VARIANTS) or any(
        not value.get("sha256") for value in variants.values()
    ):
        raise RuntimeError("PREDICTIONS_NOT_FINALIZED_BEFORE_GT")
    ground_truth = read_jsonl(Path(benchmark_root).resolve(strict=True) / "gt.jsonl")
    queries = prediction_run["queries"]
    root = Path(output_root).resolve(strict=False)
    output = {}
    for variant in VARIANTS:
        value = variants[variant]
        summary, per_query, slices, issues = evaluate(
            queries,
            value["predictions"],
            ground_truth,
            metadata={"benchmark_id": benchmark_id, "system_variant": variant},
        )
        slug = "p0" if variant == "P0_COARSE" else "p1"
        prefix = root / "evaluation" / f"{benchmark_id.casefold()}_{slug}"
        write_json(Path(f"{prefix}_summary.json"), summary)
        write_jsonl(Path(f"{prefix}_per_query.jsonl"), per_query)
        write_json(Path(f"{prefix}_slices.json"), slices)
        write_jsonl(Path(f"{prefix}_issues.jsonl"), issues)
        output[variant] = {
            "summary": summary,
            "per_query": per_query,
            "slices": slices,
            "task_diagnostics": _task_diagnostics(per_query, queries),
        }
    write_json(
        root / "diagnostics" / f"{benchmark_id.casefold()}_task_diagnostics.json",
        {variant: value["task_diagnostics"] for variant, value in output.items()},
    )
    return output


def compare_variants(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for benchmark_id, values in evaluations.items():
        p0, p1 = values["P0_COARSE"], values["P1_CANONICAL"]
        by0 = {row["query_id"]: row for row in p0["per_query"]}
        by1 = {row["query_id"]: row for row in p1["per_query"]}
        paired = []
        counts: Counter[str] = Counter()
        for query_id in sorted(by0):
            delta = by1[query_id]["final_score"] - by0[query_id]["final_score"]
            result = "P1_BETTER" if delta > 0 else "P1_WORSE" if delta < 0 else "TIE"
            counts[result] += 1
            paired.append({"query_id": query_id, "delta": delta, "result": result})
        task_delta = {}
        for task in ("KIS", "QA", "TRAKE"):
            key = f"task:{task}"
            task_delta[task] = p1["slices"][key]["final_score"] - p0["slices"][key]["final_score"]
        output[benchmark_id] = {
            "overall_delta": p1["summary"]["final_score"] - p0["summary"]["final_score"],
            "task_delta": task_delta,
            "paired_counts": dict(counts),
            "paired_queries": paired,
        }
    return output


def runtime_summary(
    prediction_runs: dict[str, dict[str, Any]], pipeline: CanonicalTriagePipeline, startup: float
) -> dict[str, Any]:
    tasks: defaultdict[str, list[float]] = defaultdict(list)
    benchmarks = {}
    for benchmark_id, run in prediction_runs.items():
        benchmarks[benchmark_id] = {}
        for variant, value in run["variants"].items():
            benchmarks[benchmark_id][variant] = value["seconds"]
            for result in value["results"]:
                tasks[result.query_plan["task"]].append(result.latency_seconds)
    return {
        "startup_seconds": startup,
        "benchmark_variant_seconds": benchmarks,
        "per_query_seconds": {
            task: {
                "count": len(values),
                "mean": mean(values),
                "median": median(values),
                "p95": float(np.percentile(values, 95)),
            }
            for task, values in sorted(tasks.items())
        },
        "devices": pipeline.runtime.runtime_manifest().get("devices", {}),
        **pipeline.runtime_diagnostics(),
    }


def failure_taxonomy(
    evaluations: dict[str, dict[str, Any]], comparison: dict[str, Any]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    records = []
    for benchmark_id, values in evaluations.items():
        p1 = values["P1_CANONICAL"]
        paired = {row["query_id"]: row for row in comparison[benchmark_id]["paired_queries"]}
        for row in p1["per_query"]:
            first = next(iter(row["prediction_diagnostics"]), {}).get("diagnostics", {})
            code = None
            if not row["prediction_diagnostics"]:
                code = "NO_PREDICTIONS"
            elif not first.get("video_correct", False):
                code = "TRAKE_WRONG_VIDEO" if row["task"] == "TRAKE" else "WRONG_VIDEO"
            elif row["task"] in {"KIS", "QA"} and not first.get("grounding_correct", False):
                code = "CORRECT_VIDEO_WRONG_FRAME"
            elif row["task"] == "QA" and not first.get("full_tuple_correct", False):
                code = "QA_GROUNDING_CORRECT_ANSWER_WRONG"
            elif row["task"] == "TRAKE" and not first.get("full_chain_correct", False):
                code = "TRAKE_PARTIAL_CHAIN"
            if paired[row["query_id"]]["result"] == "P1_WORSE":
                code = "M1_REGRESSION"
            if code:
                counts[code] += 1
                records.append(
                    {"benchmark_id": benchmark_id, "query_id": row["query_id"], "code": code}
                )
    return {"counts": dict(counts), "records": records}


def render_cross_review(
    pipeline: CanonicalTriagePipeline,
    prediction_run: dict[str, Any],
    evaluation: dict[str, Any],
    output_root: str | Path,
) -> list[Path]:
    """Render at most six compact P1 top-1 sheets per task after GT evaluation."""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return []
    root = Path(output_root)
    query_map = {row["query_id"]: row for row in prediction_run["queries"]}
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_run["variants"]["P1_CANONICAL"]["predictions"]:
        predictions[row["query_id"]].append(row)
    rendered = []
    per_query = sorted(
        evaluation["P1_CANONICAL"]["per_query"],
        key=lambda row: (row["final_score"], row["query_id"]),
    )
    for task in ("KIS", "QA", "TRAKE"):
        members = [row for row in per_query if row["task"] == task][:6]
        task_images = []
        for scored in members:
            query_id = scored["query_id"]
            candidates = sorted(predictions.get(query_id, []), key=lambda row: row["rank"])
            if not candidates:
                continue
            top = candidates[0]
            frame_id = top.get("frame_id", top.get("frame_ids", [0])[0])
            try:
                array = pipeline._decode_image(top["video_id"], int(frame_id))
            except (IndexError, OSError, RuntimeError, ValueError):
                continue
            image = Image.fromarray(array).convert("RGB")
            image.thumbnail((720, 405))
            canvas = Image.new("RGB", (760, image.height + 120), "white")
            canvas.paste(image, ((760 - image.width) // 2, 70))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), f"{query_id} | {task} | P1 rank 1", fill="black")
            draw.text((12, 30), f"{top['video_id']} frame={frame_id}", fill="black")
            draw.text((12, 50), str(query_map[query_id]["query"])[:105], fill="black")
            draw.text((12, image.height + 82), "GT overlay: post-inference review only", fill="red")
            path = root / "review" / f"{task.casefold()}_{query_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path, quality=88)
            rendered.append(path)
            task_images.append(canvas)
        if task_images:
            width = max(image.width for image in task_images)
            height = sum(image.height for image in task_images)
            montage = Image.new("RGB", (width, height), "white")
            y = 0
            for image in task_images:
                montage.paste(image, (0, y))
                y += image.height
            path = root / "montages" / f"dev_cross_60_{task.casefold()}_review.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            montage.save(path, quality=85)
            rendered.append(path)
    return rendered


def write_manifests(
    output_root: str | Path,
    *,
    pipeline: CanonicalTriagePipeline,
    raw_dataset_root: str | Path,
    team_eval_bundle: str | Path,
    branch: str,
    git_commit: str,
    prediction_runs: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    root = Path(output_root)
    runtime_manifest = pipeline.runtime.runtime_manifest()
    stage1b = runtime_manifest.get("stage1b", {})
    translator = runtime_manifest.get("translator", {})
    settings = pipeline.settings.as_dict()
    write_json(root / "config_snapshot.json", settings)
    write_json(root / "resource_manifest.json", runtime_manifest)
    write_json(
        root / "prediction_hashes.json",
        {
            benchmark: {variant: value["sha256"] for variant, value in run["variants"].items()}
            for benchmark, run in prediction_runs.items()
        },
    )
    write_json(root / "diagnostics" / "runtime_summary.json", runtime)
    write_json(
        root / "run_manifest.json",
        {
            "experiment": "TRIAGE_E2E1",
            "experiment_version": "0.1",
            "branch": branch,
            "git_commit": git_commit,
            "created_at": datetime.now(UTC).isoformat(),
            "raw_dataset_root": str(Path(raw_dataset_root).resolve()),
            "team_eval_bundle_path": str(Path(team_eval_bundle).resolve()),
            "team_eval_bundle_sha256": sha256_file(team_eval_bundle),
            "benchmark_order": list(BENCHMARKS),
            "stage1_exact": True,
            "language_router": "FROZEN_STAGE2A_EN_DIRECT_VI_OPUS_MT",
            "translator_model": translator.get("model_id"),
            "clip_model": stage1b.get("candidate_id"),
            "clip_checkpoint_sha256": stage1b.get("checkpoint_sha256"),
            "clip_device": runtime_manifest.get("devices", {}).get("clip"),
            "stage1_device": "cpu_numpy_exact",
            "m1_enabled": True,
            "t3_enabled": True,
            "m2_enabled": False,
            "m3_enabled": False,
            "event_graph_enabled": False,
            "vlm_enabled": False,
            "agent_enabled": False,
            "nvdec_default": False,
            "qa_solver": "MINIMAL_NON_VLM_BASELINE",
            "ocr_status": pipeline.ocr.status,
            "gt_available_to_inference": False,
            "sealed_content_accessed": False,
            "human_reviewed_runtime": False,
        },
    )
    (root / "README.md").write_text(
        "# TRIAGE-EG E2E-1\n\n"
        "Canonical P1 integrates frozen Stage2A exact retrieval, T3, M1, and a bounded "
        "non-VLM QA baseline. P0 is the matched no-M1 structural control. Scores are "
        "reported separately for DEV_CROSS_60 and DEV_L21_150. Prediction files were "
        "finalized and hashed before ground truth was loaded. M2, M3, Event Graph, VLM, "
        "Agent, and NVDEC default are disabled.\n",
        encoding="utf-8",
    )


def create_e2e1_bundle(output_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(output_root).resolve(strict=True)
    target = Path(zip_path).resolve(strict=False)
    forbidden = ("sealed", "notebook20", "m3_bundle")
    forbidden_suffixes = (".mp4", ".npy", ".npz", ".pt", ".pth", ".bin")
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(root).as_posix().casefold()
        if any(value in relative for value in forbidden) or relative.endswith(forbidden_suffixes):
            raise RuntimeError(f"FORBIDDEN_E2E_BUNDLE_MEMBER: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    return target


def formal_report_lines(
    *,
    git_commit: str,
    evaluations: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    runtime: dict[str, Any],
    ocr_status: str,
    zip_path: str | Path,
) -> list[str]:
    cross = evaluations["DEV_CROSS_60"]
    l21 = evaluations.get("DEV_L21_150")
    p0, p1 = cross["P0_COARSE"], cross["P1_CANONICAL"]
    lines = [
        f"HEAD={git_commit}",
        "TRIAGE_E2E1_IMPLEMENTATION=COMPLETE",
        "CANONICAL_PIPELINE=READY",
        "GT_LEAKAGE_GATE=PASS",
        "SEALED_ACCESS_GATE=PASS",
        "PREDICTION_CONTRACT_GATE=PASS",
        "STAGE1_EXACT=UNCHANGED",
        "T3=ENABLED_FROZEN",
        "M1=ENABLED_FROZEN",
        "M3=DISABLED",
        "EVENT_GRAPH=DISABLED",
        "VLM=DISABLED",
        "QA_SOLVER=MINIMAL_NON_VLM_BASELINE",
        f"OCR_STATUS={ocr_status}",
    ]
    for label, value in (("CROSS_P0", p0), ("CROSS_P1", p1)):
        summary = value["summary"]
        lines.extend(
            [
                f"{label}_QUERY_COUNT={summary['query_count']}",
                *[f"{label}_R{cutoff}={summary[f'R@{cutoff}']}" for cutoff in CUTOFFS],
                f"{label}_FINAL={summary['final_score']}",
                *[
                    f"{label}_{task}={value['slices'][f'task:{task}']['final_score']}"
                    for task in ("KIS", "QA", "TRAKE")
                ],
            ]
        )
    qa1 = p1["task_diagnostics"]["QA"]["cutoffs"]["1"]
    tr1 = p1["task_diagnostics"]["TRAKE"]["cutoffs"]["1"]
    paired = comparison["DEV_CROSS_60"]["paired_counts"]
    lines.extend(
        [
            f"CROSS_QA_GROUNDING_R1={qa1['grounding']}",
            f"CROSS_QA_FULL_R1={qa1['full_tuple']}",
            f"CROSS_TRAKE_VIDEO_R1={tr1['correct_video']}",
            f"CROSS_TRAKE_MEAN_EVENT_HIT_R1={tr1['mean_event_fraction_hit']}",
            f"CROSS_TRAKE_FULL_CHAIN_R1={tr1['full_chain']}",
            f"P1_VS_P0_BETTER={paired.get('P1_BETTER', 0)}",
            f"P1_VS_P0_TIE={paired.get('TIE', 0)}",
            f"P1_VS_P0_WORSE={paired.get('P1_WORSE', 0)}",
        ]
    )
    if l21:
        value = l21["P1_CANONICAL"]
        lines.extend(
            [
                "L21_RUN_STATUS=COMPLETE",
                f"L21_P1_QUERY_COUNT={value['summary']['query_count']}",
                *[f"L21_P1_R{cutoff}={value['summary'][f'R@{cutoff}']}" for cutoff in CUTOFFS],
                f"L21_P1_FINAL={value['summary']['final_score']}",
                *[
                    f"L21_P1_{task}={value['slices'][f'task:{task}']['final_score']}"
                    for task in ("KIS", "QA", "TRAKE")
                ],
            ]
        )
    else:
        lines.append("L21_RUN_STATUS=SKIPPED")
    per_task = runtime["per_query_seconds"]
    lines.extend(
        [
            f"STARTUP_SECONDS={runtime['startup_seconds']}",
            f"CROSS_TOTAL_SECONDS={sum(runtime['benchmark_variant_seconds']['DEV_CROSS_60'].values())}",
            f"KIS_MEDIAN_SECONDS={per_task['KIS']['median']}",
            f"QA_MEDIAN_SECONDS={per_task['QA']['median']}",
            f"TRAKE_MEDIAN_SECONDS={per_task['TRAKE']['median']}",
            f"M1_CACHE_HITS={runtime['m1_cache_hits']}",
            f"QA_FRAME_CACHE_HITS={runtime['qa_frame_embedding_cache_hits']}",
            "TEAM_EVAL_SCORES_REPORTED_SEPARATELY=YES",
            "CANONICAL_E2E_BASELINE=READY",
            f"OUTPUT_ZIP={Path(zip_path)}",
            "NEXT_ARCHITECTURE_CHANGE=NONE_IN_THIS_SPRINT",
            "RETURN_FOR_INDEPENDENT_REVIEW=YES",
        ]
    )
    return lines


__all__ = [
    "BENCHMARKS",
    "combine_prediction_variants",
    "create_e2e1_bundle",
    "compare_variants",
    "evaluate_finalized",
    "extract_development_bundle",
    "failure_taxonomy",
    "formal_report_lines",
    "materialize_inference_only",
    "render_cross_review",
    "run_predictions_only",
    "run_prediction_variant",
    "runtime_summary",
    "write_manifests",
]
