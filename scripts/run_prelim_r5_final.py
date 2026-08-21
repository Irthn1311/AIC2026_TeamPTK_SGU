"""Run the frozen Prelim R5 final sprint from Notebook 38 on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import yaml

from triage_eg.diagnostics.sca1_siglip2_complementarity import (
    Siglip2ExactBackend,
    Siglip2GroundingPipeline,
    Siglip2OfflineEncoder,
    local_only_load_smoke,
    validate_offline_asset,
)
from triage_eg.e2eg1 import SafeCoveragePipeline
from triage_eg.external_multimodal_v3.trial_smoke import OnnxE5QueryEncoder, _e5_search
from triage_eg.fs1.io import read_jsonl, sha256
from triage_eg.fs1.runner import group_predictions
from triage_eg.fs1_v11.pipeline import build_completion_arm
from triage_eg.prelim_r5 import (
    R5Settings,
    build_deterministic_qa_rows,
    build_query_views,
    build_r5_query_candidates,
    evaluate_frozen_arms,
    finalize_pre_gt_predictions,
    fuse_asr_multiview,
    fuse_multiview_branch,
    materialize_view_queries,
    qa_evidence_from_asr,
    select_production_policy,
    write_r5_artifacts,
)
from triage_eg.prelim_r5.runner import build_candidate_comparison
from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    materialize_kaggle_expanded_tokenizer,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.inputs import resolve_stage1_root
from triage_eg.retrieval.stage1d.inputs import resolve_input_root
from triage_eg.retrieval.stage2 import OperationalRetrievalRuntime, config_from_yaml
from triage_eg.trial_p1.asr_v12_loader import ASR_EXTERNAL_V3_SOURCE_TYPE, load_asr_evidence

BENCHMARK_NAMES = {"cross": "dev_cross_60", "l21": "dev_l21_150"}
EXPECTED_B0 = {
    "cross": "801e9e4a8e33916cb0430c9c391694410972a84b212d0db949d63671be39e2dc",
    "l21": "3c4dbd2bf4766b286d1efceded120c801e59696d08ab3deb19dd38669074fd16",
}
VIEW_TOP_K = 100
E5_EXACT_REVISION = "03415a4be176a1620747c692ed433219fabc3def"


def _digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_dirs(root: Path, *, max_depth: int = 6, limit: int = 4096):
    queue = [(root, 0)]
    visited = 0
    while queue:
        current, depth = queue.pop(0)
        if not current.is_dir():
            continue
        visited += 1
        if visited > limit:
            raise RuntimeError(f"R5_INPUT_DISCOVERY_LIMIT:{root}:{limit}")
        yield current
        if depth < max_depth:
            queue.extend(
                (child, depth + 1)
                for child in sorted(current.iterdir())
                if child.is_dir()
                and not child.is_symlink()
                and child.name not in {".cache", "blobs", "snapshots"}
            )


def _mount_candidates(hint: Path, aliases: tuple[str, ...] = ()) -> list[Path]:
    slugs = {hint.name, hint.name.replace("_", "-"), hint.name.replace("-", "_"), *aliases}
    values = [hint]
    for slug in slugs:
        values.extend(
            (
                Path("/kaggle/input") / slug,
                Path("/kaggle/input/datasets/irthn1311") / slug,
                Path("/kaggle/input/datasets/nadkli") / slug,
            )
        )
    return sorted({path.resolve() for path in values if path.exists()})


def _resolve_mount(hint: Path, aliases: tuple[str, ...] = ()) -> Path:
    matches = _mount_candidates(hint, aliases)
    if len(matches) != 1:
        raise RuntimeError(f"R5_EXPECTED_ONE_MOUNT:{hint}:{matches}")
    return matches[0]


def _find_root(hint: Path, marker: str, aliases: tuple[str, ...] = ()) -> Path:
    matches = []
    for mount in _mount_candidates(hint, aliases):
        for directory in _bounded_dirs(mount):
            if (directory / marker).exists():
                matches.append(directory.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"R5_EXPECTED_ONE_ROOT:{hint}:{marker}:{matches}")
    return matches[0]


def _resolve_dataset(hint: Path) -> Path:
    matches = []
    aliases = ("dataset-aic2026", "Dataset_AIC2026")
    for mount in _mount_candidates(hint, aliases):
        for directory in _bounded_dirs(mount):
            if (directory / "map-keyframes-aic25-b1/map-keyframes").is_dir() and any(
                directory.glob("Videos_*/video")
            ):
                matches.append(directory.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"R5_RAW_DATASET_DISCOVERY:{matches}")
    return matches[0]


def _materialize_team_eval(mount: Path, work_root: Path) -> Path:
    roots = [
        directory
        for directory in _bounded_dirs(mount)
        if (directory / "benchmarks/dev_cross_60/queries.jsonl").is_file()
        and (directory / "benchmarks/dev_l21_150/queries.jsonl").is_file()
    ]
    if len(roots) == 1:
        return roots[0]
    archives = [
        directory / "aic2026_team_eval_dev_v1.zip"
        for directory in _bounded_dirs(mount)
        if (directory / "aic2026_team_eval_dev_v1.zip").is_file()
    ]
    if len(archives) != 1:
        raise RuntimeError(f"R5_TEAM_EVAL_DISCOVERY:{roots}:{archives}")
    target = work_root / "team_eval"
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[0]) as archive:
        archive.extractall(target)
    return _find_root(target, "benchmarks/dev_cross_60/queries.jsonl")


def _materialize_archive_root(
    mount: Path,
    marker: str,
    archive_names: tuple[str, ...],
    target: Path,
) -> Path:
    roots = [directory for directory in _bounded_dirs(mount) if (directory / marker).exists()]
    if len(roots) == 1:
        return roots[0]
    archives = [
        directory / name
        for directory in _bounded_dirs(mount)
        for name in archive_names
        if (directory / name).is_file()
    ]
    archives = sorted(set(archives))
    if len(archives) != 1:
        raise RuntimeError(f"R5_ARCHIVE_ROOT_DISCOVERY:{marker}:{roots}:{archives}")
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[0]) as archive:
        archive.extractall(target)
    return _find_root(target, marker)


def _runtime(
    repo: Path,
    head: str,
    name: str,
    roots: dict[str, Path],
    work_root: Path,
) -> OperationalRetrievalRuntime:
    config = config_from_yaml(
        repo / "configs/retrieval/stage2_operational_runtime_gpu.yaml",
        stage1_root=roots["stage1"],
        stage1b_root=roots["stage1b"],
        stage1e_root=roots["stage1e"],
        clip_asset_root=roots["clip"],
        translator_asset_root=roots["opus"],
        output_root=work_root / f"runtime_{name}",
        stage1d_config=repo / "configs/retrieval/stage1d_translation_ablation.yaml",
        build_git_commit=head,
    )
    return OperationalRetrievalRuntime(config).load()


def _prediction_rows(pipeline: Any, query: dict[str, Any]) -> list[dict[str, Any]]:
    request = dict(query)
    if str(request["task"]).upper() == "QA":
        request["task"] = "KIS"
        request.pop("question", None)
        request.pop("answer_type", None)
        request.pop("answer_policy", None)
    result = pipeline.predict_query(request, "G1_COVERAGE_COARSE")
    return [dict(row) for row in result.predictions]


def _translate_sources(
    queries: dict[str, list[dict[str, Any]]], runtime: OperationalRetrievalRuntime
) -> dict[str, str]:
    texts = []
    for rows in queries.values():
        for query in rows:
            values = (
                [str(row["description"]) for row in query["event_descriptions"]]
                if str(query["task"]).upper() == "TRAKE"
                else [str(query["query"])]
            )
            for value in values:
                compact = " ".join(value.split())
                if compact not in texts:
                    texts.append(compact)
    translated = runtime._load_translator().translate(texts)  # noqa: SLF001
    return {
        source: str(result["translated_text_for_clip"])
        for source, result in zip(texts, translated, strict=True)
    }


def _read_completion_evidence(
    evidence_root: Path, benchmarks: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    files = {
        "action": evidence_root / "xclip_evidence_v11.jsonl",
        "object": evidence_root / "dino_evidence_v11.jsonl",
        "ocr": evidence_root / "ocr_records_v11.jsonl",
    }
    for path in files.values():
        if not path.is_file():
            raise RuntimeError(f"R5_COMPLETION_EVIDENCE_MISSING:{path}")
    evidence = {benchmark: {name: {} for name in (*files, "asr")} for benchmark in benchmarks}
    for modality, path in files.items():
        for row in read_jsonl(path):
            benchmark = str(row.get("benchmark", ""))
            if benchmark in evidence:
                evidence[benchmark][modality].setdefault(str(row["query_id"]), []).append(row)
    manifest_path = evidence_root / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plugin_path = evidence_root / "plugin_status.json"
    plugin_status = json.loads(plugin_path.read_text(encoding="utf-8"))
    required = {"xclip": {"PASS"}, "dino": {"PASS"}, "ocr": {"PASS", "OCR_LOCAL_ONLY"}}
    failures = {
        name: plugin_status.get(name, {}).get("status")
        for name, accepted in required.items()
        if plugin_status.get(name, {}).get("status") not in accepted
    }
    if failures:
        raise RuntimeError(f"R5_EXISTING_COMPLETION_EVIDENCE_GATE:{failures}")
    return evidence, {
        "path": str(manifest_path),
        "sha256": _digest(manifest_path),
        "plugin_status_sha256": _digest(plugin_path),
        **manifest,
    }


def _existing_asr_evidence(
    queries: list[dict[str, Any]],
    bcf1: list[dict[str, Any]],
    loader: Any,
) -> dict[str, list[dict[str, Any]]]:
    grouped = group_predictions(bcf1)
    output = {}
    for query in queries:
        query_id = str(query["query_id"])
        text = " | ".join(
            str(query.get(name, "")) for name in ("query", "question") if query.get(name)
        )
        spans = loader.retrieve_spans(text, max_spans=200)
        best_by_video = {}
        for span in spans:
            best_by_video.setdefault(str(span["video_id"]), span)
        output[query_id] = [
            {
                **row,
                "source": "asr",
                "source_type": ASR_EXTERNAL_V3_SOURCE_TYPE,
                "asr_span": best_by_video[str(row["video_id"])],
            }
            for row in grouped[query_id]
            if str(row["video_id"]) in best_by_video
        ]
    return output


def _safe_r4_source(
    queries: list[dict[str, Any]],
    bcf1: list[dict[str, Any]],
    evidence: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def revision_provider(query: dict[str, Any], event: Any, _: str) -> list[dict[str, Any]]:
        candidates = []
        query_id = str(query["query_id"])
        for modality in ("action", "object", "ocr", "asr"):
            candidates.extend(
                row
                for row in evidence[modality].get(query_id, [])
                if row.get("event_index") in {None, event.event_index}
            )
        unique, seen = [], set()
        for row in sorted(
            candidates,
            key=lambda item: (
                int(item.get("rank", 100)),
                str(item.get("video_id")),
                int(item.get("frame_id", item.get("anchor_frame", 0))),
            ),
        ):
            frame_id = row.get("frame_id", row.get("anchor_frame"))
            if frame_id is None and row.get("frame_ids"):
                frames = row["frame_ids"]
                frame_id = frames[min(event.event_index, len(frames) - 1)]
            if frame_id is None:
                continue
            value = {
                **row,
                "frame_id": int(frame_id),
                "source": str(row.get("source", "revision")),
            }
            key = value["video_id"], value["frame_id"], value["source"]
            if key not in seen:
                seen.add(key)
                unique.append(value)
        if not unique:
            raise RuntimeError(
                f"R5_GRAPH_REVISION_NO_EXISTING_EVIDENCE:{query_id}:{event.event_index}"
            )
        return unique[:10]

    _, safe, diagnostics = build_completion_arm(
        "M1_v11",
        queries,
        bcf1,
        evidence,
        {"asr", "ocr", "action", "object"},
        revision_provider=revision_provider,
    )
    grouped_safe, grouped_b0 = group_predictions(safe), group_predictions(bcf1)
    output = []
    for query in queries:
        query_id = str(query["query_id"])
        rows = (
            grouped_b0[query_id] if str(query["task"]).upper() == "QA" else grouped_safe[query_id]
        )
        output.extend(rows)
    return output, diagnostics


def _views_and_visual_retrieval(
    queries: list[dict[str, Any]],
    translation: dict[str, str],
    a0_pipeline: Any,
    s1_pipeline: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result, view_rows, provenance = {}, [], []
    for query in queries:
        query_id = str(query["query_id"])
        views = build_query_views(query, translator=lambda text: translation[text])
        transformed = materialize_view_queries(query, views)
        a0_by_view, s1_by_view = {}, {}
        for view_name, view_query in transformed.items():
            a0_by_view[view_name] = _prediction_rows(a0_pipeline, view_query)
            s1_by_view[view_name] = _prediction_rows(s1_pipeline, view_query)
        a0, a0_provenance = fuse_multiview_branch(
            query, a0_by_view, branch="A0", settings=R5Settings()
        )
        s1, s1_provenance = fuse_multiview_branch(
            query, s1_by_view, branch="S1", settings=R5Settings()
        )
        result[query_id] = {"views": views, "a0": a0, "s1": s1}
        view_rows.extend(views)
        provenance.extend([*a0_provenance, *s1_provenance])
    return result, view_rows, provenance


def _asr_search_all_views(
    queries: list[dict[str, Any]],
    visual: dict[str, Any],
    loader: Any,
    encoder: OnnxE5QueryEncoder,
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    records = []
    for query in queries:
        query_id = str(query["query_id"])
        for name in (
            "ORIGINAL_VI",
            "TRANSLATED_EN",
            "ENTITY_DISTINCTIVE",
            "ACTION_OBJECT",
            "CONTEXT_ANCHORS",
        ):
            selected = [row for row in visual[query_id]["views"] if row["view"] == name]
            text = " | ".join(str(row["text"]) for row in selected)
            records.append((query_id, name, text))
    e5_rows = _e5_search(loader, [row[2] for row in records], encoder, VIEW_TOP_K)
    output: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: {"lexical": {}, "e5": {}}
    )
    for (query_id, view_name, text), e5 in zip(records, e5_rows, strict=True):
        output[query_id]["lexical"][view_name] = loader.retrieve_spans(text, max_spans=VIEW_TOP_K)
        output[query_id]["e5"][view_name] = e5
    return dict(output)


def _build_predictions(
    queries: list[dict[str, Any]],
    bcf1: list[dict[str, Any]],
    safe_r4: list[dict[str, Any]],
    visual: dict[str, Any],
    asr_search: dict[str, Any],
    ocr: dict[str, list[dict[str, Any]]],
    benchmark: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    bcf1_by_id, safe_by_id = group_predictions(bcf1), group_predictions(safe_r4)
    arms = {
        name: [] for name in ("TRUE_BCF1", "SAFE_R4_LIVE_WINNER", "SAFE_R5_QE", "SAFE_R5_GATED")
    }
    diagnostics = {"views": [], "provenance": [], "head": [], "qa": []}
    for query in queries:
        query_id = str(query["query_id"])
        values = visual[query_id]
        searches = asr_search[query_id]
        original_lexical = {name: [] for name in searches["lexical"]}
        original_e5 = {name: [] for name in searches["e5"]}
        original_lexical["ORIGINAL_VI"] = searches["lexical"]["ORIGINAL_VI"]
        original_e5["ORIGINAL_VI"] = searches["e5"]["ORIGINAL_VI"]
        live_asr = fuse_asr_multiview(
            query,
            values["views"],
            original_lexical,
            original_e5,
            a0_multiview=values["a0"],
            s1_multiview=values["s1"],
            fallback_rows=bcf1_by_id[query_id],
            safe_r4_rows=safe_by_id[query_id],
            ocr_rows=ocr.get(query_id, []),
        )
        r5_asr = fuse_asr_multiview(
            query,
            values["views"],
            searches["lexical"],
            searches["e5"],
            a0_multiview=values["a0"],
            s1_multiview=values["s1"],
            fallback_rows=bcf1_by_id[query_id],
            safe_r4_rows=safe_by_id[query_id],
            ocr_rows=ocr.get(query_id, []),
        )
        built = build_r5_query_candidates(
            query,
            bcf1=bcf1_by_id[query_id],
            safe_r4_tail_source=safe_by_id[query_id],
            a0_multiview=values["a0"],
            s1_multiview=values["s1"],
            asr_multiview=r5_asr["rows"],
            live_strong_asr=live_asr["strong"],
            r5_strong_asr=r5_asr["strong"],
            gated_candidates=r5_asr["gated_candidates"],
        )
        if str(query["task"]).upper() == "QA":
            context = [*values["a0"][:10], *values["s1"][:10], *bcf1_by_id[query_id][:10]]
            evidence_rows = qa_evidence_from_asr(r5_asr, context, ocr_rows=ocr.get(query_id, []))
            context_videos = list(dict.fromkeys(str(row["video_id"]) for row in context))
            qa_rows, qa_audit = build_deterministic_qa_rows(
                query,
                evidence_rows,
                bcf1_by_id[query_id],
                context_videos=context_videos,
            )
            built["SAFE_R5_QE"] = qa_rows
            built["SAFE_R5_GATED"] = [
                {**row, "system_variant": "SAFE_R5_GATED_DETERMINISTIC_QA"} for row in qa_rows
            ]
            diagnostics["qa"].extend({**row, "benchmark": benchmark} for row in qa_audit)
        arms["TRUE_BCF1"].extend(bcf1_by_id[query_id])
        for arm in ("SAFE_R4_LIVE_WINNER", "SAFE_R5_QE", "SAFE_R5_GATED"):
            arms[arm].extend(built[arm])
        diagnostics["views"].append(
            {
                "benchmark": benchmark,
                "query_id": query_id,
                "task": query["task"],
                "views": values["views"],
                "a0_candidate_count": len(values["a0"]),
                "s1_candidate_count": len(values["s1"]),
                "asr_candidate_count": len(r5_asr["rows"]),
                "strong_asr_best_rank": (
                    r5_asr["strong"].get("rank") if r5_asr["strong"] else None
                ),
                "gt_used": False,
            }
        )
        diagnostics["provenance"].extend(
            {**row, "benchmark": benchmark} for row in r5_asr["provenance"]
        )
        diagnostics["head"].append({**built["head_override_audit"], "benchmark": benchmark})
    return arms, diagnostics


def _structural_diagnostics(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    diagnostics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    live_videos, qe_videos = [], []
    for benchmark in ("cross", "l21"):
        live = group_predictions(predictions[benchmark]["SAFE_R4_LIVE_WINNER"])
        qe = group_predictions(predictions[benchmark]["SAFE_R5_QE"])
        for query_id in live:
            live_videos.append(len({row["video_id"] for row in live[query_id][:20]}))
            qe_videos.append(len({row["video_id"] for row in qe[query_id][:20]}))
    overrides = [row for row in diagnostics["head"] if row.get("override")]
    sane = all(
        row.get("override_rank") == 5
        and row.get("pass") is True
        and all(row.get("gates", {}).values())
        for row in overrides
    )
    return {
        "all_structural_gates_pass": all(row.get("pass", True) for row in diagnostics["head"]),
        "coverage_improved": sum(qe_videos) > sum(live_videos),
        "live_unique_videos_top20_sum": sum(live_videos),
        "qe_unique_videos_top20_sum": sum(qe_videos),
        "override_audit_sane": sane,
        "override_count": len(overrides),
    }


def run() -> dict[str, Any]:
    repo = Path(os.environ.get("AIC_REPO_DIR", "/kaggle/working/AIC2026_TeamPTK_SGU"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    output_root = Path(
        os.environ.get("AIC_R5_OUTPUT_ROOT", "/kaggle/working/prelim_r5_final_sprint")
    ).resolve()
    bundle = Path(
        os.environ.get("AIC_R5_OUTPUT_ZIP", "/kaggle/working/prelim_r5_final_sprint_bundle.zip")
    ).resolve()
    working = Path("/kaggle/working").resolve()
    if output_root.parent != working or bundle.parent != working:
        raise RuntimeError("R5_OUTPUTS_MUST_BE_DIRECT_CHILDREN_OF_KAGGLE_WORKING")
    if output_root.exists():
        shutil.rmtree(output_root)
    bundle.unlink(missing_ok=True)
    output_root.mkdir(parents=True)
    work_root = working / "prelim_r5_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir()

    defaults = {
        "raw": Path(os.environ.get("AIC_DATA_ROOT", "/kaggle/input/datasets/nadkli/dataset-aic")),
        "team": Path(
            os.environ.get(
                "AIC_TEAM_EVAL_DEV_ROOT",
                "/kaggle/input/datasets/irthn1311/aic2026_team_eval_dev_v1",
            )
        ),
        "freeze": Path(
            os.environ.get(
                "AIC_FS1_MASTER_FREEZE_ROOT",
                "/kaggle/input/datasets/irthn1311/fs1-master-preparation-freeze-2026-08-18",
            )
        ),
        "stage1": Path(
            os.environ.get(
                "AIC_STAGE1_ROOT", "/kaggle/input/datasets/irthn1311/triage-eg-stage1b-input-bundle"
            )
        ),
        "stage1b": Path(
            os.environ.get(
                "AIC_STAGE1B_ROOT",
                "/kaggle/input/datasets/irthn1311/triage-eg-stage1b-encoder-compatibility-reports",
            )
        ),
        "stage1e": Path(
            os.environ.get(
                "AIC_STAGE1E_ROOT",
                "/kaggle/input/datasets/irthn1311/triage-eg-stage1e-language-path-freeze",
            )
        ),
        "clip": Path(
            os.environ.get(
                "AIC_CLIP_ROOT", "/kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32"
            )
        ),
        "opus": Path(
            os.environ.get(
                "AIC_OPUS_ROOT", "/kaggle/input/datasets/irthn1311/aic2026-opus-mt-vi-en"
            )
        ),
        "siglip": Path(
            os.environ.get(
                "AIC_SIGLIP2_ROOT",
                "/kaggle/input/datasets/irthn1311/aic2026-siglip2-base-patch16-224",
            )
        ),
        "siglip_index": Path(
            os.environ.get(
                "AIC_SCA1_INDEX_ROOT",
                "/kaggle/input/datasets/irthn1311/triage-eg-sca1-siglip2-index-v01",
            )
        ),
        "completion": Path(
            os.environ.get(
                "AIC_COMPLETION_EVIDENCE_ROOT",
                "/kaggle/input/datasets/irthn1311/triage-eg-completion-v11-evidence-bundle",
            )
        ),
        "asr": Path(
            os.environ.get(
                "AIC_ASR_EXTERNAL_V3_ROOT",
                "/kaggle/input/datasets/irthn1311/asr-external-v3-validated-bundle",
            )
        ),
        "e5": Path(
            os.environ.get(
                "AIC_E5_ROOT",
                "/kaggle/input/datasets/irthn1311/aic2026-multilingual-e5-small-onnx-query-encoder",
            )
        ),
    }
    aliases = {
        "team": ("aic2026-team-eval-dev-v1",),
        "freeze": ("FS1_MASTER_PREPARATION_FREEZE_2026-08-18",),
        "asr": ("asr-external-v3-validated",),
    }
    mounts = {
        name: _resolve_mount(path, aliases.get(name, ()))
        for name, path in defaults.items()
        if name != "raw"
    }
    dataset_root = _resolve_dataset(defaults["raw"])
    team_root = _materialize_team_eval(mounts["team"], work_root)
    prep = _materialize_archive_root(
        mounts["freeze"],
        "FS1_PROTOCOL.md",
        ("FS1_MASTER_PREPARATION_FREEZE_2026-08-18.zip",),
        work_root / "fs1_freeze",
    )
    evidence_root = _find_root(mounts["completion"], "evidence_manifest.json")
    asr_root = _find_root(mounts["asr"], "asr_external_v3_provenance.json")
    e5_root = _find_root(mounts["e5"], "model.onnx")
    index_root = _materialize_archive_root(
        mounts["siglip_index"],
        "index/siglip2_vectors.f16.npy",
        ("triage_eg_sca1_siglip2_index_v01.zip",),
        work_root / "siglip2_index",
    )
    materialized = {
        name: work_root / name
        for name in ("stage1", "stage1b", "stage1e", "clip", "opus", "siglip")
    }
    roots: dict[str, Path] = {}
    roots["stage1"] = resolve_stage1_root(
        mounts["stage1"], search_root=None, materialize_root=materialized["stage1"]
    )
    roots["stage1b"], _ = resolve_input_root(
        mounts["stage1b"],
        required=(
            "stage1b_summary.json",
            "encoder/selected_encoder_contract.json",
            "encoder/runtime_adapter_manifest.json",
        ),
        materialize_root=materialized["stage1b"],
        search_root=None,
        archive_keyword="stage1b",
    )
    roots["stage1e"], _ = resolve_input_root(
        mounts["stage1e"],
        required=("stage1e_summary.json", "language_path_contract.json"),
        materialize_root=materialized["stage1e"],
        search_root=None,
        archive_keyword="stage1e",
    )
    roots["clip"], _ = resolve_input_root(
        mounts["clip"],
        required=("checkpoint/ViT-B-32.pt", "manifests/asset_manifest.json"),
        materialize_root=materialized["clip"],
        search_root=None,
        archive_keyword="clip",
    )
    roots["opus"], _ = resolve_input_root(
        mounts["opus"],
        required=("model/config.json", "manifests/asset_manifest.json"),
        materialize_root=materialized["opus"],
        search_root=None,
        archive_keyword="opus",
    )
    roots["siglip"], _ = resolve_input_root(
        mounts["siglip"],
        required=("model/model.safetensors", "manifests/asset_manifest.json"),
        materialize_root=materialized["siglip"],
        search_root=None,
        archive_keyword="siglip2",
    )
    resolved_inputs = {
        "raw_dataset": str(dataset_root),
        "team_eval": str(team_root),
        "fs1_freeze": str(prep),
        "stage1": str(roots["stage1"]),
        "stage1b": str(roots["stage1b"]),
        "stage1e": str(roots["stage1e"]),
        "clip": str(roots["clip"]),
        "opus": str(roots["opus"]),
        "siglip2": str(roots["siglip"]),
        "siglip2_index": str(index_root),
        "completion_evidence": str(evidence_root),
        "asr_external_v3": str(asr_root),
        "e5_query_encoder": str(e5_root),
    }
    print({"resolved_inputs": resolved_inputs, "output_zip": str(bundle)})

    config_path = repo / "configs/experiments/triage_prelim_r5_final.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["fallback"] != "PRODUCTION_SAFE_R4_LIVE_WINNER":
        raise RuntimeError("R5_CONFIG_FALLBACK_CHANGED")
    queries = {
        name: read_jsonl(team_root / f"benchmarks/{folder}/queries.jsonl")
        for name, folder in BENCHMARK_NAMES.items()
    }
    b0_paths = {
        "cross": prep / "frozen_baseline/cross_b0_bcf1_f1.jsonl",
        "l21": prep / "frozen_baseline/l21_b0_bcf1_f1.jsonl",
    }
    for name, path in b0_paths.items():
        if sha256(path) != EXPECTED_B0[name]:
            raise RuntimeError(f"R5_TRUE_BCF1_HASH_MISMATCH:{name}")
    bcf1 = {name: read_jsonl(path) for name, path in b0_paths.items()}
    asr_loader = load_asr_evidence(asr_root, ASR_EXTERNAL_V3_SOURCE_TYPE)
    completion, evidence_manifest = _read_completion_evidence(evidence_root, tuple(BENCHMARK_NAMES))
    for benchmark in BENCHMARK_NAMES:
        completion[benchmark]["asr"] = _existing_asr_evidence(
            queries[benchmark], bcf1[benchmark], asr_loader
        )
    safe_r4, graph_diagnostics = {}, []
    for benchmark in BENCHMARK_NAMES:
        safe_r4[benchmark], diagnostic = _safe_r4_source(
            queries[benchmark], bcf1[benchmark], completion[benchmark]
        )
        graph_diagnostics.extend({**row, "benchmark": benchmark} for row in diagnostic)

    clip_paths = resolve_official_asset_paths(roots["clip"])
    shared_clip_source, _ = materialize_kaggle_expanded_tokenizer(
        clip_paths.source_root, work_root / "shared_openai_clip_source"
    )
    os.environ["AIC_OPENAI_CLIP_SOURCE_ROOT"] = str(shared_clip_source)
    validate_offline_asset(roots["siglip"])
    local_only_load_smoke(roots["siglip"])
    encoder = Siglip2OfflineEncoder(
        roots["siglip"],
        device=os.environ.get("AIC_SIGLIP2_DEVICE", "auto"),
        batch_size=int(os.environ.get("AIC_SIGLIP2_BATCH_SIZE", "64")),
    ).load()
    a0_runtime = _runtime(repo, head, "a0", roots, work_root)
    s1_runtime = _runtime(repo, head, "s1", roots, work_root)
    a0_pipeline = SafeCoveragePipeline(a0_runtime, dataset_root)
    s1_pipeline = Siglip2GroundingPipeline(
        s1_runtime,
        dataset_root,
        grounding_encoder=encoder,
        grounding_backend=Siglip2ExactBackend(index_root, stage1_root=roots["stage1"]),
    )
    started = time.monotonic()
    try:
        translation = _translate_sources(queries, a0_runtime)
        visual, view_diagnostics, visual_provenance = {}, [], []
        for benchmark in BENCHMARK_NAMES:
            visual[benchmark], views, provenance = _views_and_visual_retrieval(
                queries[benchmark], translation, a0_pipeline, s1_pipeline
            )
            view_diagnostics.extend({**row, "benchmark": benchmark} for row in views)
            visual_provenance.extend({**row, "benchmark": benchmark} for row in provenance)
        e5_encoder = OnnxE5QueryEncoder(e5_root, exact_revision=E5_EXACT_REVISION)
        asr_search = {
            benchmark: _asr_search_all_views(
                queries[benchmark], visual[benchmark], asr_loader, e5_encoder
            )
            for benchmark in BENCHMARK_NAMES
        }
        predictions, diagnostics = (
            {},
            {
                "views": [],
                "provenance": list(visual_provenance),
                "head": [],
                "qa": [],
            },
        )
        for benchmark in BENCHMARK_NAMES:
            predictions[benchmark], current = _build_predictions(
                queries[benchmark],
                bcf1[benchmark],
                safe_r4[benchmark],
                visual[benchmark],
                asr_search[benchmark],
                completion[benchmark]["ocr"],
                benchmark,
            )
            for name in diagnostics:
                diagnostics[name].extend(current[name])
        structural = _structural_diagnostics(predictions, diagnostics)
        pre_gt = finalize_pre_gt_predictions(
            output_root,
            queries,
            predictions,
            config={
                "experiment": config,
                "head": head,
                "resolved_inputs": resolved_inputs,
                "view_top_k": VIEW_TOP_K,
                "qwen_enabled": False,
                "whisper_run": False,
                "corpus_preprocessing": False,
            },
        )
        ground_truth = {
            name: read_jsonl(team_root / f"benchmarks/{folder}/gt.jsonl")
            for name, folder in BENCHMARK_NAMES.items()
        }
        evaluations = evaluate_frozen_arms(queries, predictions, ground_truth, pre_gt)
        decision = select_production_policy(evaluations, structural)
        comparison = build_candidate_comparison(
            queries, predictions, ground_truth, diagnostics["provenance"]
        )
        provenance = {
            "HEAD": head,
            "elapsed_seconds": time.monotonic() - started,
            "resolved_inputs": resolved_inputs,
            "asset_hashes": {
                "experiment_config": _digest(config_path),
                "cross_true_bcf1": EXPECTED_B0["cross"],
                "l21_true_bcf1": EXPECTED_B0["l21"],
                "completion_evidence_manifest": evidence_manifest["sha256"],
                "completion_plugin_status": evidence_manifest["plugin_status_sha256"],
                "asr_provenance": _digest(asr_root / "asr_external_v3_provenance.json"),
                "siglip2_index_manifest": _digest(index_root / "index_manifest.json"),
                "clip_asset_manifest": _digest(roots["clip"] / "manifests/asset_manifest.json"),
                "opus_asset_manifest": _digest(roots["opus"] / "manifests/asset_manifest.json"),
                "siglip2_asset_manifest": _digest(
                    roots["siglip"] / "manifests/asset_manifest.json"
                ),
                "stage1_summary": _digest(roots["stage1"] / "stage1_summary.json"),
                "stage1b_summary": _digest(roots["stage1b"] / "stage1b_summary.json"),
                "stage1e_summary": _digest(roots["stage1e"] / "stage1e_summary.json"),
            },
            "structural_diagnostics": structural,
            "graph_diagnostics": graph_diagnostics,
            "e5_encoder": e5_encoder.provenance,
            "qwen_enabled": False,
            "whisper_run": False,
            "corpus_preprocessing": False,
            "model_download": False,
            "submission_uploaded": False,
            "post_gt_tuning": False,
        }
        result = write_r5_artifacts(
            output_root,
            evaluations=evaluations,
            decision=decision,
            pre_gt=pre_gt,
            comparison=comparison,
            query_view_diagnostics=[*diagnostics["views"], *diagnostics["provenance"]],
            head_override_audit=diagnostics["head"],
            qa_evidence=diagnostics["qa"],
            run_provenance=provenance,
            bundle_path=bundle,
        )
    finally:
        a0_pipeline.close()
        s1_pipeline.close()
        encoder.close()
    print(
        {
            "R5_FINAL": result,
            "scores": {
                benchmark: {
                    arm: evaluations[benchmark][arm]["summary"]["final_score"]
                    for arm in evaluations[benchmark]
                }
                for benchmark in evaluations
            },
        }
    )
    return result


if __name__ == "__main__":
    run()
