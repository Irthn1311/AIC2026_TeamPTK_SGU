"""Freeze Trial visual BCF-1 and audit repaired QA without opening ground truth."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from aic2026_eval.io import sha256_file, write_json, write_jsonl
from triage_eg.diagnostics.bcf1_protected_late_fusion import candidate_key
from triage_eg.fs1_v11.contracts import QWEN_ID, QWEN_REVISION, WHISPER_ID, WHISPER_REVISION

from .qa_evidence import BoundedEvidencePackage, assess_answer_evidence

EXPECTED_SIGLIP2_INDEX_FINGERPRINT = (
    "59302ddb5fb8c4aaacc9d6945dd5e1f7c32705286f06fb0fd56493e177eaaaa3"
)
VISUAL_LABEL = "BCF1_VISUAL_GROUNDING_REPAIRED_QA"
HISTORICAL_LABEL = "BCF1_FROZEN_EXACT"


def _jsonl(archive: ZipFile, name: str) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in archive.read(name).decode("utf-8").splitlines() if line.strip()
    ]


def _canonical_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _member_hash(archive: ZipFile, name: str) -> str:
    return hashlib.sha256(archive.read(name)).hexdigest()


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    return {
        query_id: sorted(values, key=lambda value: int(value["rank"]))
        for query_id, values in grouped.items()
    }


def _freeze_visual(
    source_bundle: Path,
    plans: list[dict[str, Any]],
    a0: list[dict[str, Any]],
    s1: list[dict[str, Any]],
    f1: list[dict[str, Any]],
    provenance: dict[str, Any],
    member_hashes: dict[str, str],
) -> dict[str, Any]:
    visual_plans = [plan for plan in plans if plan["task"] in {"KIS", "TRAKE"}]
    counts = Counter(str(plan["task"]) for plan in visual_plans)
    if counts != {"KIS": 18, "TRAKE": 3}:
        raise RuntimeError(f"TRIAL_VISUAL_QUERY_COUNT_GATE_FAILED:{dict(counts)}")
    grouped = {arm: _group(rows) for arm, rows in (("A0", a0), ("S1", s1), ("F1", f1))}
    expected_ids = {str(plan["query_id"]) for plan in visual_plans}
    for arm, values in grouped.items():
        if any(len(values.get(query_id, [])) != 100 for query_id in expected_ids):
            raise RuntimeError(f"TRIAL_VISUAL_TOP100_GATE_FAILED:{arm}")
    preservation: dict[str, bool] = {}
    trake_structure: dict[str, bool] = {}
    for plan in visual_plans:
        query_id, task = str(plan["query_id"]), str(plan["task"])
        preservation[query_id] = [
            candidate_key(task, row) for row in grouped["A0"][query_id][:5]
        ] == [candidate_key(task, row) for row in grouped["F1"][query_id][:5]]
        if task == "TRAKE":
            event_count = len(plan.get("events", []))
            trake_structure[query_id] = all(
                len(row.get("frame_ids", [])) == event_count
                and all(
                    left < right
                    for left, right in zip(row["frame_ids"], row["frame_ids"][1:], strict=False)
                )
                for arm in grouped.values()
                for row in arm[query_id]
            )
    if not all(preservation.values()):
        raise RuntimeError("TRIAL_VISUAL_A0_TOP5_PRESERVATION_FAILED")
    if not all(trake_structure.values()):
        raise RuntimeError("TRIAL_VISUAL_TRAKE_STRUCTURE_FAILED")
    index_validation = provenance.get("siglip2_index_validation", {})
    manifest = index_validation.get("manifest", {})
    if (
        index_validation.get("status") != "PASS"
        or manifest.get("index_fingerprint") != EXPECTED_SIGLIP2_INDEX_FINGERPRINT
        or index_validation.get("index_rebuilt") is not False
    ):
        raise RuntimeError("TRIAL_VISUAL_SIGLIP2_INDEX_FREEZE_GATE_FAILED")
    if provenance.get("GT_OPENED") is not False:
        raise RuntimeError("TRIAL_VISUAL_GT_ISOLATION_GATE_FAILED")
    visual_ids = {str(plan["query_id"]) for plan in visual_plans}
    visual_rows = {
        arm: [row for row in rows if str(row["query_id"]) in visual_ids]
        for arm, rows in (("A0", a0), ("S1", s1), ("F1", f1))
    }
    return {
        "status": "PASS",
        "label": VISUAL_LABEL,
        "historical_exact_label": HISTORICAL_LABEL,
        "historical_exactness_status": "NOT_PROVEN_FOR_CURRENT_REPAIRED_QA",
        "historical_exactness_reason": (
            "Frozen Cross/L21 A0/S1/F1 prediction hashes were not reproduced by this "
            "GT-free post-run audit. The EXACT label is therefore withheld."
        ),
        "source_bundle": str(source_bundle),
        "source_bundle_sha256": sha256_file(source_bundle),
        "source_head": provenance.get("HEAD"),
        "query_counts": dict(counts),
        "rows_per_arm": {arm: len(rows) for arm, rows in visual_rows.items()},
        "member_sha256": member_hashes,
        "visual_subset_canonical_sha256": {
            arm: _canonical_hash(rows) for arm, rows in visual_rows.items()
        },
        "a0_top5_preserved_query_count": sum(preservation.values()),
        "a0_top5_preserved_all_21": all(preservation.values()),
        "trake_structure": trake_structure,
        "siglip2_index_fingerprint": manifest["index_fingerprint"],
        "siglip2_index_rebuilt": False,
        "gt_opened": False,
        "asr_v12_touched": False,
        "production_policy_changed": False,
    }


def _audit_qa(
    plans: list[dict[str, Any]],
    f1: list[dict[str, Any]],
    fusion_provenance: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qa_plans = {str(plan["query_id"]): plan for plan in plans if plan["task"] == "QA"}
    qa_rows = _group([row for row in f1 if str(row["query_id"]) in qa_plans])
    provenance_by_key = {
        (str(row["query_id"]), int(row["fused_rank"])): row for row in fusion_provenance
    }
    diagnostics: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for query_id, plan in qa_plans.items():
        answer_type = str(plan["answer_type"])
        for row in qa_rows.get(query_id, []):
            rank = int(row["rank"])
            fusion = provenance_by_key.get((query_id, rank), {})
            grounding_sources = tuple(
                source
                for source in (
                    "A0_OPENAI_CLIP" if fusion.get("a0_rank") else None,
                    "S1_SIGLIP2" if fusion.get("s1_rank") else None,
                )
                if source is not None
            )
            package = BoundedEvidencePackage(
                video_id=str(row["video_id"]),
                frame_id=int(row["frame_id"]),
                grounding_rank=rank,
                grounding_sources=grounding_sources,
            )
            assessment = assess_answer_evidence(str(row.get("answer", "")), answer_type, package)
            diagnostics.append(
                {
                    "query_id": query_id,
                    "rank": rank,
                    "video_id": row["video_id"],
                    "frame_id": row["frame_id"],
                    **assessment.as_dict(),
                    "fusion_provenance": {
                        "source": fusion.get("source"),
                        "a0_rank": fusion.get("a0_rank"),
                        "s1_rank": fusion.get("s1_rank"),
                    },
                    "evidence_auditability": "NOT_PERSISTED_IN_SOURCE_TRIAL_ARTIFACT",
                }
            )
        selected = [row for row in diagnostics if row["query_id"] == query_id]
        sufficient_count = sum(row["evidence_sufficient"] for row in selected)
        summaries[query_id] = {
            "answer_type": answer_type,
            "answer_policy": plan["answer_policy"],
            "top100_tuple_count": len(selected),
            "syntax_pass_count": sum(row["syntax_pass"] for row in selected),
            "evidence_sufficient_count": sufficient_count,
            "ocr_supported_count": sum(
                "OCR_CONTEXTUAL_PHRASE" in row["evidence_sources"] for row in selected
            ),
            "asr_supported_count": sum(
                "ASR_LOCAL_SPAN" in row["evidence_sources"] for row in selected
            ),
            "qwen_supported_count": sum(
                "QWEN_BOUNDED_EVIDENCE_VERIFICATION" in row["evidence_sources"] for row in selected
            ),
            "readiness": (
                "QA_READY_FOR_SERIOUS_SUBMISSION"
                if sufficient_count >= 5
                else "QA_NOT_READY_FOR_SERIOUS_SUBMISSION"
            ),
        }
    if set(summaries) != {"query-p1-15-qa", "query-p1-19-qa", "query-p1-22-qa"}:
        raise RuntimeError(f"TRIAL_QA_QUERY_SET_GATE_FAILED:{sorted(summaries)}")
    return diagnostics, {
        "status": "QA_PRE_ASR_NOT_READY",
        "qa_syntax_sanity_gate": (
            "PASS_WITH_PER_TUPLE_RESULTS" if diagnostics else "FAIL_NO_DIAGNOSTICS"
        ),
        "qa_evidence_sufficiency_gate": "FAIL_NOT_ENOUGH_VERIFIED_EVIDENCE",
        "queries": summaries,
        "ready_query_count": sum(
            row["readiness"] == "QA_READY_FOR_SERIOUS_SUBMISSION" for row in summaries.values()
        ),
        "not_ready_query_count": sum(
            row["readiness"] == "QA_NOT_READY_FOR_SERIOUS_SUBMISSION" for row in summaries.values()
        ),
        "kis_trake_packaging_blocked": False,
        "gt_opened": False,
        "asr_v12_touched": False,
        "asr_v12_mount_readiness": {
            "status": "CODE_AND_SYNTHETIC_TESTS_READY_BUNDLE_NOT_MOUNTED",
            "required_files": [
                "asr_transcripts_v12.jsonl",
                "asr_lexical_index_v12.json",
                "asr_audio_inventory_v12.jsonl",
                "asr_performance_report_v12.json",
                "*_manifest.json",
            ],
            "required_model_id": WHISPER_ID,
            "required_model_revision": WHISPER_REVISION,
            "frame_mapping": "INJECTED_CANONICAL_FRAMEMAP_ONLY",
        },
        "qwen_bounded_executor_readiness": {
            "status": "CODE_AND_SYNTHETIC_TESTS_READY_ASSET_NOT_EXECUTED",
            "required_model_id": QWEN_ID,
            "required_model_revision": QWEN_REVISION,
        },
    }


def prepare_post_bcf1_artifacts(
    source_bundle: str | Path, output_root: str | Path
) -> dict[str, Any]:
    source = Path(source_bundle).resolve(strict=True)
    root = Path(output_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    member_names = {
        "plans": "trial_p1_query_plans_v2.jsonl",
        "a0": "trial_p1_A0_predictions.jsonl",
        "s1": "trial_p1_S1_predictions.jsonl",
        "f1": "trial_p1_BCF1_F1_predictions.jsonl",
        "fusion": "trial_p1_BCF1_F1_provenance.jsonl",
        "run": "run_provenance.json",
    }
    with ZipFile(source) as archive:
        missing = [name for name in member_names.values() if name not in archive.namelist()]
        if missing:
            raise RuntimeError(f"TRIAL_POST_BCF1_MEMBERS_MISSING:{missing}")
        plans = _jsonl(archive, member_names["plans"])
        a0, s1, f1 = (_jsonl(archive, member_names[key]) for key in ("a0", "s1", "f1"))
        fusion = _jsonl(archive, member_names["fusion"])
        provenance = json.loads(archive.read(member_names["run"]))
        hashes = {key: _member_hash(archive, value) for key, value in member_names.items()}
    freeze = _freeze_visual(source, plans, a0, s1, f1, provenance, hashes)
    diagnostics, readiness = _audit_qa(plans, f1, fusion)
    visual_ids = {str(plan["query_id"]) for plan in plans if plan["task"] in {"KIS", "TRAKE"}}
    frozen_paths: dict[str, Path] = {}
    for arm, rows in (("A0", a0), ("S1", s1), ("F1", f1)):
        path = root / f"trial_p1_visual_{arm}_predictions.jsonl"
        write_jsonl(path, (row for row in rows if str(row["query_id"]) in visual_ids))
        frozen_paths[arm] = path
    freeze["frozen_artifacts"] = {
        arm: {"filename": path.name, "sha256": sha256_file(path)}
        for arm, path in frozen_paths.items()
    }
    write_json(root / "trial_p1_visual_bcf1_freeze.json", freeze)
    write_jsonl(root / "qa_evidence_sufficiency_diagnostics.jsonl", diagnostics)
    write_json(root / "qa_readiness_summary.json", readiness)
    report = "\n".join(
        [
            "# Trial P1 Post-BCF1 Status",
            "",
            f"- Frozen visual label: `{VISUAL_LABEL}`",
            f"- Historical label reserved: `{HISTORICAL_LABEL}`",
            "- Historical BCF1 exactness: **NOT PROVEN** (Cross/L21 hashes not reproduced).",
            "- KIS/TRAKE visual freeze: **PASS** (18 KIS, 3 TRAKE, Top100 per arm).",
            "- A0 Top5 preservation: **PASS for all 21 visual queries**.",
            f"- SigLIP2 index fingerprint: `{EXPECTED_SIGLIP2_INDEX_FINGERPRINT}`.",
            "- SigLIP2 index rebuilt: **NO**.",
            "- QA state: **QA_PRE_ASR_NOT_READY**.",
            "- GT opened: **NO**.",
            "- ASR v1.2 touched: **NO**.",
            "- Production policy changed: **NO**.",
            "",
            "## QA readiness",
            "",
            *[
                (
                    f"- `{query_id}` ({row['answer_type']}): "
                    f"syntax {row['syntax_pass_count']}/100; "
                    f"evidence-sufficient {row['evidence_sufficient_count']}/100; "
                    f"OCR {row['ocr_supported_count']}; ASR {row['asr_supported_count']}; "
                    f"Qwen {row['qwen_supported_count']}; **{row['readiness']}**."
                )
                for query_id, row in readiness["queries"].items()
            ],
            "",
            "The source Trial artifact did not persist tuple-level OCR/ASR/Qwen evidence. "
            "The new sufficiency gate therefore fails closed instead of treating syntax "
            "as quality.",
            "The ASR v1.2 loader and bounded Qwen executor are code-ready for a later "
            "mounted bundle; "
            "no corpus job was launched by this patch.",
        ]
    )
    (root / "TRIAL_P1_POST_BCF1_STATUS.md").write_text(report + "\n", encoding="utf-8")
    return {"freeze": freeze, "readiness": readiness, "output_root": str(root)}


__all__ = [
    "EXPECTED_SIGLIP2_INDEX_FINGERPRINT",
    "HISTORICAL_LABEL",
    "VISUAL_LABEL",
    "prepare_post_bcf1_artifacts",
]
