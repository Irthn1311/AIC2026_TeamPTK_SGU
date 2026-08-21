"""Build Trial Submission #3 from frozen M0_R4 and TRUE_BCF1 predictions only."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from triage_eg.submission.aic26_prelim import create_submission_zip, validate_submission_zip
from triage_eg.trial_p1.submit2_hybrid import (
    BCF1_MEMBER,
    BCF1_PROVENANCE_MEMBER,
    BCF1_SUMMARY_MEMBER,
    EXPECTED_BCF1_SHA256,
    QUERY_MEMBER,
    R4_PROVENANCE_MEMBER,
    SAFE_R4_MEMBER,
    _grouped,
    _json,
    _jsonl,
    _member,
    _semantic_content_hash,
    identity,
    normalize_queries,
    sha256_file,
)

CANDIDATE_NAME = "TRIAL_P1_M0_R4_QA_BCF1"
OUTPUT_ZIP_NAME = "trial_p1_M0_R4_QA_BCF1_submission.zip"
M0_R4_MEMBER = "M0_R4.jsonl"
M1_R4_MEMBER = "M1_R4.jsonl"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_submit3_sources(bcf1_bundle: str | Path, r4_bundle: str | Path) -> dict[str, Any]:
    """Load frozen sources and enforce their byte/semantic provenance contracts."""

    bcf1_path = Path(bcf1_bundle).resolve(strict=True)
    r4_path = Path(r4_bundle).resolve(strict=True)
    with zipfile.ZipFile(bcf1_path) as archive:
        bcf1_payload = _member(archive, BCF1_MEMBER)
        query_payload = _member(archive, QUERY_MEMBER)
        bcf1_provenance = _json(_member(archive, BCF1_PROVENANCE_MEMBER), BCF1_PROVENANCE_MEMBER)
        bcf1_summary = _json(_member(archive, BCF1_SUMMARY_MEMBER), BCF1_SUMMARY_MEMBER)
    with zipfile.ZipFile(r4_path) as archive:
        m0_payload = _member(archive, M0_R4_MEMBER)
        m1_payload = _member(archive, M1_R4_MEMBER)
        safe_payload = _member(archive, SAFE_R4_MEMBER)
        r4_provenance = _json(_member(archive, R4_PROVENANCE_MEMBER), R4_PROVENANCE_MEMBER)

    bcf1 = _jsonl(bcf1_payload, BCF1_MEMBER)
    m0 = _jsonl(m0_payload, M0_R4_MEMBER)
    m1 = _jsonl(m1_payload, M1_R4_MEMBER)
    safe = _jsonl(safe_payload, SAFE_R4_MEMBER)
    hashes = {
        "bcf1_bundle_file_sha256": sha256_file(bcf1_path),
        "r4_bundle_file_sha256": sha256_file(r4_path),
        "bcf1_predictions_file_sha256": _sha256(bcf1_payload),
        "query_plans_file_sha256": _sha256(query_payload),
        "m0_r4_file_sha256": _sha256(m0_payload),
        "m0_r4_semantic_sha256": _semantic_content_hash(m0),
        "m1_r4_file_sha256": _sha256(m1_payload),
        "m1_r4_semantic_sha256": _semantic_content_hash(m1),
        "safe_r4_file_sha256": _sha256(safe_payload),
        "safe_r4_semantic_sha256": _semantic_content_hash(safe),
    }
    candidate_hashes = r4_provenance.get("candidate_hashes", {})
    source_gates = {
        "known_bcf1_sha256": hashes["bcf1_predictions_file_sha256"] == EXPECTED_BCF1_SHA256,
        "bcf1_summary_hash_matches": bcf1_summary.get("f1_sha256")
        == hashes["bcf1_predictions_file_sha256"],
        "r4_bcf1_source_matches": r4_provenance.get("asset_hashes", {}).get("bcf1_predictions")
        == hashes["bcf1_predictions_file_sha256"],
        "r4_query_source_matches": r4_provenance.get("asset_hashes", {}).get("query_plans")
        == hashes["query_plans_file_sha256"],
        "m0_r4_semantic_hash_matches": candidate_hashes.get("M0_R4")
        == hashes["m0_r4_semantic_sha256"],
        "m1_r4_semantic_hash_matches": candidate_hashes.get("M1_R4")
        == hashes["m1_r4_semantic_sha256"],
        "safe_r4_semantic_hash_matches": candidate_hashes.get("SAFE_R4")
        == hashes["safe_r4_semantic_sha256"],
        "r4_canonical_inventory_pass": r4_provenance.get("canonical_inventory", {}).get("status")
        == "PASS",
        "bcf1_gt_not_opened": bcf1_provenance.get("GT_OPENED") is False
        and bcf1_summary.get("gt_opened") is False,
        "r4_gt_not_opened": r4_provenance.get("gt_opened") is False,
        "r4_whisper_not_run": r4_provenance.get("whisper_run") is False,
        "r4_submission_not_uploaded": r4_provenance.get("submission_uploaded") is False,
    }
    failed = [name for name, passed in source_gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Submission #3 frozen source gates failed: {failed}")
    return {
        "bcf1_bundle": bcf1_path,
        "r4_bundle": r4_path,
        "queries": _jsonl(query_payload, QUERY_MEMBER),
        "bcf1": bcf1,
        "m0_r4": m0,
        "m1_r4": m1,
        "safe_r4": safe,
        "r4_provenance": r4_provenance,
        "hashes": hashes,
        "source_gates": source_gates,
    }


def _trake_valid(rows: list[dict[str, Any]], event_count: int) -> bool:
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for row in rows:
        frames = tuple(int(value) for value in row.get("frame_ids", []))
        key = (str(row.get("video_id")), frames)
        if (
            len(frames) != event_count
            or any(left >= right for left, right in zip(frames, frames[1:], strict=False))
            or key in seen
        ):
            return False
        seen.add(key)
    return len(rows) == 100


def _coordinate_set(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    coordinates = set()
    for row in rows:
        video_id = str(row["video_id"])
        if "frame_id" in row:
            coordinates.add((video_id, int(row["frame_id"])))
        for frame_id in row.get("frame_ids", []):
            coordinates.add((video_id, int(frame_id)))
    return coordinates


def build_submit3_predictions(source: dict[str, Any]) -> dict[str, Any]:
    """Copy the exact frozen arms and prove all task-specific invariants."""

    queries = normalize_queries(source["queries"])
    query_ids = {query["query_id"] for query in queries}
    bcf1 = _grouped(source["bcf1"], query_ids, "TRUE_BCF1")
    m0 = _grouped(source["m0_r4"], query_ids, "M0_R4")
    validated_r4_coordinates = _coordinate_set(
        [*source["m0_r4"], *source["m1_r4"], *source["safe_r4"]]
    )
    predictions: list[dict[str, Any]] = []
    query_audits: list[dict[str, Any]] = []

    for query in queries:
        query_id, task = query["query_id"], query["task"]
        frozen = bcf1[query_id] if task == "QA" else m0[query_id]
        copied = [dict(row) for row in frozen]
        exact_copy = all(
            int(actual["rank"]) == int(expected["rank"])
            and identity(task, actual) == identity(task, expected)
            for actual, expected in zip(copied, frozen, strict=True)
        )
        coordinates = _coordinate_set(copied)
        inventory_closed = coordinates <= validated_r4_coordinates
        trake_valid = _trake_valid(copied, int(query["event_count"])) if task == "TRAKE" else None
        audit = {
            "query_id": query_id,
            "task": task,
            "source_arm": "TRUE_BCF1" if task == "QA" else "M0_R4",
            "row_count": len(copied),
            "ranks_exact_1_100": [int(row["rank"]) for row in copied] == list(range(1, 101)),
            "exact_source_identity_and_rank": exact_copy,
            "canonical_inventory_source_closure": inventory_closed,
            "trake_strict_event_count_unique": trake_valid,
            "pass": len(copied) == 100
            and exact_copy
            and inventory_closed
            and (trake_valid is not False),
        }
        query_audits.append(audit)
        predictions.extend(copied)

    gates = {
        "official_query_set_24": len(queries) == 24,
        "exact_100_rows_per_query": len(predictions) == 2400
        and all(row["row_count"] == 100 for row in query_audits),
        "kis_exact_m0_r4_1_100": all(row["pass"] for row in query_audits if row["task"] == "KIS"),
        "trake_exact_m0_r4_1_100": all(
            row["exact_source_identity_and_rank"] for row in query_audits if row["task"] == "TRAKE"
        ),
        "trake_strict_event_count_unique": all(
            row["trake_strict_event_count_unique"] for row in query_audits if row["task"] == "TRAKE"
        ),
        "qa_exact_true_bcf1_1_100": all(row["pass"] for row in query_audits if row["task"] == "QA"),
        "canonical_inventory_source_closure": all(
            row["canonical_inventory_source_closure"] for row in query_audits
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"Submission #3 construction gates failed: {failed}")
    return {
        "queries": queries,
        "predictions": predictions,
        "query_audits": query_audits,
        "construction_gates": gates,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def package_submit3(
    bcf1_bundle: str | Path,
    r4_bundle: str | Path,
    output_root: str | Path,
    *,
    head: str,
) -> dict[str, Any]:
    """Create and independently validate the real OJ Submission #3 ZIP."""

    source = load_submit3_sources(bcf1_bundle, r4_bundle)
    result = build_submit3_predictions(source)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output_zip = create_submission_zip(
        result["queries"], result["predictions"], root / OUTPUT_ZIP_NAME
    )
    official = validate_submission_zip(output_zip, result["queries"])
    final_gates = {
        **source["source_gates"],
        **result["construction_gates"],
        "official_zip_validator_pass": official.get("status") == "PASS",
        "official_zip_query_count_24": official.get("query_count") == 24,
        "official_zip_prediction_count_2400": official.get("prediction_count") == 2400,
        "gt_not_opened": True,
        "model_inference_not_run": True,
        "submission_not_uploaded": True,
    }
    decision = "READY_FOR_HUMAN_SUBMIT_3" if all(final_gates.values()) else "DO_NOT_SUBMIT_3"
    zip_sha256 = sha256_file(output_zip)
    _write_json(root / "submit3_exact_copy_audit.json", result["query_audits"])
    _write_json(root / "official_submission_validator.json", official)
    provenance = {
        "candidate": CANDIDATE_NAME,
        "HEAD": head,
        "policy": {
            "KIS": "EXACT_M0_R4_RANKS_1_100",
            "TRAKE": "EXACT_M0_R4_RANKS_1_100",
            "QA": "EXACT_TRUE_BCF1_RANKS_1_100",
        },
        "source_paths": {
            "bcf1_bundle": str(source["bcf1_bundle"]),
            "r4_bundle": str(source["r4_bundle"]),
        },
        "source_hashes": source["hashes"],
        "candidate_semantic_sha256": _semantic_content_hash(result["predictions"]),
        "output_zip": str(output_zip.resolve()),
        "output_zip_sha256": zip_sha256,
        "gates": final_gates,
        "decision": decision,
        "gt_opened": False,
        "model_inference_run": False,
        "submission_uploaded": False,
    }
    _write_json(root / "run_provenance.json", provenance)
    report = "\n".join(
        [
            "# Trial P1 Submission #3 Decision",
            "",
            f"`{decision}`",
            "",
            f"- Candidate: `{CANDIDATE_NAME}`",
            f"- OJ ZIP: `{output_zip.resolve()}`",
            f"- ZIP SHA-256: `{zip_sha256}`",
            f"- Validator: `{official['status']}`",
            f"- Queries / predictions: {official['query_count']} / {official['prediction_count']}",
            "- KIS: exact M0_R4 ranks 1..100",
            "- TRAKE: exact M0_R4 ranks 1..100",
            "- QA: exact TRUE_BCF1 ranks 1..100",
            "- Canonical inventory closure: PASS",
            "- GT opened: NO",
            "- Model inference: NO",
            "- Upload performed: NO",
            "",
            "## Gates",
            "",
            *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in final_gates.items()],
            "",
            "STOP. Human upload is the only remaining action.",
        ]
    )
    (root / "SUBMIT3_DECISION.md").write_text(report + "\n", encoding="utf-8")
    if decision != "READY_FOR_HUMAN_SUBMIT_3":
        raise RuntimeError(f"Submission #3 failed closed: {final_gates}")
    return provenance
