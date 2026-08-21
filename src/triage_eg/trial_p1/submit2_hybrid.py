"""Packaging-only Trial P1 Submission #2 conservative hybrid.

This module deliberately accepts only frozen prediction artifacts.  It does not
import a model runtime, read ground truth, or perform network/upload operations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from triage_eg.submission.aic26_prelim import create_submission_zip, validate_submission_zip

BCF1_MEMBER = "trial_p1_BCF1_F1_predictions.jsonl"
QUERY_MEMBER = "trial_p1_query_plans_v2.jsonl"
BCF1_PROVENANCE_MEMBER = "run_provenance.json"
BCF1_SUMMARY_MEMBER = "trial_p1_true_bcf1_summary.json"
SAFE_R4_MEMBER = "SAFE_R4.jsonl"
STRONG_ASR_MEMBER = "strong_asr_r4_inclusion_audit.jsonl"
R4_PROVENANCE_MEMBER = "run_provenance.json"
EXPECTED_BCF1_SHA256 = "33a6e592e0222e0c4c503dbd2d9f52fcfc3dad257730a424c8f8d365ef310acd"
CANDIDATE_NAME = "TRIAL_P1_SAFE_R4_HYBRID_QA_BCF1"
OUTPUT_ZIP_NAME = "trial_p1_SAFE_R4_HYBRID_QA_BCF1_submission.zip"
RRF_K = 60


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(archive: zipfile.ZipFile, name: str) -> bytes:
    matches = [member for member in archive.namelist() if member == name]
    if matches != [name]:
        raise RuntimeError(f"expected exactly one ZIP member {name!r}; found {matches}")
    return archive.read(name)


def _json(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"{label}:{line_number} is not a JSON object")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"{label} is empty")
    return rows


def _semantic_content_hash(rows: list[dict[str, Any]]) -> str:
    keys = ("query_id", "video_id", "frame_id", "frame_ids", "answer", "rank")
    values = [{key: row[key] for key in keys if key in row} for row in rows]
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_frozen_sources(bcf1_bundle: str | Path, r4_bundle: str | Path) -> dict[str, Any]:
    """Load and cross-check the two frozen, GT-free source bundles."""

    bcf1_path = Path(bcf1_bundle).resolve(strict=True)
    r4_path = Path(r4_bundle).resolve(strict=True)
    with zipfile.ZipFile(bcf1_path) as archive:
        bcf1_payload = _member(archive, BCF1_MEMBER)
        query_payload = _member(archive, QUERY_MEMBER)
        bcf1_provenance = _json(_member(archive, BCF1_PROVENANCE_MEMBER), BCF1_PROVENANCE_MEMBER)
        bcf1_summary = _json(_member(archive, BCF1_SUMMARY_MEMBER), BCF1_SUMMARY_MEMBER)
    with zipfile.ZipFile(r4_path) as archive:
        safe_payload = _member(archive, SAFE_R4_MEMBER)
        strong_payload = _member(archive, STRONG_ASR_MEMBER)
        r4_provenance = _json(_member(archive, R4_PROVENANCE_MEMBER), R4_PROVENANCE_MEMBER)

    bcf1_rows = _jsonl(bcf1_payload, BCF1_MEMBER)
    safe_rows = _jsonl(safe_payload, SAFE_R4_MEMBER)
    hashes = {
        "bcf1_bundle": sha256_file(bcf1_path),
        "r4_bundle": sha256_file(r4_path),
        "bcf1_predictions_file_sha256": _sha256_bytes(bcf1_payload),
        "bcf1_predictions_semantic_sha256": _semantic_content_hash(bcf1_rows),
        "query_plans": _sha256_bytes(query_payload),
        "safe_r4_file_sha256": _sha256_bytes(safe_payload),
        "safe_r4_semantic_sha256": _semantic_content_hash(safe_rows),
        "strong_asr_r4_inclusion_audit": _sha256_bytes(strong_payload),
    }
    expected_r4 = r4_provenance.get("candidate_hashes", {}).get("SAFE_R4")
    expected_query = r4_provenance.get("asset_hashes", {}).get("query_plans")
    expected_bcf1_from_r4 = r4_provenance.get("asset_hashes", {}).get("bcf1_predictions")
    expected_bcf1_from_bundle = bcf1_summary.get("f1_sha256")
    source_gates = {
        "known_bcf1_sha256": hashes["bcf1_predictions_file_sha256"] == EXPECTED_BCF1_SHA256,
        "bcf1_bundle_provenance_matches": expected_bcf1_from_bundle
        == hashes["bcf1_predictions_file_sha256"],
        "r4_bcf1_source_matches": expected_bcf1_from_r4 == hashes["bcf1_predictions_file_sha256"],
        "r4_query_source_matches": expected_query == hashes["query_plans"],
        "r4_safe_candidate_matches": expected_r4 == hashes["safe_r4_semantic_sha256"],
        "r4_inventory_gate_pass": r4_provenance.get("canonical_inventory", {}).get("status")
        == "PASS",
        "r4_gt_not_opened": r4_provenance.get("gt_opened") is False,
        "r4_whisper_not_run": r4_provenance.get("whisper_run") is False,
        "r4_submission_not_uploaded": r4_provenance.get("submission_uploaded") is False,
    }
    failed = [name for name, passed in source_gates.items() if not passed]
    if failed:
        raise RuntimeError(f"frozen source provenance gates failed: {failed}")
    return {
        "bcf1_bundle": bcf1_path,
        "r4_bundle": r4_path,
        "queries": _jsonl(query_payload, QUERY_MEMBER),
        "bcf1": bcf1_rows,
        "safe_r4": safe_rows,
        "strong_asr": _jsonl(strong_payload, STRONG_ASR_MEMBER),
        "bcf1_provenance": bcf1_provenance,
        "r4_provenance": r4_provenance,
        "hashes": hashes,
        "source_gates": source_gates,
    }


def normalize_queries(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queries = []
    for plan in plans:
        task = str(plan["task"]).upper()
        team_query = plan.get("team_query", {})
        query = {"query_id": str(plan["query_id"]), "task": task}
        if task == "TRAKE":
            query["event_count"] = int(team_query.get("event_count", len(plan.get("events", []))))
        queries.append(query)
    identifiers = [query["query_id"] for query in queries]
    if len(queries) != 24 or len(set(identifiers)) != 24:
        raise RuntimeError("official Trial query set must contain exactly 24 unique query IDs")
    counts = defaultdict(int)
    for query in queries:
        counts[query["task"]] += 1
    if dict(counts) != {"KIS": 18, "TRAKE": 3, "QA": 3}:
        raise RuntimeError(f"unexpected Trial task counts: {dict(counts)}")
    return queries


def _grouped(
    rows: list[dict[str, Any]], query_ids: set[str], label: str
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        query_id = str(row["query_id"])
        if query_id not in query_ids:
            raise RuntimeError(f"{label} contains unknown query_id {query_id}")
        output[query_id].append(dict(row))
    if set(output) != query_ids:
        raise RuntimeError(f"{label} query IDs do not match the official Trial set")
    for query_id, group in output.items():
        group.sort(key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in group]
        if ranks != list(range(1, 101)):
            raise RuntimeError(f"{label}:{query_id} must contain exact ranks 1..100")
    return dict(output)


def identity(task: str, row: dict[str, Any]) -> tuple[Any, ...]:
    task = task.upper()
    if task == "KIS":
        return (str(row["video_id"]), int(row["frame_id"]))
    if task == "QA":
        return (str(row["video_id"]), int(row["frame_id"]), str(row["answer"]))
    return (str(row["video_id"]), tuple(int(value) for value in row["frame_ids"]))


def _canonical_key(task: str, key: tuple[Any, ...]) -> tuple[Any, ...]:
    if task == "TRAKE":
        return (str(key[0]), *key[1])
    return key


def _fuse_tail(
    task: str,
    bcf1: list[dict[str, Any]],
    safe: list[dict[str, Any]],
    protected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    protected_keys = {identity(task, row) for row in protected}
    evidence: dict[tuple[Any, ...], dict[str, Any]] = {}
    for branch, rows in (("BCF1", bcf1[5:100]), ("SAFE_R4", safe[5:100])):
        for row in rows:
            key = identity(task, row)
            if key in protected_keys:
                continue
            entry = evidence.setdefault(
                key,
                {
                    "score": 0.0,
                    "best_rank": int(row["rank"]),
                    "bcf1_present": False,
                    "representative": dict(row),
                },
            )
            rank = int(row["rank"])
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            if branch == "BCF1":
                entry["bcf1_present"] = True
                entry["representative"] = dict(row)
    ordered = sorted(
        evidence.items(),
        key=lambda item: (
            -item[1]["score"],
            item[1]["best_rank"],
            0 if item[1]["bcf1_present"] else 1,
            _canonical_key(task, item[0]),
        ),
    )
    if len(ordered) < 95:
        raise RuntimeError(f"{task} fused tail has only {len(ordered)} unique candidates")
    return [dict(entry["representative"]) for _, entry in ordered[:95]]


def _renumber(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for rank, row in enumerate(rows[:100], 1):
        copied = dict(row)
        copied["rank"] = rank
        copied["system_variant"] = CANDIDATE_NAME
        output.append(copied)
    return output


def _trake_valid(rows: list[dict[str, Any]], event_count: int) -> bool:
    seen: set[tuple[Any, ...]] = set()
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


def build_hybrid(
    plans: list[dict[str, Any]],
    bcf1_rows: list[dict[str, Any]],
    safe_rows: list[dict[str, Any]],
    strong_asr_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct the frozen conservative hybrid and all independent audits."""

    queries = normalize_queries(plans)
    query_ids = {query["query_id"] for query in queries}
    bcf1 = _grouped(bcf1_rows, query_ids, "TRUE_BCF1")
    safe = _grouped(safe_rows, query_ids, "SAFE_R4")
    strong_by_id = {str(row["query_id"]): dict(row) for row in strong_asr_rows}
    predictions: list[dict[str, Any]] = []
    asr_audit: list[dict[str, Any]] = []
    exact_audit: dict[str, Any] = {"queries": {}}

    for query in queries:
        query_id, task = query["query_id"], query["task"]
        baseline, safe_group = bcf1[query_id], safe[query_id]
        if task == "QA":
            final = [dict(row) for row in baseline]
            qa_exact = all(
                identity(task, actual) == identity(task, expected)
                and int(actual["rank"]) == int(expected["rank"])
                for actual, expected in zip(final, baseline, strict=True)
            )
            exact_audit["queries"][query_id] = {
                "task": task,
                "qa_exact_bcf1_100": qa_exact,
                "top5_exact_bcf1": True,
            }
        else:
            protected = [dict(row) for row in baseline[:5]]
            final = _renumber(protected + _fuse_tail(task, baseline, safe_group, protected))
            top5_exact = all(
                identity(task, actual) == identity(task, expected)
                and int(actual["rank"]) == int(expected["rank"])
                for actual, expected in zip(final[:5], baseline[:5], strict=True)
            )
            exact_audit["queries"][query_id] = {
                "task": task,
                "qa_exact_bcf1_100": None,
                "top5_exact_bcf1": top5_exact,
            }

        if task == "KIS":
            source = strong_by_id.get(query_id)
            if source is None:
                raise RuntimeError(f"missing strong-ASR audit for KIS query {query_id}")
            qualified = int(source.get("qualified_direct_count", 0)) > 0 and bool(
                source.get("best_strong_asr_video")
            )
            video = str(source.get("best_strong_asr_video") or "")
            before_rank = next(
                (index for index, row in enumerate(final, 1) if str(row["video_id"]) == video),
                None,
            )
            intervention = False
            inserted_identity: tuple[Any, ...] | None = None
            if qualified and (before_rank is None or before_rank > 20):
                selected = next((row for row in safe_group if str(row["video_id"]) == video), None)
                if selected is None:
                    raise RuntimeError(
                        f"qualified strong-ASR video {video} has no SAFE_R4 tuple for {query_id}"
                    )
                selected_key = identity(task, selected)
                remaining = [row for row in final if identity(task, row) != selected_key]
                remaining.insert(19, dict(selected))
                final = _renumber(remaining)
                intervention = True
                inserted_identity = selected_key
            after_rank = next(
                (index for index, row in enumerate(final, 1) if str(row["video_id"]) == video),
                None,
            )
            asr_audit.append(
                {
                    "query_id": query_id,
                    "qualified": qualified,
                    "best_strong_asr_video": video or None,
                    "pre_intervention_rank": before_rank,
                    "final_hybrid_rank": after_rank,
                    "intervention": "INSERT_AT_RANK_20" if intervention else "NONE_REQUIRED",
                    "inserted_identity": list(inserted_identity) if inserted_identity else None,
                    "pass": not qualified or (after_rank is not None and after_rank <= 20),
                }
            )
            exact_audit["queries"][query_id]["top5_exact_bcf1"] = all(
                identity(task, actual) == identity(task, expected)
                and int(actual["rank"]) == int(expected["rank"])
                for actual, expected in zip(final[:5], baseline[:5], strict=True)
            )

        if len(final) != 100 or len({identity(task, row) for row in final}) != 100:
            raise RuntimeError(f"hybrid {query_id} is not an exact 100-row unique ranking")
        if task == "TRAKE" and not _trake_valid(final, int(query["event_count"])):
            raise RuntimeError(f"hybrid TRAKE structure invalid for {query_id}")
        predictions.extend(final)

    exact_audit["HYBRID_KIS_TOP5_EXACT_BCF1"] = all(
        row["top5_exact_bcf1"] for row in exact_audit["queries"].values() if row["task"] == "KIS"
    )
    exact_audit["HYBRID_TRAKE_TOP5_EXACT_BCF1"] = all(
        row["top5_exact_bcf1"] for row in exact_audit["queries"].values() if row["task"] == "TRAKE"
    )
    exact_audit["HYBRID_QA_EXACT_BCF1_100"] = all(
        row["qa_exact_bcf1_100"] for row in exact_audit["queries"].values() if row["task"] == "QA"
    )
    if not all(
        exact_audit[name]
        for name in (
            "HYBRID_KIS_TOP5_EXACT_BCF1",
            "HYBRID_TRAKE_TOP5_EXACT_BCF1",
            "HYBRID_QA_EXACT_BCF1_100",
        )
    ):
        raise RuntimeError("hybrid exact-preservation gate failed")
    if not all(row["pass"] for row in asr_audit):
        raise RuntimeError("qualified strong-ASR Top20 gate failed")
    return {
        "queries": queries,
        "predictions": predictions,
        "bcf1_grouped": bcf1,
        "hybrid_grouped": _grouped(predictions, query_ids, CANDIDATE_NAME),
        "strong_asr_audit": asr_audit,
        "exact_audit": exact_audit,
    }


def _comparison_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    strong = {row["query_id"]: row for row in result["strong_asr_audit"]}
    output = []
    for query in result["queries"]:
        query_id, task = query["query_id"], query["task"]
        baseline = result["bcf1_grouped"][query_id]
        hybrid = result["hybrid_grouped"][query_id]
        baseline_keys = [identity(task, row) for row in baseline]
        hybrid_keys = [identity(task, row) for row in hybrid]
        output.append(
            {
                "query_id": query_id,
                "task": task,
                "exact_top1_same": baseline_keys[:1] == hybrid_keys[:1],
                "exact_top5_same": baseline_keys[:5] == hybrid_keys[:5],
                "top20_overlap_count": len(set(baseline_keys[:20]) & set(hybrid_keys[:20])),
                "top50_overlap_count": len(set(baseline_keys[:50]) & set(hybrid_keys[:50])),
                "top100_overlap_count": len(set(baseline_keys) & set(hybrid_keys)),
                "best_strong_asr_video": strong.get(query_id, {}).get("best_strong_asr_video"),
                "best_strong_asr_final_hybrid_rank": strong.get(query_id, {}).get(
                    "final_hybrid_rank"
                ),
                "qa_exact100_preserved": baseline_keys == hybrid_keys if task == "QA" else None,
                "trake_valid": _trake_valid(hybrid, int(query["event_count"]))
                if task == "TRAKE"
                else None,
            }
        )
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def package_submit2_hybrid(
    bcf1_bundle: str | Path,
    r4_bundle: str | Path,
    output_root: str | Path,
    *,
    head: str,
) -> dict[str, Any]:
    """Build, validate, and report the real OJ-ready submission ZIP."""

    source = load_frozen_sources(bcf1_bundle, r4_bundle)
    result = build_hybrid(
        source["queries"], source["bcf1"], source["safe_r4"], source["strong_asr"]
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output_zip = create_submission_zip(
        result["queries"], result["predictions"], root / OUTPUT_ZIP_NAME
    )
    official_validation = validate_submission_zip(output_zip, result["queries"])
    exact_100 = official_validation.get("prediction_count") == 2400
    comparison = _comparison_rows(result)
    query_ids = {item["query_id"] for item in result["queries"]}
    safe_grouped = _grouped(source["safe_r4"], query_ids, "SAFE_R4_CHECK")
    coordinates_source_closed = all(
        identity(query["task"], row) in source_identities
        for query in result["queries"]
        for row in result["hybrid_grouped"][query["query_id"]]
        for source_identities in [
            {
                *(
                    identity(query["task"], item)
                    for item in result["bcf1_grouped"][query["query_id"]]
                ),
                *(identity(query["task"], item) for item in safe_grouped[query["query_id"]]),
            }
        ]
    )
    safety_gates = {
        **source["source_gates"],
        "official_query_set_24": len(result["queries"]) == 24,
        "hybrid_exact_100_rows_each": exact_100,
        "kis_top5_exact_bcf1": result["exact_audit"]["HYBRID_KIS_TOP5_EXACT_BCF1"],
        "trake_top5_exact_bcf1": result["exact_audit"]["HYBRID_TRAKE_TOP5_EXACT_BCF1"],
        "qa_exact_bcf1_100": result["exact_audit"]["HYBRID_QA_EXACT_BCF1_100"],
        "qualified_strong_asr_at_or_before_top20": all(
            row["pass"] for row in result["strong_asr_audit"]
        ),
        "coordinates_source_closed": coordinates_source_closed,
        "canonical_inventory_provenance_pass": source["r4_provenance"]
        .get("canonical_inventory", {})
        .get("status")
        == "PASS",
        "official_zip_validator_pass": official_validation.get("status") == "PASS",
        "gt_opened": False,
        "model_inference_run": False,
        "submission_uploaded": False,
    }
    pass_gates = {
        key: (
            not value
            if key in {"gt_opened", "model_inference_run", "submission_uploaded"}
            else value
        )
        for key, value in safety_gates.items()
    }
    decision = "READY_FOR_HUMAN_SUBMIT_2" if all(pass_gates.values()) else "DO_NOT_SUBMIT_2"
    zip_sha256 = sha256_file(output_zip)
    comparison_path = root / "hybrid_vs_bcf1_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    _write_json(root / "hybrid_vs_bcf1_comparison.json", comparison)
    _write_jsonl(root / "strong_asr_hybrid_inclusion_audit.jsonl", result["strong_asr_audit"])
    _write_json(root / "hybrid_exact_preservation_audit.json", result["exact_audit"])
    _write_json(root / "official_submission_validator.json", official_validation)
    provenance = {
        "candidate": CANDIDATE_NAME,
        "HEAD": head,
        "policy": {
            "qa": "EXACT_TRUE_BCF1_RANKS_1_100",
            "kis_prefix": "EXACT_TRUE_BCF1_RANKS_1_5",
            "trake_prefix": "EXACT_TRUE_BCF1_RANKS_1_5",
            "tail": "EQUAL_RRF60_TRUE_BCF1_PLUS_SAFE_R4_RANKS_6_100",
            "strong_asr": "KIS_BEST_QUALIFIED_VIDEO_AT_OR_BEFORE_RANK_20",
        },
        "source_paths": {
            "bcf1_bundle": str(source["bcf1_bundle"]),
            "r4_bundle": str(source["r4_bundle"]),
        },
        "source_hashes": source["hashes"],
        "output_zip": str(output_zip.resolve()),
        "output_zip_sha256": zip_sha256,
        "safety_gates": safety_gates,
        "pass_interpretation": pass_gates,
        "decision": decision,
        "gt_opened": False,
        "model_inference_run": False,
        "submission_uploaded": False,
    }
    _write_json(root / "run_provenance.json", provenance)
    changed_tail_count = sum(
        row["top100_overlap_count"] < 100 for row in comparison if row["task"] == "KIS"
    )
    query_count = official_validation["query_count"]
    prediction_count = official_validation["prediction_count"]
    insertion_count = sum(
        row["intervention"] != "NONE_REQUIRED" for row in result["strong_asr_audit"]
    )
    report = "\n".join(
        [
            "# Trial P1 Submission #2 Conservative Hybrid Decision",
            "",
            f"`{decision}`",
            "",
            f"- Candidate: `{CANDIDATE_NAME}`",
            f"- Final OJ ZIP: `{output_zip.resolve()}`",
            f"- ZIP SHA-256: `{zip_sha256}`",
            f"- Official validator: `{official_validation['status']}`",
            f"- Queries / predictions: {query_count} / {prediction_count}",
            f"- KIS queries with materially changed Top100 tail: {changed_tail_count}/18",
            f"- Strong-ASR bounded insertions: {insertion_count}",
            "- QA: exact TRUE_BCF1 ranks 1..100",
            "- KIS/TRAKE: exact TRUE_BCF1 Top5",
            "- GT opened: NO",
            "- Model inference: NO",
            "- Upload performed: NO",
            "",
            "## Safety gates",
            "",
            *[f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in pass_gates.items()],
            "",
            "Human submission is the only remaining action. No automatic upload was performed.",
        ]
    )
    (root / "SUBMIT2_HYBRID_DECISION.md").write_text(report + "\n", encoding="utf-8")
    if decision != "READY_FOR_HUMAN_SUBMIT_2":
        raise RuntimeError(f"hybrid packaging failed closed: {pass_gates}")
    return provenance
