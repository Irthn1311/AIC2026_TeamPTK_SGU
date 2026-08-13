"""Deterministic M3 registry construction with fail-closed frozen metadata."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .solver import M3InferenceCase

EXPECTED_AI_QC_SHA256 = "65062e7eb180e97ff4a9dc8462cde08bdd18b9d5ca9ca14c2f16740078cb1142"

NEW_TRANSITIONS: dict[str, dict[str, Any]] = {
    "mb1v022_c005": {
        "moment_type": "ONSET",
        "semantic_event_en": (
            "The lion dancer begins stepping/moving sideways after a brief near-stationary "
            "body position."
        ),
        "before_state_en": "The lion dancer is nearly stationary in place.",
        "after_state_en": "The lion dancer is actively stepping or moving sideways.",
        "accepted_intervals": [[1808, 1817]],
        "primary_gate": True,
        "conditional": False,
    },
    "mb1v022_c013": {
        "moment_type": "CONTACT",
        "semantic_event_en": (
            "The food/chopsticks make contact with the hot oil in the pan during transfer "
            "from the bowl."
        ),
        "before_state_en": (
            "The food and chopsticks are above the pan and not yet touching the hot oil."
        ),
        "after_state_en": "The food or chopsticks are touching the hot oil in the pan.",
        "accepted_intervals": [[5650, 5656]],
        "primary_gate": True,
        "conditional": False,
    },
    "mb1v022_c015": {
        "moment_type": "CONTACT",
        "semantic_event_en": "The man's two palms make contact.",
        "before_state_en": "The man's palms are apart and not touching.",
        "after_state_en": "The man's palms are touching each other.",
        "accepted_intervals": [[3302, 3308]],
        "primary_gate": True,
        "conditional": False,
    },
    "mb1v022_c017": {
        "moment_type": "ONSET",
        "semantic_event_en": ("The man begins raising his head from a downward-looking posture."),
        "before_state_en": "The man is looking downward with his head lowered.",
        "after_state_en": "The man is raising his head upward.",
        "accepted_intervals": [[6594, 6605]],
        "primary_gate": True,
        "conditional": False,
    },
    "mb1v022_c014": {
        "moment_type": "FIRST_OCCURRENCE",
        "semantic_event_en": (
            'The text "SÀI GÒN BẢO DỤNG" first becomes fully readable on the sign.'
        ),
        "before_state_en": "The full text on the sign is not yet readable.",
        "after_state_en": 'The full text "SÀI GÒN BẢO DỤNG" is readable on the sign.',
        "accepted_intervals": [[1933, 1943]],
        "primary_gate": False,
        "conditional": True,
    },
}

TRUSTED_REQUIRED = frozenset(
    {
        "case_id",
        "source_candidate_id",
        "source_version",
        "video_id",
        "moment_type",
        "semantic_event_vi",
        "semantic_event_en",
        "accepted_intervals",
        "candidate_anchor_frame",
        "candidate_anchor_source",
        "primary_gate",
        "conditional",
        "annotation_confidence",
        "annotation_provenance",
        "human_reviewed",
    }
)


def _zip_jsonl(path: Path, basename: str) -> list[dict[str, Any]]:
    with ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {basename} in {path}; found={matches}")
        text = archive.read(matches[0]).decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _external_rows(path: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if path is None:
        return [], None
    source = path.expanduser().resolve(strict=True)
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            matches = [
                name
                for name in archive.namelist()
                if Path(name).suffix.lower() == ".jsonl" and "sealed" not in name.casefold()
            ]
            rows = []
            for name in sorted(matches):
                for line in archive.read(name).decode("utf-8-sig").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
        return rows, str(source)
    return (
        [
            json.loads(line)
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ],
        str(source),
    )


def _intervals(row: dict[str, Any]) -> list[list[int]] | None:
    value = row.get("accepted_intervals")
    if value is None and isinstance(row.get("accepted_interval"), list):
        value = [row["accepted_interval"]]
    if value is None and all(
        key in row for key in ("acceptable_start_frame", "acceptable_end_frame")
    ):
        value = [[row["acceptable_start_frame"], row["acceptable_end_frame"]]]
    if not isinstance(value, list) or not value:
        return None
    normalized = []
    for interval in value:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(frame, int) and not isinstance(frame, bool) for frame in interval)
            or interval[0] > interval[1]
        ):
            return None
        normalized.append([int(interval[0]), int(interval[1])])
    return normalized


def validate_trusted_registry_row(row: dict[str, Any]) -> None:
    missing = sorted(key for key in TRUSTED_REQUIRED if row.get(key) is None or row.get(key) == "")
    missing.extend(key for key in ("before_state_en", "after_state_en") if key not in row)
    if row.get("moment_type") in {"ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION"}:
        missing.extend(
            key
            for key in ("before_state_en", "after_state_en")
            if row.get(key) is None or row.get(key) == ""
        )
    missing = sorted(set(missing))
    if missing:
        raise ValueError(f"M3_TRUSTED_METADATA_MISSING: {missing}")
    if _intervals(row) is None:
        raise ValueError("M3_TRUSTED_INTERVAL_INVALID")
    if (
        not isinstance(row["candidate_anchor_frame"], int)
        or isinstance(row["candidate_anchor_frame"], bool)
        or row["candidate_anchor_frame"] < 0
    ):
        raise ValueError("M3_TRUSTED_ANCHOR_INVALID")
    if row.get("registry_status") != "TRUSTED_METADATA_READY":
        raise ValueError("M3 registry row is not trusted for inference")
    if row["moment_type"] in {"ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION"} and (
        not row["before_state_en"] or not row["after_state_en"]
    ):
        raise ValueError("M3 boundary row requires explicit before/after states")


def inference_case_from_registry(row: dict[str, Any]) -> M3InferenceCase:
    validate_trusted_registry_row(row)
    return M3InferenceCase(
        case_id=str(row["case_id"]),
        video_id=str(row["video_id"]),
        moment_type=str(row["moment_type"]),
        semantic_event_en=str(row["semantic_event_en"]),
        before_state_en=(
            str(row["before_state_en"]) if row["before_state_en"] is not None else None
        ),
        after_state_en=(str(row["after_state_en"]) if row["after_state_en"] is not None else None),
        candidate_anchor_frame=int(row["candidate_anchor_frame"]),
    )


def _trusted_external_frozen(row: dict[str, Any], carry: dict[str, Any]) -> dict[str, Any] | None:
    candidate_id = str(carry["candidate_id"])
    source_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "")
    if source_id != candidate_id or str(row.get("video_id")) != str(carry["video_id"]):
        return None
    intervals = _intervals(row)
    anchor = next(
        (
            value
            for value in (
                row.get("candidate_anchor_frame"),
                row.get("proposal_center_frame"),
                row.get("source_anchor_frame"),
            )
            if value is not None
        ),
        None,
    )
    fields = {
        "moment_type": row.get("moment_type"),
        "semantic_event_vi": row.get("semantic_event_vi") or row.get("query_text_vi"),
        "semantic_event_en": row.get("semantic_event_en") or row.get("query_text"),
        "candidate_anchor_frame": anchor,
        "annotation_confidence": row.get("annotation_confidence") or row.get("confidence"),
    }
    if intervals is None or any(value is None or value == "" for value in fields.values()):
        return None
    before_state = row.get("before_state_en")
    after_state = row.get("after_state_en")
    if fields["moment_type"] in {"ONSET", "CONTACT", "FIRST_OCCURRENCE", "SEPARATION"} and (
        not before_state or not after_state
    ):
        return None
    return {
        "case_id": f"m3_{candidate_id}",
        "source_candidate_id": candidate_id,
        "source_version": "MB1_V02_FROZEN",
        "video_id": carry["video_id"],
        **fields,
        "before_state_en": before_state,
        "after_state_en": after_state,
        "accepted_intervals": intervals,
        "candidate_anchor_source": "ORIGINAL_FROZEN_SEED_METADATA",
        "primary_gate": True,
        "conditional": False,
        "annotation_provenance": row.get("annotation_provenance")
        or "ORIGINAL_FROZEN_SEED_ANNOTATION",
        "human_reviewed": bool(row.get("human_reviewed", False)),
        "registry_status": "TRUSTED_METADATA_READY",
        "eligible_for_inference": True,
        "reason_code_used_to_infer_semantics": False,
    }


def build_case_registry(
    *,
    ai_qc_zip: str | Path,
    notebook20_candidates_zip: str | Path,
    frozen_seed_metadata: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qc_zip = Path(ai_qc_zip).expanduser().resolve(strict=True)
    candidates_zip = Path(notebook20_candidates_zip).expanduser().resolve(strict=True)
    new_qc = _zip_jsonl(qc_zip, "mb1_v022_ai_qc_new_candidates_v01.jsonl")
    carryover = _zip_jsonl(qc_zip, "mb1_v022_frozen_seed_carryover_v01.jsonl")
    candidates = {
        str(row["candidate_id"]): row
        for row in _zip_jsonl(candidates_zip, "mb1_v022_candidate_manifest.jsonl")
    }
    external, external_source = _external_rows(
        Path(frozen_seed_metadata) if frozen_seed_metadata is not None else None
    )
    external_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in external:
        key = str(row.get("source_candidate_id") or row.get("candidate_id") or "")
        if key:
            external_by_id.setdefault(key, []).append(row)

    registry = []
    recovered = 0
    for carry in sorted(carryover, key=lambda row: str(row["candidate_id"])):
        candidate_id = str(carry["candidate_id"])
        recovered_row = next(
            (
                value
                for source in external_by_id.get(candidate_id, [])
                if (value := _trusted_external_frozen(source, carry)) is not None
            ),
            None,
        )
        if recovered_row is not None:
            validate_trusted_registry_row(recovered_row)
            registry.append(recovered_row)
            recovered += 1
            continue
        registry.append(
            {
                "case_id": f"m3_{candidate_id}",
                "source_candidate_id": candidate_id,
                "source_version": "MB1_V02_FROZEN",
                "video_id": carry["video_id"],
                "moment_type": None,
                "semantic_event_vi": None,
                "semantic_event_en": None,
                "before_state_en": None,
                "after_state_en": None,
                "accepted_intervals": None,
                "candidate_anchor_frame": None,
                "candidate_anchor_source": None,
                "primary_gate": False,
                "intended_primary_gate": True,
                "conditional": False,
                "annotation_confidence": None,
                "annotation_provenance": "FROZEN_SEED_CARRYOVER_ONLY",
                "human_reviewed": False,
                "registry_status": "FROZEN_SEED_METADATA_UNAVAILABLE",
                "eligible_for_inference": False,
                "prior_reason_code": carry.get("prior_reason_code"),
                "reason_code_used_to_infer_semantics": False,
            }
        )

    qc_by_id = {str(row["candidate_id"]): row for row in new_qc}
    for candidate_id, transition in NEW_TRANSITIONS.items():
        qc = qc_by_id.get(candidate_id)
        candidate = candidates.get(candidate_id)
        if qc is None or candidate is None:
            raise ValueError(f"M3 required new case missing from source artifacts: {candidate_id}")
        expected_qc = "CONDITIONAL" if transition["conditional"] else "USABLE"
        if qc.get("ai_qc") != expected_qc or str(qc["video_id"]) != str(candidate["video_id"]):
            raise ValueError(f"M3 source mismatch for {candidate_id}")
        row = {
            "case_id": f"m3_{candidate_id}",
            "source_candidate_id": candidate_id,
            "source_version": "MB1_V022_AI_QC_V01",
            "video_id": qc["video_id"],
            "moment_type": transition["moment_type"],
            "semantic_event_vi": qc["semantic_event_vi"],
            "semantic_event_en": transition["semantic_event_en"],
            "before_state_en": transition["before_state_en"],
            "after_state_en": transition["after_state_en"],
            "accepted_intervals": transition["accepted_intervals"],
            "candidate_anchor_frame": int(candidate["proposal_center_frame"]),
            "candidate_anchor_source": "NOTEBOOK20_PROPOSAL_CENTER_FRAME",
            "primary_gate": transition["primary_gate"],
            "conditional": transition["conditional"],
            "annotation_confidence": qc["confidence"],
            "annotation_provenance": qc.get("interval_basis") or "MB1_V022_AI_QC_DENSE_SHEET",
            "human_reviewed": False,
            "registry_status": "TRUSTED_METADATA_READY",
            "eligible_for_inference": True,
            "reason_code_used_to_infer_semantics": False,
        }
        validate_trusted_registry_row(row)
        registry.append(row)

    registry.sort(key=lambda row: (str(row["source_version"]), str(row["source_candidate_id"])))
    eligible = [row for row in registry if row["eligible_for_inference"]]
    primary = [row for row in eligible if row["primary_gate"]]
    secondary = [row for row in eligible if row["conditional"]]
    type_counts = Counter(str(row["moment_type"]) for row in primary)
    summary = {
        "status": "READY" if recovered == len(carryover) else "PARTIAL",
        "registry_row_count": len(registry),
        "frozen_seed_count": len(carryover),
        "frozen_seed_metadata_found": recovered,
        "frozen_seed_metadata_unavailable": len(carryover) - recovered,
        "new_primary_cases": 4,
        "secondary_conditional_cases": len(secondary),
        "primary_case_count": len(primary),
        "eligible_case_count": len(eligible),
        "type_counts": dict(sorted(type_counts.items())),
        "frozen_seed_annotation_source": external_source,
        "benchmark_coverage": (
            "ADEQUATE_FOR_BOUNDED_KEEP_DROP" if len(primary) >= 8 else "TOO_SMALL_FOR_KEEP_DROP"
        ),
    }
    return registry, summary


__all__ = [
    "EXPECTED_AI_QC_SHA256",
    "NEW_TRANSITIONS",
    "TRUSTED_REQUIRED",
    "build_case_registry",
    "inference_case_from_registry",
    "validate_trusted_registry_row",
]
