"""Rank-only R5 fusion with a frozen live-winner fallback and bounded gated head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aic2026_eval.validation import validate_predictions


@dataclass(frozen=True)
class R5Settings:
    rrf_k: int = 60
    protected_qe: int = 5
    protected_gated: int = 3
    max_predictions: int = 100
    strong_asr_max_rank: int = 20
    gated_insert_rank: int = 5
    cross_material_delta: float = 0.005
    cross_flat_tolerance: float = 0.002
    l21_catastrophic_delta: float = -0.02
    gated_meaningful_delta: float = 0.005
    gated_min_query_wins: int = 3

    def __post_init__(self) -> None:
        if (
            self.rrf_k != 60
            or self.protected_qe != 5
            or self.protected_gated != 3
            or self.max_predictions != 100
            or self.strong_asr_max_rank != 20
            or self.gated_insert_rank != 5
            or self.cross_material_delta != 0.005
            or self.cross_flat_tolerance != 0.002
            or self.l21_catastrophic_delta != -0.02
            or self.gated_meaningful_delta != 0.005
            or self.gated_min_query_wins != 3
        ):
            raise ValueError("R5 frozen policy constants were changed")


def candidate_key(task: str, row: dict[str, Any]) -> tuple[Any, ...]:
    task = task.upper()
    if task in {"KIS", "QA"}:
        return str(row["video_id"]), int(row["frame_id"])
    if task == "TRAKE":
        return str(row["video_id"]), tuple(int(value) for value in row["frame_ids"])
    raise ValueError(f"unsupported R5 task: {task}")


def _canonical(key: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(key) == 2 and isinstance(key[1], tuple):
        return key[0], *key[1]
    return key


def _renumber(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "query_id": str(row["query_id"]).split("__r5__", 1)[0],
            "rank": rank,
            "system_variant": variant,
        }
        for rank, row in enumerate(rows[:100], 1)
    ]


def fuse_multiview_branch(
    query: dict[str, Any],
    rows_by_view: dict[str, list[dict[str, Any]]],
    *,
    settings: R5Settings | None = None,
    branch: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Equal RRF60 within one modality, retaining all view provenance."""

    settings = settings or R5Settings()
    task, query_id = str(query["task"]).upper(), str(query["query_id"])
    evidence: dict[tuple[Any, ...], dict[str, Any]] = {}
    for view, rows in rows_by_view.items():
        seen = set()
        for row in sorted(rows, key=lambda value: int(value["rank"]))[: settings.max_predictions]:
            key = candidate_key(task, row)
            if key in seen:
                continue
            seen.add(key)
            rank = int(row["rank"])
            item = evidence.setdefault(
                key,
                {
                    "score": 0.0,
                    "best_rank": rank,
                    "representative": dict(row),
                    "view_ranks": {},
                    "high_anchor_match_count": 0,
                },
            )
            item["score"] += 1.0 / (settings.rrf_k + rank)
            item["best_rank"] = min(item["best_rank"], rank)
            item["view_ranks"][view] = rank
            item["high_anchor_match_count"] = max(
                int(item["high_anchor_match_count"]),
                int(row.get("high_anchor_match_count", 0)),
            )
            if rank < int(item["representative"].get("rank", 101)):
                item["representative"] = dict(row)
    ordered = sorted(
        evidence.items(),
        key=lambda item: (
            -item[1]["score"],
            item[1]["best_rank"],
            -len(item[1]["view_ranks"]),
            _canonical(item[0]),
        ),
    )[: settings.max_predictions]
    predictions, provenance = [], []
    for rank, (key, item) in enumerate(ordered, 1):
        predictions.append(
            {
                **item["representative"],
                "query_id": query_id,
                "rank": rank,
                "system_variant": f"R5_{branch}_MULTIVIEW",
            }
        )
        provenance.append(
            {
                "query_id": query_id,
                "task": task,
                "branch": branch,
                "candidate_key": list(_canonical(key)),
                "fused_rank": rank,
                "rrf_k": settings.rrf_k,
                "rrf_score": item["score"],
                "best_view_rank": item["best_rank"],
                "views_agreeing": len(item["view_ranks"]),
                "view_ranks": item["view_ranks"],
                "high_anchor_match_count": item["high_anchor_match_count"],
                "gt_used": False,
            }
        )
    return predictions, provenance


def _fuse_final_tail(
    query: dict[str, Any],
    protected: list[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    *,
    settings: R5Settings,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task, query_id = str(query["task"]).upper(), str(query["query_id"])
    protected_keys = {candidate_key(task, row) for row in protected}
    evidence: dict[tuple[Any, ...], dict[str, Any]] = {}
    branch_order = {name: index for index, name in enumerate(branches)}
    for branch, rows in branches.items():
        seen = set()
        for row in sorted(rows, key=lambda value: int(value["rank"])):
            key = candidate_key(task, row)
            if key in protected_keys or key in seen:
                continue
            seen.add(key)
            rank = int(row["rank"])
            item = evidence.setdefault(
                key,
                {
                    "score": 0.0,
                    "best_rank": rank,
                    "representative": dict(row),
                    "representative_branch": branch,
                    "source_ranks": {},
                },
            )
            item["score"] += 1.0 / (settings.rrf_k + rank)
            item["best_rank"] = min(item["best_rank"], rank)
            item["source_ranks"][branch] = rank
            current = item["representative_branch"]
            if (rank, branch_order[branch]) < (
                int(item["representative"].get("rank", 101)),
                branch_order[current],
            ):
                item["representative"] = dict(row)
                item["representative_branch"] = branch
    ordered = sorted(
        evidence.items(),
        key=lambda item: (
            -item[1]["score"],
            item[1]["best_rank"],
            min(branch_order[name] for name in item[1]["source_ranks"]),
            _canonical(item[0]),
        ),
    )
    needed = settings.max_predictions - len(protected)
    if len(ordered) < needed:
        raise RuntimeError(f"R5_FINAL_TAIL_TOO_SHORT:{query_id}:{len(ordered)}:{needed}")
    output = [dict(row) for row in protected]
    provenance = []
    for key, item in ordered[:needed]:
        output.append(dict(item["representative"]))
        provenance.append(
            {
                "query_id": query_id,
                "task": task,
                "candidate_key": list(_canonical(key)),
                "rrf_score": item["score"],
                "best_source_rank": item["best_rank"],
                "source_ranks": item["source_ranks"],
                "sources_agreeing": len(item["source_ranks"]),
                "gt_used": False,
            }
        )
    return _renumber(output, variant), provenance


def _ensure_strong_asr(
    query: dict[str, Any],
    rows: list[dict[str, Any]],
    strong: dict[str, Any] | None,
    *,
    settings: R5Settings,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_id = str(query["query_id"])
    if str(query["task"]).upper() != "KIS" or not strong or not strong.get("qualified"):
        return rows, {
            "query_id": query_id,
            "qualified": False,
            "intervention": "NOT_APPLICABLE",
            "pass": True,
        }
    video = str(strong["video_id"])
    before = next(
        (index for index, row in enumerate(rows, 1) if str(row["video_id"]) == video), None
    )
    intervention = "NONE_REQUIRED"
    if before is None or before > settings.strong_asr_max_rank:
        representative = strong.get("representative")
        if not isinstance(representative, dict):
            raise RuntimeError(f"R5_STRONG_ASR_REPRESENTATIVE_MISSING:{query_id}:{video}")
        key = candidate_key("KIS", representative)
        remaining = [row for row in rows if candidate_key("KIS", row) != key]
        remaining.insert(settings.strong_asr_max_rank - 1, dict(representative))
        rows = _renumber(remaining, variant)
        intervention = f"INSERT_AT_RANK_{settings.strong_asr_max_rank}"
    after = next(
        (index for index, row in enumerate(rows, 1) if str(row["video_id"]) == video), None
    )
    passed = after is not None and after <= settings.strong_asr_max_rank
    if not passed:
        raise RuntimeError(f"R5_STRONG_ASR_TOP20_GATE_FAILED:{query_id}:{video}:{after}")
    return rows, {
        "query_id": query_id,
        "qualified": True,
        "video_id": video,
        "pre_intervention_rank": before,
        "final_rank": after,
        "intervention": intervention,
        "tier": strong.get("tier"),
        "view_ranks": strong.get("view_ranks", {}),
        "pass": passed,
    }


def _gated_override(
    query: dict[str, Any],
    bcf1: list[dict[str, Any]],
    qe: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    settings: R5Settings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_id, task = str(query["query_id"]), str(query["task"]).upper()
    if task != "KIS":
        return qe, {
            "query_id": query_id,
            "task": task,
            "override": False,
            "reason": "HEAD_OVERRIDE_FORBIDDEN_FOR_TASK",
            "pass": True,
        }
    protected_keys = {candidate_key(task, row) for row in bcf1[:5]}
    eligible = []
    for row in candidates:
        reasons = {
            "tier_a_direct": row.get("tier") == "TIER_A_DIRECT",
            "exact_high_anchor": int(row.get("high_anchor_match_count", 0)) > 0,
            "visual_corroboration": bool(row.get("visual_support")),
            "dual_text_or_ocr_visual": bool(row.get("lexical_e5_agreement"))
            or bool(row.get("ocr_exact_high_with_visual")),
            "not_object_or_weak_ocr_only": not bool(row.get("object_only"))
            and not bool(row.get("weak_ocr_only")),
            "not_already_protected": candidate_key(task, row["representative"])
            not in protected_keys,
        }
        if all(reasons.values()):
            eligible.append(
                (
                    int(row.get("rank", 101)),
                    _canonical(candidate_key(task, row["representative"])),
                    row,
                    reasons,
                )
            )
    if not eligible:
        return qe, {
            "query_id": query_id,
            "task": task,
            "override": False,
            "reason": "NO_CANDIDATE_PASSED_ALL_FROZEN_GATES",
            "pass": candidate_key(task, qe[0]) == candidate_key(task, bcf1[0]),
        }
    _, _, selected, reasons = min(eligible, key=lambda item: (item[0], item[1]))
    representative = dict(selected["representative"])
    selected_key = candidate_key(task, representative)
    prefix = [dict(row) for row in bcf1[: settings.gated_insert_rank - 1]]
    tail = [
        row
        for row in qe[settings.protected_gated :]
        if candidate_key(task, row) != selected_key
        and candidate_key(task, row) not in {candidate_key(task, item) for item in prefix}
    ]
    displaced = bcf1[settings.gated_insert_rank - 1]
    if candidate_key(task, displaced) not in {candidate_key(task, row) for row in tail}:
        tail.insert(0, displaced)
    output = _renumber([*prefix, representative, *tail], "SAFE_R5_GATED")
    if candidate_key(task, output[0]) != candidate_key(task, bcf1[0]):
        raise RuntimeError(f"R5_GATED_RANK1_CHANGED:{query_id}")
    return output, {
        "query_id": query_id,
        "task": task,
        "override": True,
        "override_rank": settings.gated_insert_rank,
        "candidate_key": list(_canonical(selected_key)),
        "replaced_bcf1_key": list(_canonical(candidate_key(task, displaced))),
        "gates": reasons,
        "deterministic_reason": "ALL_R5_GATED_HEAD_CONDITIONS_PASS",
        "pass": True,
    }


def build_r5_query_candidates(
    query: dict[str, Any],
    *,
    bcf1: list[dict[str, Any]],
    safe_r4_tail_source: list[dict[str, Any]],
    a0_multiview: list[dict[str, Any]],
    s1_multiview: list[dict[str, Any]],
    asr_multiview: list[dict[str, Any]] | None = None,
    live_strong_asr: dict[str, Any] | None = None,
    r5_strong_asr: dict[str, Any] | None = None,
    gated_candidates: list[dict[str, Any]] | None = None,
    settings: R5Settings | None = None,
) -> dict[str, Any]:
    """Build immutable SAFE_R4 fallback plus SAFE_R5_QE/GATED for one query."""

    settings = settings or R5Settings()
    task, query_id = str(query["task"]).upper(), str(query["query_id"])
    for label, rows in (("BCF1", bcf1), ("SAFE_R4_SOURCE", safe_r4_tail_source)):
        if len(rows) != 100 or [int(row["rank"]) for row in rows] != list(range(1, 101)):
            raise RuntimeError(f"R5_{label}_TOP100_CONTRACT:{query_id}")
    if task == "QA":
        return {
            "SAFE_R4_LIVE_WINNER": [dict(row) for row in bcf1],
            "SAFE_R5_QE": [dict(row) for row in bcf1],
            "SAFE_R5_GATED": [dict(row) for row in bcf1],
            "tail_provenance": [],
            "live_strong_asr_audit": {"query_id": query_id, "pass": True},
            "r5_strong_asr_audit": {"query_id": query_id, "pass": True},
            "head_override_audit": {
                "query_id": query_id,
                "task": task,
                "override": False,
                "reason": "QA_DETERMINISTIC_PATH_OR_BCF1_FALLBACK",
                "pass": True,
            },
        }
    live, live_provenance = _fuse_final_tail(
        query,
        [dict(row) for row in bcf1[: settings.protected_qe]],
        {"BCF1": bcf1, "SAFE_R4_EXISTING": safe_r4_tail_source},
        settings=settings,
        variant="SAFE_R4_LIVE_WINNER",
    )
    live, live_asr_audit = _ensure_strong_asr(
        query,
        live,
        live_strong_asr,
        settings=settings,
        variant="SAFE_R4_LIVE_WINNER",
    )
    branches = {
        "BCF1": bcf1,
        "SAFE_R4_LIVE_WINNER": live,
        "A0_MULTIVIEW": a0_multiview,
        "S1_MULTIVIEW": s1_multiview,
    }
    if asr_multiview:
        branches["ASR_E5_MULTIVIEW"] = asr_multiview
    qe, qe_provenance = _fuse_final_tail(
        query,
        [dict(row) for row in live[: settings.protected_qe]],
        branches,
        settings=settings,
        variant="SAFE_R5_QE",
    )
    qe, r5_asr_audit = _ensure_strong_asr(
        query, qe, r5_strong_asr, settings=settings, variant="SAFE_R5_QE"
    )
    gated, override_audit = _gated_override(
        query,
        bcf1,
        qe,
        gated_candidates or [],
        settings=settings,
    )
    if len(gated) != 100:
        raise RuntimeError(f"R5_GATED_TOP100_CONTRACT:{query_id}:{len(gated)}")
    return {
        "SAFE_R4_LIVE_WINNER": live,
        "SAFE_R5_QE": qe,
        "SAFE_R5_GATED": gated,
        "tail_provenance": [*live_provenance, *qe_provenance],
        "live_strong_asr_audit": live_asr_audit,
        "r5_strong_asr_audit": r5_asr_audit,
        "head_override_audit": override_audit,
    }


def validate_r5_arms(
    queries: list[dict[str, Any]], arms: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    summaries = {}
    for arm, rows in arms.items():
        summary, issues = validate_predictions(queries, rows)
        if summary["status"] != "PASS" or issues:
            raise RuntimeError(f"R5_ARM_VALIDATION_FAILED:{arm}:{issues}")
        summaries[arm] = summary
    return summaries


__all__ = [
    "R5Settings",
    "build_r5_query_candidates",
    "candidate_key",
    "fuse_multiview_branch",
    "validate_r5_arms",
]
