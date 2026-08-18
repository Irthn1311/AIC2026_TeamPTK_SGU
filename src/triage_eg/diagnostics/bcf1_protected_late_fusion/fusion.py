"""Exact A0-Top5-protected equal RRF60 fusion over final prediction lists."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from aic2026_eval.validation import validate_predictions

from .contracts import BCF1Settings

CandidateKey = tuple[Any, ...]


def normalize_qa_answer(value: str) -> str:
    """BCF-1 identity normalization: whitespace collapse and casefold only."""

    return " ".join(str(value).split()).casefold()


def candidate_key(task: str, row: dict[str, Any]) -> CandidateKey:
    if task == "KIS":
        return str(row["video_id"]), int(row["frame_id"])
    if task == "QA":
        return (
            str(row["video_id"]),
            int(row["frame_id"]),
            normalize_qa_answer(str(row["answer"])),
        )
    if task == "TRAKE":
        return str(row["video_id"]), tuple(int(value) for value in row["frame_ids"])
    raise ValueError(f"unsupported BCF-1 task: {task}")


def _rank_map(task: str, rows: list[dict[str, Any]]) -> dict[CandidateKey, dict[str, Any]]:
    output: dict[CandidateKey, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda value: int(value["rank"])):
        output.setdefault(candidate_key(task, row), row)
    return output


def fuse_query(
    query: dict[str, Any],
    a0_rows: list[dict[str, Any]],
    s1_rows: list[dict[str, Any]],
    *,
    settings: BCF1Settings | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = settings or BCF1Settings()
    task, query_id = str(query["task"]), str(query["query_id"])
    a0 = sorted(a0_rows, key=lambda row: int(row["rank"]))
    s1 = sorted(s1_rows, key=lambda row: int(row["rank"]))
    if (
        len(a0) != settings.max_predictions
        or len(s1) != settings.max_predictions
        or any(row.get("query_id") != query_id for row in [*a0, *s1])
        or [row["rank"] for row in a0] != list(range(1, 101))
        or [row["rank"] for row in s1] != list(range(1, 101))
    ):
        raise RuntimeError(f"BCF1_INPUT_TOP100_CONTRACT_FAILED: {query_id}")
    a0_map, s1_map = _rank_map(task, a0), _rank_map(task, s1)
    protected = a0[: settings.protected_prefix]
    protected_keys = [candidate_key(task, row) for row in protected]
    if len(set(protected_keys)) != settings.protected_prefix:
        raise RuntimeError(f"BCF1_A0_PROTECTED_PREFIX_DUPLICATE: {query_id}")
    output = [dict(row) for row in protected]
    provenance = [
        {
            "query_id": query_id,
            "fused_rank": int(row["rank"]),
            "protected_a0_prefix": True,
            "a0_rank": int(row["rank"]),
            "s1_rank": (
                int(s1_map[candidate_key(task, row)]["rank"])
                if candidate_key(task, row) in s1_map
                else None
            ),
            "rrf_k": settings.rrf_k,
            "rrf_score": None,
            "source": "A0_PROTECTED",
        }
        for row in protected
    ]
    tail = []
    for key in (set(a0_map) | set(s1_map)) - set(protected_keys):
        a0_row, s1_row = a0_map.get(key), s1_map.get(key)
        a0_rank = int(a0_row["rank"]) if a0_row else None
        s1_rank = int(s1_row["rank"]) if s1_row else None
        score = (1 / (settings.rrf_k + a0_rank) if a0_rank else 0.0) + (
            1 / (settings.rrf_k + s1_rank) if s1_rank else 0.0
        )
        best_rank = min(rank for rank in (a0_rank, s1_rank) if rank is not None)
        source = "BOTH" if a0_row and s1_row else "A0_ONLY" if a0_row else "S1_ONLY"
        tail.append((key, a0_row, s1_row, score, best_rank, source))
    tail.sort(
        key=lambda item: (
            -item[3],
            item[4],
            0 if item[1] else 1,
            item[0],
        )
    )
    for _key, a0_row, s1_row, score, _, source in tail:
        if len(output) >= settings.max_predictions:
            break
        representative = a0_row or s1_row
        if representative is None:  # pragma: no cover - union construction guarantees this
            raise RuntimeError("BCF1_FUSION_REPRESENTATIVE_MISSING")
        rank = len(output) + 1
        output.append({name: value for name, value in representative.items() if name != "rank"})
        output[-1]["rank"] = rank
        provenance.append(
            {
                "query_id": query_id,
                "fused_rank": rank,
                "protected_a0_prefix": False,
                "a0_rank": int(a0_row["rank"]) if a0_row else None,
                "s1_rank": int(s1_row["rank"]) if s1_row else None,
                "rrf_k": settings.rrf_k,
                "rrf_score": score,
                "source": source,
            }
        )
    if [row["rank"] for row in output] != list(range(1, len(output) + 1)):
        raise RuntimeError(f"BCF1_STRICT_RENUMBERING_FAILED: {query_id}")
    return output, provenance


def fuse_predictions(
    queries: list[dict[str, Any]],
    a0_predictions: list[dict[str, Any]],
    s1_predictions: list[dict[str, Any]],
    *,
    settings: BCF1Settings | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fuse in query input order without accepting or consulting ground truth."""

    settings = settings or BCF1Settings()
    for predictions, label in ((a0_predictions, "A0"), (s1_predictions, "S1")):
        validation, issues = validate_predictions(queries, predictions)
        if validation["status"] != "PASS":
            codes = sorted({issue["code"] for issue in issues})
            raise RuntimeError(f"BCF1_{label}_PREDICTION_VALIDATION_FAILED: {codes}")
    grouped_a0: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_s1: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in a0_predictions:
        grouped_a0[str(row["query_id"])].append(row)
    for row in s1_predictions:
        grouped_s1[str(row["query_id"])].append(row)
    fused, provenance = [], []
    for query in queries:
        query_id = str(query["query_id"])
        rows, sources = fuse_query(
            query,
            grouped_a0[query_id],
            grouped_s1[query_id],
            settings=settings,
        )
        fused.extend(rows)
        provenance.extend(sources)
    validation, issues = validate_predictions(queries, fused)
    if validation["status"] != "PASS":
        codes = sorted({issue["code"] for issue in issues})
        raise RuntimeError(f"BCF1_F1_PREDICTION_VALIDATION_FAILED: {codes}")
    return fused, provenance


__all__ = ["candidate_key", "fuse_predictions", "fuse_query", "normalize_qa_answer"]
