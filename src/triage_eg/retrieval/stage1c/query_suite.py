"""Deterministic loading, validation, filtering, and fingerprinting of Stage 1C queries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from triage_eg.retrieval.stage1c.contracts import QUERY_SUITE_VERSION, QueryRecord


def query_suite_fingerprint(records: list[QueryRecord]) -> str:
    canonical = [asdict(record) for record in sorted(records, key=lambda item: item.query_id)]
    payload = {
        "query_suite_version": QUERY_SUITE_VERSION,
        "queries": canonical,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_query_suite(records: list[QueryRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("QUERY_SUITE_INVALID: suite is empty")
    ids = [item.query_id for item in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise ValueError(f"QUERY_SUITE_INVALID: duplicate query_id {duplicates}")
    normalized_texts = [item.text.strip() for item in records]
    if len(normalized_texts) != len(set(normalized_texts)):
        raise ValueError("QUERY_SUITE_INVALID: duplicate exact query text")
    pairs: defaultdict[str, list[QueryRecord]] = defaultdict(list)
    for item in records:
        pairs[item.pair_id].append(item)
    for pair_id, items in sorted(pairs.items()):
        languages = sorted(item.language for item in items)
        if languages != ["en", "vi"]:
            raise ValueError(
                f"QUERY_PAIR_INVALID: {pair_id} must contain exactly one en and one vi query"
            )
    return {
        "version": QUERY_SUITE_VERSION,
        "fingerprint": query_suite_fingerprint(records),
        "query_count": len(records),
        "pair_count": len(pairs),
        "languages": sorted({item.language for item in records}),
        "categories": sorted({item.category for item in records}),
    }


def load_query_suite(path: str | Path) -> tuple[list[QueryRecord], dict[str, object]]:
    source = Path(path).expanduser().resolve(strict=True)
    records: list[QueryRecord] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"QUERY_SUITE_INVALID: malformed JSONL at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"QUERY_SUITE_INVALID: line {line_number} is not an object")
            records.append(QueryRecord.from_dict(value))
    return records, validate_query_suite(records)


def filter_query_suite(
    records: list[QueryRecord],
    *,
    query_ids: tuple[str, ...] = (),
    languages: tuple[str, ...] = (),
    categories: tuple[str, ...] = (),
) -> list[QueryRecord]:
    requested_ids = set(query_ids)
    if requested_ids:
        missing = sorted(requested_ids - {item.query_id for item in records})
        if missing:
            raise ValueError(f"QUERY_SUITE_INVALID: unknown query_ids {missing}")
    selected = [
        item
        for item in records
        if (not requested_ids or item.query_id in requested_ids)
        and (not languages or item.language in languages)
        and (not categories or item.category in categories)
    ]
    if not selected:
        raise ValueError("QUERY_SUITE_INVALID: filters selected no queries")
    return selected

