"""Fail-closed parser for the official 24-query Trial P1 ZIP."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

_FILENAME = re.compile(r"^query-p1-(\d+)-(kis|qa|trake)\.txt$", re.ASCII | re.I)
_EVENT = re.compile(r"^\s*(E\d+)\s*:\s*(\S.*)\s*$", re.I)
_EXPECTED = {"KIS": 18, "QA": 3, "TRAKE": 3}


def _normalized(text: str) -> str:
    return "\n".join(
        " ".join(line.split()) for line in unicodedata.normalize("NFC", text).splitlines()
    ).strip()


def _parse_events(lines: list[str]) -> tuple[str, list[dict[str, Any]]]:
    context: list[str] = []
    events: list[dict[str, Any]] = []
    started = False
    for line_number, line in enumerate(lines, 1):
        match = _EVENT.match(line)
        if match:
            started = True
            events.append(
                {
                    "event_index": len(events),
                    "event_id": f"E{len(events) + 1}",
                    "raw_event_label": match.group(1).upper(),
                    "description": " ".join(match.group(2).split()),
                    "source_line_number": line_number,
                }
            )
        elif line.strip():
            if started:
                raise ValueError("non-event content encountered after the first TRAKE event")
            context.append(line.strip())
    if not events:
        raise ValueError("TRAKE query has no event lines")
    return " ".join(context), events


def parse_trial_zip(path: str | Path) -> dict[str, Any]:
    """Return a deterministic manifest; reject every package-contract violation."""

    source = Path(path).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with ZipFile(source) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) != 24:
            raise ValueError(f"TRIAL_FILE_COUNT_MISMATCH: expected 24, found {len(members)}")
        for info in sorted(members, key=lambda item: item.filename.casefold()):
            if "/" in info.filename or "\\" in info.filename:
                raise ValueError(f"TRIAL_NESTED_OR_UNSAFE_MEMBER: {info.filename}")
            match = _FILENAME.fullmatch(info.filename)
            if not match:
                raise ValueError(f"TRIAL_UNSUPPORTED_MEMBER: {info.filename}")
            try:
                raw_text = archive.read(info).decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"TRIAL_NOT_UTF8: {info.filename}") from error
            task = match.group(2).upper()
            query_id = info.filename[:-4]
            row: dict[str, Any] = {
                "query_id": query_id,
                "numeric_id": int(match.group(1)),
                "task": task,
                "filename": info.filename,
                "raw_text": raw_text,
                "normalized_text": _normalized(raw_text),
                "original_lines": raw_text.splitlines(keepends=True),
            }
            if not row["normalized_text"]:
                raise ValueError(f"TRIAL_EMPTY_QUERY: {info.filename}")
            if task == "TRAKE":
                context, events = _parse_events(raw_text.splitlines())
                row.update(
                    {
                        "context": context,
                        "event_count": len(events),
                        "events": events,
                        "raw_event_labels": [event["raw_event_label"] for event in events],
                    }
                )
            rows.append(row)

    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("TRIAL_DUPLICATE_QUERY_ID")
    counts = Counter(row["task"] for row in rows)
    if dict(counts) != _EXPECTED:
        raise ValueError(f"TRIAL_TASK_COUNTS_MISMATCH: {dict(counts)}")
    rows.sort(key=lambda row: (row["numeric_id"], row["task"]))
    return {
        "contract": "AIC2026_TRIAL_P1_QUERY_MANIFEST_V1",
        "source_zip": source.name,
        "query_count": len(rows),
        "task_counts": _EXPECTED,
        "queries": rows,
    }


__all__ = ["parse_trial_zip"]
