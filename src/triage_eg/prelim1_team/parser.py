"""Fail-closed parser for the current blind SOTUYEN1 query package."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

CURRENT_PACKAGE_SHA256 = "0ac8289ae4578969b80e8f6fda90d4e10e24ece1352aca61c5918137343ee8d2"
CURRENT_CONTENT_SHA256 = "3774527ba8b9313ebe1557a1eb2ae36dcb2f1458d02d156c89f7555d9848049a"
_FILENAME = re.compile(r"^query-p1-(\d+)-(kis|qa|trake)\.txt$", re.ASCII | re.I)
_EVENT = re.compile(r"^\s*E(\d+)(?:\s*[:.\-)]+\s*|\s+)(\S.*)\s*$", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return "\n".join(
        " ".join(line.split()) for line in unicodedata.normalize("NFC", text).splitlines()
    ).strip()


def _answer_type(text: str) -> str:
    folded = text.casefold()
    if any(
        value in folded
        for value in ("con số hiển thị", "con số được ghi", "số được ghi", "biển báo")
    ):
        return "VISIBLE_NUMBER"
    if "có bao nhiêu" in folded or "đếm" in folded:
        return "VISUAL_COUNT"
    if "bao nhiêu" in folded or "con số" in folded:
        return "NUMBER_OR_COUNT"
    if any(value in folded for value in ("tên của", "tên gì", "tên là gì")):
        return "LOCATION_OR_NAME"
    if any(value in folded for value in ("màu gì", "màu nào")):
        return "COLOR"
    if any(value in folded for value in ("ghi gì", "dòng chữ", "tiêu đề")):
        return "TEXT_PRESERVING"
    return "UNKNOWN_MANUAL"


def _parse_events(lines: list[str]) -> tuple[str, list[dict[str, Any]]]:
    context: list[str] = []
    events: list[dict[str, Any]] = []
    started = False
    for line_number, line in enumerate(lines, 1):
        match = _EVENT.match(line)
        if match:
            started = True
            expected = len(events) + 1
            actual = int(match.group(1))
            if actual != expected:
                raise ValueError(f"PRELIM1_TRAKE_EVENT_ORDINAL:{actual}:expected={expected}")
            events.append(
                {
                    "event_index": len(events),
                    "event_id": f"E{expected}",
                    "description": " ".join(match.group(2).split()),
                    "source_line_number": line_number,
                }
            )
        elif line.strip():
            if started:
                raise ValueError("PRELIM1_NON_EVENT_CONTENT_AFTER_EVENT")
            context.append(" ".join(line.split()))
    if not events:
        raise ValueError("PRELIM1_TRAKE_NO_EVENTS")
    return " ".join(context), events


def parse_prelim1_zip(
    path: str | Path,
    *,
    expected_sha256: str = CURRENT_PACKAGE_SHA256,
    expected_content_sha256: str = CURRENT_CONTENT_SHA256,
) -> dict[str, Any]:
    """Parse the exact current package without deduplicating identical query text."""

    source = Path(path).expanduser().resolve(strict=True)
    digest = _sha256(source)
    if expected_sha256 and digest != expected_sha256.casefold():
        raise ValueError(
            f"PRELIM1_PACKAGE_SHA256_MISMATCH: expected={expected_sha256} actual={digest}"
        )
    rows: list[dict[str, Any]] = []
    content_digest = hashlib.sha256()
    try:
        with ZipFile(source) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in members]
            if len(names) != len(set(names)):
                raise ValueError("PRELIM1_DUPLICATE_ZIP_MEMBER")
            for info in sorted(members, key=lambda item: item.filename.casefold()):
                if "/" in info.filename or "\\" in info.filename:
                    raise ValueError(f"PRELIM1_NESTED_OR_UNSAFE_MEMBER:{info.filename}")
                match = _FILENAME.fullmatch(info.filename)
                if not match:
                    raise ValueError(f"PRELIM1_UNSUPPORTED_MEMBER:{info.filename}")
                try:
                    payload = archive.read(info)
                    raw_text = payload.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(f"PRELIM1_NOT_UTF8:{info.filename}") from error
                normalized = _normalized(raw_text)
                if not normalized:
                    raise ValueError(f"PRELIM1_EMPTY_QUERY:{info.filename}")
                task = match.group(2).upper()
                query_id = info.filename[:-4]
                content_digest.update(info.filename.encode("utf-8"))
                content_digest.update(b"\0")
                content_digest.update(hashlib.sha256(payload).digest())
                content_digest.update(b"\n")
                event_lines = [line for line in raw_text.splitlines() if _EVENT.match(line)]
                if task != "TRAKE" and event_lines:
                    raise ValueError(f"PRELIM1_TASK_CONTENT_CONFLICT:{info.filename}:EVENT_LINES")
                if task == "QA" and not normalized.rstrip().endswith("?"):
                    raise ValueError(f"PRELIM1_TASK_CONTENT_CONFLICT:{info.filename}:QA_MARK")
                row: dict[str, Any] = {
                    "query_id": query_id,
                    "numeric_id": int(match.group(1)),
                    "task": task,
                    "filename": info.filename,
                    "raw_text": raw_text,
                    "normalized_text": normalized,
                    "query": normalized,
                    "language": "vi",
                }
                if task == "QA":
                    row.update(
                        {
                            "question": normalized,
                            "answer_type": _answer_type(normalized),
                            "answer_policy": "EVIDENCE_ONLY_OR_MANUAL_REVIEW",
                        }
                    )
                elif task == "TRAKE":
                    context, events = _parse_events(raw_text.splitlines())
                    row.update(
                        {
                            "context": context,
                            "event_count": len(events),
                            "events": events,
                            "event_descriptions": [
                                {"event_id": event["event_id"], "description": event["description"]}
                                for event in events
                            ],
                        }
                    )
                rows.append(row)
    except BadZipFile as error:
        raise ValueError("PRELIM1_BAD_ZIP") from error

    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("PRELIM1_DUPLICATE_QUERY_ID")
    numeric_ids = [int(row["numeric_id"]) for row in rows]
    if sorted(numeric_ids) != list(range(1, len(rows) + 1)):
        raise ValueError(f"PRELIM1_NUMERIC_ID_SET_INVALID:{sorted(numeric_ids)}")
    counts = dict(sorted(Counter(row["task"] for row in rows).items()))
    if counts != {"KIS": 20, "QA": 4, "TRAKE": 1}:
        raise ValueError(f"PRELIM1_TASK_COUNTS_INVALID:{counts}")
    rows.sort(key=lambda row: (int(row["numeric_id"]), str(row["task"])))
    content_sha256 = content_digest.hexdigest()
    if expected_content_sha256 and content_sha256 != expected_content_sha256:
        raise ValueError(
            "PRELIM1_PACKAGE_CONTENT_MISMATCH: "
            f"expected={expected_content_sha256} actual={content_sha256}"
        )
    text_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        text_groups[str(row["normalized_text"])].append(str(row["query_id"]))
    duplicate_text_groups = [ids for ids in text_groups.values() if len(ids) > 1]
    return {
        "contract": "AIC2026_PRELIM1_SOTUYEN1_QUERY_MANIFEST_V1",
        "source_zip": source.name,
        "source_zip_sha256": digest,
        "official_package_sha256": CURRENT_PACKAGE_SHA256,
        "archive_is_byte_exact_official": digest == CURRENT_PACKAGE_SHA256,
        "content_sha256": content_sha256,
        "query_count": len(rows),
        "task_counts": counts,
        "duplicate_text_groups": duplicate_text_groups,
        "duplicate_text_queries_preserved": sum(map(len, duplicate_text_groups)),
        "queries": rows,
        "ground_truth_opened": False,
        "leaderboard_used": False,
    }


__all__ = ["CURRENT_CONTENT_SHA256", "CURRENT_PACKAGE_SHA256", "parse_prelim1_zip"]
