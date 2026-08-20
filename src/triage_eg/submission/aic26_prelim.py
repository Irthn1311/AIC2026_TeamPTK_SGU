"""Fail-closed AIC2026 preliminary per-query CSV submission contract."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

VIDEO = re.compile(r"^L\d+_V\d+$", re.ASCII)


def _filename(query_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", query_id):
        raise ValueError("unsafe query_id")
    return f"{query_id}.csv"


def _values(task: str, row: dict[str, Any], event_count: int | None) -> list[str]:
    video = str(row["video_id"])
    if not VIDEO.fullmatch(video) or video.endswith(".mp4"):
        raise ValueError("invalid submission video_id")
    if task == "KIS":
        return [video, str(int(row["frame_id"]))]
    if task == "QA":
        answer = str(row["answer"])
        if len(answer) > 100 or "\n" in answer or "\r" in answer:
            raise ValueError("QA answer exceeds 100 Unicode characters or contains newline")
        return [video, str(int(row["frame_id"])), answer]
    frames = [int(value) for value in row["frame_ids"]]
    if event_count is None or len(frames) != event_count:
        raise ValueError("TRAKE frame count mismatch")
    if any(left >= right for left, right in zip(frames, frames[1:], strict=False)):
        raise ValueError("TRAKE frames must be strictly increasing")
    return [video, *map(str, frames)]


def create_submission_zip(
    queries: list[dict[str, Any]], predictions: list[dict[str, Any]], output_zip: Path
) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["query_id"])].append(row)
    target = Path(output_zip)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for query in queries:
            query_id, task = str(query["query_id"]), str(query["task"]).upper()
            rows = sorted(grouped.get(query_id, []), key=lambda row: int(row["rank"]))
            if not rows or len(rows) > 100:
                raise ValueError(f"submission query {query_id} requires 1..100 rows")
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            for row in rows:
                writer.writerow(_values(task, row, query.get("event_count")))
            archive.writestr(f"submission/{_filename(query_id)}", stream.getvalue())
    validate_submission_zip(target, queries)
    return target


def validate_submission_zip(path: Path, queries: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {f"submission/{_filename(str(query['query_id']))}": query for query in queries}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if names != set(expected):
            missing = set(expected) - names
            extra = names - set(expected)
            raise ValueError(f"submission members mismatch: missing={missing} extra={extra}")
        row_count = 0
        for name, query in expected.items():
            raw = archive.read(name).decode("utf-8")
            if not raw or raw.startswith("video_id,"):
                raise ValueError("header rows and empty CSVs are forbidden")
            try:
                rows = list(csv.reader(io.StringIO(raw), strict=True))
            except csv.Error as error:
                raise ValueError(f"malformed CSV quoting: {name}") from error
            if not 1 <= len(rows) <= 100:
                raise ValueError("CSV row count outside 1..100")
            expected_columns = 2 if query["task"] == "KIS" else 3
            if query["task"] == "TRAKE":
                expected_columns = 1 + int(query["event_count"])
            for values in rows:
                if len(values) != expected_columns:
                    raise ValueError(f"wrong column count: {name}")
                _values(
                    str(query["task"]),
                    {
                        "video_id": values[0],
                        "frame_id": values[1] if query["task"] != "TRAKE" else None,
                        "answer": values[2] if query["task"] == "QA" else None,
                        "frame_ids": values[1:] if query["task"] == "TRAKE" else None,
                    },
                    query.get("event_count"),
                )
            row_count += len(rows)
    return {"status": "PASS", "query_count": len(expected), "prediction_count": row_count}
