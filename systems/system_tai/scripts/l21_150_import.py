"""Import the L21-150 diagnostic benchmark from its source DOCX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = SYSTEM_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from system_tai.quality.l21_150_answers import source_answer_aliases  # noqa: E402
from system_tai.quality.l21_150_schema import (  # noqa: E402
    BENCHMARK_ID,
    BENCHMARK_ROLE,
    FRAME_GT_STATUS,
    FrameInterval,
    L21150Benchmark,
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEEvent,
    L21150TRAKEQuery,
    serialize_l21_150_benchmark,
)

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{WORD_NS}}}"
SPLIT_SEED = "system_tai_l21_150_v1"
EXPECTED_HEADERS = (
    ("ID", "Truy vấn", "Video GT", "Thời gian GT", "Frame GT đề xuất", "Nhánh test", "Mức"),
    (
        "ID",
        "Mô tả sự kiện + câu hỏi",
        "Video GT",
        "Thời gian GT",
        "Frame GT đề xuất",
        "Đáp án",
        "Nhánh test",
        "Mức",
    ),
    (
        "ID",
        "Chuỗi sự kiện cần tìm theo thứ tự",
        "Video GT",
        "GT thời gian + frame tham chiếu",
        "Nhánh test",
        "Mức",
    ),
)
FRAME_PATTERN = re.compile(
    r"^f\s*=\s*([\d,]+)\s*\|\s*\[\s*([\d,]+)\s*-\s*([\d,]+)\s*\]$"
)
EVENT_DESCRIPTION_PATTERN = re.compile(
    r"(?:^|→)\s*E(\d+)\s*:\s*(.*?)(?=\s*→\s*E\d+\s*:|$)"
)
EVENT_REFERENCE_PATTERN = re.compile(
    r"^E(\d+)\s*:\s*([0-9]{2}:[0-9]{2})\s*"
    r"\(\s*f\s*=\s*([\d,]+)\s*;\s*±\s*(\d+)f\s*\)$"
)


class L21150ImportError(ValueError):
    """The source DOCX does not match the expected deterministic layout."""


def _clean_text(value: str) -> str:
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _cell_text(cell: ElementTree.Element) -> str:
    paragraphs: list[str] = []
    for paragraph in cell.findall(f"./{W}p"):
        parts: list[str] = []
        for element in paragraph.iter():
            if element.tag == f"{W}t" and element.text:
                parts.append(element.text)
            elif element.tag == f"{W}tab":
                parts.append("\t")
            elif element.tag in {f"{W}br", f"{W}cr"}:
                parts.append("\n")
        paragraphs.append("".join(parts))
    return _clean_text("\n".join(paragraphs))


def read_docx_tables(path: Path) -> tuple[tuple[tuple[str, ...], ...], ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"source DOCX does not exist: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise L21150ImportError("source is not a readable DOCX document") from exc
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise L21150ImportError("word/document.xml is malformed") from exc

    tables: list[tuple[tuple[str, ...], ...]] = []
    for table in root.iter(f"{W}tbl"):
        rows: list[tuple[str, ...]] = []
        for row in table.findall(f"./{W}tr"):
            rows.append(tuple(_cell_text(cell) for cell in row.findall(f"./{W}tc")))
        tables.append(tuple(rows))
    return tuple(tables)


def _number(value: str, context: str) -> int:
    compact = value.replace(",", "")
    if not compact.isdigit():
        raise L21150ImportError(f"{context} is not an integer: {value!r}")
    return int(compact)


def _frame_reference(value: str, context: str) -> tuple[int, FrameInterval]:
    match = FRAME_PATTERN.fullmatch(value)
    if match is None:
        raise L21150ImportError(f"{context} has malformed frame reference: {value!r}")
    center, start, end = (_number(part, context) for part in match.groups())
    interval = FrameInterval(start, end)
    if not interval.start_frame_id <= center <= interval.end_frame_id:
        raise L21150ImportError(f"{context} center is outside its proposed interval")
    return center, interval


def _video_splits(video_ids: set[str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    ranked = sorted(
        (
            hashlib.sha256(f"{SPLIT_SEED}|{video_id}".encode()).hexdigest(),
            video_id,
        )
        for video_id in video_ids
    )
    if len(ranked) != 16:
        raise L21150ImportError(f"expected 16 unique videos, found {len(ranked)}")
    assignments: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    for rank, (digest, video_id) in enumerate(ranked, start=1):
        split = "DEV" if rank <= 12 else "HOLDOUT"
        assignments[video_id] = split
        manifest_rows.append(
            {
                "split_rank": rank,
                "video_id": video_id,
                "split_hash": digest,
                "split": split,
            }
        )
    return assignments, manifest_rows


def _validate_table_shape(tables: tuple[tuple[tuple[str, ...], ...], ...]) -> None:
    if len(tables) != 3:
        raise L21150ImportError(f"expected exactly 3 tables, found {len(tables)}")
    for index, (table, expected_header) in enumerate(zip(tables, EXPECTED_HEADERS)):
        if len(table) != 51:
            raise L21150ImportError(
                f"table {index} must contain one header and 50 data rows"
            )
        if table[0] != expected_header:
            raise L21150ImportError(
                f"table {index} header mismatch: expected {expected_header}, got {table[0]}"
            )
        for row_number, row in enumerate(table[1:], start=2):
            if len(row) != len(expected_header):
                raise L21150ImportError(
                    f"table {index} row {row_number} has {len(row)} cells; "
                    f"expected {len(expected_header)}"
                )
            if any(not cell for cell in row):
                raise L21150ImportError(f"table {index} row {row_number} contains an empty cell")


def _parse_trake_events(
    descriptions: str,
    references: str,
    context: str,
) -> tuple[L21150TRAKEEvent, ...]:
    description_parts = EVENT_DESCRIPTION_PATTERN.findall(descriptions)
    reference_lines = references.splitlines()
    if not description_parts or len(description_parts) != len(reference_lines):
        raise L21150ImportError(f"{context} event description/reference count mismatch")

    events: list[L21150TRAKEEvent] = []
    for offset, ((description_index, description), reference_line) in enumerate(
        zip(description_parts, reference_lines), start=1
    ):
        reference_match = EVENT_REFERENCE_PATTERN.fullmatch(reference_line)
        if reference_match is None:
            raise L21150ImportError(
                f"{context} has malformed event reference: {reference_line!r}"
            )
        reference_index, timestamp, center_text, radius_text = reference_match.groups()
        if int(description_index) != offset or int(reference_index) != offset:
            raise L21150ImportError(f"{context} event indexes must be contiguous and ordered")
        center = _number(center_text, context)
        radius = _number(radius_text, context)
        events.append(
            L21150TRAKEEvent(
                event_index=offset,
                description_vi=description.strip(),
                reference_timestamp=timestamp,
                proposed_frame_center=center,
                proposed_interval=FrameInterval(max(0, center - radius), center + radius),
            )
        )
    return tuple(events)


def import_l21_150_docx(path: Path) -> tuple[L21150Benchmark, dict[str, Any]]:
    tables = read_docx_tables(path)
    _validate_table_shape(tables)
    video_ids = {row[2] for table in tables for row in table[1:]}
    splits, split_rows = _video_splits(video_ids)

    queries: list[L21150KISQuery | L21150QAQuery | L21150TRAKEQuery] = []
    for row_number, row in enumerate(tables[0][1:], start=2):
        query_id, query_vi, video_id, timestamp, frame_text, branch, difficulty = row
        center, interval = _frame_reference(frame_text, f"KIS row {row_number}")
        queries.append(
            L21150KISQuery(
                query_id=query_id,
                query_vi=query_vi,
                video_id=video_id,
                reference_timestamp=timestamp,
                proposed_frame_center=center,
                proposed_interval=interval,
                branch=branch,
                difficulty=difficulty,
                split=splits[video_id],
            )
        )

    for row_number, row in enumerate(tables[1][1:], start=2):
        (
            query_id,
            question_vi,
            video_id,
            timestamp,
            frame_text,
            source_answer,
            branch,
            difficulty,
        ) = row
        center, interval = _frame_reference(frame_text, f"QA row {row_number}")
        aliases = source_answer_aliases(source_answer)
        queries.append(
            L21150QAQuery(
                query_id=query_id,
                question_vi=question_vi,
                video_id=video_id,
                reference_timestamp=timestamp,
                proposed_frame_center=center,
                proposed_interval=interval,
                source_answer=source_answer,
                canonical_answer=aliases[0],
                accepted_answers=aliases,
                branch=branch,
                difficulty=difficulty,
                split=splits[video_id],
            )
        )

    for row_number, row in enumerate(tables[2][1:], start=2):
        query_id, descriptions, video_id, references, branch, difficulty = row
        queries.append(
            L21150TRAKEQuery(
                query_id=query_id,
                video_id=video_id,
                events=_parse_trake_events(descriptions, references, f"TRAKE row {row_number}"),
                branch=branch,
                difficulty=difficulty,
                split=splits[video_id],
            )
        )

    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise L21150ImportError("source contains duplicate query IDs")
    task_counts = Counter(query.task_type for query in queries)
    if len(queries) != 150 or task_counts != {"kis": 50, "qa": 50, "trake": 50}:
        raise L21150ImportError(
            f"expected 150 queries and 50/50/50 tasks, got {len(queries)} and {task_counts}"
        )

    benchmark = L21150Benchmark(
        schema_version=1,
        benchmark_id=BENCHMARK_ID,
        benchmark_role=BENCHMARK_ROLE,
        official_ground_truth=False,
        dataset_scope="L21 16-video subset",
        frame_gt_status=FRAME_GT_STATUS,
        description=(
            "Internal diagnostic benchmark imported from the human-reviewed L21-150 "
            "source. Proposed KIS/Q&A intervals use the source ±1 second trial windows; "
            "TRAKE intervals use ±4 frames. This is not official BTC ground truth, and "
            "no ASR transcript ground truth is fabricated."
        ),
        queries=tuple(queries),
    )
    benchmark_bytes = serialize_l21_150_benchmark(benchmark)
    source_path = Path(path)
    manifest = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_role": BENCHMARK_ROLE,
        "official_ground_truth": False,
        "source_document_name": source_path.name,
        "source_document_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "importer_version": "l21_150_import_v1",
        "query_count": len(queries),
        "task_counts": {task: task_counts[task] for task in ("kis", "qa", "trake")},
        "dataset_scope": "L21 16-video subset",
        "unique_video_count": len(video_ids),
        "frame_gt_status": FRAME_GT_STATUS,
        "split_seed": SPLIT_SEED,
        "split_algorithm": "sha256(seed + '|' + video_id); first 12 DEV, last 4 HOLDOUT",
        "dev_video_count": 12,
        "holdout_video_count": 4,
        "video_assignments": split_rows,
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
    }
    return benchmark, manifest


def _write_manifest(payload: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-docx", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark, manifest = import_l21_150_docx(args.source_docx)
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_bytes(serialize_l21_150_benchmark(benchmark))
        _write_manifest(manifest, args.manifest_output)
    except (FileNotFoundError, L21150ImportError, ValueError, OSError) as exc:
        print(f"L21-150 import failed: {exc}", file=sys.stderr)
        return 2
    print(
        "L21-150 import complete: "
        f"queries={len(benchmark.queries)} benchmark_sha256={manifest['benchmark_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
