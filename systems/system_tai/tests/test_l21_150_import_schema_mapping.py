from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any

import pytest

from system_tai.quality.l21_150_answers import (
    answer_matches,
    normalize_answer,
    source_answer_aliases,
)
from system_tai.quality.l21_150_mapping import (
    VideoMetadata,
    validate_l21_150_mapping,
)
from system_tai.quality.l21_150_schema import (
    BENCHMARK_ID,
    BENCHMARK_ROLE,
    FRAME_GT_STATUS,
    FrameInterval,
    L21150Benchmark,
    L21150FormatError,
    L21150KISQuery,
    L21150QAQuery,
    L21150TRAKEEvent,
    L21150TRAKEQuery,
    benchmark_to_payload,
    load_l21_150_benchmark,
    parse_l21_150_payload,
    serialize_l21_150_benchmark,
)

SYSTEM_ROOT = Path(__file__).parents[1]
BENCHMARK_PATH = SYSTEM_ROOT / "benchmarks" / "l21_150_diagnostic" / "benchmark.json"
MANIFEST_PATH = SYSTEM_ROOT / "benchmarks" / "l21_150_diagnostic" / "manifest.json"
IMPORT_SCRIPT = SYSTEM_ROOT / "scripts" / "l21_150_import.py"
SPEC = importlib.util.spec_from_file_location("system_tai_l21_150_import_tests", IMPORT_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)


def _benchmark(queries: tuple[Any, ...]) -> L21150Benchmark:
    return L21150Benchmark(
        schema_version=1,
        benchmark_id=BENCHMARK_ID,
        benchmark_role=BENCHMARK_ROLE,
        official_ground_truth=False,
        dataset_scope="L21 16-video subset",
        frame_gt_status=FRAME_GT_STATUS,
        description="Internal diagnostic fixture, not official BTC GT.",
        queries=queries,
    )


def _kis(
    query_id: str = "KIS-X",
    *,
    video_id: str = "L21_V001",
    start: int = 90,
    end: int = 110,
    center: int = 100,
    timestamp: str = "00:10",
    split: str = "DEV",
) -> L21150KISQuery:
    return L21150KISQuery(
        query_id=query_id,
        query_vi="Một sự kiện nhìn thấy được.",
        video_id=video_id,
        reference_timestamp=timestamp,
        proposed_frame_center=center,
        proposed_interval=FrameInterval(start, end),
        branch="Visual",
        difficulty="Dễ",
        split=split,
    )


def _qa(query_id: str = "QA-X", *, video_id: str = "L21_V001") -> L21150QAQuery:
    return L21150QAQuery(
        query_id=query_id,
        question_vi="Vật thể có màu gì?",
        video_id=video_id,
        reference_timestamp="00:10",
        proposed_frame_center=100,
        proposed_interval=FrameInterval(90, 110),
        source_answer="Đỏ / màu đỏ",
        canonical_answer="Đỏ",
        accepted_answers=("Đỏ", "màu đỏ"),
        branch="Visual",
        difficulty="Dễ",
        split="DEV",
    )


def _trake(query_id: str = "TR-X", *, video_id: str = "L21_V001") -> L21150TRAKEQuery:
    return L21150TRAKEQuery(
        query_id=query_id,
        video_id=video_id,
        events=(
            L21150TRAKEEvent(1, "Sự kiện một", "00:10", 100, FrameInterval(96, 104)),
            L21150TRAKEEvent(2, "Sự kiện hai", "00:20", 200, FrameInterval(196, 204)),
            L21150TRAKEEvent(3, "Sự kiện ba", "00:30", 300, FrameInterval(296, 304)),
        ),
        branch="Temporal + Mixed",
        difficulty="Trung bình",
        split="DEV",
    )


def _paragraph(text: str) -> str:
    chunks = text.split("\n")
    body = []
    for index, chunk in enumerate(chunks):
        if index:
            body.append("<w:br/>")
        body.append(f'<w:r><w:t xml:space="preserve">{escape(chunk)}</w:t></w:r>')
    return f"<w:p>{''.join(body)}</w:p>"


def _table(rows: list[list[str]]) -> str:
    return "<w:tbl>" + "".join(
        "<w:tr>"
        + "".join(f"<w:tc>{_paragraph(cell)}</w:tc>" for cell in row)
        + "</w:tr>"
        for row in rows
    ) + "</w:tbl>"


def _source_tables() -> list[list[list[str]]]:
    videos = [f"L21_V{value:03d}" for value in range(1, 17)]
    kis = [list(IMPORTER.EXPECTED_HEADERS[0])]
    qa = [list(IMPORTER.EXPECTED_HEADERS[1])]
    trake = [list(IMPORTER.EXPECTED_HEADERS[2])]
    for index in range(1, 51):
        video = videos[(index - 1) % len(videos)]
        center = index * 100
        kis.append(
            [
                f"KIS-{index:02d}",
                f"Tìm sự kiện số {index}.",
                video,
                f"00:{index % 60:02d}",
                f"f={center:,} | [{center - 30:,}-{center + 30:,}]",
                "Visual scene",
                "Dễ",
            ]
        )
        qa.append(
            [
                f"QA-{index:02d}",
                f"Câu hỏi số {index}?",
                video,
                f"00:{index % 60:02d}",
                f"f={center:,} | [{center - 30:,}-{center + 30:,}]",
                "Đỏ / màu đỏ" if index == 1 else f"Đáp án {index}",
                "Visual",
                "Trung bình",
            ]
        )
        trake.append(
            [
                f"TR-{index:02d}",
                "E1: sự kiện đầu → E2: sự kiện giữa → E3: sự kiện cuối",
                video,
                (
                    f"E1: 00:10 (f={center:,}; ±4f)\n"
                    f"E2: 00:20 (f={center + 100:,}; ±4f)\n"
                    f"E3: 00:30 (f={center + 200:,}; ±4f)"
                ),
                "Temporal + Mixed",
                "Khó",
            ]
        )
    return [kis, qa, trake]


def _write_docx(path: Path, tables: list[list[list[str]]]) -> Path:
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{IMPORTER.WORD_NS}"><w:body>'
        + "".join(_table(rows) for rows in tables)
        + "<w:sectPr/></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml.encode("utf-8"))
    return path


def test_committed_benchmark_has_exact_source_counts_and_videos() -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    assert len(benchmark.queries) == 150
    assert Counter(query.task_type for query in benchmark.queries) == {
        "kis": 50,
        "qa": 50,
        "trake": 50,
    }
    assert len({query.video_id for query in benchmark.queries}) == 16


def test_committed_manifest_matches_benchmark_bytes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["query_count"] == 150
    assert manifest["task_counts"] == {"kis": 50, "qa": 50, "trake": 50}
    assert manifest["source_document_sha256"] == (
        "09732560be1b467f63b4566e5ad943f461838fedb2cc650ff0f67c064c3ddaf0"
    )
    assert manifest["benchmark_sha256"] == hashlib.sha256(
        BENCHMARK_PATH.read_bytes()
    ).hexdigest()


def test_importer_is_byte_deterministic(tmp_path: Path) -> None:
    source = _write_docx(tmp_path / "source.docx", _source_tables())
    first, first_manifest = IMPORTER.import_l21_150_docx(source)
    second, second_manifest = IMPORTER.import_l21_150_docx(source)
    assert serialize_l21_150_benchmark(first) == serialize_l21_150_benchmark(second)
    assert first_manifest == second_manifest


def test_importer_rejects_duplicate_query_id(tmp_path: Path) -> None:
    tables = _source_tables()
    tables[1][1][0] = "KIS-01"
    source = _write_docx(tmp_path / "duplicate.docx", tables)
    with pytest.raises(IMPORTER.L21150ImportError, match="duplicate query IDs"):
        IMPORTER.import_l21_150_docx(source)


def test_importer_rejects_malformed_frame_reference(tmp_path: Path) -> None:
    tables = _source_tables()
    tables[0][1][4] = "not-a-frame"
    source = _write_docx(tmp_path / "malformed.docx", tables)
    with pytest.raises(IMPORTER.L21150ImportError, match="malformed frame reference"):
        IMPORTER.import_l21_150_docx(source)


def test_importer_preserves_vietnamese_and_source_answer_alternatives(tmp_path: Path) -> None:
    source = _write_docx(tmp_path / "source.docx", _source_tables())
    benchmark, _ = IMPORTER.import_l21_150_docx(source)
    assert benchmark.queries[0].query_vi == "Tìm sự kiện số 1."
    qa = benchmark.queries[50]
    assert isinstance(qa, L21150QAQuery)
    assert qa.source_answer == "Đỏ / màu đỏ"
    assert qa.accepted_answers == ("Đỏ", "màu đỏ")


def test_video_split_is_exact_deterministic_12_dev_4_holdout() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dev = [row["video_id"] for row in manifest["video_assignments"] if row["split"] == "DEV"]
    holdout = [
        row["video_id"]
        for row in manifest["video_assignments"]
        if row["split"] == "HOLDOUT"
    ]
    assert dev == [
        "L21_V005",
        "L21_V010",
        "L21_V012",
        "L21_V016",
        "L21_V009",
        "L21_V007",
        "L21_V017",
        "L21_V003",
        "L21_V011",
        "L21_V015",
        "L21_V008",
        "L21_V001",
    ]
    assert holdout == ["L21_V006", "L21_V002", "L21_V014", "L21_V013"]
    assert set(dev).isdisjoint(holdout)


def test_every_video_has_one_split_across_all_tasks() -> None:
    benchmark = load_l21_150_benchmark(BENCHMARK_PATH)
    splits: dict[str, set[str]] = {}
    for query in benchmark.queries:
        splits.setdefault(query.video_id, set()).add(query.split)
    assert all(len(value) == 1 for value in splits.values())


@pytest.mark.parametrize("field", ["unknown", "missing"])
def test_schema_rejects_unknown_or_missing_fields(field: str) -> None:
    payload = benchmark_to_payload(_benchmark((_kis(),)))
    if field == "unknown":
        payload["unexpected"] = True
    else:
        del payload["description"]
    with pytest.raises(L21150FormatError, match="fields mismatch"):
        parse_l21_150_payload(payload)


def test_schema_rejects_invalid_frames() -> None:
    payload = benchmark_to_payload(_benchmark((_kis(),)))
    payload["queries"][0]["proposed_interval"] = [10, 9]
    with pytest.raises(L21150FormatError, match="start_frame_id"):
        parse_l21_150_payload(payload)


def test_schema_rejects_qa_without_answers() -> None:
    payload = benchmark_to_payload(_benchmark((_qa(),)))
    payload["queries"][0]["accepted_answers"] = []
    with pytest.raises(L21150FormatError, match="accepted_answers"):
        parse_l21_150_payload(payload)


def test_schema_rejects_malformed_trake_event_order() -> None:
    payload = benchmark_to_payload(_benchmark((_trake(),)))
    payload["queries"][0]["events"][1]["event_index"] = 3
    with pytest.raises(L21150FormatError, match="contiguous"):
        parse_l21_150_payload(payload)


def test_schema_rejects_duplicate_query_ids() -> None:
    with pytest.raises(L21150FormatError, match="query_id values must be unique"):
        _benchmark((_kis(), replace(_kis(), query_vi="Khác")))


def test_schema_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(L21150FormatError, match="duplicate JSON key"):
        load_l21_150_benchmark(path)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  ĐỎ  ", "đỏ"),
        ("2,60 %", "2.60%"),
        ("400.000 - 600.000 đồng", "400000-600000 đồng"),
        ("“BRAVO!”", "bravo"),
    ],
)
def test_answer_normalization_is_conservative_and_deterministic(
    value: str, expected: str
) -> None:
    assert normalize_answer(value) == expected
    assert normalize_answer(value) == normalize_answer(value)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Song sắt / khung cửa sắt", ("Song sắt", "khung cửa sắt")),
        ("Cánh đồng/ruộng trồng theo luống", ("Cánh đồng", "ruộng trồng theo luống")),
        ("Đu dây / zipline", ("Đu dây", "zipline")),
        ("Vàng miếng / thỏi vàng", ("Vàng miếng", "thỏi vàng")),
    ],
)
def test_explicit_source_alternatives(source: str, expected: tuple[str, ...]) -> None:
    assert source_answer_aliases(source) == expected


def test_answer_matcher_does_not_invent_synonyms() -> None:
    assert answer_matches("màu đỏ", ("Đỏ", "màu đỏ"))
    assert not answer_matches("scarlet", ("Đỏ", "màu đỏ"))


def _write_mapping(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])
        for index, (frame_idx, pts_time) in enumerate(rows, start=1):
            writer.writerow([index, pts_time, 10, frame_idx])


def _write_raw_video_placeholder(root: Path, video_id: str = "L21_V001") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{video_id}.mp4"
    path.write_bytes(b"raw-video-fixture")
    return path


def test_mapping_validator_validates_nearest_frame_without_shift(tmp_path: Path) -> None:
    _write_mapping(tmp_path / "L21_V001.csv", [(0, 0.0), (100, 10.0), (100, 10.1)])
    report = validate_l21_150_mapping(_benchmark((_kis(),)), tmp_path)
    record = report["records"][0]
    assert record["status"] == "VALIDATED"
    assert record["nearest_mapping_frame_idx"] == 100
    assert record["nearest_keyframe_inside_proposed_interval"] is True
    assert record["keyframe_overlap_status"] == "IN_GT_INTERVAL"
    assert record["source_proposed_frame_center"] == 100
    assert record["frame_shift_applied"] == 0


def test_sparse_keyframe_outside_gt_is_only_a_proximity_diagnostic(
    tmp_path: Path,
) -> None:
    mapping_root = tmp_path / "mappings"
    video_root = tmp_path / "videos"
    video_path = _write_raw_video_placeholder(video_root)
    _write_mapping(mapping_root / "L21_V001.csv", [(711, 23.7)])
    query = _kis(start=720, end=780, center=750, timestamp="00:25")

    report = validate_l21_150_mapping(
        _benchmark((query,)),
        mapping_root,
        video_root=video_root,
        video_probe=lambda path: VideoMetadata(30.0, 37_849, path),
    )
    record = report["records"][0]

    assert report["schema_version"] == 2
    assert record["status"] == "VALIDATED"
    assert record["expected_frame_from_timestamp"] == 750
    assert record["timestamp_coordinate_consistent"] is True
    assert record["nearest_mapping_frame_idx"] == 711
    assert record["nearest_mapping_pts_time"] == pytest.approx(23.7)
    assert record["nearest_keyframe_inside_proposed_interval"] is False
    assert record["keyframe_overlap_status"] == "OUTSIDE_GT_INTERVAL"
    assert record["raw_video_path"] == str(video_path)
    assert report["coordinate_status_counts"]["VALIDATED"] == 1
    assert report["keyframe_overlap_counts"]["OUTSIDE_GT_INTERVAL"] == 1


def test_mapping_validator_reports_out_of_range(tmp_path: Path) -> None:
    mapping_root = tmp_path / "mappings"
    video_root = tmp_path / "videos"
    _write_raw_video_placeholder(video_root)
    _write_mapping(mapping_root / "L21_V001.csv", [(0, 0.0), (100, 10.0)])
    query = _kis(start=200, end=220, center=210)
    report = validate_l21_150_mapping(
        _benchmark((query,)),
        mapping_root,
        video_root=video_root,
        video_probe=lambda path: VideoMetadata(10.0, 205, path),
    )
    assert report["records"][0]["status"] == "OUT_OF_RANGE"


def test_mapping_validator_reports_missing_mapping(tmp_path: Path) -> None:
    report = validate_l21_150_mapping(_benchmark((_kis(),)), tmp_path)
    assert report["records"][0]["status"] == "MISSING_MAPPING"


def test_mapping_validator_reports_missing_video_metadata_when_requested(
    tmp_path: Path,
) -> None:
    mapping_root = tmp_path / "mappings"
    video_root = tmp_path / "videos"
    video_root.mkdir()
    _write_mapping(mapping_root / "L21_V001.csv", [(0, 0.0), (100, 10.0)])
    report = validate_l21_150_mapping(
        _benchmark((_kis(),)), mapping_root, video_root=video_root
    )
    assert report["records"][0]["status"] == "MISSING_VIDEO_METADATA"


def test_mapping_validator_uses_raw_video_bounds(tmp_path: Path) -> None:
    mapping_root = tmp_path / "mappings"
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "L21_V001.mp4"
    video_path.write_bytes(b"fixture")
    _write_mapping(mapping_root / "L21_V001.csv", [(0, 0.0), (100, 10.0)])
    report = validate_l21_150_mapping(
        _benchmark((_kis(),)),
        mapping_root,
        video_root=video_root,
        video_probe=lambda path: VideoMetadata(10.0, 105, path),
    )
    assert report["records"][0]["status"] == "OUT_OF_RANGE"


def test_whole_second_timestamp_tolerance_is_one_second(tmp_path: Path) -> None:
    mapping_root = tmp_path / "mappings"
    video_root = tmp_path / "videos"
    _write_raw_video_placeholder(video_root)
    _write_mapping(mapping_root / "L21_V001.csv", [(750, 25.0)])
    within = _kis(
        "KIS-WITHIN",
        start=740,
        end=780,
        center=779,
        timestamp="00:25",
    )
    outside = _kis(
        "KIS-OUTSIDE",
        start=740,
        end=790,
        center=781,
        timestamp="00:25",
    )

    report = validate_l21_150_mapping(
        _benchmark((within, outside)),
        mapping_root,
        video_root=video_root,
        video_probe=lambda path: VideoMetadata(30.0, 1_000, path),
    )

    assert report["timestamp_coordinate_tolerance_seconds"] == 1.0
    assert report["records"][0]["status"] == "VALIDATED"
    assert report["records"][0]["center_timestamp_delta_frames"] == 29
    assert report["records"][1]["status"] == "INVALID_COORDINATE"
    assert report["records"][1]["center_timestamp_delta_frames"] == 31


def test_trake_sparse_keyframes_do_not_invalidate_narrow_intervals(
    tmp_path: Path,
) -> None:
    mapping_root = tmp_path / "mappings"
    video_root = tmp_path / "videos"
    _write_raw_video_placeholder(video_root)
    _write_mapping(mapping_root / "L21_V001.csv", [(90, 10.0), (190, 20.0), (290, 30.0)])

    report = validate_l21_150_mapping(
        _benchmark((_trake(),)),
        mapping_root,
        video_root=video_root,
        video_probe=lambda path: VideoMetadata(10.0, 1_000, path),
    )

    assert [record["status"] for record in report["records"]] == [
        "VALIDATED",
        "VALIDATED",
        "VALIDATED",
    ]
    assert report["keyframe_overlap_counts"] == {
        "IN_GT_INTERVAL": 0,
        "OUTSIDE_GT_INTERVAL": 3,
        "UNAVAILABLE": 0,
    }


def test_invalid_mapping_is_distinct_from_missing_mapping(tmp_path: Path) -> None:
    mapping = tmp_path / "L21_V001.csv"
    mapping.write_text("n,pts_time\n1,10\n", encoding="utf-8")

    report = validate_l21_150_mapping(_benchmark((_kis(),)), tmp_path)

    assert report["records"][0]["status"] == "INVALID_MAPPING"


def test_mapping_validator_never_mutates_source_gt(tmp_path: Path) -> None:
    benchmark = _benchmark((_kis(),))
    before = serialize_l21_150_benchmark(benchmark)
    _write_mapping(tmp_path / "L21_V001.csv", [(0, 0.0), (101, 10.0)])
    report = validate_l21_150_mapping(benchmark, tmp_path)
    assert report["automatic_frame_shift_applied"] is False
    assert report["records"][0]["source_proposed_start_frame_id"] == 90
    assert report["records"][0]["source_proposed_end_frame_id"] == 110
    assert serialize_l21_150_benchmark(benchmark) == before
