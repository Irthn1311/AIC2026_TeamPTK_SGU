"""Fail-closed audit and repack of the frozen external multimodal archive.

This module deliberately treats the archive as untrusted input.  It never extracts
the archive tree wholesale and never changes the frozen visual retrieval assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tarfile
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .contracts import (
    ASR_SOURCE_TYPE,
    EMBEDDING_MODEL,
    OBJECT_SOURCE_TYPE,
    OCR_SOURCE_TYPE,
    PROVENANCE_LEVEL,
    ImportResult,
)

EXPECTED_SOURCE_SHA256 = "1b1493a46fd167fae80ed826ffe31b91df52ef644da3dd456435b01a7f979c26"
ASR_CORPUS_SUFFIX = "artifacts/indexes/asr_v3/l21_asr_v3_corpus.parquet"
ASR_INDEX_SUFFIX = "artifacts/indexes/asr_v3/l21_asr_v3_flat_ip.faiss"
ASR_METADATA_SUFFIX = "artifacts/indexes/asr_v3/l21_asr_v3_metadata.json"
BTC_MAP_SUFFIX = "artifacts/keyframe_btc_full/indexes/keyframe_btc_global_map.parquet"
OBJECT_CORPUS_SUFFIX = "artifacts/keyframe_btc_full/object_btc/l21_objects_btc.parquet"
OBJECT_STATS_SUFFIX = "artifacts/keyframe_btc_full/object_btc/l21_objects_btc_stats.json"
NESTED_INDEX_ZIP_SUFFIX = "artifacts/kaggle_outputs_indices.zip"
TIMESTAMP_TOLERANCE_SECONDS = 1.0


def _sha256_path(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(handle: BinaryIO, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while block := handle.read(block_size):
        digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL_ROW_NOT_OBJECT:{path}:{number}")
        rows.append(row)
    return rows


def _safe_tar_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _member_type(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "file"
    if member.isdir():
        return "directory"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    return "other"


def _find_member(members: list[tarfile.TarInfo], suffix: str) -> tarfile.TarInfo:
    matches = [member for member in members if member.isfile() and member.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"EXPECTED_EXACTLY_ONE_ARCHIVE_MEMBER:{suffix}:{len(matches)}")
    return matches[0]


def _copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> str:
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"ARCHIVE_MEMBER_NOT_READABLE:{member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source, target.open("wb") as output:
        while block := source.read(8 * 1024 * 1024):
            output.write(block)
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    return tuple(dict.fromkeys(re.findall(r"[^\W_]+", normalized, re.UNICODE)))


def _load_durations(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    rows = _read_jsonl(path)
    durations: dict[str, float] = {}
    for row in rows:
        video_id = str(row.get("video_id", ""))
        duration = row.get("duration_seconds")
        if not video_id or video_id in durations or not isinstance(duration, int | float):
            raise RuntimeError(f"CANONICAL_VIDEO_MANIFEST_INVALID:{video_id}")
        duration_float = float(duration)
        if not math.isfinite(duration_float) or duration_float <= 0:
            raise RuntimeError(f"CANONICAL_VIDEO_DURATION_INVALID:{video_id}:{duration}")
        durations[video_id] = duration_float
    if len(durations) != 873:
        raise RuntimeError(f"CANONICAL_VIDEO_COUNT_NOT_873:{len(durations)}")
    return durations, {
        "path": str(path.resolve()),
        "sha256": _sha256_path(path),
        "video_count": len(durations),
        "duration_field": "duration_seconds",
        "source": "STAGE0_FFPROBE_VIDEO_MANIFEST",
    }


def _pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PYARROW_REQUIRED_FOR_EXTERNAL_V3_IMPORT") from error
    return pa, pq


def _table_rows(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    _, pq = _pyarrow_modules()
    table = pq.read_table(path)
    return table, table.to_pylist()


def _archive_audit(
    source: Path, output_root: Path, members: list[tarfile.TarInfo]
) -> dict[str, Any]:
    names = [member.name for member in members]
    unsafe = sorted(name for name in names if not _safe_tar_name(name))
    link_members = sorted(member.name for member in members if member.issym() or member.islnk())
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    inventory_path = output_root / "source_archive_inventory.jsonl"
    _write_jsonl(
        inventory_path,
        (
            {
                "name": member.name,
                "type": _member_type(member),
                "size_bytes": member.size,
                "mode": oct(member.mode),
            }
            for member in members
        ),
    )
    result = {
        "source_path": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": _sha256_path(source),
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "member_count": len(members),
        "file_count": sum(member.isfile() for member in members),
        "uncompressed_file_bytes": sum(member.size for member in members if member.isfile()),
        "unsafe_path_count": len(unsafe),
        "unsafe_paths": unsafe,
        "link_member_count": len(link_members),
        "link_members": link_members,
        "duplicate_member_name_count": len(duplicates),
        "duplicate_member_names": duplicates,
        "inventory_path": str(inventory_path.resolve()),
    }
    if result["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "SOURCE_ARCHIVE_SHA256_MISMATCH:"
            f"expected={EXPECTED_SOURCE_SHA256}:actual={result['source_sha256']}"
        )
    if unsafe or link_members or duplicates:
        raise RuntimeError("SOURCE_ARCHIVE_SAFETY_GATE_FAILED")
    _write_json(output_root / "source_archive_audit.json", result)
    return result


def _audit_nested_zip(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    temporary_root: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    nested_path = temporary_root / "kaggle_outputs_indices.zip"
    nested_sha = _copy_member(archive, member, nested_path)
    with zipfile.ZipFile(nested_path) as nested:
        bad_crc = nested.testzip()
        infos = nested.infolist()
        unsafe = sorted(
            info.filename
            for info in infos
            if not _safe_tar_name(info.filename) or info.is_dir() and info.filename.startswith("/")
        )
        duplicates = sorted(
            name for name, count in Counter(info.filename for info in infos).items() if count > 1
        )
        inventory = [
            {
                "name": info.filename,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "is_directory": info.is_dir(),
            }
            for info in infos
        ]
    result = {
        "source_member": member.name,
        "source_member_size_bytes": member.size,
        "sha256": nested_sha,
        "member_count": len(inventory),
        "crc_status": "PASS" if bad_crc is None else "FAIL",
        "first_bad_crc_member": bad_crc,
        "unsafe_path_count": len(unsafe),
        "unsafe_paths": unsafe,
        "duplicate_member_name_count": len(duplicates),
        "duplicate_member_names": duplicates,
        "members": inventory,
    }
    if bad_crc is not None or unsafe or duplicates:
        raise RuntimeError("NESTED_ZIP_SAFETY_OR_CRC_GATE_FAILED")
    _write_json(output_root / "nested_zip_audit.json", result)
    return nested_path, result


def _clean_asr(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    durations: dict[str, float],
    duration_provenance: dict[str, Any],
    temporary_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    corpus_member = _find_member(members, ASR_CORPUS_SUFFIX)
    index_member = _find_member(members, ASR_INDEX_SUFFIX)
    metadata_member = _find_member(members, ASR_METADATA_SUFFIX)
    source_corpus = temporary_root / "source_asr_corpus.parquet"
    source_metadata = temporary_root / "source_asr_metadata.json"
    source_index = temporary_root / "source_asr_index.faiss"
    corpus_sha = _copy_member(archive, corpus_member, source_corpus)
    metadata_sha = _copy_member(archive, metadata_member, source_metadata)
    index_sha = _copy_member(archive, index_member, source_index)
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    table, rows = _table_rows(source_corpus)
    schema = [f"{field.name}:{field.type}" for field in table.schema]
    vector_count = int(metadata.get("total_chunks", -1))
    dimension = int(metadata.get("dimension", metadata.get("dim", 384)))
    if vector_count < 1 or vector_count > len(rows):
        raise RuntimeError(f"ASR_E5_VECTOR_COUNT_INVALID:{vector_count}:{len(rows)}")

    original_ids = [str(row.get("chunk_id", "")) for row in rows]
    duplicate_ids = sorted(name for name, count in Counter(original_ids).items() if count > 1)
    cleaned: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    seen: set[str] = set()
    video_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_time_rows: list[int] = []
    excluded_invalid_rows: list[dict[str, Any]] = []
    duration_exceeds: list[dict[str, Any]] = []
    unknown_videos: set[str] = set()
    no_speech_count = 0
    indexed_count = 0

    for source_row_index, source_row in enumerate(rows):
        video_id = str(source_row.get("video_id", ""))
        chunk_id = str(source_row.get("chunk_id", ""))
        text = str(
            source_row.get("text_clean")
            or source_row.get("text_normalized")
            or source_row.get("text")
            or ""
        ).strip()
        try:
            start = float(source_row.get("start"))
            end = float(source_row.get("end"))
        except (TypeError, ValueError):
            start = end = math.nan
        is_no_speech = text == "[NO_SPEECH]"
        is_indexed = source_row_index < vector_count
        if is_no_speech:
            no_speech_count += 1
        if video_id not in durations:
            unknown_videos.add(video_id)
        if not math.isfinite(start + end) or start < 0 or end < start:
            invalid_time_rows.append(source_row_index)
            excluded_invalid_rows.append(
                {
                    "source_row_index": source_row_index,
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "reason": "INVALID_TIMESTAMP_ORDER",
                    "semantic_vector_row_excluded": is_indexed,
                }
            )
            # Never invent or swap timestamps.  The source row and its aligned vector are
            # quarantined; the supplied FAISS file remains byte-exact with an exclusion mask.
            continue
        elif video_id in durations and end > durations[video_id] + TIMESTAMP_TOLERANCE_SECONDS:
            duration_exceeds.append(
                {
                    "source_row_index": source_row_index,
                    "video_id": video_id,
                    "end_seconds": end,
                    "duration_seconds": durations[video_id],
                    "excess_seconds": end - durations[video_id],
                }
            )
        repaired_from = None
        if chunk_id in seen:
            if is_no_speech and not is_indexed and video_id in durations:
                repaired_from = chunk_id
                chunk_id = f"{video_id}_nospeech_0000"
                repaired.append(
                    {
                        "source_row_index": source_row_index,
                        "video_id": video_id,
                        "old_chunk_id": repaired_from,
                        "new_chunk_id": chunk_id,
                        "semantic_indexed": False,
                    }
                )
            else:
                raise RuntimeError(f"ASR_DUPLICATE_SEMANTIC_CHUNK_NOT_REPAIRABLE:{chunk_id}")
        if not chunk_id or chunk_id in seen:
            raise RuntimeError(f"ASR_CLEAN_CHUNK_ID_INVALID:{source_row_index}:{chunk_id}")
        seen.add(chunk_id)
        if is_indexed:
            indexed_count += 1
            if is_no_speech or repaired_from is not None:
                raise RuntimeError(f"ASR_E5_INDEX_CONTAINS_PLACEHOLDER:{source_row_index}")
        clean = {
            "source_type": ASR_SOURCE_TYPE,
            "video_id": video_id,
            "chunk_id": chunk_id,
            "start_seconds": start,
            "end_seconds": end,
            "text": text,
            "status": "NO_SPEECH" if is_no_speech else "PASS",
            "is_no_speech": is_no_speech,
            "semantic_indexed": is_indexed,
            "e5_row_index": source_row_index if is_indexed else None,
            "source_row_index": source_row_index,
            "source_chunk_id": repaired_from or chunk_id,
            "retrieval_metadata": {
                "words_count": source_row.get("words_count"),
                "avg_probability": source_row.get("avg_probability"),
                "segment_ids": source_row.get("segment_ids"),
            },
            "provenance_level": PROVENANCE_LEVEL,
        }
        cleaned.append(clean)
        video_rows[video_id].append(clean)

    canonical = set(durations)
    represented = set(video_rows)
    useful_videos = {
        video_id
        for video_id, video_chunks in video_rows.items()
        if any(row["status"] == "PASS" and row["text"].strip() for row in video_chunks)
    }
    monotonic_failures: list[str] = []
    for video_id, video_chunks in video_rows.items():
        ordered = sorted(
            video_chunks,
            key=lambda row: (
                row["start_seconds"],
                row["end_seconds"],
                row["chunk_id"],
                row["source_row_index"],
            ),
        )
        starts = [row["start_seconds"] for row in ordered]
        if any(right < left for left, right in zip(starts, starts[1:], strict=False)):
            monotonic_failures.append(video_id)

    e5_rows = [row for row in cleaned if row["semantic_indexed"]]
    excluded_vector_rows = sorted(
        row["source_row_index"]
        for row in excluded_invalid_rows
        if row["semantic_vector_row_excluded"]
    )
    e5_alignment = (
        indexed_count == vector_count - len(excluded_vector_rows)
        and len({row["chunk_id"] for row in e5_rows}) == indexed_count
        and all(row["semantic_indexed"] for row in e5_rows)
        and all(not row["is_no_speech"] for row in e5_rows)
        and len({int(row["e5_row_index"]) for row in e5_rows}) == indexed_count
    )
    gates = {
        "canonical_video_count_873": len(canonical) == 873,
        "all_canonical_videos_represented": represented == canonical,
        "no_unknown_videos": not unknown_videos,
        "finite_nonnegative_ordered_timestamps": not any(
            not math.isfinite(row["start_seconds"] + row["end_seconds"])
            or row["start_seconds"] < 0
            or row["end_seconds"] < row["start_seconds"]
            for row in cleaned
        ),
        "per_video_monotonic_after_deterministic_sort": not monotonic_failures,
        "timestamps_within_known_duration_tolerance": not duration_exceeds,
        "useful_text_substantial_majority": len(useful_videos) / len(canonical) > 0.5,
        "no_speech_explicit": no_speech_count == 14,
        "semantic_chunk_ids_unique_after_repair": len(seen) == len(cleaned),
        "e5_nonplaceholder_rows_align_exactly": e5_alignment,
    }
    if not all(gates.values()):
        failed = sorted(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"EXTERNAL_ASR_HARD_GATE_FAILED:{failed}")

    transcripts_path = output_root / "asr_external_v3_transcripts.jsonl"
    _write_jsonl(transcripts_path, cleaned)
    chunk_table = {
        row["chunk_id"]: {
            "video_id": row["video_id"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "text": row["text"],
            "e5_row_index": row["e5_row_index"],
        }
        for row in e5_rows
    }
    postings: dict[str, list[str]] = defaultdict(list)
    for row in e5_rows:
        for token in _tokens(row["text"]):
            postings[token].append(row["chunk_id"])
    lexical_path = output_root / "asr_external_v3_lexical_index.json"
    _write_json(
        lexical_path,
        {
            "schema_version": "ASR_EXTERNAL_V3_LEXICAL_V1",
            "source_type": ASR_SOURCE_TYPE,
            "chunk_count": len(chunk_table),
            "term_count": len(postings),
            "chunks": chunk_table,
            "postings": dict(sorted(postings.items())),
        },
    )
    cleaned_corpus_path = output_root / "asr_external_v3_e5_corpus.parquet"
    _, pq = _pyarrow_modules()
    pq.write_table(table.slice(0, vector_count), cleaned_corpus_path, compression="zstd")
    cleaned_index_path = output_root / "asr_external_v3_e5_flat_ip.faiss"
    shutil.copyfile(source_index, cleaned_index_path)
    e5_manifest_path = output_root / "asr_external_v3_e5_manifest.json"
    e5_manifest = {
        "source_type": ASR_SOURCE_TYPE,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_exact_revision": None,
        "provenance_level": PROVENANCE_LEVEL,
        "metric": metadata.get("metric", "inner_product"),
        "normalized": metadata.get("normalized"),
        "dimension": dimension,
        "source_vector_count": vector_count,
        "vector_count": indexed_count,
        "usable_vector_count": indexed_count,
        "source_corpus_row_count": len(rows),
        "corpus_row_count": vector_count,
        "alignment_status": "PASS_WITH_EXCLUSION_MASK",
        "excluded_vector_rows": excluded_vector_rows,
        "excluded_vector_row_count": len(excluded_vector_rows),
        "placeholder_rows_excluded": len(rows) - vector_count,
        "source_members": {
            "corpus": corpus_member.name,
            "index": index_member.name,
            "metadata": metadata_member.name,
        },
        "source_sha256": {
            "corpus": corpus_sha,
            "index": index_sha,
            "metadata": metadata_sha,
        },
        "clean_files": {
            cleaned_corpus_path.name: {
                "size_bytes": cleaned_corpus_path.stat().st_size,
                "sha256": _sha256_path(cleaned_corpus_path),
            },
            cleaned_index_path.name: {
                "size_bytes": cleaned_index_path.stat().st_size,
                "sha256": _sha256_path(cleaned_index_path),
            },
        },
    }
    _write_json(e5_manifest_path, e5_manifest)

    coverage_path = output_root / "asr_external_v3_video_coverage.json"
    coverage = {
        "source_type": ASR_SOURCE_TYPE,
        "canonical_video_count": len(canonical),
        "represented_video_count": len(represented),
        "useful_transcript_video_count": len(useful_videos),
        "no_speech_only_video_count": len(canonical - useful_videos),
        "missing_video_ids": sorted(canonical - represented),
        "unknown_video_ids": sorted(unknown_videos),
        "coverage_complete": represented == canonical,
    }
    _write_json(coverage_path, coverage)
    timestamp_path = output_root / "asr_external_v3_timestamp_audit.json"
    timestamp_audit = {
        "source_type": ASR_SOURCE_TYPE,
        "decode_tolerance_seconds": TIMESTAMP_TOLERANCE_SECONDS,
        "duration_provenance": duration_provenance,
        "row_count": len(cleaned),
        "source_invalid_timestamp_row_count": len(invalid_time_rows),
        "invalid_timestamp_source_rows": invalid_time_rows,
        "excluded_invalid_rows": excluded_invalid_rows,
        "clean_invalid_timestamp_row_count": 0,
        "duration_exceed_count": len(duration_exceeds),
        "duration_exceeds": duration_exceeds,
        "monotonic_failure_video_count": len(monotonic_failures),
        "monotonic_failure_video_ids": monotonic_failures,
        "frame_mapping_policy": "SECONDS_TO_CANONICAL_FRAMEMAP_BTC_ONLY",
        "seconds_times_fps_forbidden": True,
        "status": "PASS_WITH_SOURCE_ROW_EXCLUSIONS",
    }
    _write_json(timestamp_path, timestamp_audit)
    provenance_path = output_root / "asr_external_v3_provenance.json"
    provenance = {
        "source_type": ASR_SOURCE_TYPE,
        "transcription_engine": "UNKNOWN_EXTERNAL",
        "transcription_exact_revision": None,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_exact_revision": None,
        "provenance_level": PROVENANCE_LEVEL,
        "is_whisper_v12_exact": False,
        "source_archive_member_paths": [
            corpus_member.name,
            index_member.name,
            metadata_member.name,
        ],
        "duplicate_source_chunk_ids": duplicate_ids,
        "deterministic_repairs": repaired,
        "deterministic_exclusions": excluded_invalid_rows,
        "semantic_reembedding_performed": False,
    }
    _write_json(provenance_path, provenance)
    report_path = output_root / "asr_external_v3_import_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# ASR External V3 Import Report",
                "",
                "Decision: **ACCEPT_EXTERNAL_ASR_WITH_WARNINGS**",
                "",
                f"- Source type: `{ASR_SOURCE_TYPE}`",
                f"- Clean transcript rows: {len(cleaned):,}",
                f"- Usable E5-indexed rows: {indexed_count:,}/{vector_count:,}",
                f"- Canonical videos represented: {len(represented)}/{len(canonical)}",
                f"- Videos with useful transcript text: {len(useful_videos)}/{len(canonical)}",
                f"- Explicit no-speech placeholders: {no_speech_count}",
                f"- Deterministically repaired placeholder IDs: {len(repaired)}",
                "- Invalid source timestamp rows quarantined without repair: "
                f"{len(invalid_time_rows)}",
                "- Timestamp gate: PASS against Stage 0 ffprobe durations (+1.0 s tolerance)",
                "- E5 row alignment gate: PASS; supplied vectors preserved without re-embedding",
                "- Warning: transcription engine/checkpoint and E5 exact revision are absent.",
                "- This artifact is not and must never be aliased to `ASR_V12_EXACT`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    audit = {
        "decision": "ACCEPT_EXTERNAL_ASR_WITH_WARNINGS",
        "source_type": ASR_SOURCE_TYPE,
        "schema": schema,
        "source_row_count": len(rows),
        "clean_row_count": len(cleaned),
        "semantic_indexed_row_count": indexed_count,
        "original_duplicate_chunk_ids": duplicate_ids,
        "repaired_placeholder_count": len(repaired),
        "gates": gates,
        "coverage": coverage,
        "timestamp_summary": {
            "invalid": len(invalid_time_rows),
            "invalid_excluded_from_clean_retrieval": len(excluded_invalid_rows),
            "duration_exceeds": len(duration_exceeds),
            "monotonic_failures": len(monotonic_failures),
        },
        "e5_manifest": e5_manifest,
    }
    _write_json(output_root / "asr_external_v3_audit.json", audit)
    return (
        audit,
        {
            "transcripts": transcripts_path,
            "lexical": lexical_path,
            "e5_manifest": e5_manifest_path,
            "e5_corpus": cleaned_corpus_path,
            "e5_index": cleaned_index_path,
            "coverage": coverage_path,
            "timestamp": timestamp_path,
            "provenance": provenance_path,
            "report": report_path,
        },
        cleaned,
    )


def _audit_ocr(nested_zip: Path, output_root: Path) -> dict[str, Any]:
    _, pq = _pyarrow_modules()
    with zipfile.ZipFile(nested_zip) as archive:
        corpus_infos = [
            info
            for info in archive.infolist()
            if info.filename.endswith("l21_ocr_temporal_v3_corpus.parquet")
            and info.file_size > 1_000_000
        ]
        index_infos = [
            info
            for info in archive.infolist()
            if info.filename.endswith("l21_ocr_temporal_v3_flat_ip.faiss")
            and info.file_size > 1_000_000
        ]
        if len(corpus_infos) != 1 or len(index_infos) != 1:
            raise RuntimeError(
                f"OCR_REAL_INDEX_DISCOVERY_FAILED:corpus={len(corpus_infos)}:index={len(index_infos)}"
            )
        with archive.open(corpus_infos[0]) as handle:
            corpus_bytes = handle.read()
        import io

        table = pq.read_table(io.BytesIO(corpus_bytes))
        rows = table.to_pylist()
        video_ids = {str(row.get("video_id", "")) for row in rows if row.get("video_id")}
        text_fields = (
            "corrected_text",
            "combined_text",
            "text_consensus",
            "semantic_search_text",
            "text_search",
        )
        nonempty = sum(
            any(str(row.get(field, "")).strip() for field in text_fields) for row in rows
        )
        status_counts = Counter(str(row.get("ocr_status", "UNKNOWN")) for row in rows)
        result = {
            "source_type": OCR_SOURCE_TYPE,
            "decision": "ACCEPT_PARTIAL",
            "source_container_member": NESTED_INDEX_ZIP_SUFFIX,
            "corpus_member": corpus_infos[0].filename,
            "index_member": index_infos[0].filename,
            "corpus_size_bytes": corpus_infos[0].file_size,
            "index_size_bytes": index_infos[0].file_size,
            "row_count": len(rows),
            "covered_video_count": len(video_ids),
            "canonical_video_count": 873,
            "coverage_rate": len(video_ids) / 873,
            "nonempty_text_row_count": nonempty,
            "empty_text_row_count": len(rows) - nonempty,
            "schema": [f"{field.name}:{field.type}" for field in table.schema],
            "status_counts": dict(sorted(status_counts.items())),
            "policy": {
                "evidence_only": True,
                "direct_final_qa_answer_forbidden": True,
                "requires_evidence_sufficiency_and_qwen_verifier": True,
                "does_not_block_asr_acceptance": True,
            },
            "warning": "Partial coverage and source backend failures; nested ZIP is authoritative.",
        }
    _write_json(output_root / "ocr_external_partial_manifest.json", result)
    _write_json(
        output_root / "ocr_external_partial_coverage.json",
        {
            key: result[key]
            for key in (
                "source_type",
                "decision",
                "covered_video_count",
                "canonical_video_count",
                "coverage_rate",
                "nonempty_text_row_count",
                "empty_text_row_count",
                "status_counts",
            )
        },
    )
    return result


def _audit_objects(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    temporary_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    object_member = _find_member(members, OBJECT_CORPUS_SUFFIX)
    stats_member = _find_member(members, OBJECT_STATS_SUFFIX)
    map_member = _find_member(members, BTC_MAP_SUFFIX)
    object_path = temporary_root / "objects.parquet"
    map_path = temporary_root / "btc_map.parquet"
    stats_path = temporary_root / "object_stats.json"
    object_sha = _copy_member(archive, object_member, object_path)
    map_sha = _copy_member(archive, map_member, map_path)
    stats_sha = _copy_member(archive, stats_member, stats_path)
    _, pq = _pyarrow_modules()
    object_table = pq.read_table(object_path)
    map_table = pq.read_table(map_path)
    object_rows = object_table.to_pylist()
    map_rows = map_table.to_pylist()

    def key(row: dict[str, Any]) -> tuple[str, int]:
        video_id = str(row.get("video_id", ""))
        frame = row.get("frame_idx", row.get("actual_frame_id"))
        return video_id, int(frame)

    map_keys = {key(row) for row in map_rows}
    object_keys = {key(row) for row in object_rows}
    missing = sorted(object_keys - map_keys)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    result = {
        "source_type": OBJECT_SOURCE_TYPE,
        "decision": "ACCEPT_PARTIAL",
        "object_corpus_member": object_member.name,
        "canonical_btc_map_member": map_member.name,
        "stats_member": stats_member.name,
        "object_row_count": len(object_rows),
        "btc_map_row_count": len(map_rows),
        "object_key_count": len(object_keys),
        "btc_key_count": len(map_keys),
        "unmapped_object_key_count": len(missing),
        "unmapped_object_key_sample": [list(item) for item in missing[:20]],
        "mapping_status": "PASS" if not missing else "FAIL",
        "source_sha256": {
            "object_corpus": object_sha,
            "btc_map": map_sha,
            "stats": stats_sha,
        },
        "reported_stats": stats,
        "policy": {
            "evidence_only": True,
            "raw_label_as_final_qa_answer_forbidden": True,
            "does_not_replace_frozen_object_evidence": True,
        },
        "warning": "Raw detection labels are noisy and model revision provenance is incomplete.",
    }
    if missing:
        result["decision"] = "REJECT"
    _write_json(output_root / "object_external_manifest.json", result)
    _write_json(
        output_root / "object_external_mapping_audit.json",
        {
            key: result[key]
            for key in (
                "source_type",
                "decision",
                "object_row_count",
                "btc_map_row_count",
                "object_key_count",
                "btc_key_count",
                "unmapped_object_key_count",
                "unmapped_object_key_sample",
                "mapping_status",
            )
        },
    )
    return result


def _write_bundle(files: dict[str, Path], bundle_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files.values(), key=lambda item: item.name):
            archive.write(path, path.name)
    with zipfile.ZipFile(bundle_path) as archive:
        bad_crc = archive.testzip()
        members = sorted(info.filename for info in archive.infolist())
    if bad_crc is not None:
        raise RuntimeError(f"OUTPUT_BUNDLE_CRC_FAILED:{bad_crc}")
    return {
        "path": str(bundle_path.resolve()),
        "size_bytes": bundle_path.stat().st_size,
        "sha256": _sha256_path(bundle_path),
        "crc_status": "PASS",
        "members": members,
    }


def repack_external_ocr_object_evidence(
    source_archive: str | Path, output_root: str | Path
) -> Path:
    """Extract only the accepted OCR/object runtime corpora into a small clean ZIP.

    The source archive is never extracted wholesale.  This is a byte-preserving
    repack for the Trial dry-run, not a regeneration of either external model.
    """

    source = Path(source_archive).resolve(strict=True)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    if _sha256_path(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("SOURCE_ARCHIVE_SHA256_MISMATCH")
    runtime = destination / "runtime_evidence"
    runtime.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if any(not _safe_tar_name(member.name) for member in members):
            raise RuntimeError("SOURCE_ARCHIVE_UNSAFE_PATH")
        for key, suffix, filename in (
            ("objects", OBJECT_CORPUS_SUFFIX, "object_records_external_v3.parquet"),
            ("btc_map", BTC_MAP_SUFFIX, "btc_keyframe_map_external_v3.parquet"),
            ("object_stats", OBJECT_STATS_SUFFIX, "object_stats_external_v3.json"),
        ):
            target = runtime / filename
            _copy_member(archive, _find_member(members, suffix), target)
            files[key] = target
        nested_member = _find_member(members, NESTED_INDEX_ZIP_SUFFIX)
        with tempfile.TemporaryDirectory(prefix="triage_eg_external_evidence_") as temporary:
            nested_path = Path(temporary) / "indices.zip"
            _copy_member(archive, nested_member, nested_path)
            with zipfile.ZipFile(nested_path) as nested:
                infos = [
                    info
                    for info in nested.infolist()
                    if info.filename.endswith("l21_ocr_temporal_v3_corpus.parquet")
                    and info.file_size > 1_000_000
                ]
                if len(infos) != 1 or nested.testzip() is not None:
                    raise RuntimeError("OCR_RUNTIME_CORPUS_DISCOVERY_OR_CRC_FAILED")
                target = runtime / "ocr_records_external_v3.parquet"
                digest = hashlib.sha256()
                with nested.open(infos[0]) as input_stream, target.open("wb") as output:
                    while block := input_stream.read(8 * 1024 * 1024):
                        output.write(block)
                        digest.update(block)
                files["ocr"] = target
    manifest = {
        "contract": "TRIAGE_EG_EXTERNAL_RUNTIME_EVIDENCE_V3",
        "source_archive_sha256": EXPECTED_SOURCE_SHA256,
        "source_types": {
            "ocr": OCR_SOURCE_TYPE,
            "object": OBJECT_SOURCE_TYPE,
        },
        "policy": {
            "evidence_only": True,
            "ocr_direct_final_answer_forbidden": True,
            "object_label_direct_final_answer_forbidden": True,
        },
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256_path(path)}
            for path in files.values()
        },
    }
    manifest_path = runtime / "external_runtime_evidence_manifest.json"
    _write_json(manifest_path, manifest)
    files["manifest"] = manifest_path
    bundle = destination / "external_multimodal_runtime_evidence_v3.zip"
    _write_bundle(files, bundle)
    return bundle


def run_external_multimodal_import(
    source_archive: str | Path,
    canonical_video_manifest: str | Path,
    output_root: str | Path,
) -> ImportResult:
    """Audit and repack the frozen source; abort before outputs on any ASR hard gate."""

    source = Path(source_archive).resolve(strict=True)
    duration_manifest = Path(canonical_video_manifest).resolve(strict=True)
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    durations, duration_provenance = _load_durations(duration_manifest)
    with tempfile.TemporaryDirectory(prefix="triage_eg_external_v3_") as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(source, "r:gz") as archive:
            members = archive.getmembers()
            source_audit = _archive_audit(source, destination, members)
            nested_path, nested_audit = _audit_nested_zip(
                archive,
                _find_member(members, NESTED_INDEX_ZIP_SUFFIX),
                temporary_root,
                destination,
            )
            asr, bundle_files, _ = _clean_asr(
                archive,
                members,
                durations,
                duration_provenance,
                temporary_root,
                destination,
            )
            ocr = _audit_ocr(nested_path, destination)
            objects = _audit_objects(archive, members, temporary_root, destination)
    bundle_path = destination / "asr_external_v3_validated_bundle.zip"
    bundle = _write_bundle(bundle_files, bundle_path)
    _write_json(destination / "asr_external_v3_bundle_manifest.json", bundle)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_archive": source_audit,
        "nested_zip_crc_status": nested_audit["crc_status"],
        "asr": asr,
        "ocr": ocr,
        "object": objects,
        "visual_policy": {
            "external_visual_artifacts_audit_only": True,
            "frozen_a0_replaced": False,
            "frozen_s1_replaced": False,
            "canonical_btc_framemap_replaced": False,
        },
        "ground_truth_opened": False,
        "whisper_873_job_started": False,
        "bundle": bundle,
    }
    _write_json(destination / "external_multimodal_audit_summary.json", summary)
    status_path = destination / "EXTERNAL_MULTIMODAL_AUDIT_STATUS.md"
    status_path.write_text(
        "\n".join(
            [
                "# External Multimodal Audit V3 — Status",
                "",
                "- ASR: **ACCEPT_EXTERNAL_ASR_WITH_WARNINGS**",
                f"- ASR runtime label: `{ASR_SOURCE_TYPE}`",
                "- OCR: **ACCEPT_PARTIAL**",
                f"- OCR runtime label: `{OCR_SOURCE_TYPE}`",
                f"- Object: **{objects['decision']}**",
                f"- Object runtime label: `{OBJECT_SOURCE_TYPE}`",
                "- Source SHA-256 gate: **PASS**",
                "- Archive path/link/duplicate gate: **PASS**",
                "- Nested ZIP CRC gate: **PASS**",
                "- ASR 873-video coverage/timestamp/E5 alignment gates: **PASS**",
                "- GT opened: **NO**",
                "- Full Whisper rerun started: **NO**",
                "- Frozen A0/S1/BTC visual assets changed: **NO**",
                "",
                "The accepted ASR is validated external evidence, not reproducible Whisper v1.2.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ImportResult(
        output_root=str(destination),
        asr_bundle=str(bundle_path),
        asr_decision=str(asr["decision"]),
        ocr_decision=str(ocr["decision"]),
        object_decision=str(objects["decision"]),
        summary=summary,
    )


__all__ = ["EXPECTED_SOURCE_SHA256", "ImportResult", "run_external_multimodal_import"]
