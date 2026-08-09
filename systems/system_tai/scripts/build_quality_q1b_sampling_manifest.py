"""Build the deterministic Q1-B candidate-video sampling manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from system_tai.data.corpus_discovery import (
    VIDEO_ID_PATTERN,
    CorpusDiscoveryError,
    CorpusManifest,
    DiscoveryValidation,
    discover_corpus,
    load_corpus_manifest,
)

SAMPLING_SEED = "system_tai_q1b_v1"
CURRENT_Q1B_VIDEO_COUNT = 873
CURRENT_Q1B_FEATURE_ROW_COUNT = 177_321
CURRENT_Q1B_CORPUS_FINGERPRINT = (
    "b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4"
)
CSV_COLUMNS = ("sample_rank", "video_id", "selection_hash")


class SamplingRecord(NamedTuple):
    """One immutable row in deterministic sampling order."""

    sample_rank: int
    video_id: str
    selection_hash: str


def _validate_inventory(video_ids: Iterable[str]) -> tuple[str, ...]:
    materialized = tuple(video_ids)
    if not materialized:
        raise ValueError("video inventory must not be empty")

    seen: set[str] = set()
    for video_id in materialized:
        if not isinstance(video_id, str):
            raise ValueError("every inventory record must be a canonical video_id string")
        if video_id != video_id.strip() or VIDEO_ID_PATTERN.fullmatch(video_id) is None:
            raise ValueError(f"malformed or noncanonical video_id: {video_id!r}")
        if video_id in seen:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen.add(video_id)
    return tuple(sorted(materialized))


def selection_hash(video_id: str) -> str:
    """Return the frozen Q1-B sampling digest for one validated video ID."""

    validated = _validate_inventory((video_id,))[0]
    material = f"{SAMPLING_SEED}|{validated}".encode()
    return hashlib.sha256(material).hexdigest()


def build_sampling_records(video_ids: Iterable[str]) -> tuple[SamplingRecord, ...]:
    """Rank validated IDs using only the fixed seed and canonical video ID."""

    canonical_inventory = _validate_inventory(video_ids)
    ranked = sorted(
        ((selection_hash(video_id), video_id) for video_id in canonical_inventory),
        key=lambda item: (item[0], item[1]),
    )
    return tuple(
        SamplingRecord(rank, video_id, digest)
        for rank, (digest, video_id) in enumerate(ranked, start=1)
    )


def validate_current_q1b_corpus(manifest: CorpusManifest) -> None:
    """Fail closed unless the loaded manifest is the accepted Q1-B snapshot."""

    failures: list[str] = []
    if len(manifest.videos) != CURRENT_Q1B_VIDEO_COUNT:
        failures.append(
            f"video_count={len(manifest.videos)} (expected {CURRENT_Q1B_VIDEO_COUNT})"
        )
    if manifest.total_rows != CURRENT_Q1B_FEATURE_ROW_COUNT:
        failures.append(
            "feature_row_count="
            f"{manifest.total_rows} (expected {CURRENT_Q1B_FEATURE_ROW_COUNT})"
        )
    if manifest.fingerprint != CURRENT_Q1B_CORPUS_FINGERPRINT:
        failures.append(
            f"manifest_fingerprint={manifest.fingerprint} "
            f"(expected {CURRENT_Q1B_CORPUS_FINGERPRINT})"
        )
    if failures:
        raise ValueError("current Q1-B corpus identity mismatch: " + "; ".join(failures))


def write_sampling_manifest(
    records: Iterable[SamplingRecord],
    destination: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic UTF-8 CSV without source paths or semantic fields."""

    rows = tuple(records)
    if not rows:
        raise ValueError("sampling records must not be empty")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return destination


def _load_manifest(args: argparse.Namespace) -> CorpusManifest:
    if args.manifest is not None:
        return load_corpus_manifest(
            args.manifest,
            input_root=args.input_root,
            max_root_depth=args.max_root_depth,
        )
    if args.input_root is None:
        raise ValueError("provide --manifest or --input-root")
    return discover_corpus(
        args.input_root,
        expected_dimension=args.expected_dimension,
        max_root_depth=args.max_root_depth,
        validation_mode=DiscoveryValidation.STRICT,
        portable=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-current-q1b-corpus", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-dimension", type=int, default=512)
    parser.add_argument("--max-root-depth", type=int, default=4)
    return parser


def run(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args)
    if args.require_current_q1b_corpus:
        validate_current_q1b_corpus(manifest)
    records = build_sampling_records(video.video_id for video in manifest.videos)
    destination = write_sampling_manifest(records, args.output, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "output": str(destination),
                "sampling_seed": SAMPLING_SEED,
                "video_count": len(manifest.videos),
                "feature_row_count": manifest.total_rows,
                "manifest_fingerprint": manifest.fingerprint,
                "current_q1b_identity_required": args.require_current_q1b_corpus,
                "source_artifacts_copied": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (CorpusDiscoveryError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Q1-B sampling manifest failed: {exc}", file=sys.stderr)
        if isinstance(exc, CorpusDiscoveryError):
            for issue in exc.issues:
                print(f"- {issue}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
