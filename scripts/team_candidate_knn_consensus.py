"""Merge two to five blind member packets with deterministic unsupervised consensus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from triage_eg.prelim1_team.consensus import consensus_rows


def _mapping(values: list[str], label: str) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use MEMBER_ID=/path syntax")
        member_id, path = value.split("=", 1)
        if not member_id or member_id in output:
            raise ValueError(f"invalid or duplicate {label} member_id: {member_id}")
        output[member_id] = Path(path).expanduser().resolve(strict=True)
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"query_id", "candidate_rank", "video_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"member candidate CSV contract mismatch: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True, help="MEMBER_ID=/path/top5.csv")
    parser.add_argument(
        "--embedding", action="append", default=[], help="MEMBER_ID=/path/vectors.npz"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--near-frame-tolerance", type=int, default=96)
    parser.add_argument("--cosine-threshold", type=float, default=0.92)
    args = parser.parse_args()
    member_paths = _mapping(args.member, "member")
    embedding_paths = _mapping(args.embedding, "embedding")
    if not set(embedding_paths).issubset(member_paths):
        raise ValueError("embedding member IDs must be a subset of candidate member IDs")
    members = {member_id: _read_csv(path) for member_id, path in member_paths.items()}
    embeddings = {}
    for member_id, path in embedding_paths.items():
        with np.load(path) as archive:
            embeddings[member_id] = {key: np.asarray(archive[key]) for key in archive.files}
    rows = consensus_rows(
        members,
        embeddings=embeddings,
        near_frame_tolerance=args.near_frame_tolerance,
        cosine_threshold=args.cosine_threshold,
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "team_consensus_top3.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "team_consensus_top3.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        {
            "members": sorted(members),
            "queries": len({row["query_id"] for row in rows}),
            "consensus_rows": len(rows),
            "automatic_submission": False,
            "output_dir": str(output),
        }
    )


if __name__ == "__main__":
    main()
