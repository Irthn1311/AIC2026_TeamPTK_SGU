"""Validator for the proposed UTF-8 JSONL KIS checkpoint boundary."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from system_tai.common.schemas import ValidationIssue, ValidationResult
from system_tai.features.btc_clip_store import FeatureStoreRegistry

REQUIRED_FIELDS = ("query_id", "rank", "video_id", "frame_id")


class CheckpointValidator:
    def validate(
        self,
        checkpoint_path: Path,
        registry: FeatureStoreRegistry | None = None,
    ) -> ValidationResult:
        path = Path(checkpoint_path)
        if not path.is_file():
            return ValidationResult(
                valid=False,
                errors=(
                    ValidationIssue(
                        code="FILE_NOT_FOUND",
                        message=f"checkpoint file not found: {path}",
                    ),
                ),
            )
        try:
            text = path.read_bytes().decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            return ValidationResult(
                valid=False,
                errors=(
                    ValidationIssue(
                        code="INVALID_UTF8",
                        message=f"checkpoint is not valid UTF-8: {exc}",
                    ),
                ),
            )

        errors: list[ValidationIssue] = []
        records_by_query: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        nonempty_line_count = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            nonempty_line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    ValidationIssue(
                        code="INVALID_JSON",
                        message=f"invalid JSON object: {exc.msg}",
                        line_number=line_number,
                    )
                )
                continue
            if not isinstance(record, dict):
                errors.append(
                    ValidationIssue(
                        code="NOT_AN_OBJECT",
                        message="each non-empty line must contain one JSON object",
                        line_number=line_number,
                    )
                )
                continue
            query_value = record.get("query_id")
            query_id = query_value if isinstance(query_value, str) else None
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            for field in missing:
                errors.append(
                    ValidationIssue(
                        code="MISSING_FIELD",
                        message=f"missing required field: {field}",
                        line_number=line_number,
                        query_id=query_id,
                    )
                )
            if missing:
                continue
            valid_record = True
            if not isinstance(record["query_id"], str) or not record["query_id"].strip():
                errors.append(
                    self._issue(
                        "INVALID_QUERY_ID",
                        "query_id must be a non-empty string",
                        line_number,
                        query_id,
                    )
                )
                valid_record = False
            if not isinstance(record["video_id"], str) or not record["video_id"].strip():
                errors.append(
                    self._issue(
                        "INVALID_VIDEO_ID",
                        "video_id must be a non-empty string",
                        line_number,
                        query_id,
                    )
                )
                valid_record = False
            if type(record["rank"]) is not int or record["rank"] < 1:
                errors.append(
                    self._issue(
                        "INVALID_RANK",
                        "rank must be an integer greater than or equal to one",
                        line_number,
                        query_id,
                    )
                )
                valid_record = False
            if type(record["frame_id"]) is not int or record["frame_id"] < 0:
                errors.append(
                    self._issue(
                        "INVALID_FRAME_ID",
                        "frame_id must be a non-negative integer",
                        line_number,
                        query_id,
                    )
                )
                valid_record = False
            if not valid_record:
                continue
            query_id = record["query_id"]
            video_id = record["video_id"]
            frame_id = record["frame_id"]
            if registry is not None:
                try:
                    store = registry.get(video_id)
                except KeyError:
                    errors.append(
                        self._issue(
                            "UNKNOWN_VIDEO_ID",
                            f"video_id is not present in the feature registry: {video_id}",
                            line_number,
                            query_id,
                        )
                    )
                else:
                    if not store.contains_frame(frame_id):
                        errors.append(
                            self._issue(
                                "UNKNOWN_FRAME_ID",
                                f"frame_id is not mapped for {video_id}: {frame_id}",
                                line_number,
                                query_id,
                            )
                        )
            records_by_query[query_id].append((line_number, record))

        if nonempty_line_count == 0:
            errors.append(
                ValidationIssue(code="EMPTY_CHECKPOINT", message="checkpoint has no records")
            )
        for query_id, entries in records_by_query.items():
            if len(entries) > 100:
                errors.append(
                    self._issue(
                        "TOO_MANY_RESULTS",
                        f"query has {len(entries)} records; maximum is 100",
                        entries[100][0],
                        query_id,
                    )
                )
            seen_ranks: dict[int, int] = {}
            seen_pairs: dict[tuple[str, int], int] = {}
            for line_number, record in entries:
                rank = int(record["rank"])
                pair = (str(record["video_id"]), int(record["frame_id"]))
                if rank in seen_ranks:
                    errors.append(
                        self._issue(
                            "DUPLICATE_RANK",
                            f"rank {rank} is duplicated; first seen at line {seen_ranks[rank]}",
                            line_number,
                            query_id,
                        )
                    )
                else:
                    seen_ranks[rank] = line_number
                if pair in seen_pairs:
                    errors.append(
                        self._issue(
                            "DUPLICATE_VIDEO_FRAME",
                            f"video/frame pair {pair} is duplicated; first seen at line "
                            f"{seen_pairs[pair]}",
                            line_number,
                            query_id,
                        )
                    )
                else:
                    seen_pairs[pair] = line_number
            expected = list(range(1, len(entries) + 1))
            if sorted(seen_ranks) != expected:
                errors.append(
                    self._issue(
                        "NON_CONTIGUOUS_RANKS",
                        f"ranks must be unique and contiguous from one; observed "
                        f"{sorted(seen_ranks)}",
                        entries[0][0],
                        query_id,
                    )
                )
        return ValidationResult(valid=not errors, errors=tuple(errors))

    @staticmethod
    def _issue(
        code: str,
        message: str,
        line_number: int | None,
        query_id: str | None,
    ) -> ValidationIssue:
        return ValidationIssue(
            code=code,
            message=message,
            line_number=line_number,
            query_id=query_id,
        )
