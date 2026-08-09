from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from triage_eg.retrieval.stage1c.artifacts import (
    contact_sheet_label,
    grouped_video_diagnostics,
    render_contact_sheet,
    write_review_template,
)
from triage_eg.retrieval.stage1c.contracts import ISSUE_CODES, QueryRecord, Stage1CConfig
from triage_eg.retrieval.stage1c.metrics import (
    exact_vector_diagnostics,
    initial_frame_diagnostics,
    paired_language_diagnostic,
    score_diagnostics,
    video_concentration_diagnostics,
)
from triage_eg.retrieval.stage1c.query_suite import (
    filter_query_suite,
    load_query_suite,
    query_suite_fingerprint,
    validate_query_suite,
)
from triage_eg.retrieval.stage1c.review import score_human_review


def query(query_id: str, pair_id: str, language: str, **changes) -> QueryRecord:
    values = {
        "query_id": query_id,
        "pair_id": pair_id,
        "language": language,
        "category": "OBJECT",
        "difficulty": "EASY",
        "text": f"text {query_id}",
        "notes": "",
    }
    values.update(changes)
    return QueryRecord(**values)


def frame(rank: int, *, video: str = "L01_V001", n: int | None = None, original=None):
    ordinal = n if n is not None else rank
    original_idx = rank if original is None else original
    return {
        "query_id": "q_en",
        "rank": rank,
        "global_row": rank - 1,
        "video_id": video,
        "n": ordinal,
        "original_frame_idx": original_idx,
        "score": 1.0 - rank / 1000,
        "keyframe_relative_path": f"keyframes/{video}/{ordinal:03d}.jpg",
        "is_initial_frame": ordinal == 1 and original_idx == 0,
        "same_score_as_previous": False,
    }


def test_default_query_suite_is_valid_and_complete() -> None:
    records, manifest = load_query_suite("configs/retrieval/stage1c_qualitative_queries.jsonl")
    assert len(records) == 28
    assert manifest["pair_count"] == 14
    assert set(manifest["categories"]) == {
        "OBJECT",
        "ACTION",
        "SCENE",
        "ATTRIBUTE",
        "SPATIAL_RELATION",
        "MULTI_CONCEPT",
        "EVENT",
        "DIFFICULT",
    }


def test_duplicate_query_id_is_rejected() -> None:
    records = [query("q", "p", "en"), query("q", "p", "vi", text="khac")]
    with pytest.raises(ValueError, match="duplicate query_id"):
        validate_query_suite(records)


@pytest.mark.parametrize("language", ["en", "vi"])
def test_missing_pair_language_is_rejected(language: str) -> None:
    with pytest.raises(ValueError, match="QUERY_PAIR_INVALID"):
        validate_query_suite([query(f"q_{language}", "p", language)])


def test_invalid_query_language_and_category_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid language"):
        query("q", "p", "fr")
    with pytest.raises(ValueError, match="invalid category"):
        query("q", "p", "en", category="UNKNOWN")


def test_empty_text_and_invalid_difficulty_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        query("q", "p", "en", text="  ")
    with pytest.raises(ValueError, match="invalid difficulty"):
        query("q", "p", "en", difficulty="IMPOSSIBLE")


def test_query_suite_fingerprint_is_canonical_and_deterministic() -> None:
    records = [query("q_en", "p", "en"), query("q_vi", "p", "vi")]
    assert query_suite_fingerprint(records) == query_suite_fingerprint(list(reversed(records)))


def test_query_filters_are_explicit_and_fail_on_unknown_ids() -> None:
    records = [query("q_en", "p", "en"), query("q_vi", "p", "vi")]
    assert [item.query_id for item in filter_query_suite(records, languages=("vi",))] == [
        "q_vi"
    ]
    with pytest.raises(ValueError, match="unknown query_ids"):
        filter_query_suite(records, query_ids=("missing",))


def test_stage1c_limits_enforce_raw_review_and_internal_kis_contract(tmp_path: Path) -> None:
    values = {
        "repo_root": tmp_path,
        "dataset_root": tmp_path / "data",
        "stage0_root": tmp_path / "stage0",
        "stage1_root": tmp_path / "stage1",
        "stage1b_root": tmp_path / "stage1b",
        "encoder_asset_root": tmp_path / "asset",
        "query_suite": tmp_path / "queries.jsonl",
        "output_root": tmp_path / "output",
    }
    with pytest.raises(ValueError, match="kis_top_k"):
        Stage1CConfig(**values, frame_top_k=50, kis_top_k=20)
    with pytest.raises(ValueError, match="review_top_k"):
        Stage1CConfig(**values, frame_top_k=5, review_top_k=10)


@pytest.mark.parametrize(
    "code",
    [
        "STAGE1_INDEX_NOT_READY",
        "STAGE1B_ENCODER_NOT_VERIFIED",
        "STAGE1B_MODEL_SPACE_NOT_VERIFIED",
        "ENCODER_ASSET_NOT_FOUND",
        "ENCODER_LOAD_FAILED",
        "QUERY_SUITE_INVALID",
        "QUERY_PAIR_INVALID",
        "QUERY_ENCODING_FAILED",
        "QUERY_TOKENIZATION_FAILED",
        "QUERY_SEARCH_FAILED",
        "KEYFRAME_RESOLUTION_FAILED",
        "CONTACT_SHEET_RENDER_FAILED",
        "KIS_EXPORT_FAILED",
        "HIGH_INITIAL_FRAME_CONCENTRATION",
        "HIGH_SINGLE_VIDEO_CONCENTRATION",
        "HIGH_EXACT_VECTOR_DUPLICATION",
        "REVIEW_LABEL_INVALID",
        "REVIEW_ROW_IDENTITY_MISMATCH",
        "REVIEW_DUPLICATE_JUDGMENT",
        "REVIEW_INCOMPLETE",
    ],
)
def test_required_issue_code_vocabulary(code: str) -> None:
    assert code in ISSUE_CODES


def test_initial_frame_contract_requires_n_one_and_original_zero() -> None:
    frames = [frame(1, n=1, original=0), frame(2, n=1, original=9)]
    diagnostics = initial_frame_diagnostics(frames)
    assert diagnostics["initial_frame_count_top5"] == 1
    assert diagnostics["initial_frame_rate_top5"] == 0.5


def test_initial_frame_rates_respect_all_cutoffs() -> None:
    frames = [frame(index, n=1, original=0) for index in range(1, 51)]
    diagnostics = initial_frame_diagnostics(frames)
    for cutoff in (5, 10, 20, 50):
        assert diagnostics[f"initial_frame_count_top{cutoff}"] == cutoff
        assert diagnostics[f"initial_frame_rate_top{cutoff}"] == 1.0


def test_exact_duplicate_detection_is_exact_not_tolerant() -> None:
    vectors = np.eye(3, dtype=np.float32)
    vectors[1] = vectors[0]
    vectors[2, 0] = np.nextafter(vectors[0, 0], np.float32(2.0))
    diagnostics = exact_vector_diagnostics(vectors)
    assert diagnostics["unique_exact_vectors_top5"] == 2
    assert diagnostics["equality_basis"] == "EXACT_CANONICAL_STORED_VECTOR_BYTES"


def test_video_concentration_counts_unique_max_and_share() -> None:
    frames = [frame(index, video="same" if index <= 12 else f"v{index}") for index in range(1, 21)]
    diagnostics = video_concentration_diagnostics(frames)
    assert diagnostics["unique_videos_top20"] == 9
    assert diagnostics["max_frames_from_single_video_top20"] == 12
    assert diagnostics["top_video_share_top20"] == 0.6


def test_grouped_video_view_uses_best_raw_frame_and_stable_order() -> None:
    frames = [frame(1, video="b"), frame(2, video="a"), frame(3, video="b")]
    grouped = grouped_video_diagnostics(frames)
    assert [item["video_id"] for item in grouped] == ["b", "a"]
    assert grouped[0]["best_frame_rank"] == 1
    assert grouped[0]["frames_in_raw_top50"] == 2


def test_score_diagnostics_are_descriptive_only() -> None:
    values = score_diagnostics([frame(index) for index in range(1, 51)])
    assert values["top1_score"] == pytest.approx(0.999)
    assert values["score_gap_1_20"] == pytest.approx(0.019)
    assert not any("relevance" in key.lower() for key in values)


def test_pair_diagnostics_cover_embedding_frame_and_video_overlap() -> None:
    en = [frame(index, video=f"v{index}") for index in range(1, 21)]
    vi = [dict(item) for item in en]
    for item in vi:
        item["query_id"] = "q_vi"
    result = paired_language_diagnostic("p", np.array([1, 0]), np.array([1, 0]), en, vi)
    assert result["text_embedding_cosine_en_vi"] == 1.0
    assert result["top10_global_row_overlap_count"] == 10
    assert result["top20_global_row_jaccard"] == 1.0
    assert result["top20_video_jaccard"] == 1.0


def test_pair_diagnostics_are_safe_for_zero_overlap() -> None:
    en = [frame(index, video=f"en{index}") for index in range(1, 6)]
    vi = [frame(index + 10, video=f"vi{index}") for index in range(1, 6)]
    for item in vi:
        item["query_id"] = "q_vi"
    result = paired_language_diagnostic("p", np.array([1, 0]), np.array([0, 1]), en, vi)
    assert result["top5_global_row_jaccard"] == 0.0
    assert result["top5_video_jaccard"] == 0.0


def test_contact_sheet_has_required_label_and_preserves_source(tmp_path: Path) -> None:
    item = frame(1, n=1, original=77)
    source = tmp_path / item["keyframe_relative_path"]
    source.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), "red").save(source)
    before = source.read_bytes()
    output = tmp_path / "sheet.jpg"
    assert "frame=77" in contact_sheet_label(item)
    assert not render_contact_sheet([item], tmp_path, output)
    assert output.is_file()
    assert source.read_bytes() == before


def test_missing_contact_sheet_image_yields_issue(tmp_path: Path) -> None:
    issues = render_contact_sheet([frame(1)], tmp_path, tmp_path / "sheet.jpg")
    assert issues[0]["code"] == "KEYFRAME_RESOLUTION_FAILED"


def review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "stage1c"
    queries = [query("q_en", "p", "en"), query("q_vi", "p", "vi")]
    frames = {
        item.query_id: [{**frame(rank), "query_id": item.query_id} for rank in range(1, 3)]
        for item in queries
    }
    template = root / "review/review_template.csv"
    write_review_template(template, queries, frames, 2)
    filled = tmp_path / "filled.csv"
    filled.write_bytes(template.read_bytes())
    return root, filled


def update_review(path: Path, mutate) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_review_template_row_count_and_not_reviewed_state(tmp_path: Path) -> None:
    root, filled = review_fixture(tmp_path)
    metrics = score_human_review(root, filled)
    assert metrics["judgments_expected"] == 4
    assert metrics["human_review_status"] == "NOT_REVIEWED"
    assert metrics["retrieval_quality_status"] == "NOT_REVIEWED"


def test_review_invalid_label_is_rejected(tmp_path: Path) -> None:
    root, filled = review_fixture(tmp_path)
    update_review(filled, lambda rows: rows[0].update(review_label="GOOD"))
    with pytest.raises(ValueError, match="REVIEW_LABEL_INVALID"):
        score_human_review(root, filled)


def test_review_missing_row_and_invalid_failure_tag_are_rejected(tmp_path: Path) -> None:
    root, filled = review_fixture(tmp_path)

    def remove_row(rows):
        rows.pop()

    update_review(filled, remove_row)
    with pytest.raises(ValueError, match="REVIEW_INCOMPLETE"):
        score_human_review(root, filled)
    root, filled = review_fixture(tmp_path / "tag")
    update_review(filled, lambda rows: rows[0].update(failure_tags="MADE_UP_TAG"))
    with pytest.raises(ValueError, match="REVIEW_LABEL_INVALID"):
        score_human_review(root, filled)


@pytest.mark.parametrize("label", ["RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN"])
def test_all_review_labels_are_accepted(tmp_path: Path, label: str) -> None:
    root, filled = review_fixture(tmp_path)
    update_review(filled, lambda rows: rows[0].update(review_label=label))
    metrics = score_human_review(root, filled)
    assert metrics["judgments_completed"] == 1
    assert metrics["human_review_status"] == "PARTIALLY_REVIEWED"


def test_review_duplicate_and_modified_identity_are_rejected(tmp_path: Path) -> None:
    root, filled = review_fixture(tmp_path)
    update_review(filled, lambda rows: rows.__setitem__(1, dict(rows[0])))
    with pytest.raises(ValueError, match="REVIEW_DUPLICATE_JUDGMENT"):
        score_human_review(root, filled)
    root, filled = review_fixture(tmp_path / "second")
    update_review(filled, lambda rows: rows[0].update(video_id="changed"))
    with pytest.raises(ValueError, match="REVIEW_ROW_IDENTITY_MISMATCH"):
        score_human_review(root, filled)


def test_review_partial_grade_and_uncertain_are_separate(tmp_path: Path) -> None:
    root, filled = review_fixture(tmp_path)

    def labels(rows):
        values = ["RELEVANT", "PARTIAL", "UNCERTAIN", "IRRELEVANT"]
        for row, label in zip(rows, values, strict=True):
            row["review_label"] = label

    update_review(filled, labels)
    metrics = score_human_review(root, filled)
    en = metrics["per_query"]["q_en"]
    vi = metrics["per_query"]["q_vi"]
    assert en["human_graded_relevance_top5"] == 0.75
    assert vi["uncertain_count_top5"] == 1
    assert vi["human_graded_relevance_top5"] == 0.0
    serialized = json.dumps(metrics)
    assert "Recall" in serialized  # only a non-claim; no metric key uses Recall.
    assert not any("recall" in key.lower() for key in en)


def test_query_record_serialization_is_human_readable() -> None:
    record = query("q_en", "p", "en", text="a red car")
    assert asdict(record)["text"] == "a red car"
