from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from triage_eg.retrieval.stage1c.contracts import QueryRecord
from triage_eg.retrieval.stage1d import create_stage1d_bundle
from triage_eg.retrieval.stage1d.contracts import ReviewConfig
from triage_eg.retrieval.stage1d.review import score_stage1d_review, write_blinded_review
from triage_eg.retrieval.stage1d.review_visuals import (
    ARM_FILES,
    blinded_header_text,
    blinded_tile_label,
    patch_blinded_review_visuals,
)

ARMS = ("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN")
COLORS = {
    "EN_DIRECT": (220, 20, 20),
    "VI_DIRECT": (20, 210, 20),
    "VI_TRANSLATED_EN": (20, 20, 220),
}


def _write(path: Path, value: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    _write(path, "".join(json.dumps(value) + "\n" for value in values))


def _record(pair_id: str, arm: str, rank: int, row: int) -> dict[str, object]:
    video_id = f"{pair_id}_{arm[:2]}_V{rank:02d}"
    return {
        "query_id": f"{pair_id}_{arm.lower()}",
        "rank": rank,
        "global_row": row,
        "video_id": video_id,
        "n": rank + 10,
        "original_frame_idx": row * 3,
        "score": 0.9 - rank / 100,
        "keyframe_relative_path": f"keyframes/{pair_id}/{arm}/{rank:03d}.jpg",
        "is_initial_frame": False,
        "same_score_as_previous": False,
    }


def _frozen_output(root: Path, dataset: Path, pair_count: int = 14) -> Path:
    pairs = {}
    frames_by_pair_arm = {}
    comparisons = []
    translations = []
    for pair_index in range(pair_count):
        pair_id = f"pair_{pair_index:02d}"
        en_text = f"English intent {pair_index}"
        vi_text = f"Ý định tiếng Việt {pair_index}"
        pairs[pair_id] = {
            "en": QueryRecord(f"{pair_id}_en", pair_id, "en", "OBJECT", "EASY", en_text, ""),
            "vi": QueryRecord(f"{pair_id}_vi", pair_id, "vi", "OBJECT", "EASY", vi_text, ""),
        }
        arm_frames = {}
        for arm_index, arm in enumerate(ARMS):
            records = [
                _record(pair_id, arm, rank, pair_index * 10000 + arm_index * 100 + rank)
                for rank in range(1, 21)
            ]
            arm_frames[arm] = records
            _write_jsonl(root / "comparisons" / pair_id / ARM_FILES[arm], records)
            for item in records[:5]:
                target = dataset / str(item["keyframe_relative_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (160, 100), COLORS[arm]).save(target)
        frames_by_pair_arm[pair_id] = arm_frames
        comparisons.append({"pair_id": pair_id, "category": "OBJECT", "difficulty": "EASY"})
        translations.append(
            {
                "pair_id": pair_id,
                "original_vi_text": vi_text,
                "translated_text_for_clip": f"Machine translation secret {pair_index}",
            }
        )
        translated = root / "translated_queries" / pair_id
        _write(translated / "translation.json")
        _write(translated / "query.json")
        _write_jsonl(translated / "ranked_frames.jsonl", arm_frames["VI_TRANSLATED_EN"])
        _write_jsonl(translated / "ranked_videos.jsonl", [])
        _write(translated / "kis_candidates.csv", "video_id,frame_id\n")
        _write(translated / "retrieval_diagnostics.json")
    write_blinded_review(
        root,
        pairs=pairs,
        frames_by_pair_arm=frames_by_pair_arm,
        config=ReviewConfig(top_k=5, seed=2026),
    )
    _write_jsonl(root / "comparisons/pair_comparisons.jsonl", comparisons)
    _write_jsonl(root / "translations/translations.jsonl", translations)
    summary = {
        "stage1d_version": "0.1.0",
        "execution_status": "COMPLETE_WITH_WARNINGS",
        "build_git_commit": "frozen-commit",
        "stage1_index_fingerprint": "frozen-index",
        "stage1c_frozen_baseline": {
            "query_suite_fingerprint": "frozen-suite",
            "pairs_selected": pair_count,
        },
        "human_review": {
            "status": "NOT_REVIEWED",
            "judgments_expected": pair_count * 15,
            "judgments_completed": 0,
            "blinded": True,
        },
        "language_bridge_quality_status": "NOT_REVIEWED",
    }
    _write(root / "stage1d_summary.json", json.dumps(summary))
    _write(
        root / "run_manifest.json",
        json.dumps(
            {
                "stage1d_version": "0.1.0",
                "build_git_commit": "frozen-commit",
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:01:00Z",
            }
        ),
    )
    _write(root / "stage1d_report.md", "# Frozen Stage 1D report\n")
    for relative in (
        "translator/translator_contract.json",
        "translator/translator_runtime_manifest.json",
        "translator/asset_validation.json",
        "issues.jsonl",
    ):
        _write(root / relative)
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_patch_builds_14_blinded_sheets_without_changing_frozen_records(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    root = _frozen_output(tmp_path / "frozen", dataset)
    template = root / "review/review_template_blinded.csv"
    key = root / "review/review_key.json"
    frozen_files = [
        root / "translations/translations.jsonl",
        *sorted((root / "comparisons").glob("*/**/*.jsonl")),
        *sorted((root / "translated_queries").glob("*/ranked_frames.jsonl")),
    ]
    before = {path: _sha(path) for path in frozen_files}
    template_hash, key_hash = _sha(template), _sha(key)

    result = patch_blinded_review_visuals(root, dataset)

    assert result["patch_scope"] == "REVIEW_PRESENTATION_ONLY"
    assert result["retrieval_source"] == "FROZEN_STAGE1D_V0_1_0"
    assert result["retrieval_regenerated"] is False
    assert result["translation_regenerated"] is False
    assert result["baseline_regenerated"] is False
    assert result["blinded_sheet_count"] == 14
    assert result["review_rows"] == 210
    sheets = sorted((root / "review/blinded_sheets").glob("*.jpg"))
    assert len(sheets) == 14
    assert all(Image.open(path).size == (1080, 1430) for path in sheets)
    assert before == {path: _sha(path) for path in frozen_files}
    assert _sha(template) == template_hash
    assert _sha(key) == key_hash

    summary = json.loads((root / "stage1d_summary.json").read_text())
    assert summary["stage1d_version"] == "0.1.1"
    assert summary["original_retrieval_run"]["stage1d_version"] == "0.1.0"
    assert summary["human_review"]["formal_executability"] == "READY"
    report = (root / "stage1d_report.md").read_text()
    assert "ENGINEERING_UNBLINDED_SHEET" in report
    assert "HUMAN_REVIEW_BLINDED_SHEET" in report
    assert "REVIEW_PRESENTATION_ONLY" in report
    assert "FROZEN_STAGE1D_V0_1_0" in report

    with (root / "review/blinded_sheet_index.csv").open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    assert len(index) == 14
    assert all(row["condition_codes"] == "C01|C02|C03" for row in index)
    assert all(row["ranks"] == "1|2|3|4|5" for row in index)
    assert not any("DIRECT" in json.dumps(row) for row in index)


def test_sheet_condition_order_follows_key_and_labels_do_not_leak(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    root = _frozen_output(tmp_path / "frozen", dataset, pair_count=1)
    patch_blinded_review_visuals(root, dataset)
    key = json.loads((root / "review/review_key.json").read_text())
    mapping = key["pairs"][0]["conditions"]
    with Image.open(root / "review/blinded_sheets/pair_00_top5.jpg") as sheet:
        for column, code in enumerate(("C01", "C02", "C03")):
            pixel = sheet.convert("RGB").getpixel((column * 360 + 180, 250))
            expected = COLORS[mapping[code]]
            assert max(abs(pixel[index] - expected[index]) for index in range(3)) < 20
    header = blinded_header_text("pair_00", "English intent", "Ý định tiếng Việt")
    label = blinded_tile_label(
        "C02",
        {
            "rank": 3,
            "video_id": "L26_V440",
            "original_frame_idx": 5633,
        },
    )
    displayed = header + label
    assert "C02" in displayed and "#3" in displayed and "frame=5633" in displayed
    assert not any(arm in displayed for arm in ARMS)
    assert "Machine translation" not in displayed
    image_bytes = (root / "review/blinded_sheets/pair_00_top5.jpg").read_bytes()
    assert not any(arm.encode() in image_bytes for arm in ARMS)


def test_review_patch_rejects_csv_identity_drift(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    root = _frozen_output(tmp_path / "frozen", dataset, pair_count=1)
    template = root / "review/review_template_blinded.csv"
    with template.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["original_frame_idx"] = "999999"
    with template.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="REVIEW_IDENTITY_MISMATCH"):
        patch_blinded_review_visuals(root, dataset)
    assert not (root / "review/blinded_sheets").exists()


def test_patched_bundle_and_existing_scorer_remain_compatible(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    root = _frozen_output(tmp_path / "frozen", dataset)
    patch_blinded_review_visuals(root, dataset)
    archive = create_stage1d_bundle(root, tmp_path / "patched.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    blinded = [
        name
        for name in names
        if name.startswith("review/blinded_sheets/") and name.endswith(".jpg")
    ]
    assert len(blinded) == 14
    assert "review/blinded_sheet_index.csv" in names
    assert not any(name.endswith((".bin", ".npy", ".mp4")) for name in names)

    template = root / "review/review_template_blinded.csv"
    with template.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["review_label"] = "RELEVANT"
    reviewed = tmp_path / "reviewed.csv"
    with reviewed.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = score_stage1d_review(root, reviewed)
    assert metrics["human_review_status"] == "COMPLETE"


def test_patch_module_has_no_translation_encoding_or_search_invocation() -> None:
    source = Path("src/triage_eg/retrieval/stage1d/review_visuals.py").read_text()
    for forbidden in (
        "from .translator",
        "OfflineViEnTranslator",
        "load_multimodal_encoder",
        "encode_text(",
        ".search(",
        "build_index(",
    ):
        assert forbidden not in source
