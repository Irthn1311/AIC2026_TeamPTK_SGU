from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from triage_eg.retrieval.stage1b.writers import write_json
from triage_eg.retrieval.stage1d.review import REVIEW_FIELDS
from triage_eg.retrieval.stage1e import (
    run_stage1e_language_path_freeze,
    validate_and_score_ai_review,
)
from triage_eg.retrieval.stage1e.contracts import (
    CLIP_CANDIDATE,
    EVALUATION_MODE,
    JUDGE_MODEL,
    PAIR_METRIC_FIELDS,
    TRANSLATOR_MODEL_ID,
    TRANSLATOR_REVISION,
)

ARMS = ("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN")


def _write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    frozen, ai = tmp_path / "stage1d", tmp_path / "ai"
    template: list[dict[str, str]] = []
    judged: list[dict[str, str]] = []
    key_pairs = []
    comparisons = []
    for pair_index in range(14):
        pair_id = "difficult_01" if pair_index == 0 else f"pair_{pair_index:02d}"
        if pair_index == 1:
            pair_id = "obj_01"
        mapping = {"C01": "EN_DIRECT", "C02": "VI_DIRECT", "C03": "VI_TRANSLATED_EN"}
        key_pairs.append({"pair_id": pair_id, "conditions": mapping})
        comparisons.append({"pair_id": pair_id, "category": "OBJECT", "difficulty": "HARD"})
        for condition, arm in mapping.items():
            for rank in range(1, 6):
                score = repr(0.5 - pair_index / 100 - rank / 1000)
                row = {
                    "review_row_id": f"{pair_id}:{condition}:{rank:02d}",
                    "pair_id": pair_id,
                    "condition_code": condition,
                    "en_reference_text": f"English intent {pair_index}",
                    "vi_original_text": f"Vietnamese intent {pair_index}",
                    "rank": str(rank),
                    "video_id": f"L{pair_index:02d}_V{rank:03d}",
                    "global_row": str(pair_index * 100 + rank),
                    "n": str(rank + 1),
                    "original_frame_idx": str(rank * 25),
                    "score": score,
                    "review_label": "",
                    "review_notes": "",
                }
                template.append(row)
                label = "RELEVANT" if arm in {"EN_DIRECT", "VI_TRANSLATED_EN"} else "IRRELEVANT"
                if pair_index == 0 and arm == "VI_TRANSLATED_EN":
                    label = "IRRELEVANT"
                if pair_index == 1 and arm in {"EN_DIRECT", "VI_TRANSLATED_EN"}:
                    label = "PARTIAL" if rank == 1 else "IRRELEVANT"
                judged.append(
                    {
                        **row,
                        "score": (
                            repr(math.nextafter(float(score), math.inf))
                            if pair_index == 1 and condition == "C01" and rank == 1
                            else score
                        ),
                        "review_label": label,
                        "review_notes": "AI_JUDGED",
                    }
                )
    _write_csv(frozen / "review/review_template_blinded.csv", template, REVIEW_FIELDS)
    write_json(
        frozen / "review/review_key.json",
        {"stage1d_review_version": "0.1.0", "blinded": True, "pairs": key_pairs},
    )
    (frozen / "comparisons").mkdir(parents=True)
    (frozen / "comparisons/pair_comparisons.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in comparisons), encoding="utf-8"
    )
    write_json(
        frozen / "stage1d_summary.json",
        {
            "execution_status": "COMPLETE_WITH_WARNINGS",
            "stage1_index_fingerprint": "frozen-index",
            "stage1b_encoder": {
                "candidate_id": CLIP_CANDIDATE,
                "compatibility_status": "VERIFIED",
                "model_space_status": "MODEL_SPACE_VERIFIED",
            },
            "translator": {
                "model_id": TRANSLATOR_MODEL_ID,
                "exact_revision": TRANSLATOR_REVISION,
            },
        },
    )
    _write_csv(ai / "review_template_ai_judged.csv", judged, REVIEW_FIELDS)
    metrics, _, pairs = validate_and_score_ai_review(frozen, ai / "review_template_ai_judged.csv")
    write_json(
        ai / "ai_review_metrics.json",
        {
            "evaluation_mode": EVALUATION_MODE,
            "judge": JUDGE_MODEL,
            "judgments_completed": 210,
            "ai_review_status": "COMPLETE",
            "human_review_status": "NOT_PERFORMED",
            "per_arm": metrics["per_arm"],
            "pair_comparison": metrics["pair_comparison"],
        },
    )
    _write_csv(ai / "ai_pair_metrics.csv", pairs, PAIR_METRIC_FIELDS)
    (ai / "ai_review_metrics.md").write_text(
        f"{EVALUATION_MODE}\n{JUDGE_MODEL}\n", encoding="utf-8"
    )
    return frozen, ai, judged


def test_valid_ai_review_freezes_paths_without_modifying_stage1d(tmp_path: Path) -> None:
    frozen, ai, _ = _fixture(tmp_path)
    before = _hash_tree(frozen)
    output = tmp_path / "stage1e"

    result = run_stage1e_language_path_freeze(frozen, ai, output, build_git_commit="abc")

    assert result["stage1e_execution"] == "COMPLETE"
    assert result["ai_review_status"] == "COMPLETE"
    assert result["human_review_status"] == "NOT_PERFORMED"
    assert result["stage2_readiness"] == "READY"
    assert (
        result["ai_metrics"]["identity_validation"]["score_strings_canonicalized_within_one_ulp"]
        == 1
    )
    assert _hash_tree(frozen) == before
    language = json.loads((output / "language_path_contract.json").read_text())
    assert language["english_path"]["mode"] == "DIRECT"
    assert language["english_path"]["text_encoder"] == CLIP_CANDIDATE
    assert language["vietnamese_path"]["mode"] == "TRANSLATE_TO_ENGLISH_THEN_CLIP"
    assert language["vietnamese_path"]["translator"]["exact_revision"] == TRANSLATOR_REVISION
    assert language["stage1_index_fingerprint"] == "frozen-index"
    contract = json.loads((output / "ai_evaluation/ai_evaluation_contract.json").read_text())
    assert contract["judge"] == {"provider": "OpenAI", "model": JUDGE_MODEL}
    assert contract["human_review_performed"] is False
    assert contract["judgments_completed"] == 210
    assert all(
        "human_relevance" not in key
        for arm in result["ai_metrics"]["per_arm"].values()
        for key in arm
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("identity", "AI_REVIEW_IDENTITY_MISMATCH"),
        ("missing", "AI_REVIEW_INCOMPLETE"),
        ("invalid_label", "AI_REVIEW_LABEL_INVALID"),
        ("duplicate", "AI_REVIEW_DUPLICATE"),
        ("score_drift", "AI_REVIEW_IDENTITY_MISMATCH"),
    ],
)
def test_invalid_ai_review_is_rejected(tmp_path: Path, mutation: str, error: str) -> None:
    frozen, ai, rows = _fixture(tmp_path)
    if mutation == "identity":
        rows[0]["video_id"] = "MUTATED"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "invalid_label":
        rows[0]["review_label"] = "MAYBE"
    elif mutation == "score_drift":
        rows[0]["score"] = str(float(rows[0]["score"]) + 0.01)
    else:
        rows[-1] = dict(rows[0])
    _write_csv(ai / "review_template_ai_judged.csv", rows, REVIEW_FIELDS)
    with pytest.raises(ValueError, match=error):
        validate_and_score_ai_review(frozen, ai / "review_template_ai_judged.csv")


def test_supplied_metric_drift_is_rejected(tmp_path: Path) -> None:
    frozen, ai, _ = _fixture(tmp_path)
    supplied = json.loads((ai / "ai_review_metrics.json").read_text())
    supplied["per_arm"]["VI_DIRECT"]["ai_graded_relevance_top5"] = 0.5
    write_json(ai / "ai_review_metrics.json", supplied)
    with pytest.raises(ValueError, match="AI_REVIEW_METRIC_MISMATCH"):
        run_stage1e_language_path_freeze(frozen, ai, tmp_path / "stage1e")


def test_stage1e_source_has_no_model_or_retrieval_invocation() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/triage_eg/retrieval/stage1e").glob("*.py")
    )
    for forbidden in (
        "OfflineViEnTranslator",
        "from_pretrained(",
        "load_multimodal_encoder",
        "encode_text(",
        ".search(",
        "build_index(",
        "snapshot_download(",
    ):
        assert forbidden not in source
