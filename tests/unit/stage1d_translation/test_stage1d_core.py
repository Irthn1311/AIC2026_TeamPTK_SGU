from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.stage1c.contracts import QueryRecord
from triage_eg.retrieval.stage1d.artifacts import create_stage1d_bundle
from triage_eg.retrieval.stage1d.contracts import (
    TRANSLATOR_MODEL_ID,
    TRANSLATOR_REVISION,
    GenerationConfig,
    ReviewConfig,
    TranslatorConfig,
)
from triage_eg.retrieval.stage1d.inputs import (
    TRANSLATOR_REQUIRED,
    resolve_input_root,
    validate_translator_asset,
)
from triage_eg.retrieval.stage1d.metrics import arm_overlap, pair_comparison
from triage_eg.retrieval.stage1d.review import score_stage1d_review, write_blinded_review
from triage_eg.retrieval.stage1d.translator import (
    OfflineViEnTranslator,
    translator_dependency_versions,
)


def _translator_asset(root: Path) -> Path:
    for relative in TRANSLATOR_REQUIRED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "manifests/MODEL_REVISION.txt":
            path.write_text(TRANSLATOR_REVISION, encoding="utf-8")
        elif relative not in {
            "manifests/asset_manifest.json",
            "manifests/file_inventory.jsonl",
        }:
            path.write_bytes((relative + "\n").encode())
    hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in TRANSLATOR_REQUIRED
        if relative.startswith("model/")
    }
    (root / "manifests/asset_manifest.json").write_text(
        json.dumps(
            {
                "model_id": TRANSLATOR_MODEL_ID,
                "exact_revision": TRANSLATOR_REVISION,
                "architecture": "MarianMT / transformer-align",
                "runtime_model_path": "model",
                "internet_required_at_runtime": False,
                "file_hashes": hashes,
            }
        ),
        encoding="utf-8",
    )
    (root / "manifests/file_inventory.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "path": relative,
                    "size_bytes": (root / relative).stat().st_size,
                    "sha256": digest,
                }
            )
            + "\n"
            for relative, digest in hashes.items()
        ),
        encoding="utf-8",
    )
    return root


def test_translator_asset_preflight_and_fail_closed_cases(tmp_path: Path) -> None:
    asset = _translator_asset(tmp_path / "asset")
    validated = validate_translator_asset(asset)
    assert validated["status"] == "VALID"
    assert validated["hash_verification"] == "PASS"
    assert validated["model_root"].is_absolute()
    assert len(validated["runtime_files"]) == 7

    (asset / "model/config.json").write_text("tampered")
    with pytest.raises(ValueError, match="TRANSLATOR_FILE_HASH_MISMATCH"):
        validate_translator_asset(asset)
    with pytest.raises(FileNotFoundError, match="TRANSLATOR_ASSET_NOT_FOUND"):
        validate_translator_asset(tmp_path / "missing")


def test_translator_asset_rejects_wrong_revision_and_manifest(tmp_path: Path) -> None:
    asset = _translator_asset(tmp_path / "asset")
    (asset / "manifests/MODEL_REVISION.txt").write_text("main")
    with pytest.raises(ValueError, match="TRANSLATOR_REVISION_MISMATCH"):
        validate_translator_asset(asset)
    asset = _translator_asset(tmp_path / "asset2")
    (asset / "manifests/asset_manifest.json").write_text("not json")
    with pytest.raises(ValueError, match="TRANSLATOR_ASSET_MANIFEST_INVALID"):
        validate_translator_asset(asset)


def test_input_resolution_supports_direct_nested_and_zip(tmp_path: Path) -> None:
    direct = _translator_asset(tmp_path / "direct")
    root, mode = resolve_input_root(
        direct,
        required=TRANSLATOR_REQUIRED,
        materialize_root=tmp_path / "unused",
        archive_keyword="opus-mt-vi-en",
    )
    assert (root, mode) == (direct.resolve(), "DIRECT_DIRECTORY")

    nested = _translator_asset(tmp_path / "mount/outer/asset")
    root, mode = resolve_input_root(
        tmp_path / "mount",
        required=TRANSLATOR_REQUIRED,
        materialize_root=tmp_path / "unused2",
        archive_keyword="opus-mt-vi-en",
    )
    assert (root, mode) == (nested.resolve(), "DIRECT_DIRECTORY")

    archive = tmp_path / "aic2026-opus-mt-vi-en.zip"
    with ZipFile(archive, "w") as stream:
        for path in direct.rglob("*"):
            if path.is_file():
                stream.write(path, Path("bundle") / path.relative_to(direct))
    root, mode = resolve_input_root(
        archive,
        required=TRANSLATOR_REQUIRED,
        materialize_root=tmp_path / "materialized",
        archive_keyword="opus-mt-vi-en",
    )
    assert mode == "EXTRACTED_ZIP"
    assert root.name == "bundle"


def test_input_resolution_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "opus-mt-vi-en.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("../escape/model/config.json", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        resolve_input_root(
            archive,
            required=TRANSLATOR_REQUIRED,
            materialize_root=tmp_path / "materialized",
            archive_keyword="opus-mt-vi-en",
        )


class _Tensor:
    def to(self, device: str) -> _Tensor:
        self.device = device
        return self


class _Tokenizer:
    loaded: list[tuple[str, bool]] = []

    @classmethod
    def from_pretrained(cls, path: str, *, local_files_only: bool):
        cls.loaded.append((path, local_files_only))
        return cls()

    def __call__(self, texts, **kwargs):
        self.texts = list(texts)
        self.tokenize_kwargs = kwargs
        return {"input_ids": _Tensor()}

    def batch_decode(self, generated, *, skip_special_tokens: bool):
        assert skip_special_tokens
        return [f"  translated {index}  " for index in range(len(generated))]


class _Model:
    loaded: list[tuple[str, bool]] = []

    def __init__(self):
        self.generation_config = SimpleNamespace()
        self.generate_calls = []

    @classmethod
    def from_pretrained(cls, path: str, *, local_files_only: bool):
        cls.loaded.append((path, local_files_only))
        return cls()

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [object(), object()][: len(getattr(kwargs["input_ids"], "batch", [1, 2]))]


class _Inference:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _module_loader(name: str):
    if name == "torch":
        return SimpleNamespace(
            __version__="test",
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
            inference_mode=lambda: _Inference(),
        )
    if name == "transformers":
        return SimpleNamespace(
            __version__="test",
            AutoTokenizer=_Tokenizer,
            AutoModelForSeq2SeqLM=_Model,
        )
    return SimpleNamespace(__version__="test")


def test_offline_translator_uses_local_absolute_path_and_deterministic_generation(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    translator = OfflineViEnTranslator(
        model,
        TranslatorConfig(batch_size=2),
        GenerationConfig(),
        module_loader=_module_loader,
    ).load()
    result = translator.translate(["một", "hai"])
    assert [item["translated_text_for_clip"] for item in result] == [
        "translated 0",
        "translated 1",
    ]
    assert _Tokenizer.loaded[-1] == (str(model.resolve()), True)
    assert _Model.loaded[-1] == (str(model.resolve()), True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert translator.model.generate_calls[0]["do_sample"] is False
    assert translator.model.generate_calls[0]["num_beams"] == 4
    assert translator.load() is translator


def test_missing_dependency_is_reported() -> None:
    def loader(name: str):
        if name in {"torch", "sentencepiece"}:
            raise ImportError(name)
        return SimpleNamespace(__version__="test")

    with pytest.raises(ImportError, match="sentencepiece, torch"):
        translator_dependency_versions(loader)


def test_empty_translation_is_rejected(tmp_path: Path) -> None:
    class EmptyTokenizer(_Tokenizer):
        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool):
            return cls()

        def batch_decode(self, generated, *, skip_special_tokens: bool):
            return ["   "]

    class SingleModel(_Model):
        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool):
            return cls()

        def generate(self, **kwargs):
            return [object()]

    def loader(name: str):
        if name == "transformers":
            return SimpleNamespace(
                __version__="test",
                AutoTokenizer=EmptyTokenizer,
                AutoModelForSeq2SeqLM=SingleModel,
            )
        return _module_loader(name)

    model = tmp_path / "model"
    model.mkdir()
    translator = OfflineViEnTranslator(
        model,
        TranslatorConfig(),
        GenerationConfig(),
        module_loader=loader,
    ).load()
    with pytest.raises(ValueError, match="TRANSLATION_EMPTY"):
        translator.translate(["nội dung"])


def _frame(rank: int, row: int, video: str = "V1") -> dict[str, object]:
    return {
        "rank": rank,
        "global_row": row,
        "video_id": video,
        "n": rank,
        "original_frame_idx": row * 3,
        "score": 1.0 / rank,
        "keyframe_relative_path": f"keyframes/{video}/{rank:03d}.jpg",
    }


def test_overlap_and_text_space_diagnostics_are_safe() -> None:
    left = [_frame(i + 1, i, f"V{i // 2}") for i in range(50)]
    same = [_frame(i + 1, i, f"V{i // 2}") for i in range(50)]
    other = [_frame(i + 1, i + 100, f"X{i}") for i in range(50)]
    assert arm_overlap(left, same)["frame"]["top50"]["jaccard"] == 1.0
    assert arm_overlap(left, other)["frame"]["top50"]["jaccard"] == 0.0
    item = pair_comparison(
        pair_id="p1",
        category="OBJECT",
        difficulty="EASY",
        en_text="car",
        vi_text="xe",
        translated_text="car",
        en_embedding=np.array([1, 0]),
        vi_embedding=np.array([0, 1]),
        translated_embedding=np.array([1, 0]),
        en_frames=left,
        vi_frames=other,
        translated_frames=same,
        structural={arm: {} for arm in ("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN")},
    )
    assert item["text_space"]["clip_text_cosine_en_translated"] == 1.0
    assert item["ranking_alignment"]["translated_minus_vi_frame_jaccard"] == 1.0


def _pairs_and_frames(count: int = 14):
    pairs = {}
    frames = {}
    for index in range(count):
        pair_id = f"p{index:02d}"
        pairs[pair_id] = {
            "en": QueryRecord(f"{pair_id}_en", pair_id, "en", "OBJECT", "EASY", "car", ""),
            "vi": QueryRecord(f"{pair_id}_vi", pair_id, "vi", "OBJECT", "EASY", "xe", ""),
        }
        frames[pair_id] = {
            arm: [_frame(rank, index * 1000 + offset * 10 + rank) for rank in range(1, 6)]
            for offset, arm in enumerate(("EN_DIRECT", "VI_DIRECT", "VI_TRANSLATED_EN"))
        }
    return pairs, frames


def test_blinded_review_has_210_rows_and_scores_without_recall(tmp_path: Path) -> None:
    pairs, frames = _pairs_and_frames()
    count = write_blinded_review(
        tmp_path, pairs=pairs, frames_by_pair_arm=frames, config=ReviewConfig()
    )
    assert count == 210
    template = tmp_path / "review/review_template_blinded.csv"
    with template.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "actual_arm" not in rows[0]
    assert "translated" not in rows[0]
    for index, row in enumerate(rows):
        row["review_label"] = ("RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN")[index % 4]
    reviewed = tmp_path / "reviewed.csv"
    with reviewed.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = [
        {"pair_id": pair_id, "category": "OBJECT", "difficulty": "EASY"}
        for pair_id in sorted(pairs)
    ]
    (tmp_path / "comparisons").mkdir()
    (tmp_path / "comparisons/pair_comparisons.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in comparisons)
    )
    metrics = score_stage1d_review(tmp_path, reviewed)
    assert metrics["human_review_status"] == "COMPLETE"
    assert metrics["per_arm"]["EN_DIRECT"]["uncertain_count"] > 0
    metric_names = json.dumps(
        {"per_arm": metrics["per_arm"], "ablation_deltas": metrics["ablation_deltas"]}
    )
    assert "Recall" not in metric_names


def test_review_rejects_invalid_label_and_identity_mutation(tmp_path: Path) -> None:
    pairs, frames = _pairs_and_frames(1)
    write_blinded_review(tmp_path, pairs=pairs, frames_by_pair_arm=frames, config=ReviewConfig())
    template = tmp_path / "review/review_template_blinded.csv"
    with template.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["review_label"] = "BAD"
    target = tmp_path / "bad.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="REVIEW_LABEL_INVALID"):
        score_stage1d_review(tmp_path, target)
    rows[0]["review_label"] = ""
    rows[0]["global_row"] = "999"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="REVIEW_IDENTITY_MISMATCH"):
        score_stage1d_review(tmp_path, target)


def test_bundle_allowlist_excludes_weights_indexes_videos_and_logs(tmp_path: Path) -> None:
    root = tmp_path / "output"
    core = (
        "run_manifest.json",
        "stage1d_summary.json",
        "stage1d_report.md",
        "translator/translator_contract.json",
        "translator/translator_runtime_manifest.json",
        "translator/asset_validation.json",
        "translations/translations.jsonl",
        "comparisons/pair_comparisons.jsonl",
        "review/review_template_blinded.csv",
        "review/review_key.json",
        "review/review_instructions.md",
        "issues.jsonl",
    )
    for relative in core:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    for relative in (
        "translated_queries/p1/translation.json",
        "translated_queries/p1/query.json",
        "translated_queries/p1/ranked_frames.jsonl",
        "translated_queries/p1/ranked_videos.jsonl",
        "translated_queries/p1/kis_candidates.csv",
        "translated_queries/p1/retrieval_diagnostics.json",
        "comparisons/p1/en_direct_top20.jsonl",
        "comparisons/p1/vi_direct_top20.jsonl",
        "comparisons/p1/vi_translated_en_top20.jsonl",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    for relative in ("pytorch_model.bin", "index/vectors.npy", "raw/video.mp4", "logs/run.txt"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"forbidden")
    archive = create_stage1d_bundle(root, tmp_path / "bundle.zip")
    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert "translated_queries/p1/ranked_frames.jsonl" in names
    assert "comparisons/p1/en_direct_top20.jsonl" in names
    assert not any(
        name.endswith((".bin", ".npy", ".mp4")) or name.startswith("logs/") for name in names
    )
