"""Regression tests for the canonical VinAI Vietnamese-to-English path."""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from system_tai.kis.session import build_parser, session_config_from_args
from system_tai.kis.session_schema import SessionConfig
from system_tai.translation.provider import (
    TokenBudgetGuard,
    TranslationError,
    VinAITranslateProvider,
)


class _FakeBatch(dict[str, Any]):
    def to(self, device: str) -> _FakeBatch:
        self["moved_to"] = device
        return self


def _install_fake_inference_modules(
    monkeypatch: pytest.MonkeyPatch,
    records: dict[str, Any],
) -> None:
    class FakeTokenizer:
        lang_code_to_id = {"en_XX": 42}

        def __call__(self, texts: list[str], **kwargs: Any) -> _FakeBatch:
            records["tokenize_texts"] = texts
            records["tokenize_kwargs"] = kwargs
            return _FakeBatch(input_ids=list(range(len(texts))))

        def batch_decode(
            self,
            generated: list[int],
            *,
            skip_special_tokens: bool,
        ) -> list[str]:
            records["decode_skip_special_tokens"] = skip_special_tokens
            return [f"translated-{index}" for index in generated]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> FakeTokenizer:
            records["tokenizer_load"] = (model_name, kwargs)
            return FakeTokenizer()

    class FakeModel:
        def to(self, device: str) -> FakeModel:
            records["model_device"] = device
            return self

        def eval(self) -> None:
            records["model_eval"] = True

        def generate(self, **kwargs: Any) -> list[int]:
            records["generate_kwargs"] = kwargs
            return list(kwargs["input_ids"])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> FakeModel:
            records["model_load"] = (model_name, kwargs)
            return FakeModel()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        no_grad=nullcontext,
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForSeq2SeqLM=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_vinai_provider_uses_public_mbart_language_contract_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: dict[str, Any] = {}
    _install_fake_inference_modules(monkeypatch, records)

    provider = VinAITranslateProvider(device="cpu", allow_model_download=False)
    translated = provider.translate_many(["câu một", "câu hai"])

    assert translated == ("translated-0", "translated-1")
    assert provider.provider_name.startswith("vinai-translate:vinai/")
    tokenizer_model, tokenizer_kwargs = records["tokenizer_load"]
    model_name, model_kwargs = records["model_load"]
    assert tokenizer_model == model_name == "vinai/vinai-translate-vi2en-v2"
    assert tokenizer_kwargs["src_lang"] == "vi_VN"
    assert tokenizer_kwargs["local_files_only"] is True
    assert model_kwargs["local_files_only"] is True
    assert tokenizer_kwargs["revision"] == VinAITranslateProvider.DEFAULT_PINNED_REVISION
    assert records["generate_kwargs"]["decoder_start_token_id"] == 42
    assert records["generate_kwargs"]["num_beams"] == 5
    assert records["model_eval"] is True


def test_vinai_provider_download_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: dict[str, Any] = {}
    _install_fake_inference_modules(monkeypatch, records)
    VinAITranslateProvider(device="cpu", allow_model_download=True)
    assert records["tokenizer_load"][1]["local_files_only"] is False
    assert records["model_load"][1]["local_files_only"] is False


def test_vinai_provider_rejects_missing_target_language_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TokenizerWithoutTarget:
        lang_code_to_id: dict[str, int] = {}

        @staticmethod
        def convert_tokens_to_ids(_value: str) -> int:
            return -1

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: Any) -> TokenizerWithoutTarget:
            return TokenizerWithoutTarget()

    class Model:
        def to(self, _device: str) -> Model:
            return self

        def eval(self) -> None:
            return None

    class AutoModel:
        @staticmethod
        def from_pretrained(_name: str, **_kwargs: Any) -> Model:
            return Model()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            no_grad=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForSeq2SeqLM=AutoModel,
            AutoTokenizer=AutoTokenizer,
        ),
    )
    with pytest.raises(TranslationError, match="en_XX"):
        VinAITranslateProvider(device="cpu")


class _WordTokenizer:
    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()


def test_clip_segmentation_is_lossless_and_never_truncates() -> None:
    guard = TokenBudgetGuard(max_tokens=5)
    guard._clip_tokenizer = _WordTokenizer()
    source = "one two three four five, six seven eight nine ten eleven twelve"

    segments = guard.split_for_clip(source)

    assert len(segments) == 3
    assert " ".join(segments).split() == source.split()
    assert all(guard.count_tokens(segment) <= 7 for segment in segments)


def test_production_config_selects_vinai_and_explicit_download(tmp_path: Path) -> None:
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        """
system:
  device: cpu
kis:
  enable_dynamic_translation: true
  translation_model_name: vinai/vinai-translate-vi2en-v2
  translation_allow_model_download: true
  translation_max_clip_tokens: 75
""".strip(),
        encoding="utf-8",
    )

    config = SessionConfig.from_yaml(config_path)

    assert config.enable_dynamic_translation is True
    assert config.translation_model_name == "vinai/vinai-translate-vi2en-v2"
    assert config.translation_allow_model_download is True
    assert config.translation_max_clip_tokens == 75


def test_invalid_clip_segment_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 1 and 75"):
        SessionConfig(translation_max_clip_tokens=76)


def test_session_cli_exposes_explicit_vinai_translation_flags(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--input-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "output"),
            "--enable-dynamic-translation",
            "--translation-cache-dir",
            str(tmp_path / "vinai-cache"),
            "--translation-device",
            "cpu",
            "--translation-allow-model-download",
        ]
    )

    config = session_config_from_args(args)

    assert config.enable_dynamic_translation is True
    assert config.translation_model_name == "vinai/vinai-translate-vi2en-v2"
    assert config.translation_cache_dir == tmp_path / "vinai-cache"
    assert config.translation_device == "cpu"
    assert config.translation_allow_model_download is True
