"""Production-grade offline translation provider using Helsinki-NLP/opus-mt-vi-en."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when translation generation fails."""


class TranslationProvider(Protocol):
    """Protocol interface for query translation engines."""

    @property
    def provider_name(self) -> str: ...

    @property
    def device(self) -> str: ...

    def translate(self, text: str) -> str: ...


class MarianOfflineTranslator:
    """Offline translation provider backed by Helsinki-NLP/opus-mt-vi-en.

    Uses direct AutoTokenizer + AutoModelForSeq2SeqLM for deterministic
    offline execution (compatible with modern transformers releases).
    """

    DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"

    def __init__(
        self,
        *,
        model_name_or_path: str | Path | None = None,
        device: str = "auto",
        cache_dir: Path | None = None,
        max_length: int = 128,
        num_beams: int = 4,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                f"transformers and torch must be installed to use MarianOfflineTranslator: {exc}"
            ) from exc

        self.model_name = str(model_name_or_path or self.DEFAULT_MODEL_NAME)
        self.max_length = max_length
        self.num_beams = num_beams

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in {"cuda", "cpu"}:
            if device == "cuda" and not torch.cuda.is_available():
                raise TranslationError("CUDA requested for translation but unavailable")
            self._device = device
        else:
            raise ValueError(f"Unsupported device '{device}', must be 'auto', 'cuda', or 'cpu'")

        logger.info(
            "Loading Marian MT model '%s' on %s (cache_dir=%s)...",
            self.model_name,
            self._device,
            cache_dir,
        )

        resolved_cache = str(cache_dir) if cache_dir else None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=resolved_cache,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=resolved_cache,
            ).to(self._device)
            self.model.eval()
        except Exception as exc:
            raise TranslationError(
                f"Failed to load Marian MT model from '{self.model_name}': {exc}"
            ) from exc

        self._torch = torch

    @property
    def provider_name(self) -> str:
        return f"marian-mt:{self.model_name}"

    @property
    def device(self) -> str:
        return self._device

    def translate(self, text: str) -> str:
        """Translate Vietnamese text to English.

        Args:
            text: Raw Vietnamese input text.

        Returns:
            Translated English string.

        Raises:
            TranslationError: if input is empty or generation fails.
        """
        cleaned = text.strip()
        if not cleaned:
            raise TranslationError("Cannot translate empty or whitespace-only text")

        try:
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)

            with self._torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            translated = self.tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
            ).strip()

            if not translated:
                raise TranslationError(f"Translation produced empty output for input: {cleaned!r}")

            return translated
        except Exception as exc:
            if isinstance(exc, TranslationError):
                raise
            raise TranslationError(f"Marian translation generation failed: {exc}") from exc


class TokenBudgetGuard:
    """Validates and enforces that translated English queries fit within CLIP's context budget.

    OpenAI CLIP ViT-B/32 has a maximum context window of 77 tokens (including <start_of_text>
    and <end_of_text>). Usable content tokens must be <= 75.
    """

    SAFE_CLIP_TOKEN_LIMIT = 75

    def __init__(self, max_tokens: int = SAFE_CLIP_TOKEN_LIMIT) -> None:
        if max_tokens <= 0 or max_tokens > 75:
            raise ValueError(f"max_tokens must be in range 1..75, got {max_tokens}")
        self.max_tokens = max_tokens
        self._clip_tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        if self._clip_tokenizer is None:
            try:
                import clip
                self._clip_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
            except ImportError:
                import clip
                self._clip_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        return self._clip_tokenizer

    def count_tokens(self, text: str) -> int:
        """Count exact CLIP BPE tokens for given text (including SOT and EOT)."""
        tokenizer = self._get_tokenizer()
        bpe_tokens = tokenizer.encode(text)
        return len(bpe_tokens) + 2

    def guard_and_compact(self, text: str) -> tuple[str, int, bool]:
        """Guard text against CLIP token budget.

        If text fits within safe budget (<= 75 tokens), returns (text, token_count, False).
        If text exceeds safe budget (> 75 tokens), compacts to boundary without exceeding budget
        and returns (compacted_text, new_token_count, True).
        """
        tokenizer = self._get_tokenizer()
        bpe_tokens = tokenizer.encode(text)
        raw_count = len(bpe_tokens) + 2

        if raw_count <= (self.max_tokens + 2):
            return text, raw_count, False

        # Compaction / safe boundary truncation
        kept_tokens = bpe_tokens[: self.max_tokens]
        compacted = tokenizer.decode(kept_tokens).strip()
        new_count = len(kept_tokens) + 2
        logger.warning(
            "Query exceeded token budget (%d > %d). Compacted text: %r",
            raw_count,
            self.max_tokens + 2,
            compacted,
        )
        return compacted, new_count, True
