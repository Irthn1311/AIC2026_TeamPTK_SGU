"""Offline translation providers and lossless CLIP query segmentation."""

from __future__ import annotations

import hashlib
import logging
import os
import re
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


class VinAITranslateProvider:
    """Vietnamese-to-English provider backed by VinAI Translate v2.

    The implementation follows VinAI's public mBART inference contract:
    ``src_lang='vi_VN'`` and ``decoder_start_token_id`` for ``en_XX``.
    Model download is opt-in; a missing local checkpoint fails clearly when
    ``allow_model_download`` is false.
    """

    DEFAULT_MODEL_NAME = "vinai/vinai-translate-vi2en-v2"
    DEFAULT_PINNED_REVISION = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    SOURCE_LANGUAGE = "vi_VN"
    TARGET_LANGUAGE = "en_XX"

    def __init__(
        self,
        *,
        model_name_or_path: str | Path | None = None,
        device: str = "auto",
        cache_dir: Path | None = None,
        allow_model_download: bool = False,
        revision: str | None = None,
        max_length: int = 1024,
        num_beams: int = 5,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                "transformers and torch must be installed to use "
                f"VinAITranslateProvider: {exc}"
            ) from exc

        self.model_name = str(model_name_or_path or self.DEFAULT_MODEL_NAME)
        self.max_length = max_length
        self.num_beams = num_beams
        self.allow_model_download = allow_model_download
        self.revision = revision or self.DEFAULT_PINNED_REVISION

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
            "Loading VinAI Translate model '%s' on %s "
            "(cache_dir=%s, allow_model_download=%s, revision=%s)...",
            self.model_name,
            self._device,
            cache_dir,
            allow_model_download,
            self.revision,
        )

        resolved_cache = str(cache_dir) if cache_dir else None
        load_kwargs = {
            "cache_dir": resolved_cache,
            "local_files_only": not allow_model_download,
            "revision": self.revision,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                src_lang=self.SOURCE_LANGUAGE,
                **load_kwargs,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            ).to(self._device)
            self.model.eval()
        except Exception as exc:
            download_hint = (
                " Enable translation_allow_model_download explicitly to provision it."
                if not allow_model_download
                else ""
            )
            raise TranslationError(
                "Failed to load VinAI Translate model "
                f"'{self.model_name}' (revision={self.revision}): {exc}.{download_hint}"
            ) from exc

        language_ids = getattr(self.tokenizer, "lang_code_to_id", None) or {}
        target_language_id = language_ids.get(self.TARGET_LANGUAGE)
        if target_language_id is None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            target_language_id = self.tokenizer.convert_tokens_to_ids(self.TARGET_LANGUAGE)
        if not isinstance(target_language_id, int) or target_language_id < 0:
            raise TranslationError(
                f"VinAI tokenizer does not expose target language token {self.TARGET_LANGUAGE!r}"
            )
        self.target_language_id = target_language_id

        self._torch = torch

    @property
    def provider_name(self) -> str:
        return f"vinai-translate:{self.model_name}@{self.revision[:8]}"

    @property
    def device(self) -> str:
        return self._device

    def get_artifact_fingerprint(self) -> dict[str, Any]:
        """Compute artifact provenance and SHA256 fingerprint strictly from local disk."""
        info: dict[str, Any] = {
            "model_name": self.model_name,
            "pinned_revision": self.revision,
            "device": self._device,
        }

        repo_id_normalized = f"models--{self.model_name.replace('/', '--')}"
        possible_roots = [
            Path.home() / ".cache" / "huggingface" / "hub" / repo_id_normalized,
            Path("/root/.cache/huggingface/hub") / repo_id_normalized,
            Path("/kaggle/working/.cache/huggingface/hub") / repo_id_normalized,
        ]

        snapshot_dir: Path | None = None
        for r in possible_roots:
            if r.exists():
                snaps = r / "snapshots"
                if snaps.exists():
                    # 1. Prioritize exact pinned revision
                    if self.revision and (snaps / self.revision).is_dir():
                        snapshot_dir = snaps / self.revision
                        break
                    # Fall back to an existing snapshot if the revision was
                    # cached under the repository's default branch.
                    for snap in snaps.iterdir():
                        if snap.is_dir():
                            snapshot_dir = snap
                            break
            if snapshot_dir:
                break

        if snapshot_dir and snapshot_dir.exists():
            info["resolved_snapshot_dir"] = str(snapshot_dir)
            info["snapshot_commit_hash"] = snapshot_dir.name
            info["revision_matches_snapshot"] = (
                True if (self.revision and snapshot_dir.name == self.revision) else False
            )

            # Determine primary weight artifact
            primary_weights = (
                "model.safetensors"
                if (snapshot_dir / "model.safetensors").exists()
                else "pytorch_model.bin"
            )
            info["primary_weight_artifact"] = primary_weights

            # Scan all files in snapshot_dir as well as any component artifacts in repo cache
            scanned_files: dict[str, Path] = {}
            for fpath in snapshot_dir.iterdir():
                if fpath.is_file():
                    scanned_files[fpath.name] = fpath

            # Check parent snapshots for tokenizer artifacts stored elsewhere.
            repo_root_dir = snapshot_dir.parent.parent
            if repo_root_dir.exists():
                for fpath in repo_root_dir.rglob("*"):
                    if fpath.is_file() and fpath.name not in scanned_files:
                        artifact_markers = (
                            "spm",
                            "json",
                            "safetensors",
                            "bin",
                            "model",
                            "txt",
                        )
                        if any(ext in fpath.name for ext in artifact_markers):
                            scanned_files[fpath.name] = fpath

            for fname, fpath in sorted(scanned_files.items()):
                try:
                    info[f"{fname}_size_bytes"] = fpath.stat().st_size
                    h = hashlib.sha256()
                    with open(fpath, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    info[f"{fname}_sha256"] = h.hexdigest()
                except Exception as exc:
                    info[f"{fname}_hash_error"] = str(exc)
        else:
            info["fingerprint_warning"] = (
                "Local snapshot directory not found in standard cache locations."
            )

        return info

    def translate(self, text: str) -> str:
        """Translate one Vietnamese string to English."""
        return self.translate_many((text,))[0]

    def translate_many(self, texts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Translate a batch while loading and reusing the model only once."""
        cleaned = tuple(text.strip() for text in texts)
        if not cleaned or any(not text for text in cleaned):
            raise TranslationError("Cannot translate an empty batch or whitespace-only text")

        try:
            inputs = self.tokenizer(
                list(cleaned),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)

            with self._torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    decoder_start_token_id=self.target_language_id,
                    max_length=self.max_length,
                    num_return_sequences=1,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            translated = tuple(
                value.strip()
                for value in self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
            )
            if len(translated) != len(cleaned):
                raise TranslationError(
                    "VinAI translation returned "
                    f"{len(translated)} rows for {len(cleaned)} inputs"
                )
            if any(not value for value in translated):
                raise TranslationError("VinAI translation produced an empty output")
            return translated
        except Exception as exc:
            if isinstance(exc, TranslationError):
                raise
            raise TranslationError(f"VinAI translation generation failed: {exc}") from exc


class NLLBOfflineTranslator:
    """Experimental Multilingual VI->EN Translator using NLLB-200 distilled 600M (P1B candidate).

    Default model: 'facebook/nllb-200-distilled-600M'
    Source language: 'vie_Latn'
    Target language: 'eng_Latn'
    """

    DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "auto",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        max_length: int = 256,
        num_beams: int = 4,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name_or_path
        self.cache_dir = cache_dir or Path(
            os.environ.get(
                "HF_HOME",
                Path("/kaggle/working/hf_cache")
                if Path("/kaggle/working").exists()
                else Path.home() / ".cache" / "huggingface",
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.num_beams = num_beams

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        logger.info(
            "Initializing NLLBOfflineTranslator with %s on device %s",
            self.model_name,
            self._device,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                src_lang="vie_Latn",
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            ).to(self._device)
            self.model.eval()
            if hasattr(self.tokenizer, "lang_code_to_id") and self.tokenizer.lang_code_to_id:
                self.target_lang_id = self.tokenizer.lang_code_to_id.get("eng_Latn")
            else:
                self.target_lang_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
            if self.target_lang_id is None:
                self.target_lang_id = self.tokenizer.get_vocab().get("eng_Latn")
        except Exception as exc:
            raise TranslationError(f"Failed to load NLLB model: {exc}") from exc

    def translate(self, text: str) -> str:
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
                    forced_bos_token_id=self.target_lang_id,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )

            translated = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )[0].strip()

            if not translated:
                raise TranslationError(
                    f"NLLB translation produced empty output for input: {cleaned!r}"
                )
            return translated
        except Exception as exc:
            if isinstance(exc, TranslationError):
                raise
            raise TranslationError(f"NLLB translation generation failed: {exc}") from exc



class TokenBudgetGuard:
    """Split translated English into lossless CLIP-sized query segments.

    OpenAI CLIP ViT-B/32 has a maximum context window of 77 tokens (including <start_of_text>
    and <end_of_text>). Usable content tokens must be <= 75.
    """

    SAFE_CLIP_TOKEN_LIMIT = 75
    _BOUNDARY_RE = re.compile(r"(?<=[.!?;:,])\s+")

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
            except ImportError as exc:
                raise TranslationError(
                    "OpenAI CLIP must be installed before segmenting translated queries"
                ) from exc
        return self._clip_tokenizer

    def count_tokens(self, text: str) -> int:
        """Count exact CLIP BPE tokens for given text (including SOT and EOT)."""
        tokenizer = self._get_tokenizer()
        bpe_tokens = tokenizer.encode(text)
        return len(bpe_tokens) + 2

    def count_clip_tokens(self, text: str) -> int:
        """Alias for count_tokens."""
        return self.count_tokens(text)

    def split_for_clip(self, text: str) -> tuple[str, ...]:
        """Return CLIP-sized segments without dropping translated words."""
        cleaned = " ".join(text.split())
        if not cleaned:
            raise TranslationError("Cannot segment empty translated text")
        if self.count_tokens(cleaned) <= self.max_tokens + 2:
            return (cleaned,)

        clauses = tuple(
            clause.strip()
            for clause in self._BOUNDARY_RE.split(cleaned)
            if clause.strip()
        )
        segments: list[str] = []
        for clause in clauses:
            current: list[str] = []
            for word in clause.split():
                proposed = " ".join((*current, word))
                if current and self.count_tokens(proposed) > self.max_tokens + 2:
                    segments.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
                if self.count_tokens(" ".join(current)) > self.max_tokens + 2:
                    raise TranslationError(
                        "A single translated token cannot fit within the CLIP context budget"
                    )
            if current:
                segments.append(" ".join(current))

        if not segments or any(
            self.count_tokens(segment) > self.max_tokens + 2 for segment in segments
        ):
            raise TranslationError("Failed to segment translation within CLIP token budget")
        if " ".join(segments).split() != cleaned.split():
            raise TranslationError("Lossless query segmentation invariant failed")
        return tuple(segments)
