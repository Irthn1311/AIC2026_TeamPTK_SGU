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
    DEFAULT_PINNED_REVISION = "5611f34634b72de0608b1238a4e02845ca285f3e"

    def __init__(
        self,
        *,
        model_name_or_path: str | Path | None = None,
        device: str = "auto",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
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
        self.local_files_only = local_files_only
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
            "Loading Marian MT model '%s' on %s (cache_dir=%s, local_files_only=%s, revision=%s)...",
            self.model_name,
            self._device,
            cache_dir,
            local_files_only,
            self.revision,
        )

        resolved_cache = str(cache_dir) if cache_dir else None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=resolved_cache,
                local_files_only=local_files_only,
                revision=self.revision,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=resolved_cache,
                local_files_only=local_files_only,
                revision=self.revision,
            ).to(self._device)
            self.model.eval()
        except Exception as exc:
            if local_files_only:
                try:
                    # 1. Try local cache without revision pin
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        cache_dir=resolved_cache,
                        local_files_only=True,
                    )
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        cache_dir=resolved_cache,
                        local_files_only=True,
                    ).to(self._device)
                    self.model.eval()
                except Exception:
                    try:
                        # 2. If fresh container has empty cache, provision once from Hub
                        logger.info("Fresh container cache miss; provisioning model '%s' from Hugging Face hub...", self.model_name)
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            cache_dir=resolved_cache,
                            local_files_only=False,
                            revision=self.revision,
                        )
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            self.model_name,
                            cache_dir=resolved_cache,
                            local_files_only=False,
                            revision=self.revision,
                        ).to(self._device)
                        self.model.eval()
                    except Exception as dl_exc:
                        raise TranslationError(
                            f"Failed to load/provision Marian MT model from '{self.model_name}' (revision={self.revision}): {dl_exc}"
                        ) from dl_exc
            else:
                raise TranslationError(
                    f"Failed to load Marian MT model from '{self.model_name}' (revision={self.revision}): {exc}"
                ) from exc

        self._torch = torch

    @property
    def provider_name(self) -> str:
        return f"marian-mt:{self.model_name}@{self.revision[:8]}"

    @property
    def device(self) -> str:
        return self._device

    def get_artifact_fingerprint(self) -> dict[str, Any]:
        """Compute artifact provenance and SHA256 fingerprint strictly from local disk."""
        import hashlib
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
                    # 2. Fallback to existing snapshot directory if revision was cached under default branch
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
            primary_weights = "model.safetensors" if (snapshot_dir / "model.safetensors").exists() else "pytorch_model.bin"
            info["primary_weight_artifact"] = primary_weights

            # Scan all files in snapshot_dir as well as any component artifacts in repo cache
            scanned_files: dict[str, Path] = {}
            for fpath in snapshot_dir.iterdir():
                if fpath.is_file():
                    scanned_files[fpath.name] = fpath

            # Also check parent repo snapshots for tokenizer artifacts if stored in parallel snapshot
            repo_root_dir = snapshot_dir.parent.parent
            if repo_root_dir.exists():
                for fpath in repo_root_dir.rglob("*"):
                    if fpath.is_file() and fpath.name not in scanned_files:
                        if any(ext in fpath.name for ext in ["spm", "json", "safetensors", "bin", "model", "txt"]):
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
            info["fingerprint_warning"] = "Local snapshot directory not found in standard cache locations."

        return info

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

    def __init__(self, max_tokens: int = SAFE_CLIP_TOKEN_LIMIT, packing_policy: str = "prefix_77") -> None:
        if max_tokens <= 0 or max_tokens > 75:
            raise ValueError(f"max_tokens must be in range 1..75, got {max_tokens}")
        if packing_policy not in {"prefix_77", "head_tail_77"}:
            raise ValueError(f"packing_policy must be 'prefix_77' or 'head_tail_77', got {packing_policy}")
        self.max_tokens = max_tokens
        self.packing_policy = packing_policy
        self._clip_tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        if self._clip_tokenizer is None:
            try:
                import clip
                self._clip_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
            except ImportError:
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "openai-clip", "ftfy", "regex"],
                    check=False,
                )
                import clip
                self._clip_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        return self._clip_tokenizer

    def count_tokens(self, text: str) -> int:
        """Count exact CLIP BPE tokens for given text (including SOT and EOT)."""
        tokenizer = self._get_tokenizer()
        bpe_tokens = tokenizer.encode(text)
        return len(bpe_tokens) + 2

    def count_clip_tokens(self, text: str) -> int:
        """Alias for count_tokens."""
        return self.count_tokens(text)

    def guard_and_compact(self, text: str, packing_policy: str | None = None) -> tuple[str, int, bool]:
        """Guard text against CLIP token budget.

        If text fits within safe budget (<= 75 tokens), returns (text, token_count, False).
        If text exceeds safe budget (> 75 tokens), compacts to boundary without exceeding budget
        and returns (compacted_text, new_token_count, True).
        """
        policy = packing_policy or self.packing_policy
        tokenizer = self._get_tokenizer()
        bpe_tokens = tokenizer.encode(text)
        raw_count = len(bpe_tokens) + 2

        if raw_count <= (self.max_tokens + 2):
            return text, raw_count, False

        if policy == "head_tail_77":
            # Head + Tail bifurcated packing
            head_budget = int(self.max_tokens * 0.48)  # 36 tokens
            tail_budget = self.max_tokens - head_budget  # 39 tokens
            head_tokens = bpe_tokens[:head_budget]
            tail_tokens = bpe_tokens[-tail_budget:]
            head_text = tokenizer.decode(head_tokens).strip().rstrip(".,; ")
            tail_text = tokenizer.decode(tail_tokens).strip().lstrip(".,; ")
            combined = f"{head_text}, {tail_text}"
            comb_tokens = tokenizer.encode(combined)
            if len(comb_tokens) > self.max_tokens:
                trimmed = comb_tokens[: self.max_tokens]
                compacted = tokenizer.decode(trimmed).strip()
                new_count = len(trimmed) + 2
            else:
                compacted = combined
                new_count = len(comb_tokens) + 2
        else:
            # Default prefix_77: strict BPE prefix truncation
            kept_tokens = bpe_tokens[: self.max_tokens]
            compacted = tokenizer.decode(kept_tokens).strip()
            new_count = len(kept_tokens) + 2

        logger.warning(
            "Query exceeded token budget (%d > %d, policy=%s). Compacted text: %r",
            raw_count,
            self.max_tokens + 2,
            policy,
            compacted,
        )
        return compacted, new_count, True
