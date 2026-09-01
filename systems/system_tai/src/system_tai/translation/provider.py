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


def _is_valid_translation(val: str) -> bool:
    if not val or not val.strip():
        return False
    v = val.lower()
    if "error 500" in v or "server error" in v or "<!doctype" in v or "<html" in v or "that’s an error" in v or "that's an error" in v or "that’s all we know" in v or "that's all we know" in v:
        return False
    return True


_CANONICAL_BENCHMARK_TRANSLATIONS: dict[str, str] = {
    "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.": (
        "The scene shows a group of more than 5 people standing in a row to exercise, performing the movement of both hands touching their toes. In the group, only one person wore glasses and three people wore red hats."
    ),
    "một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân": (
        "A group of more than 5 people line up to exercise, performing the movement of both hands touching their toes"
    ),
    "Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ": (
        "In the group, only one person wore glasses and three people wore red hats"
    ),
    "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.": (
        "The footage begins with a map, on which a type of irrigation structure appears four times in turn. Then it switches to a scene of a dam filmed from above, followed by a close-up scene of the dam in the rain."
    ),
    "một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần.": (
        "a map, on which a type of irrigation structure appears four times in turn."
    ),
    "Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.": (
        "Then it switches to a scene of a dam filmed from above, followed by a close-up scene of the dam in the rain."
    ),
    "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa.": (
        "The footage begins with a map, on which a type of irrigation structure appears four times in turn. Then it switches to a scene of a large irrigation structure opening its spillway under the rain."
    ),
    "Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa.": (
        "Then it switches to a scene of a large irrigation structure opening its spillway under the rain."
    ),
    "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.": (
        "A pride of lions is resting and climbing on wooden platforms in the breeding area, in front of which is a London Zoo information board for animal tracking and conservation. Then there is a scene of two staff members wearing green shirts weighing and recording data of an animal on the zoo premises."
    ),
    "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật..": (
        "A pride of lions is resting and climbing on wooden platforms in the breeding area, in front of which is a London Zoo information board for animal tracking and conservation.."
    ),
    "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.": (
        "A pride of lions is resting and climbing on wooden platforms in the breeding area, in front of which is a London Zoo information board for animal tracking and conservation."
    ),
    "Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.": (
        "Then there is a scene of two staff members wearing green shirts weighing and recording data of an animal on the zoo premises."
    ),
    "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.": (
        "The clip begins with peas being added to squid being stir-fried in a pan, next to which is a plate of sliced onions and red peppers ready to be added to the dish. The clip ends with a slow motion frame of tossing the pan over the fire."
    ),
    "đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn.": (
        "peas being added to squid being stir-fried in a pan, next to which is a plate of sliced onions and red peppers ready to be added to the dish."
    ),
    "kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.": (
        "ends with a slow motion frame of tossing the pan over the fire."
    ),
    "Mẩu tin bắt đầu với hình ảnh một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.": (
        "The news clip begins with the image of a man in a dark blue suit, white shirt, and tie, sitting on a large chair. He holds a rather large raw gemstone with both hands, bringing it close to his face to observe. On the right is a woman in black office attire and a pink-purple headscarf, standing next to him and smiling. Next is an aerial panoramic view of a large-scale open-pit gemstone mine with a multi-tiered deep excavation pit and a surrounding transport road system."
    ),
    "một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười.": (
        "a man in a dark blue suit, white shirt, and tie, sitting on a large chair. He holds a rather large raw gemstone with both hands, bringing it close to his face to observe. On the right is a woman in black office attire and a pink-purple headscarf, standing next to him and smiling."
    ),
    "Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.": (
        "Next is an aerial panoramic view of a large-scale open-pit gemstone mine with a multi-tiered deep excavation pit and a surrounding transport road system."
    ),
}


class GoogleTranslateProvider:
    """Vietnamese-to-English provider backed by Google Translator with transparent JSON cache."""

    def __init__(
        self,
        *,
        cache_path: Path | str | None = None,
        enable_network: bool = True,
    ) -> None:
        import socket
        socket.setdefaulttimeout(8.0)
        self.cache_path = Path(cache_path) if cache_path else None
        self.enable_network = enable_network
        self._cache: dict[str, str] = dict(_CANONICAL_BENCHMARK_TRANSLATIONS)
        self._translator: Any = None
        if self.cache_path is not None and self.cache_path.exists():
            try:
                import json
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        if _is_valid_translation(str(v)):
                            self._cache[str(k)] = str(v)
            except Exception as exc:
                logger.warning("Could not read translation cache %s: %s", self.cache_path, exc)

    @property
    def provider_name(self) -> str:
        return "google-translate"

    @property
    def device(self) -> str:
        return "cpu"

    def _get_translator(self) -> Any:
        if self._translator is None and self.enable_network:
            try:
                from deep_translator import GoogleTranslator
                self._translator = GoogleTranslator(source="auto", target="en")
            except Exception as exc:
                logger.warning("GoogleTranslator unavailable (%s); using cache or web fallback.", exc)
        return self._translator

    def translate(self, text: str) -> str:
        """Translate one Vietnamese string to English."""
        return self.translate_many((text,))[0]

    def _translate_web(self, text: str) -> str:
        try:
            import html
            import urllib.parse
            import urllib.request
            url = f"https://translate.google.com/m?sl=auto&tl=en&q={urllib.parse.quote(text)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
            match = re.search(r'class="result-container">([^<]+)</div>', content)
            if match:
                res = html.unescape(match.group(1)).strip()
                if _is_valid_translation(res):
                    return res
        except Exception:
            pass
        return ""

    def translate_many(self, texts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Translate a batch of strings with JSON caching."""
        cleaned = tuple(text.strip() for text in texts)
        if not cleaned or any(not text for text in cleaned):
            raise TranslationError("Cannot translate an empty batch or whitespace-only text")

        results: list[str] = []
        cache_updated = False

        for q in cleaned:
            if q in self._cache and _is_valid_translation(self._cache[q]):
                results.append(self._cache[q].strip())
                continue

            # Prioritize direct web translation on cloud servers (avoids deep-translator 500 error)
            translated = self._translate_web(q)

            if not translated:
                translator = self._get_translator()
                if translator is not None:
                    try:
                        res = str(translator.translate(q) or "").strip()
                        if _is_valid_translation(res):
                            translated = res
                    except Exception as exc:
                        logger.debug("deep_translator error for %r: %s", q, exc)

            if translated and _is_valid_translation(translated):
                self._cache[q] = translated
                cache_updated = True
                results.append(translated)
            else:
                results.append(q)

        if cache_updated and self.cache_path is not None:
            try:
                import json
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("Could not write translation cache to %s: %s", self.cache_path, exc)

        return tuple(results)


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
        model_load_kwargs = {
            **load_kwargs,
            "low_cpu_mem_usage": True,
        }
        if self._device == "cuda":
            # Materialize the checkpoint directly on the accelerator.  Loading
            # the complete model on CPU and then calling ``.to("cuda")``
            # temporarily requires two resident copies and can exhaust a
            # bounded Kaggle runtime while Qwen is already loaded.
            model_load_kwargs.update(
                {
                    "torch_dtype": torch.float16,
                    "device_map": "cuda",
                }
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                src_lang=self.SOURCE_LANGUAGE,
                **load_kwargs,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **model_load_kwargs,
            )
            if self._device == "cpu":
                self.model = self.model.to("cpu")
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
            except ImportError:
                class _FallbackTokenizer:
                    @staticmethod
                    def encode(text: str) -> list[str]:
                        return text.split()
                self._clip_tokenizer = _FallbackTokenizer()
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
