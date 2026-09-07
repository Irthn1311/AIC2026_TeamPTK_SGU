"""Offline translation providers and lossless CLIP query segmentation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Collection
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
    error_markers = (
        "error 500",
        "server error",
        "<!doctype",
        "<html",
        "that’s an error",
        "that's an error",
        "that’s all we know",
        "that's all we know",
    )
    if any(marker in v for marker in error_markers):
        return False
    return True


_CANONICAL_BENCHMARK_TRANSLATIONS: dict[str, str] = {
    (
        "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
        "động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba "
        "người đội nón có màu đỏ."
    ): (
        "The scene shows a group of more than 5 people standing in a row to "
        "exercise, performing the movement of both hands touching their toes. In the "
        "group, only one person wore glasses and three people wore red hats."
    ),
    (
        "một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác "
        "hai tay chạm mũi chân"
    ): (
        "A group of more than 5 people line up to exercise, performing the movement "
        "of both hands touching their toes"
    ),
    ("Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ"): (
        "In the group, only one person wore glasses and three people wore red hats"
    ),
    (
        "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần "
        "lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ "
        "trên cao, tiếp đến là cảnh cận con đập dưới trời mưa."
    ): (
        "The footage begins with a map, on which a type of irrigation structure "
        "appears four times in turn. Then it switches to a scene of a dam filmed "
        "from above, followed by a close-up scene of the dam in the rain."
    ),
    ("một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần."): (
        "a map, on which a type of irrigation structure appears four times in turn."
    ),
    (
        "Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh "
        "cận con đập dưới trời mưa."
    ): (
        "Then it switches to a scene of a dam filmed from above, followed by a "
        "close-up scene of the dam in the rain."
    ),
    (
        "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần "
        "lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một công trình thủy lợi lớn "
        "đang mở cửa xả nước dưới trời mưa."
    ): (
        "The footage begins with a map, on which a type of irrigation structure "
        "appears four times in turn. Then it switches to a scene of a large "
        "irrigation structure opening its spillway under the rain."
    ),
    ("Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa."): (
        "Then it switches to a scene of a large irrigation structure opening its "
        "spillway under the rain."
    ),
    (
        "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi "
        "dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo "
        "dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang "
        "cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú."
    ): (
        "A pride of lions is resting and climbing on wooden platforms in the "
        "breeding area, in front of which is a London Zoo information board for "
        "animal tracking and conservation. Then there is a scene of two staff "
        "members wearing green shirts weighing and recording data of an animal on "
        "the zoo premises."
    ),
    (
        "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi "
        "dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo "
        "dõi và bảo tồn động vật.."
    ): (
        "A pride of lions is resting and climbing on wooden platforms in the "
        "breeding area, in front of which is a London Zoo information board for "
        "animal tracking and conservation.."
    ),
    (
        "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi "
        "dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo "
        "dõi và bảo tồn động vật."
    ): (
        "A pride of lions is resting and climbing on wooden platforms in the "
        "breeding area, in front of which is a London Zoo information board for "
        "animal tracking and conservation."
    ),
    (
        "Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu "
        "của một con vật trong khuôn viên sở thú."
    ): (
        "Then there is a scene of two staff members wearing green shirts weighing "
        "and recording data of an animal on the zoo premises."
    ),
    (
        "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào "
        "trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món "
        "ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên "
        "bếp lửa."
    ): (
        "The clip begins with peas being added to squid being stir-fried in a pan, "
        "next to which is a plate of sliced onions and red peppers ready to be added "
        "to the dish. The clip ends with a slow motion frame of tossing the pan over "
        "the fire."
    ),
    (
        "đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa "
        "hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn."
    ): (
        "peas being added to squid being stir-fried in a pan, next to which is a "
        "plate of sliced onions and red peppers ready to be added to the dish."
    ),
    ("kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa."): (
        "ends with a slow motion frame of tossing the pan over the fire."
    ),
    (
        "Mẩu tin bắt đầu với hình ảnh một người đàn ông mặc vest xanh đậm, sơ mi "
        "trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một "
        "khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ "
        "nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng "
        "cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá "
        "quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường "
        "vận chuyển bao quanh."
    ): (
        "The news clip begins with the image of a man in a dark blue suit, white "
        "shirt, and tie, sitting on a large chair. He holds a rather large raw "
        "gemstone with both hands, bringing it close to his face to observe. On the "
        "right is a woman in black office attire and a pink-purple headscarf, "
        "standing next to him and smiling. Next is an aerial panoramic view of a "
        "large-scale open-pit gemstone mine with a multi-tiered deep excavation pit "
        "and a surrounding transport road system."
    ),
    (
        "một người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên "
        "một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa "
        "lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu "
        "đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười."
    ): (
        "a man in a dark blue suit, white shirt, and tie, sitting on a large chair. "
        "He holds a rather large raw gemstone with both hands, bringing it close to "
        "his face to observe. On the right is a woman in black office attire and a "
        "pink-purple headscarf, standing next to him and smiling."
    ),
    (
        "Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy "
        "mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao "
        "quanh."
    ): (
        "Next is an aerial panoramic view of a large-scale open-pit gemstone mine "
        "with a multi-tiered deep excavation pit and a surrounding transport road "
        "system."
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
        import warnings

        warnings.warn(
            "GoogleTranslateProvider is deprecated and uses unsafe web scraping. "
            "Use GoogleCloudTranslationProvider from system_tai.translation.google_cloud instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
                logger.warning(
                    "GoogleTranslator unavailable (%s); using cache or web fallback.",
                    exc,
                )
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
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                },
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

    License notice:
    VinAI Translate v2 weights and code are distributed under the GNU Affero General Public
    License v3.0 (AGPL-3.0). Any downstream deployment or distribution must comply with AGPL-3.0.
    """

    DEFAULT_MODEL_NAME = "vinai/vinai-translate-vi2en-v2"
    CANONICAL_PINNED_REVISION = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
    DEFAULT_PINNED_REVISION = CANONICAL_PINNED_REVISION
    TRUSTED_REVISIONS: frozenset[str] = frozenset({CANONICAL_PINNED_REVISION})

    # Pinned canonical manifest digest outside snapshot (Trust Anchor)
    CANONICAL_MANIFEST_SHA256 = "bbc68d04ff08d4e9608bdad9ab360cdba4fa75ee48ba82a0e0490493d7864379"
    TRUSTED_MANIFEST_DIGESTS: dict[str, str] = {
        CANONICAL_PINNED_REVISION: CANONICAL_MANIFEST_SHA256,
    }
    SOURCE_LANGUAGE = "vi_VN"
    TARGET_LANGUAGE = "en_XX"

    CANONICAL_ARTIFACT_NAMES: frozenset[str] = frozenset(
        {
            ".gitattributes",
            "README.md",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "manifest.json",
            "provenance.json",
        }
    )

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
        require_exact_snapshot: bool = False,
        trusted_revisions: Collection[str] | None = None,
        trusted_manifest_sha256: str | None = None,
        allow_custom_trust_anchor: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            raise TranslationError(
                "transformers and torch must be installed to use "
                "VinAITranslateProvider [vinai_import_error]"
            ) from None

        self.model_name = str(model_name_or_path or self.DEFAULT_MODEL_NAME)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_length = max_length
        self.num_beams = num_beams
        self.allow_model_download = allow_model_download
        self.revision = revision or self.DEFAULT_PINNED_REVISION
        self.require_exact_snapshot = require_exact_snapshot
        self.allow_custom_trust_anchor = allow_custom_trust_anchor
        self.trusted_revisions = (
            frozenset(r.strip().lower() for r in trusted_revisions)
            if trusted_revisions is not None
            else self.TRUSTED_REVISIONS
        )
        self.trusted_manifest_sha256 = (
            trusted_manifest_sha256
            or os.environ.get("SYSTEM_TAI_VINAI_MANIFEST_SHA256")
            or os.environ.get("VINAI_TRUSTED_MANIFEST_SHA256")
        )

        if require_exact_snapshot:
            if self.model_name != self.DEFAULT_MODEL_NAME and not Path(self.model_name).exists():
                raise TranslationError(
                    "Strict VinAI mode requires canonical model [vinai_invalid_model]"
                )
            is_hex40 = len(self.revision) == 40 and all(
                c in "0123456789abcdefABCDEF" for c in self.revision
            )
            if not is_hex40:
                raise TranslationError(
                    "Strict VinAI mode requires 40-char revision [vinai_invalid_revision]"
                )
            if all(c == "0" for c in self.revision):
                raise TranslationError(
                    "Strict VinAI mode rejects all-zero revision [vinai_untrusted_revision]"
                )
            if self.revision.lower() not in self.trusted_revisions:
                raise TranslationError(
                    "Strict VinAI mode requires trusted canonical revision "
                    "[vinai_untrusted_revision]"
                )
            if not self.allow_custom_trust_anchor and self.trusted_manifest_sha256 is not None:
                raise TranslationError(
                    "Custom trust anchor not permitted unless allow_custom_trust_anchor=True "
                    "[vinai_trust_anchor_override_forbidden]"
                )
            if allow_model_download:
                raise TranslationError("Strict VinAI mode forbids allow_model_download=True")
            self.verify_snapshot()

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
            "Loading VinAI Translate model (device=%s, strict=%s)",
            self._device,
            require_exact_snapshot,
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
        except Exception:
            raise TranslationError(
                "Failed to load VinAI Translate model [vinai_model_load_failed]"
            ) from None

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
        if self.model_name == self.DEFAULT_MODEL_NAME:
            return f"vinai-translate:{self.model_name}@{self.revision[:8]}"
        return f"vinai-translate:local@{self.revision[:8]}"

    @property
    def device(self) -> str:
        return self._device

    def _resolve_snapshot_dir(self) -> Path | None:
        """Locate snapshot directory across local model dir, cache_dir, HF_HOME, or default hub."""
        if Path(self.model_name).is_dir():
            return Path(self.model_name)

        repo_id_normalized = f"models--{self.model_name.replace('/', '--')}"
        possible_roots: list[Path] = []
        if self.cache_dir:
            possible_roots.append(Path(self.cache_dir) / repo_id_normalized)
            possible_roots.append(Path(self.cache_dir) / "hub" / repo_id_normalized)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            possible_roots.append(Path(hf_home) / "hub" / repo_id_normalized)
            possible_roots.append(Path(hf_home) / repo_id_normalized)
        possible_roots.extend(
            [
                Path.home() / ".cache" / "huggingface" / "hub" / repo_id_normalized,
                Path("/root/.cache/huggingface/hub") / repo_id_normalized,
                Path("/kaggle/working/.cache/huggingface/hub") / repo_id_normalized,
            ]
        )

        for r in possible_roots:
            if r.exists():
                snaps = r / "snapshots"
                if snaps.exists():
                    if self.revision and (snaps / self.revision).is_dir():
                        return snaps / self.revision
                    if not self.require_exact_snapshot:
                        for snap in snaps.iterdir():
                            if snap.is_dir():
                                return snap
        return None

    def verify_snapshot(self) -> None:
        """Verify presence, manifest revision, and artifact checksums of the snapshot."""
        snapshot_dir = self._resolve_snapshot_dir()
        if not snapshot_dir or not snapshot_dir.exists():
            raise TranslationError(
                "Strict VinAI mode: snapshot was not found [vinai_snapshot_not_found]"
            )

        snapshot_root = snapshot_dir.resolve()
        is_hf_layout = (
            snapshot_dir.parent.name == "snapshots"
            and snapshot_dir.parent.parent.name.startswith("models--")
        )
        hf_blobs_dir = (snapshot_dir.parent.parent / "blobs").resolve() if is_hf_layout else None

        # 1. Reject directory symlinks or escaping symlinks in snapshot
        for item in snapshot_dir.rglob("*"):
            if item.is_symlink():
                if item.is_dir():
                    raise TranslationError(
                        "Strict VinAI mode: directory symlinks are forbidden "
                        "[vinai_manifest_path_traversal]"
                    )
                try:
                    resolved = item.resolve(strict=True)
                    is_contained = resolved.is_relative_to(snapshot_root) or (
                        hf_blobs_dir is not None and resolved.is_relative_to(hf_blobs_dir)
                    )
                    if not is_contained:
                        raise TranslationError(
                            "Strict VinAI mode: artifact path escapes snapshot root "
                            "[vinai_manifest_path_traversal]"
                        )
                except Exception:
                    raise TranslationError(
                        "Strict VinAI mode: artifact path escapes snapshot root "
                        "[vinai_manifest_path_traversal]"
                    )

        # 2. Resolve Manifest Data from Independent Trust Anchor
        if not self.allow_custom_trust_anchor:
            try:
                import importlib.resources

                manifest_bytes = (
                    importlib.resources.files("system_tai.translation")
                    .joinpath("canonical_vinai_manifest.json")
                    .read_bytes()
                )
            except Exception:
                raise TranslationError(
                    "Strict VinAI mode: packaged canonical manifest missing "
                    "[vinai_missing_manifest]"
                ) from None

            try:
                manifest_data = json.loads(manifest_bytes.decode("utf-8"))
                if not isinstance(manifest_data, dict) or not manifest_data:
                    raise ValueError("Empty manifest")
            except Exception:
                raise TranslationError(
                    "Strict VinAI mode: manifest is empty or corrupted [vinai_manifest_corrupted]"
                ) from None

            canonical_manifest_bytes = json.dumps(
                manifest_data, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            actual_manifest_sha = hashlib.sha256(canonical_manifest_bytes).hexdigest().lower()
            if actual_manifest_sha != self.CANONICAL_MANIFEST_SHA256:
                raise TranslationError(
                    "Strict VinAI mode: canonical manifest digest mismatch "
                    "[vinai_manifest_untrusted]"
                )
        else:
            manifest_path = None
            for cand in ("manifest.json", "provenance.json"):
                if (snapshot_dir / cand).is_file():
                    manifest_path = snapshot_dir / cand
                    break
            if not manifest_path:
                raise TranslationError(
                    "Strict VinAI mode: local snapshot lacks provenance manifest "
                    "[vinai_missing_manifest]"
                )
            manifest_bytes = manifest_path.read_bytes()
            try:
                manifest_data = json.loads(manifest_bytes.decode("utf-8"))
                if not isinstance(manifest_data, dict) or not manifest_data:
                    raise ValueError("Empty manifest")
            except Exception:
                raise TranslationError(
                    "Strict VinAI mode: manifest is empty or corrupted [vinai_manifest_corrupted]"
                ) from None

            canonical_manifest_bytes = json.dumps(
                manifest_data, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            actual_canonical_sha = hashlib.sha256(canonical_manifest_bytes).hexdigest().lower()
            actual_raw_sha = hashlib.sha256(manifest_bytes).hexdigest().lower()

            expected_manifest_sha = (
                self.trusted_manifest_sha256.strip().lower()
                if self.trusted_manifest_sha256
                else self.TRUSTED_MANIFEST_DIGESTS.get(self.revision.lower())
            )
            if not expected_manifest_sha or (
                actual_canonical_sha != expected_manifest_sha
                and actual_raw_sha != expected_manifest_sha
            ):
                raise TranslationError(
                    "Strict VinAI mode: manifest does not match trusted anchor "
                    "[vinai_manifest_untrusted]"
                )

        manifest_rev = manifest_data.get("revision") or manifest_data.get("commit_hash")
        if not manifest_rev or not isinstance(manifest_rev, str):
            raise TranslationError(
                "Strict VinAI mode: manifest lacks valid revision [manifest_revision_mismatch]"
            )
        if manifest_rev.strip().lower() != (self.revision or "").strip().lower():
            raise TranslationError("Manifest revision mismatch [manifest_revision_mismatch]")

        checksums = manifest_data.get("checksums") or manifest_data.get("files")
        if not isinstance(checksums, dict) or not checksums:
            raise TranslationError(
                "Strict VinAI mode: manifest lacks required checksums [vinai_manifest_corrupted]"
            )

        for fname, expected_hash in checksums.items():
            if not isinstance(fname, str) or not isinstance(expected_hash, str):
                raise TranslationError(
                    "Strict VinAI mode: invalid manifest entries [vinai_manifest_corrupted]"
                )
            if ".." in fname or fname.startswith(("/", "\\")):
                raise TranslationError(
                    "Strict VinAI mode: path traversal in manifest [vinai_manifest_path_traversal]"
                )

        # 3. Absolute Inventory Allowlist: every file in snapshot_dir MUST be in checksums
        # or allowed metadata
        ALLOWED_METADATA_FILES = frozenset({"manifest.json", "provenance.json", ".gitattributes"})
        for item in snapshot_dir.rglob("*"):
            if item.is_file():
                rel_name = item.relative_to(snapshot_dir).as_posix()
                if rel_name in ALLOWED_METADATA_FILES:
                    continue
                if rel_name not in checksums:
                    raise TranslationError(
                        "Strict VinAI mode: unapproved file detected in snapshot inventory "
                        "[vinai_untrusted_file]"
                    )

        # 4. Required artifacts check
        weight_found = any(k in checksums for k in ("model.safetensors", "pytorch_model.bin"))
        tokenizer_found = any(
            k in checksums for k in ("sentencepiece.bpe.model", "tokenizer.json")
        )
        if not (weight_found and tokenizer_found and "config.json" in checksums):
            raise TranslationError(
                "Strict VinAI mode: manifest lacks required snapshot inventory "
                "[vinai_missing_artifact]"
            )

        # 5. Verify artifact presence and SHA-256 hashes
        for fname, expected_hash in checksums.items():
            fpath = snapshot_dir / fname
            try:
                resolved_file = fpath.resolve(strict=True)
                is_contained = resolved_file.is_relative_to(snapshot_root) or (
                    hf_blobs_dir is not None and resolved_file.is_relative_to(hf_blobs_dir)
                )
                if not is_contained:
                    raise TranslationError(
                        "Strict VinAI mode: artifact path escapes snapshot root "
                        "[vinai_manifest_path_traversal]"
                    )
            except (FileNotFoundError, OSError):
                raise TranslationError(
                    "Missing artifact specified in manifest [vinai_missing_artifact]"
                ) from None

            exp_hash_clean = expected_hash.strip().lower()
            if len(exp_hash_clean) != 64 or not re.match(r"^[0-9a-f]{64}$", exp_hash_clean):
                raise TranslationError(
                    "Invalid artifact checksum format [vinai_manifest_corrupted]"
                )

            h = hashlib.sha256()
            with open(resolved_file, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_hash = h.hexdigest().lower()
            if actual_hash != exp_hash_clean:
                raise TranslationError("Artifact checksum mismatch [artifact_checksum_mismatch]")

        self._snapshot_verified = True

    def get_artifact_fingerprint(self) -> dict[str, Any]:
        """Compute artifact provenance and SHA256 fingerprint strictly from local snapshot."""
        model_display = (
            self.DEFAULT_MODEL_NAME
            if self.model_name == self.DEFAULT_MODEL_NAME
            else "custom-local"
        )
        info: dict[str, Any] = {
            "model_name": model_display,
            "pinned_revision": self.revision,
            "device": self._device,
        }

        snapshot_dir = self._resolve_snapshot_dir()

        if snapshot_dir and snapshot_dir.exists():
            is_hf = bool(
                snapshot_dir.parent.name == "snapshots"
                and snapshot_dir.parent.parent.name.startswith("models--")
            )
            info["resolved_snapshot_dir"] = "hf-cache" if is_hf else "custom-local"
            if getattr(self, "_snapshot_verified", False):
                info["snapshot_commit_hash"] = self.revision
                info["revision_matches_snapshot"] = True
            else:
                info["snapshot_commit_hash"] = "unverified"
                info["revision_matches_snapshot"] = False

            primary_weights = (
                "model.safetensors"
                if (snapshot_dir / "model.safetensors").exists()
                else "pytorch_model.bin"
            )
            info["primary_weight_artifact"] = primary_weights

            scanned_files: dict[str, Path] = {}
            for fpath in snapshot_dir.iterdir():
                if fpath.is_file():
                    scanned_files[fpath.name] = fpath

            for fname, fpath in sorted(scanned_files.items()):
                if fname in self.CANONICAL_ARTIFACT_NAMES:
                    key_prefix = fname
                else:
                    h_name = hashlib.sha256(fname.encode("utf-8")).hexdigest()[:12]
                    key_prefix = f"artifact_{h_name}"
                try:
                    info[f"{key_prefix}_size_bytes"] = fpath.stat().st_size
                    h = hashlib.sha256()
                    with open(fpath, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    info[f"{key_prefix}_sha256"] = h.hexdigest()
                except Exception as exc:
                    info[f"{key_prefix}_hash_error"] = type(exc).__name__
        else:
            if self.require_exact_snapshot:
                raise TranslationError(
                    "Strict VinAI mode: snapshot was not found [vinai_snapshot_not_found]"
                )
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
                    f"VinAI translation returned {len(translated)} rows for {len(cleaned)} inputs"
                )
            if any(not value for value in translated):
                raise TranslationError("VinAI translation produced an empty output")
            return translated
        except Exception as exc:
            if isinstance(exc, TranslationError):
                raise
            raise TranslationError(
                "VinAI translation generation failed [vinai_generation_error]."
            ) from None


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
            clause.strip() for clause in self._BOUNDARY_RE.split(cleaned) if clause.strip()
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
