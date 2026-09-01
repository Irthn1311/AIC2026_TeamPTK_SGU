"""Immutable Translation Sidecar Provider with strict integrity and zero network isolation."""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

from .provider import TranslationError

logger = logging.getLogger(__name__)


def canonical_sidecar_sha256(path: Path | str) -> str:
    """Compute deterministic canonical JSON content SHA256 (platform independent)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Translation sidecar file not found: {p}")
    raw_text = p.read_text(encoding="utf-8")
    obj = json.loads(raw_text)
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ImmutableSidecarTranslationProvider:
    """Zero-network, fail-fast translation provider backed by an immutable JSON sidecar."""

    def __init__(
        self,
        sidecar_path: Path | str,
        expected_content_sha256: str | None = None,
    ) -> None:
        self.sidecar_path = Path(sidecar_path)
        if not self.sidecar_path.exists():
            raise FileNotFoundError(
                f"Sidecar file does not exist: {self.sidecar_path}"
            )

        self._actual_content_sha256 = canonical_sidecar_sha256(self.sidecar_path)
        if expected_content_sha256 is not None:
            norm_expected = expected_content_sha256.strip().lower()
            if self._actual_content_sha256.lower() != norm_expected:
                raise ValueError(
                    f"Sidecar canonical content SHA256 mismatch for {self.sidecar_path.name}: "
                    f"expected {norm_expected}, got {self._actual_content_sha256}"
                )

        raw_obj = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        self._schema_version = str(raw_obj.get("$schema_version", raw_obj.get("schema_version", "")))
        self._sidecar_id = str(raw_obj.get("sidecar_id", ""))
        self._policy = str(raw_obj.get("translation_policy", ""))
        
        target_count = raw_obj.get("target_queries_count")
        queries_dict = raw_obj.get("queries", {})
        if target_count is not None and int(target_count) != len(queries_dict):
            raise ValueError(
                f"Sidecar declared target_queries_count={target_count} but found {len(queries_dict)} queries"
            )

        # Build in-memory lookup table: (query_id, vi_text_sha256) -> en_text
        # and query metadata lookup: query_id -> {expected_hash, expected_variants, ...}
        self._query_meta: dict[str, dict[str, Any]] = {}
        self._translation_table: dict[tuple[str, str], str] = {}
        self._global_translation_table: dict[str, str] = {}

        for qid, qdata in queries_dict.items():
            qid_str = str(qid).strip()
            exp_hash = qdata.get("expected_semantic_variant_sha256")
            exp_var_count = qdata.get("expected_variant_count")
            q_vi = qdata.get("query_vi", "")
            q_vi_nfc = unicodedata.normalize("NFC", q_vi)
            q_vi_sha = hashlib.sha256(q_vi_nfc.encode("utf-8")).hexdigest()
            
            declared_q_sha = qdata.get("query_vi_sha256")
            if declared_q_sha and declared_q_sha.lower() != q_vi_sha:
                raise ValueError(
                    f"Query {qid_str} declared query_vi_sha256={declared_q_sha} "
                    f"does not match computed NFC SHA={q_vi_sha}"
                )

            self._query_meta[qid_str] = {
                "query_vi": q_vi_nfc,
                "query_vi_sha256": q_vi_sha,
                "expected_semantic_variant_sha256": exp_hash,
                "expected_variant_count": int(exp_var_count) if exp_var_count is not None else None,
            }

            units = qdata.get("units", [])
            seen_unit_hashes: set[str] = set()
            for u in units:
                vi_text = u.get("vi_text", "")
                en_text = u.get("en_text", "")
                if not vi_text or not vi_text.strip():
                    raise ValueError(f"Query {qid_str} contains empty vi_text in unit: {u}")
                if not en_text or not en_text.strip():
                    raise ValueError(f"Query {qid_str} contains empty en_text in unit: {u}")

                vi_norm = unicodedata.normalize("NFC", vi_text)
                u_sha = hashlib.sha256(vi_norm.encode("utf-8")).hexdigest()
                declared_u_sha = u.get("vi_text_sha256")
                if declared_u_sha and declared_u_sha.lower() != u_sha:
                    raise ValueError(
                        f"Unit in {qid_str} declared vi_text_sha256={declared_u_sha} "
                        f"does not match computed NFC SHA={u_sha}"
                    )

                if u_sha in seen_unit_hashes:
                    pass
                seen_unit_hashes.add(u_sha)

                self._translation_table[(qid_str, u_sha)] = en_text
                if u_sha in self._global_translation_table and self._global_translation_table[u_sha] != en_text:
                    raise ValueError(
                        f"Translation collision detected for text SHA {u_sha}: "
                        f"{self._global_translation_table[u_sha]!r} != {en_text!r}"
                    )
                self._global_translation_table[u_sha] = en_text

    @property
    def provider_name(self) -> str:
        return f"immutable_sidecar::{self._sidecar_id}"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def sidecar_id(self) -> str:
        return self._sidecar_id

    @property
    def content_sha256(self) -> str:
        return self._actual_content_sha256

    def expected_semantic_hash(self, query_id: str) -> str | None:
        """Return expected golden compiled semantic variant SHA256 for a query."""
        meta = self._query_meta.get(query_id.strip())
        if meta:
            return meta.get("expected_semantic_variant_sha256")
        return None

    def expected_variant_count(self, query_id: str) -> int | None:
        """Return expected compiled semantic variant count for a query."""
        meta = self._query_meta.get(query_id.strip())
        if meta:
            return meta.get("expected_variant_count")
        return None

    def sidecar_metadata(self) -> dict[str, object]:
        """Return audit telemetry dictionary for candidates.json and summaries."""
        return {
            "sidecar_id": self._sidecar_id,
            "sidecar_file_path": str(self.sidecar_path),
            "sidecar_content_sha256": self._actual_content_sha256,
            "sidecar_schema_version": self._schema_version,
            "translation_policy": self._policy,
            "queries_registered": list(self._query_meta.keys()),
        }

    def translate(self, text: str, query_id: str | None = None) -> str:
        """Translate a single string using exact NFC SHA256 key matching (fail-fast, no network)."""
        if not text or not text.strip():
            raise TranslationError("Cannot translate empty Vietnamese text")

        norm_text = unicodedata.normalize("NFC", text)
        text_sha = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

        if query_id:
            key = (query_id.strip(), text_sha)
            if key in self._translation_table:
                return self._translation_table[key]

        if text_sha in self._global_translation_table:
            return self._global_translation_table[text_sha]

        raise TranslationError(
            f"Sidecar translation miss for query_id={query_id!r}, "
            f"text_sha={text_sha}, text={repr(text[:60])}"
        )

    def translate_many(
        self,
        texts: tuple[str, ...],
        query_id: str | None = None,
    ) -> tuple[str, ...]:
        """Batch translate strings using exact key matching."""
        return tuple(self.translate(t, query_id=query_id) for t in texts)
