"""Immutable Paraphrase Ensemble Translation Sidecar Provider with group-level isolation and zero network access."""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .provider import TranslationError
from .sidecar_provider import canonical_sidecar_sha256

logger = logging.getLogger(__name__)


class ImmutableParaphraseEnsembleSidecarProvider:
    """Zero-network, fail-fast paraphrase ensemble translation provider backed by an immutable JSON sidecar."""

    def __init__(
        self,
        sidecar_path: Path | str,
        expected_content_sha256: str | None = None,
    ) -> None:
        self.sidecar_path = Path(sidecar_path)
        if not self.sidecar_path.exists():
            raise FileNotFoundError(
                f"Paraphrase sidecar file does not exist: {self.sidecar_path}"
            )

        self._actual_content_sha256 = canonical_sidecar_sha256(self.sidecar_path)
        if expected_content_sha256 is not None:
            norm_expected = expected_content_sha256.strip().lower()
            if self._actual_content_sha256.lower() != norm_expected:
                raise ValueError(
                    f"Paraphrase sidecar canonical content SHA256 mismatch for {self.sidecar_path.name}: "
                    f"expected {norm_expected}, got {self._actual_content_sha256}"
                )

        raw_obj = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        self._schema_version = str(raw_obj.get("$schema_version", raw_obj.get("schema_version", "1.0.0")))
        self._sidecar_id = str(raw_obj.get("sidecar_id", ""))
        self._policy = str(raw_obj.get("translation_policy", "PARAPHRASE_ENSEMBLE_INVARIANT_FUSION"))

        target_count = raw_obj.get("target_queries_count")
        queries_dict = raw_obj.get("queries", {})
        if target_count is not None and int(target_count) != len(queries_dict):
            raise ValueError(
                f"Paraphrase sidecar declared target_queries_count={target_count} but found {len(queries_dict)} queries"
            )

        # Lookup structures:
        # 1. query_meta: query_id -> query metadata & ordered groups
        # 2. translation_table: (query_id, group_id, vi_text_sha256) -> en_text
        self._query_meta: dict[str, dict[str, Any]] = {}
        self._translation_table: dict[tuple[str, str, str], str] = {}
        self._group_meta: dict[tuple[str, str], dict[str, Any]] = {}

        for qid, qdata in queries_dict.items():
            qid_str = str(qid).strip()
            q_vi = qdata.get("query_vi", "")
            q_vi_nfc = unicodedata.normalize("NFC", q_vi)
            q_vi_sha = hashlib.sha256(q_vi_nfc.encode("utf-8")).hexdigest()

            declared_q_sha = qdata.get("query_vi_sha256")
            if declared_q_sha and declared_q_sha.lower() != q_vi_sha:
                raise ValueError(
                    f"Query {qid_str} declared query_vi_sha256={declared_q_sha} "
                    f"does not match computed NFC SHA={q_vi_sha}"
                )

            groups = qdata.get("paraphrase_groups", [])
            if not groups:
                raise ValueError(f"Query {qid_str} must have at least one paraphrase group")

            seen_group_ids: set[str] = set()
            group_list: list[dict[str, Any]] = []

            for grp in groups:
                gid = str(grp.get("group_id", "")).strip()
                if not gid:
                    raise ValueError(f"Query {qid_str} contains paraphrase group with empty group_id")
                if gid in seen_group_ids:
                    raise ValueError(f"Query {qid_str} contains duplicate group_id: '{gid}'")
                seen_group_ids.add(gid)

                exp_hash = grp.get("expected_semantic_variant_sha256")
                exp_var_count = grp.get("expected_variant_count")
                src_text = grp.get("source_text", q_vi_nfc)

                self._group_meta[(qid_str, gid)] = {
                    "group_id": gid,
                    "source_text": src_text,
                    "expected_semantic_variant_sha256": exp_hash,
                    "expected_variant_count": int(exp_var_count) if exp_var_count is not None else None,
                }

                units = grp.get("units", [])
                seen_unit_hashes: set[str] = set()
                for u in units:
                    vi_text = u.get("vi_text", "")
                    en_text = u.get("en_text", "")
                    vi_nfc = unicodedata.normalize("NFC", vi_text)
                    u_sha = hashlib.sha256(vi_nfc.encode("utf-8")).hexdigest()

                    composite_key = (qid_str, gid, u_sha)
                    if u_sha in seen_unit_hashes:
                        existing_en = self._translation_table[composite_key]
                        if existing_en != en_text:
                            raise ValueError(
                                f"Conflicting translation for same Vietnamese unit within group '{gid}' "
                                f"of query '{qid_str}': '{existing_en}' vs '{en_text}'"
                            )
                    else:
                        seen_unit_hashes.add(u_sha)
                        self._translation_table[composite_key] = en_text

                group_list.append(grp)

            self._query_meta[qid_str] = {
                "query_vi": q_vi_nfc,
                "query_vi_sha256": q_vi_sha,
                "paraphrase_groups": tuple(group_list),
            }

        logger.info(
            "Initialized ImmutableParaphraseEnsembleSidecarProvider: "
            "sidecar_id=%s, queries=%d, canonical_sha256=%s",
            self._sidecar_id,
            len(self._query_meta),
            self._actual_content_sha256,
        )

    @property
    def sidecar_id(self) -> str:
        return self._sidecar_id

    @property
    def canonical_content_sha256(self) -> str:
        return self._actual_content_sha256

    def sidecar_metadata(self) -> dict[str, Any]:
        return {
            "sidecar_id": self._sidecar_id,
            "schema_version": self._schema_version,
            "translation_policy": self._policy,
            "sidecar_content_sha256": self._actual_content_sha256,
            "sidecar_path": str(self.sidecar_path),
            "target_queries_count": len(self._query_meta),
        }

    def get_paraphrase_groups(self, query_id: str) -> tuple[dict[str, Any], ...]:
        qid_clean = query_id.strip()
        if qid_clean not in self._query_meta:
            # Fallback for prefixed IDs e.g. "query-p1-1-kis" -> "p1-1"
            short_id = qid_clean.replace("query-", "").replace("-kis", "")
            if short_id in self._query_meta:
                qid_clean = short_id
            else:
                raise TranslationError(
                    f"Query '{query_id}' not found in paraphrase sidecar '{self._sidecar_id}'"
                )
        return self._query_meta[qid_clean]["paraphrase_groups"]

    def translate_unit(
        self,
        query_id: str,
        group_id: str,
        vi_text: str,
    ) -> str:
        qid_clean = query_id.strip()
        if qid_clean not in self._query_meta:
            short_id = qid_clean.replace("query-", "").replace("-kis", "")
            if short_id in self._query_meta:
                qid_clean = short_id
            else:
                raise TranslationError(
                    f"Query '{query_id}' not found in paraphrase sidecar '{self._sidecar_id}'"
                )

        vi_nfc = unicodedata.normalize("NFC", vi_text)
        u_sha = hashlib.sha256(vi_nfc.encode("utf-8")).hexdigest()
        composite_key = (qid_clean, group_id.strip(), u_sha)

        if composite_key in self._translation_table:
            return self._translation_table[composite_key]

        raise TranslationError(
            f"Unit translation miss in sidecar '{self._sidecar_id}' for query '{query_id}', "
            f"group '{group_id}', text='{vi_text[:40]}...' (SHA={u_sha[:12]})"
        )

    def expected_group_hashes(self, query_id: str) -> dict[str, str | None]:
        groups = self.get_paraphrase_groups(query_id)
        qid_clean = query_id.strip().replace("query-", "").replace("-kis", "")
        return {
            grp["group_id"]: self._group_meta.get((qid_clean, grp["group_id"]), {}).get("expected_semantic_variant_sha256")
            for grp in groups
        }

    def expected_group_variant_counts(self, query_id: str) -> dict[str, int | None]:
        groups = self.get_paraphrase_groups(query_id)
        qid_clean = query_id.strip().replace("query-", "").replace("-kis", "")
        return {
            grp["group_id"]: self._group_meta.get((qid_clean, grp["group_id"]), {}).get("expected_variant_count")
            for grp in groups
        }
