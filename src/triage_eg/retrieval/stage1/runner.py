"""Public Stage 1 vector/text query runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.contracts import EncoderContract, SearchConfig, TextEncoder
from triage_eg.retrieval.stage1.encoder import (
    compatibility_gate,
    validate_encoder_output,
    write_compatibility_report,
)
from triage_eg.retrieval.stage1.search import rank_query, write_query_outputs


def load_query_vector(path: str | Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape == (512,):
        value = value.reshape(1, 512)
    if value.shape != (1, 512):
        raise ValueError("Query vector must have shape (512,) or (1,512)")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ValueError("Query vector must be finite and nonzero")
    return value


def search_vector(
    query: np.ndarray, config: SearchConfig
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    candidates, latency = rank_query(query, config)
    paths = write_query_outputs(
        config.stage1_root,
        config,
        candidates,
        search_latency_seconds=latency,
        encoder_status="NOT_APPLICABLE_VECTOR_QUERY",
    )
    return candidates, paths


def search_text(
    text: str,
    config: SearchConfig,
    contract: EncoderContract,
    encoder: TextEncoder,
    *,
    allow_unverified: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    if not text.strip():
        raise ValueError("query text must not be empty")
    status = compatibility_gate(contract, allow_unverified=allow_unverified)
    query = validate_encoder_output(encoder.encode_text([text]), 1)
    if contract.normalize_text_embedding:
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
    candidates, latency = rank_query(query, config, encoder_status=status)
    paths = write_query_outputs(
        config.stage1_root,
        config,
        candidates,
        search_latency_seconds=latency,
        encoder_status=status,
    )
    reason = (
        "Text search completed with verified encoder compatibility"
        if status == "VERIFIED"
        else "Text search completed under explicit unverified encoder override"
    )
    write_compatibility_report(config.stage1_root, contract, status, reason)
    return candidates, paths
