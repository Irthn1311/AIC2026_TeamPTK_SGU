"""Verified-only Stage 1B text retrieval smoke testing."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.retrieval.stage1.contracts import EncoderContract, SearchConfig
from triage_eg.retrieval.stage1.encoder import compatibility_gate, validate_encoder_output
from triage_eg.retrieval.stage1.search import rank_query, write_query_outputs
from triage_eg.retrieval.stage1b.contracts import CandidateContract, MultimodalEncoder


def stage1_contract(candidate: CandidateContract) -> EncoderContract:
    return EncoderContract(
        implementation=candidate.implementation,
        model_name=candidate.architecture,
        pretrained=candidate.pretrained,
        checkpoint_path=candidate.checkpoint_path,
        tokenizer=candidate.tokenizer,
        text_preprocessing=json.dumps(candidate.text_preprocessing, sort_keys=True),
        image_preprocessing=json.dumps(candidate.image_preprocessing, sort_keys=True),
        output_dimension=candidate.output_dimension,
        normalize_text_embedding=candidate.text_embedding_normalization,
        evidence_source="EMPIRICAL_PROBE",
        compatibility_status=candidate.compatibility_status,
    )


def run_text_smoke(
    candidate: CandidateContract,
    encoder: MultimodalEncoder,
    queries: list[dict[str, Any]],
    stage1_root: Path,
    output_root: Path,
    top_k: int = 20,
) -> tuple[list[dict], str]:
    contract = stage1_contract(candidate)
    compatibility_gate(contract, allow_unverified=False)
    texts = []
    for item in queries:
        text = str(item["text"])
        if candidate.text_preprocessing.get("strip"):
            text = text.strip()
        if candidate.text_preprocessing.get("lowercase"):
            text = text.lower()
        normalization = candidate.text_preprocessing.get("unicode_normalization")
        if normalization:
            text = unicodedata.normalize(str(normalization), text)
        texts.append(text)
    try:
        embeddings = validate_encoder_output(encoder.encode_text(texts), len(texts))
    except ValueError as error:
        raise ValueError(f"TEXT_EMBEDDING_INVALID: {error}") from error
    if candidate.text_embedding_normalization:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    results = []
    for index, query in enumerate(queries):
        config = SearchConfig(stage1_root, str(query["query_id"]), top_k=top_k)
        candidates, latency = rank_query(
            embeddings[index : index + 1], config, encoder_status="VERIFIED"
        )
        paths = write_query_outputs(
            output_root / "smoke",
            config,
            candidates,
            search_latency_seconds=latency,
            encoder_status="VERIFIED",
            query_directory_name="query_artifacts",
        )
        results.append(
            {
                **query,
                "encoder_status": "VERIFIED",
                "top_k": top_k,
                "latency_seconds": latency,
                "result_artifacts": {
                    key: str(path.relative_to(output_root)) for key, path in paths.items()
                },
                "qualitative_label": "NOT_REVIEWED",
            }
        )
    return results, "PASS" if results else "PASS_WITH_WARNINGS"
