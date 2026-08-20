"""GT-free Trial P1 evidence smoke for the validated external ASR source."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from triage_eg.trial_p1.asr_v12_loader import ASRExternalV3Loader
from triage_eg.trial_p1.parser import parse_trial_zip

QueryEncoder = Callable[[list[str]], np.ndarray]


class OnnxE5QueryEncoder:
    """Pinned local ONNX encoder used only to query the preserved external index."""

    def __init__(self, model_root: str | Path, *, exact_revision: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
            from tokenizers import Tokenizer  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("ONNX_E5_QUERY_ENCODER_DEPENDENCY_MISSING") from error
        root = Path(model_root).resolve(strict=True)
        self.session = ort.InferenceSession(
            str(root / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()
        self.provenance = {
            "model_id": "intfloat/multilingual-e5-small",
            "exact_revision": exact_revision,
            "runtime": "onnxruntime_cpu",
            "model_file": str((root / "model.onnx").resolve()),
            "purpose": "TRIAL_QUERY_ENCODING_ONLY_NO_CORPUS_REEMBEDDING",
            "source_index_exact_encoder_revision_known": False,
        }

    def __call__(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.asarray([row.ids for row in encoded], dtype=np.int64)
        attention_mask = np.asarray([row.attention_mask for row in encoded], dtype=np.int64)
        token_type_ids = np.asarray([row.type_ids for row in encoded], dtype=np.int64)
        hidden = self.session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]
        mask = attention_mask[..., None].astype(np.float32)
        return np.asarray((hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1, None))


def _query_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_text") or row.get("query_text") or "").strip()


def _e5_search(
    loader: ASRExternalV3Loader,
    query_texts: list[str],
    encoder: QueryEncoder,
    top_k: int,
) -> list[list[dict[str, Any]]]:
    manifest = loader.e5_manifest
    count = int(manifest["source_vector_count"])
    usable_count = int(manifest["usable_vector_count"])
    dimension = int(manifest["dimension"])
    index_path = loader.root / "asr_external_v3_e5_flat_ip.faiss"
    payload_bytes = count * dimension * 4
    offset = index_path.stat().st_size - payload_bytes
    if offset < 0:
        raise RuntimeError("ASR_EXTERNAL_V3_FAISS_PAYLOAD_SIZE_INVALID")
    vectors = np.memmap(
        index_path, dtype=np.float32, mode="r", offset=offset, shape=(count, dimension)
    )
    queries = np.asarray(encoder([f"query: {text}" for text in query_texts]), dtype=np.float32)
    if queries.shape != (len(query_texts), dimension):
        raise RuntimeError(
            f"ASR_EXTERNAL_V3_QUERY_VECTOR_SHAPE_INVALID:{queries.shape}:"
            f"expected={(len(query_texts), dimension)}"
        )
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("ASR_EXTERNAL_V3_QUERY_VECTOR_NORM_INVALID")
    queries = queries / norms
    chunk_by_row = {
        int(row["e5_row_index"]): chunk_id
        for chunk_id, row in loader.lexical_chunks.items()
    }
    if len(chunk_by_row) != usable_count:
        raise RuntimeError("ASR_EXTERNAL_V3_E5_CHUNK_ORDER_COUNT_MISMATCH")
    excluded = np.asarray(sorted(set(range(count)) - set(chunk_by_row)), dtype=np.int64)
    results: list[list[dict[str, Any]]] = []
    for query in queries:
        scores = np.asarray(vectors @ query)
        scores[excluded] = -np.inf
        candidate_count = min(top_k, usable_count)
        candidate = np.argpartition(scores, -candidate_count)[-candidate_count:]
        ordered = candidate[np.argsort(-scores[candidate], kind="stable")]
        rows: list[dict[str, Any]] = []
        for rank, index in enumerate(ordered, 1):
            chunk_id = chunk_by_row[int(index)]
            chunk = loader.lexical_chunks[chunk_id]
            rows.append(
                {
                    "rank": rank,
                    "source_type": "ASR_EXTERNAL_V3_VALIDATED",
                    "branch": "E5",
                    "video_id": chunk["video_id"],
                    "chunk_id": chunk_id,
                    "start_seconds": chunk["start_seconds"],
                    "end_seconds": chunk["end_seconds"],
                    "text": chunk["text"],
                    "score": float(scores[int(index)]),
                }
            )
        results.append(rows)
    return results


def run_trial_asr_smoke(
    loader: ASRExternalV3Loader,
    trial_zip: str | Path,
    output_root: str | Path,
    *,
    e5_query_encoder: QueryEncoder | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve KIS and QA evidence without opening GT or producing answers."""

    parsed = parse_trial_zip(trial_zip)
    queries = [row for row in parsed["queries"] if row["task"] in {"KIS", "QA"}]
    query_texts = [_query_text(row) for row in queries]
    e5_results = (
        _e5_search(loader, query_texts, e5_query_encoder, top_k)
        if e5_query_encoder is not None
        else [None] * len(queries)
    )
    rows: list[dict[str, Any]] = []
    for query, text, e5 in zip(queries, query_texts, e5_results, strict=True):
        lexical = loader.retrieve_spans(text, max_spans=top_k)
        rows.append(
            {
                "query_id": query["query_id"],
                "task": query["task"],
                "query_text": text,
                "lexical": lexical,
                "lexical_nonempty": bool(lexical),
                "e5": e5,
                "e5_status": "PASS" if e5 is not None else "NOT_EXECUTED_ENCODER_UNAVAILABLE",
                "evidence_nonempty": bool(lexical or e5),
                "final_answer_generated": False,
                "ground_truth_opened": False,
            }
        )
    qa_ids = {"query-p1-15-qa", "query-p1-19-qa", "query-p1-22-qa"}
    qa_summary = {
        row["query_id"]: {
            "lexical_nonempty": row["lexical_nonempty"],
            "e5_status": row["e5_status"],
            "evidence_nonempty": row["evidence_nonempty"],
            "answer_generated": False,
        }
        for row in rows
        if row["query_id"] in qa_ids
    }
    result = {
        "source_type": "ASR_EXTERNAL_V3_VALIDATED",
        "trial_query_count": parsed["query_count"],
        "retrieved_query_count": len(rows),
        "kis_query_count": sum(row["task"] == "KIS" for row in rows),
        "qa_query_count": sum(row["task"] == "QA" for row in rows),
        "top_k": top_k,
        "lexical_nonempty_query_count": sum(row["lexical_nonempty"] for row in rows),
        "e5_status": "PASS" if e5_query_encoder is not None else "NOT_EXECUTED_ENCODER_UNAVAILABLE",
        "e5_query_encoder_provenance": getattr(e5_query_encoder, "provenance", None),
        "qa_evidence": qa_summary,
        "final_answers_generated": False,
        "ground_truth_opened": False,
        "queries": rows,
    }
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "trial_p1_asr_evidence_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Trial P1 ASR Evidence Smoke",
        "",
        "This is a GT-free evidence smoke. It does not generate final QA answers.",
        "",
        "- Source: `ASR_EXTERNAL_V3_VALIDATED`",
        f"- KIS queries: {result['kis_query_count']}",
        f"- QA queries: {result['qa_query_count']}",
        f"- Lexical non-empty: {result['lexical_nonempty_query_count']}/{len(rows)}",
        f"- E5 status: {result['e5_status']}",
        "",
        "## Required QA evidence checks",
        "",
    ]
    for query_id in sorted(qa_ids):
        summary = qa_summary[query_id]
        lines.append(
            f"- `{query_id}`: lexical={summary['lexical_nonempty']}, "
            f"E5={summary['e5_status']}, evidence={summary['evidence_nonempty']}"
        )
    lines.extend(["", "## Top lexical evidence", ""])
    for row in rows:
        lines.append(f"### {row['query_id']} ({row['task']})")
        if not row["lexical"]:
            lines.append("- No lexical evidence.")
        for span in row["lexical"]:
            text = str(span["text"]).replace("\n", " ")[:180]
            lines.append(
                f"- #{span['asr_rank']} `{span['video_id']}` "
                f"{span['start_seconds']:.2f}-{span['end_seconds']:.2f}s: {text}"
            )
        lines.append("")
    (destination / "trial_p1_asr_evidence_smoke.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


__all__ = ["OnnxE5QueryEncoder", "QueryEncoder", "run_trial_asr_smoke"]
