# Stage 1C — Qualitative Text Retrieval Evaluation v0.1

Stage 1C evaluates the raw text-to-BTC baseline after Stage 1B has selected a
`VERIFIED` encoder. It reuses the Stage 1A exact cosine backend and canonical
catalog without rebuilding Stage 0, Stage 1A, or changing Stage 1B compatibility
logic. It performs no translation, expansion, reranking, diversification,
multilingual projection, object fusion, temporal reasoning, or model training.

## Inputs and fail-closed boundary

The runner requires complete saved Stage 0 manifests, a complete Stage 1A index,
and a Stage 1B result whose selected candidate is `VERIFIED`, model-space status is
`MODEL_SPACE_VERIFIED`, and text retrieval is
`READY_FOR_QUALITATIVE_TESTING`. The Stage 1A fingerprint recorded by Stage 1B must
match the loaded index. The selected Stage 1B contract is the only encoder contract
used; local source and checkpoint assets are loaded through the existing offline
Stage 1B adapter.

## Query and retrieval contracts

The default UTF-8 JSONL suite contains 14 English/Vietnamese pairs, or 28 queries,
covering object, action, scene, attribute, spatial relation, multi-concept, event,
and difficult cases. Query IDs and texts are unique, each pair contains exactly one
English and one Vietnamese record, and the suite is fingerprinted from canonical
serialization.

All selected queries are encoded in one batch and normalized to finite 512-D
vectors. The existing Stage 1A exact cosine backend retrieves an internal raw
Top-100. The first 50 rows are preserved unchanged for qualitative review; the
Top-100 supplies the existing KIS export, whose `frame_id` is the authoritative
`original_frame_idx`, never `n` or a reconstructed timestamp.

## Diagnostics, not ranking changes

For each query, Stage 1C writes raw frames, a grouped-by-video view, score
descriptors, initial-frame concentration, same-video concentration, and exact
stored-vector duplication within returned Top-50 only. Exact duplication uses
canonical stored vector bytes; it does not use tolerance or scan the corpus.
English/Vietnamese pairs record embedding cosine plus frame/video overlap and
Jaccard. None of these diagnostics determines semantic relevance or modifies raw
ranking.

The optional structural warning thresholds are `PROJECT_REVIEW_HEURISTIC`. They
only prioritize human inspection and never reject successful Stage 1C execution.

## Human review boundary

Contact sheets show raw Top-20 frames and one representative per Top-12 video.
The review template contains Top-10 raw frames per query. Reviewers assign only
`RELEVANT`, `PARTIAL`, `IRRELEVANT`, or `UNCERTAIN`; no score, CLIP model, LLM, or
VLM assigns relevance automatically. Before those labels are complete, the only
valid status is:

```text
RETRIEVAL_QUALITY_STATUS = NOT_REVIEWED
```

Human relevance rates are qualitative measurements, not competition Recall@K.

## Output bundle

The qualitative ZIP contains manifests, report, query suite, pair diagnostics,
review files, per-query JSONL/CSV diagnostics, and resized contact sheets. It
excludes model weights, source checkout, Stage 1 vectors, NPY files, raw videos,
the full keyframe corpus, logs, and caches.

