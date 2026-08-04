# system_tai Context

## Purpose

`system_tai` is Nguyen Tai's independent end-to-end AIC 2026 system. Its reviewed
personal design emphasizes retrieval-first candidate generation, video ranking, local
temporal refinement, evidence-first Q&A, ordered TRAKE chains, ranked output, and
validation. The Phase 1 catalog, mapping, feature-store, and input-audit foundation is
implemented; query encoding and semantic retrieval have not started.

TRIAGE-EG belongs to Huu Tri. Its architecture and code are reference material only.

## Scope boundary

The team shares benchmark inputs, query sets, ground truth, original-frame coordinates,
selected checkpoint schemas, a validator, and an evaluator. Everything inside the
`system_tai` boundary may be designed independently.

UTF-8 JSONL is the current proposed checkpoint format, not yet an accepted schema.
Serialization must remain behind an adapter so an accepted team schema can replace it.

## Frame model

`system_tai.actual_frame_id` must equal the original frame index in the original BTC
video. At a shared boundary, `frame_id` is an alias for that value.

For BTC keyframes, `actual_frame_id` preserves `frame_idx` from the map-keyframes CSV
exactly; no one-based correction is applied. Zero-based indexing is the working
interpretation until decoded raw-video calibration confirms it. Keyframe order, CLIP
row, local frame index, and filename number are separate identifiers and must be
connected to the raw-video coordinate by validated mapping data.

## First implementation scope

The first implementation is KIS-only and library/CLI-oriented:

mapping CSV + BTC CLIP NPY
-> video catalog and validated mapping
-> compatible text encoder
-> vector retrieval
-> original-frame candidates
-> grouping, deduplication, and Top-100 ranking
-> proposed JSONL export
-> validation and fixture evaluation.

Gate A proves integration on one real video, mapping, feature file, and query.

Gate B checks retrieval sanity on a small positive/distractor corpus with at least five
manually verified queries when data is available. It reports Video Recall@K and observed
rankings, not official performance.

## Phase 1.5A calibration scope

The private Dataset_AIC2026 remains attached under a runtime-discovered child of
`/kaggle/input`. Phase 1.5A discovers per-video artifacts, compares preserved
`frame_idx` values with decoded `f-1/f/f+1`, and optionally compares three ViT-B/32
implementation interfaces against BTC image-feature rows. BTC currently confirms only
the clip-ViT-B-32 family label and feature-row order matching keyframe order. Exact
weights, preprocessing, normalization, similarity metric, and text compatibility remain
unverified until real multi-video Kaggle reports satisfy the identification gate.

## Deferred scope

Agent, GNN, Event Graph, VLM, OCR, ASR, Q&A, TRAKE, backend API, production frontend,
distributed services, and deployment are outside the first slice.

## Intended isolated layout

- `src/system_tai/`: system code
- `scripts/`: CLI entry points
- `configs/`: local examples
- `tests/`: unit and later integration tests
- `docs/`: system-specific documentation

The repository placement is resolved: work remains isolated under `systems/system_tai`
in the shared repository on a personal branch. TRIAGE-EG remains untouched.

## Blocking uncertainties

- Exact BTC CLIP model, weights, tokenizer, preprocessing, and text/visual compatibility.
- Decoded raw-frame agreement and dataset-wide confirmation of the zero-based working
  interpretation.
- Accepted shared JSONL schema and required version fields.
- Official BTC submission format.
- Authoritative mapping between CSV rows, keyframe order, feature rows, and raw frames.
- Ownership and executable interface of the shared validator and evaluator.
