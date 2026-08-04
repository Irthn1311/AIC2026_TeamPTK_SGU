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
exactly; no one-based correction is applied. Zero-based bounds and decoded behavior are
verified for `L21_V001`, but dataset-wide confirmation still requires more videos.
`keyframe_visual_frame_id = decimal_round_half_up(pts_time * fps)` is a separate
diagnostic and never changes the shared coordinate. Decimal floor, binary-float
truncation, and Decimal nearest are candidate numeric models, not mapping-validity
requirements. Keyframe order, CLIP row, local frame index, and filename number are
separate identifiers and must be connected to the raw-video coordinate by validated
mapping data.

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

## Phase 1.5B calibration scope

The private Dataset_AIC2026 remains attached somewhere within the nested runtime layout
under `/kaggle/input`. Phase 1.5B discovers per-video artifacts, audits Decimal timestamp
rounding, compares binary-float truncation, validates preserved `frame_idx` bounds, and
separately compares JPEGs with decoded `f-1/f/f+1`. The full `L21_V001` mapping has 303
Decimal-floor matches and four valid `-1` differences; the Decimal-nearest offset is
`0` for 233 rows and `+1` for 74. All 15 visually decoded samples match that nearest
prediction. Binary truncation and nearest timestamp extraction remain inferred rules.
Optional ViT-B/32 implementation comparison remains gated on three passing videos. BTC
currently confirms only the clip-ViT-B-32 family label and feature-row order matching
keyframe order. Exact weights, preprocessing, normalization, similarity metric, and text
compatibility remain unverified.

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
- Reproduction of decoded-frame, rounding, and mapping behavior on at least three real
  videos; current real evidence covers only `L21_V001`.
- Accepted shared JSONL schema and required version fields.
- Official BTC submission format.
- Authoritative mapping between CSV rows, keyframe order, feature rows, and raw frames.
- Ownership and executable interface of the shared validator and evaluator.
