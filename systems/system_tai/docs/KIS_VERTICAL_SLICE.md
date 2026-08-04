# KIS Real Vertical Slice

## Objective

Build the smallest real KIS path using BTC mapping and CLIP data. No dummy semantic
embedding is allowed.

## Flow

mapping CSV + BTC CLIP NPY
→ `BenchmarkVideoCatalog`
→ validated `FrameRecord` mapping
→ compatible query encoder
→ vector retrieval
→ `CandidateFrame(actual_frame_id)`
→ grouping and deduplication
→ ranked Top-100
→ checkpoint exporter adapter
→ validator
→ fixture evaluation.

## Module 0 — Benchmark Video Catalog

- **Input:** original BTC video files or authoritative BTC video metadata.
- **Output:** `video_id`, video path or content locator, FPS, duration seconds,
  total frames, frame-index base, and optional codec/resolution.
- **Intended source:** `src/system_tai/data/video_catalog.py`
- **Status:** `IMPLEMENTED` for Phase 1 input loading and validation.
- **Dependencies:** authoritative BTC inventory and an accepted index-base definition.
- **Unit tests:** duplicate `video_id`; missing file; invalid FPS; non-positive frame count;
  duration/frame-count inconsistency; unknown frame-index base.
- **Acceptance:** every retrieval/export `video_id` resolves to authoritative frame bounds,
  and validation can prove that `actual_frame_id` is in range.

## Module 1 — Mapping ingestion and validation

- **Input:** mapping CSV and `BenchmarkVideoCatalog`.
- **Output:** validated records linking keyframe order, timestamp, FPS, CLIP row, and
  original-video `actual_frame_id`.
- **Intended source:** `src/system_tai/data/frame_mapping.py`
- **Status:** `IMPLEMENTED` for Phase 1 CSV loading, catalog validation, and explicit
  frame/feature-row identifiers.
- **Dependencies:** video catalog, mapping schema, feature-row convention.
- **Unit tests:** missing fields; duplicate rows; invalid video; frame bounds;
  timestamp/frame inconsistency; ambiguous row mapping; index-base cases.
- **Acceptance:** every accepted row resolves to exactly one in-range
  `(video_id, actual_frame_id)`; ambiguity is rejected.

## Module 2 — BTC CLIP feature store

- **Input:** BTC CLIP NPY and validated frame records.
- **Output:** typed feature matrix with explicit row-to-frame mapping.
- **Intended source:** `src/system_tai/features/btc_clip_store.py`
- **Status:** `IMPLEMENTED` for Phase 1 NPY loading, validation, statistics, and
  row-to-frame access.
- **Dependencies:** NumPy, mapping records, documented BTC feature layout.
- **Unit tests:** rank/dimension; row count; dtype; non-finite values; vector norms;
  mapping coverage; mismatched row count.
- **Acceptance:** every vector maps to one frame record; a CLIP row is never used as
  `actual_frame_id`.

## Module 3 — Compatible query encoder

- **Input:** KIS text and encoder configuration.
- **Output:** finite normalized query vector in the BTC visual-feature space.
- **Intended source:** `src/system_tai/features/query_encoder.py`
- **Status:** `UNKNOWN`
- **Dependencies:** exact BTC model, weights, tokenizer, preprocessing, projection, and
  normalization convention.
- **Unit tests:** determinism; expected dimension; finite values; normalization;
  empty query; model/config identity.
- **Acceptance:** compatibility is established by authoritative metadata or a reproducible
  sanity benchmark. Equal vector dimension alone is insufficient.

No guessed encoder or dummy embedding may be substituted.

## Phase 1.5B — Kaggle-native calibration

### Module 2A — Kaggle input discovery

- **Input:** `/kaggle/input` or a test root, discovery hints, and `video_id`.
- **Output:** compact JSON manifest resolving the dataset root, original video,
  map-keyframes CSV, CLIP NPY, keyframes, and optional media/object artifacts.
- **Intended source:** `scripts/discover_kaggle_inputs.py`
- **Status:** `IMPLEMENTED` and locally tested; real `L21_V001` discovery is verified,
  while two additional videos remain pending.
- **Dependencies:** attached private Dataset_AIC2026 and runtime directory layout.
- **Unit tests:** zero/one/multiple dataset roots; missing artifacts; ambiguous matches;
  optional artifacts; custom input root.
- **Acceptance:** no slug is hard-coded, ambiguity fails clearly, and no source artifact
  is copied.

### Module 2B — Raw-frame coordinate calibration

- **Input:** original BTC video, mapping CSV, BTC keyframes, `video_id`, deterministic
  sample selection, and offset candidates.
- **Output:** JSON report comparing mapped frame `f` against valid `f-1`, `f`, and `f+1`
  decodes with aggregate and per-sample scores.
- **Intended source:** `scripts/calibrate_frame_mapping.py`
- **Status:** `IMPLEMENTED` with local mechanics tests and a real 15-sample
  `L21_V001` calibration; multi-video reproduction remains pending.
- **Dependencies:** original video decoder, map-keyframes CSV, keyframe images.
- **Unit tests:** beginning/middle/end sampling; invalid offsets; deterministic pixel
  scores; exact zero-offset explanation; timestamp-explained `+1`; systematic
  unexplained offset failure; decoder disagreement; bounds failure; batch aggregation.
- **Acceptance:** `frame_idx` is never silently corrected; bounds and zero-based mapping
  policy are validated separately; timestamp-explained visual offsets pass; systematic
  unexplained offsets or material decoder disagreement fail; and at least three real
  videos can be processed in one batch.

Phase 1.5B revises this acceptance model: mapping-coordinate validity is independent of
JPEG visual alignment. `actual_frame_id` always preserves `frame_idx`. A JPEG best match
at `frame_idx + 1` is explained, rather than failed, when it equals
`decimal_round_half_up(pts_time * fps)`. Numeric-generation-rule identification is
diagnostic and cannot invalidate an in-bounds `frame_idx`. The implementation exposes
separate mapping-policy, numeric-model, decoder-agreement, and visual-agreement results.

### Module 2D — Mapping rounding audit

- **Input:** BTC map-keyframes CSV and `video_id`.
- **Output:** JSON summary or CSV row diagnostics for Decimal-exact product/floor/
  nearest, binary-float product/truncation/floor, agreement counts and ratios, and all
  required frame-delta distributions.
- **Intended source:** `scripts/audit_mapping_rounding.py`
- **Status:** `IMPLEMENTED` with local Decimal boundary tests; real `L21_V001` evidence
  is documented separately from dataset-wide claims.
- **Dependencies:** Python `decimal`, documented Python binary-float behavior, and
  map-keyframes columns `n`, `pts_time`, `fps`, and `frame_idx`.
- **Unit tests:** integer timestamps; fractional products at 0.001, 0.49, 0.50, and
  0.99; non-30 FPS; the verified `260.4`, `1024.1`, `1031.1`, and `1058.6` binary-float
  regression cases; malformed and non-finite values.
- **Acceptance:** Decimal and binary-float models are reported separately; a Decimal
  floor difference is not unresolved when binary-float truncation matches; numeric-rule
  status does not gate mapping validity; and diagnostics never modify shared
  `actual_frame_id` or `frame_id`.

For real `L21_V001`, the 307-row audit verifies 303 Decimal-floor matches and four
`frame_idx - Decimal floor = -1` cases. Decimal-nearest offset is `0` for 233 rows and
`+1` for 74. Only 15 rows have been visually decoded; all 15 visual best offsets match
the Decimal-nearest prediction, and random/sequential decoding agrees for all 15.
Binary-float truncation as the mapping-generation rule and nearest timestamp alignment
as the JPEG extraction rule remain inferred until reproduced on `L21_V002` and
`L22_V001`.

### Module 2C — BTC CLIP pipeline identification

- **Input:** validated mapping/NPY alignment, sampled keyframes, and optional OpenAI
  CLIP, OpenCLIP, or Hugging Face CLIP image encoders.
- **Output:** JSON comparison report with row-wise cosine/L2/difference metrics,
  self-match statistics, norms, implementation identifiers, and preprocessing details.
- **Intended source:** `scripts/identify_btc_clip_pipeline.py`
- **Status:** `IMPLEMENTED` as optional adapters and metric gates with synthetic tests;
  compatibility remains `UNVERIFIED` until reproducible multi-video Kaggle calibration
  succeeds.
- **Dependencies:** optional backend libraries and exact weights, validated feature-row
  mapping, BTC keyframes and CLIP NPY.
- **Unit tests:** metric correctness; dimension mismatch; rank calculation; skipped
  optional backend; multi-video identification gate.
- **Acceptance:** dimension alone never identifies a pipeline, unavailable backends are
  reported as `SKIPPED`, and `IDENTIFIED` requires correct self-match plus strong
  row-wise agreement reproduced across multiple videos.

## Module 4 — Vector retrieval

- **Input:** compatible query vector and BTC visual matrix.
- **Output:** ranked feature-row hits with scores.
- **Intended source:** `src/system_tai/retrieval/vector_search.py`
- **Status:** `PLANNED`
- **Dependencies:** query encoder and feature store.
- **Unit tests:** known-vector ranking; ties; top-k bounds; dimension mismatch;
  non-finite vectors; deterministic ordering.
- **Acceptance:** exact search is reproducible. NumPy exact search is sufficient before
  adding a scalable index.

## Module 5 — Candidate construction

- **Input:** retrieval hits and validated row-to-frame mapping.
- **Output:** `CandidateFrame` records with `video_id`, `actual_frame_id`, score, rank,
  and internal provenance.
- **Intended source:** `src/system_tai/retrieval/candidates.py`
- **Status:** `PLANNED`
- **Dependencies:** retrieval, frame mapping, common schemas.
- **Unit tests:** correct row mapping; missing/duplicate mapping; score/rank preservation;
  prevention of row-as-frame substitution.
- **Acceptance:** every candidate identifies an in-range original BTC frame.

## Module 6 — Grouping and KIS ranking

- **Input:** `CandidateFrame` records.
- **Output:** grouped, deduplicated, deterministically ranked records, maximum 100.
- **Intended source:** `src/system_tai/ranking/kis_ranker.py`
- **Status:** `PLANNED`
- **Dependencies:** candidates and an explicit ranking/dedup policy.
- **Unit tests:** grouping; exact duplicates; ties; empty input; stable ranks;
  maximum 100; repeated-run determinism.
- **Acceptance:** ranks start at one, remain unique and ordered, and exact
  `(video_id, actual_frame_id)` duplicates are removed.

## Module 7 — Checkpoint exporter adapter

- **Input:** ranked internal KIS records.
- **Output:** current proposed UTF-8 JSONL checkpoint.
- **Intended source:** `src/system_tai/checkpointing/exporter.py`
- **Status:** `PLANNED`
- **Dependencies:** accepted or proposed boundary schema.
- **Unit tests:** UTF-8; one object per line; field mapping; deterministic order;
  no internal-field leakage.
- **Acceptance:** exported `frame_id` equals `actual_frame_id` without numeric conversion,
  and a future accepted schema can replace serialization without changing retrieval.

## Module 8 — Shared-output validator

- **Input:** checkpoint JSONL, query set, `BenchmarkVideoCatalog`, and validated mapping.
- **Output:** deterministic errors, warnings, and validity result.
- **Intended source:** `src/system_tai/validation/checkpoint_validator.py`
- **Status:** `PLANNED`
- **Dependencies:** video catalog, mapping, shared schema, team validator boundary.
- **Unit tests:** malformed JSON; types; missing fields; unknown query/video;
  out-of-range frame; duplicate rank/record; unsorted ranks; more than 100 rows.
- **Acceptance:** invalid output cannot reach evaluation; bounds are proven from the
  authoritative catalog.

## Module 9 — KIS fixture evaluator

- **Input:** validated predictions and fixture ground-truth intervals.
- **Output:** per-query score, R@1/5/20/50/100, observed ranks, and aggregate report.
- **Intended source:** `src/system_tai/evaluation/kis_fixture.py`
- **Status:** `PLANNED`
- **Dependencies:** fixture schema, validated output, interval semantics.
- **Unit tests:** correct/wrong video; in/out-of-range frame; boundaries; top-k best hit;
  empty predictions.
- **Acceptance:** fixture results are manually verifiable and reproducible. They are not
  labeled official BTC performance.

## Validation gates

### Gate A — Integration smoke test

- one real video;
- one real mapping CSV;
- one real CLIP NPY;
- one query;
- valid end-to-end checkpoint output.

### Gate B — Retrieval sanity benchmark

- a small corpus containing positive and distractor videos;
- at least five manually verified queries when data is available;
- expected video and/or interval for every query;
- Video Recall@K and observed ranking reported;
- no official-performance claim.

## Exclusions

Agent, GNN, Event Graph, VLM, OCR, ASR, Q&A, TRAKE, API server, backend, and
production frontend are excluded.

## Remaining decisions

- Exact compatible BTC text encoder.
- Dataset-wide raw-video confirmation of the zero-based working interpretation.
- Optional checkpoint envelope/version fields.
- Shared validator/evaluator interfaces.
- Official BTC submission format.
- Final repository destination; this blocks merge, not isolated local implementation.
