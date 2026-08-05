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
- **Status:** `IMPLEMENTED` as the Phase 2 canonical official OpenAI CLIP ViT-B/32
  adapter behind an optional-dependency protocol.
- **Dependencies:** optional official OpenAI CLIP and Torch packages, locally cached or
  explicitly downloadable weights, identified 512-dimensional compatibility.
- **Unit tests:** determinism; expected dimension; finite values; normalization;
  empty query; model/config identity.
- **Acceptance:** the public OpenAI APIs are used, the model is loaded once, output is a
  finite non-zero normalized float32 vector of dimension 512, and no fallback model is
  selected silently.

No guessed encoder or dummy embedding may be substituted.

## Phase 1.5C — Kaggle-native compatibility calibration

### Module 2A — Kaggle input discovery

- **Input:** `/kaggle/input` or a test root, discovery hints, and `video_id`.
- **Output:** compact JSON manifest resolving the dataset root, original video,
  map-keyframes CSV, CLIP NPY, keyframes, and optional media/object artifacts.
- **Intended source:** `scripts/discover_kaggle_inputs.py`
- **Status:** `IMPLEMENTED` and locally tested; discovery and input gates are verified
  for all three calibration videos.
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
- **Status:** `IMPLEMENTED` with local mechanics tests and real 45-sample calibration
  across three videos.
- **Dependencies:** original video decoder, map-keyframes CSV, keyframe images.
- **Unit tests:** beginning/middle/end sampling; invalid offsets; deterministic pixel
  scores; exact zero-offset explanation; timestamp-explained `+1`; systematic
  unexplained offset failure; exact/below/above superiority-margin decisions; decoder
  disagreement; bounds failure; batch aggregation.
- **Acceptance:** `frame_idx` is never silently corrected; bounds and zero-based mapping
  policy are validated separately; ambiguous ties are excluded from explained and
  contradictory counts; all decisive samples must match the timestamp prediction for
  `VISUAL_ALIGNMENT_EXPLAINED`; and material decoder disagreement still fails.

Phase 1.5B established that mapping-coordinate validity is independent of
JPEG visual alignment. `actual_frame_id` always preserves `frame_idx`. A JPEG best match
at `frame_idx + 1` is explained, rather than failed, when it equals
`decimal_round_half_up(pts_time * fps)`. Numeric-generation-rule identification is
diagnostic and cannot invalidate an in-bounds `frame_idx`. The implementation exposes
separate mapping-policy, numeric-model, decoder-agreement, and visual-agreement results.
Phase 1.5C adds margin-aware ambiguity. With `superiority_margin = 0.0001`, the three
raw-best mismatches have margins
`0.000007`, `0.000014`, and `0.000017` and are classified as ambiguous. The verified
three-video aggregate is 42 explained decisive, 3 ambiguous, and 0 contradictory
decisive samples.

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
All three calibration videos have binary-float-truncation ratio `1.0`. Binary-float
truncation as the mapping-generation implementation and nearest timestamp alignment as
the JPEG extraction implementation remain inferred rather than proven.

### Module 2C — BTC CLIP pipeline identification

- **Input:** validated mapping/NPY alignment, sampled keyframes, official OpenAI CLIP,
  distinct OpenCLIP model/pretrained variants, and Hugging Face CLIP image encoders.
- **Output:** JSON comparison report with row-wise cosine/L2/difference metrics,
  self-match statistics, norms, implementation identifiers, and preprocessing details.
- **Intended source:** `scripts/identify_btc_clip_pipeline.py`
- **Status:** `IMPLEMENTED`; the full-corpus three-video run identifies official OpenAI
  CLIP ViT-B/32 and OpenCLIP ViT-B-32-quickgelu/openai as compatible candidates.
- **Dependencies:** optional backend libraries and exact weights, validated feature-row
  mapping, BTC keyframes and CLIP NPY.
- **Unit tests:** metric correctness; public OpenAI module without `_MODELS`; direct
  Tensor, pooled ModelOutput, and unsupported Hugging Face outputs; separate standard
  and QuickGELU OpenCLIP candidates; dynamic candidate discovery; multi-video gate.
- **Acceptance:** dimension alone never identifies a pipeline, unavailable backends are
  reported as `SKIPPED`, variants remain separate, and `IDENTIFIED` requires at least
  three unique videos, validated mapping, correct self-match, plus near-exact or clearly
  superior row-wise agreement.

Across all 867 audited rows, official OpenAI CLIP and OpenCLIP QuickGELU/OpenAI both
have mean cosine approximately `0.999162`, minimum p05 `0.997200`, self-match Top-1
`1.0`, mean rank `1.0`, and dimension 512. They are numerically equivalent within
approximately `1e-10`. Official OpenAI CLIP is canonical. This is three-video
compatibility evidence, not BTC-official preprocessing or dataset-wide evidence.

## Module 4 — Vector retrieval

- **Input:** compatible query vector and BTC visual matrix.
- **Output:** ranked feature-row hits with scores.
- **Intended source:** `src/system_tai/retrieval/vector_search.py`
- **Status:** `BASELINE`; exact chunked multi-video NumPy cosine retrieval is
  implemented for Phase 2.
- **Dependencies:** query encoder protocol, validated feature registry, NumPy.
- **Unit tests:** known-vector ranking; ties; top-k bounds; dimension mismatch;
  non-finite vectors; deterministic ordering.
- **Acceptance:** chunked cosine equals a non-chunked reference, global Top-K is exact,
  and ties use score descending then video/frame/CLIP-row ascending. FAISS is deferred.

## Module 5 — Candidate construction

- **Input:** retrieval hits and validated row-to-frame mapping.
- **Output:** immutable `CandidateFrame` records with shared `frame_id` copied from CSV
  `frame_idx` plus internal CLIP-row/keyframe provenance.
- **Intended source:** `src/system_tai/common/schemas.py` and
  `src/system_tai/retrieval/vector_search.py`
- **Status:** `IMPLEMENTED` in the exact retriever and immutable domain schemas.
- **Dependencies:** exact retrieval and validated row-to-frame mapping.
- **Unit tests:** correct row mapping; missing/duplicate mapping; score/rank preservation;
  prevention of row-as-frame substitution.
- **Acceptance:** every candidate identifies an in-range original BTC frame.

## Module 6 — Grouping and KIS ranking

- **Input:** `CandidateFrame` records.
- **Output:** optionally temporally suppressed, deterministically re-ranked candidates
  plus removal counts.
- **Intended source:** `src/system_tai/ranking/kis_ranker.py`
- **Status:** `BASELINE`; exact ordering is canonical and optional suppression is
  implemented but disabled by default.
- **Dependencies:** ranked candidates and explicit minimum-gap/per-video limits.
- **Unit tests:** grouping; exact duplicates; ties; empty input; stable ranks;
  maximum 100; repeated-run determinism.
- **Acceptance:** disabled mode preserves the exact baseline; enabled mode preserves
  order and frame IDs while reporting every removal.

## Module 7 — Checkpoint exporter adapter

- **Input:** one or more `KISResult` objects.
- **Output:** current proposed UTF-8 JSONL checkpoint.
- **Intended source:** `src/system_tai/checkpointing/exporter.py`
- **Status:** `IMPLEMENTED` for the current proposed checkpoint boundary.
- **Dependencies:** accepted or proposed boundary schema.
- **Unit tests:** UTF-8; one object per line; field mapping; deterministic order;
  no internal-field leakage.
- **Acceptance:** ranks are contiguous from one, at most 100 unique video/frame pairs
  are emitted per query, core mode leaks no internal fields, and internal fields appear
  only under a separate internal object when explicitly requested.

## Module 8 — Shared-output validator

- **Input:** checkpoint JSONL and optional loaded feature registry.
- **Output:** deterministic errors, warnings, and validity result.
- **Intended source:** `src/system_tai/validation/checkpoint_validator.py`
- **Status:** `IMPLEMENTED` as the local Phase 2 proposed-boundary validator.
- **Dependencies:** proposed core schema and optional validated feature registry.
- **Unit tests:** malformed JSON; types; missing fields; unknown query/video;
  out-of-range frame; duplicate rank/record; unsorted ranks; more than 100 rows.
- **Acceptance:** errors contain line, query ID, code, and message; valid files enforce
  types, unique contiguous ranks, Top-100, unique pairs, and optional registry existence.

## Module 8A — Feature-store registry

- **Input:** explicit JSON manifest containing per-video mapping CSV and CLIP NPY paths.
- **Output:** immutable store descriptors, memory-mapped matrices, and physical-row to
  `FrameMappingRecord` mappings.
- **Intended source:** `src/system_tai/features/btc_clip_store.py`
- **Status:** `IMPLEMENTED` with read-only memory mapping and explicit manifests.
- **Dependencies:** NumPy and the verified physical-row alignment rule.
- **Unit tests:** CSV row alignment; exact `frame_idx`; missing files; duplicate videos;
  invalid/non-finite/zero-norm matrices; row and dimension mismatch.
- **Acceptance:** every feature row resolves to exactly one mapping record, all stores
  share configured dimension 512, and source NPY arrays are memory-mapped read-only.

## Module 8B — Phase 2 CLI

- **Input:** manifest, query ID/text, Top-K, device, chunk size, and destination.
- **Output:** proposed KIS JSONL plus a concise runtime summary.
- **Intended source:** `src/system_tai/kis/retrieve.py`
- **Status:** `IMPLEMENTED` for exact retrieval with official OpenAI CLIP.
- **Dependencies:** registry, canonical encoder, exact retrieval, exporter, validator.
- **Unit tests:** parser/config failures and fake-encoder integration through library
  components; no network or BTC fixture dependency.
- **Acceptance:** missing optional packages/weights fail clearly, source data remains
  read-only, no vectors are printed, and only the requested output is written.

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

## Phase 2.5 — Ground-truth KIS benchmark

### Module 10 — Benchmark schema and loader

- **Input:** UTF-8 YAML or JSON containing human-authored query variants and positive
  `(video_id, frame_id)` labels.
- **Output:** immutable benchmark/query/relevant-frame domain records.
- **Intended source:** `src/system_tai/evaluation/benchmark_schema.py`
- **Status:** `IMPLEMENTED` with typed immutable schemas and UTF-8 YAML/JSON loading.
- **Dependencies:** PyYAML and the shared `frame_id = frame_idx` rule.
- **Unit tests:** schema parsing, enums, draft/verified status, duplicate identifiers and
  label pairs, invalid frame identifiers.
- **Acceptance:** no CLIP row, keyframe order, filename, or physical CSV row can be used
  as ground-truth `frame_id`; verified labels are human-authored only.

### Module 11 — Registry-aware benchmark validator

- **Input:** benchmark file and loaded `FeatureStoreRegistry`.
- **Output:** typed benchmark plus structured errors and draft/verified counts.
- **Intended source:** `src/system_tai/evaluation/benchmark_validator.py`
- **Status:** `IMPLEMENTED` with structured registry-aware errors and warnings.
- **Dependencies:** Module 10 and the validated feature registry.
- **Unit tests:** unknown videos, labels absent from mapping, incomparable variants,
  missing verified labels, and draft exclusion.
- **Acceptance:** invalid data is reported explicitly; drafts are never scored by
  default; every scored frame exists in the corresponding mapping CSV.

### Module 12 — Exact benchmark evaluator and paired comparison

- **Input:** verified benchmark queries, canonical `ExactNumpyRetriever`, and configured
  cutoffs.
- **Output:** binary query Recall@K, ground-truth coverage, hit/rank/MRR and
  relevant-video coverage metrics, grouped aggregates, and Vietnamese-versus-English
  paired comparisons.
- **Intended source:** `src/system_tai/evaluation/kis_benchmark.py`
- **Status:** `IMPLEMENTED` over the unchanged canonical unsuppressed exact retriever.
- **Dependencies:** unsuppressed Phase 2 exact retrieval and Modules 10–11.
- **Unit tests:** exact Recall@K, first relevant rank, reciprocal rank, aggregates,
  paired win/tie/loss, missing variants, deterministic output, suppression isolation.
- **Acceptance:** per-query Recall@K is one when any exact positive occurs in Top-K and
  zero otherwise; aggregate Recall@K is its mean over valid verified queries;
  multi-label coverage is separately named `ground_truth_coverage_at_k`; only verified
  queries are scored; zero verified queries returns `no_verified_queries`; the evaluator
  never invokes temporal suppression. Paired comparisons require exactly one verified
  Vietnamese-direct query and one verified English variant in the same comparable
  semantic group; deltas are English minus Vietnamese; Recall and first-rank
  win/tie/loss use the English perspective. Missing or draft variants are reported,
  while duplicate or otherwise invalid variants block evaluation with structured
  validation errors. Translation claims require verified paired measurements.

### Module 13 — Reports, CLI, and bounded annotation helper

- **Input:** validation/evaluation results, runtime metadata, output directory, and
  optional candidate keyframe directories.
- **Output:** deterministic JSON/CSV/Markdown reports and unverified draft annotation
  review records outside Git.
- **Intended source:** `src/system_tai/evaluation/reports.py`,
  `src/system_tai/evaluation/annotation.py`, and `src/system_tai/kis/benchmark.py`.
- **Status:** `IMPLEMENTED`; generated artifacts default outside Git and annotation
  candidates remain explicitly unreviewed.
- **Dependencies:** Modules 10–12 and existing bounded Kaggle artifact layout.
- **Unit tests:** report serialization, validation-only CLI, bounded path resolution,
  and draft-only annotation output.
- **Acceptance:** report defaults target `/kaggle/working/system_tai_outputs/kis_benchmark/`;
  helper output never marks a frame relevant; no dataset image or generated report is
  checked into Git.

## Exclusions

Agent, GNN, Event Graph, VLM, OCR, ASR, Q&A, TRAKE, API server, backend, and
production frontend are excluded.

## Remaining decisions

- Dataset-wide compatibility and real text-query retrieval quality.
- Dataset-wide raw-video confirmation of the zero-based working interpretation.
- Optional checkpoint envelope/version fields.
- Shared validator/evaluator interfaces.
- Official BTC submission format.
- Final repository destination; this blocks merge, not isolated local implementation.
