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

## Phase 2.6 — Opt-in multilingual Weighted RRF pilot

### Module 14 — Explicit query variants and multi-query retrieval

- **Input:** one query ID, at least one immutable explicit query variant
  `(variant_id, text, language, variant_type, weight)`, per-variant Top-K, output Top-K,
  and positive finite RRF constant.
- **Output:** one deterministically ranked `KISResult` whose candidates are deduplicated
  by exact `(video_id, frame_id)` and retain internal per-variant provenance.
- **Intended source:** `src/system_tai/retrieval/multi_query.py`
- **Status:** `BASELINE`; implemented as an opt-in layer without changing exact
  single-query retrieval.
- **Dependencies:** unchanged canonical `ExactNumpyRetriever`, `KISQuery`, `KISResult`,
  `CandidateFrame`, and explicit caller-provided query variants.
- **Unit tests:** validation; one/three-variant hits; duplicate frame pairs; weighted RRF
  arithmetic with one-based ranks; deterministic tie-breaking; Top-100; contiguous
  ranks; exact frame preservation; raw-score scale isolation; exporter provenance
  isolation; unchanged single-query behavior.
- **Acceptance:** fusion score is exactly
  `sum(weight / (rrf_constant + one_based_rank))` over variants containing a candidate;
  ordering is fusion score descending, variant-hit count descending, best individual
  rank ascending, then video ID, frame ID, and CLIP row ascending; raw cosine scores are
  diagnostic only; temporal suppression is absent; fusion is never an implicit default.

### Module 15 — Fusion pilot evaluator, reports, and CLI

- **Input:** a valid human-authored benchmark, comparable verified variants grouped by
  `semantic_group_id`, canonical exact retriever, RRF configuration, and metric cutoffs.
- **Output:** per-group fused rank/reciprocal-rank/Recall/hit/ground-truth-coverage
  metrics plus deterministic JSON, CSV, and Markdown pilot reports outside Git.
- **Intended source:** `src/system_tai/evaluation/fusion_benchmark.py`,
  `src/system_tai/evaluation/fusion_reports.py`, and
  `src/system_tai/kis/benchmark_fusion.py`.
- **Status:** `BASELINE`; evaluator, deterministic reports, and a separate CLI are
  implemented and synthetically tested. Real Kaggle fusion measurement is pending.
- **Dependencies:** Modules 10–14, official OpenAI CLIP ViT-B/32, and the validated
  three-video feature manifest at runtime.
- **Unit tests:** metric correctness; missing, draft, duplicate, and incomparable groups;
  deterministic repeated reports; validation-only/no-valid-group states; CLI parsing.
- **Acceptance:** only comparable verified variants with identical positive labels and
  source scope are fused; existing Phase 2.5 metric definitions do not change; no valid
  group fails explicitly; reports identify pilot scope and retrieval-selection bias;
  no official or dataset-wide performance claim is made.

### Phase 2.6 pilot evidence boundary

`config/kis_benchmark.pilot_three_groups.yaml` contains exactly three semantic groups,
nine comparable human-verified variants, and six excluded drafts. Its positives use only
official/shared `(video_id, frame_id)` coordinates. The labels were selected after
retrieval inspection, so the pilot has retrieval-selection bias.

Canonical unsuppressed per-variant observations are: city pedestrians (Vietnamese miss,
translation rank 1, expansion rank 3), conference attendees (Vietnamese miss,
translation rank 14, expansion rank 14), and landslide warning sign (Vietnamese rank 1,
translation rank 4, expansion rank 6). These three intents over three videos do not
establish official or dataset-wide quality. The next measured milestone is opt-in
Weighted RRF; Gate B remains incomplete.

## Phase 3 — Contest-ready Textual KIS CLI MVP

### Module 16 — Bounded full-corpus discovery and reusable manifest

- **Input:** Kaggle input root containing one dataset root and bounded mapping, CLIP,
  keyframe, and optional raw-video artifact families.
- **Output:** immutable discovered-video records and deterministic
  `feature_manifest.json` with schema/discovery versions and a SHA-256 fingerprint.
- **Intended source:** `src/system_tai/data/corpus_discovery.py` and
  `src/system_tai/kis/build_manifest.py`.
- **Status:** `IMPLEMENTED`; bounded family discovery, deterministic fingerprinted
  manifests, reuse validation, and a standalone manifest CLI are present.
- **Dependencies:** BTC artifact naming conventions as discovery hints, NumPy header
  loading, UTF-8-SIG mapping parsing, and the existing feature-store loader.
- **Unit tests:** multiple videos, incomplete and ambiguous artifacts, deterministic
  ordering/fingerprint, row-count mismatch, manifest reuse, Windows/POSIX-safe paths,
  and proof that source artifacts are never copied.
- **Acceptance:** discovery never recursively scans unrelated Kaggle input trees;
  mapping rows equal NPY rows; each video has one mapping, one NPY, and one keyframe
  source; raw video is optional; reuse revalidates the fingerprint and source paths.

### Module 17 — Contest query schema

- **Input:** one explicit Vietnamese query, optional caller-authored English translation
  and expansion, positive finite variant weights, output Top-K, and optional metadata;
  or an equivalent UTF-8 YAML/JSON batch.
- **Output:** immutable contest-query records and explicit Phase 2.6 `QueryVariant`
  tuples.
- **Intended source:** `src/system_tai/kis/contest_schema.py`.
- **Status:** `IMPLEMENTED`; single and safe UTF-8 YAML/JSON batch inputs are supported.
- **Dependencies:** safe YAML loading and existing query-variant enums.
- **Unit tests:** single and batch parsing, missing Vietnamese text, duplicate query IDs,
  invalid weights/Top-K, malformed UTF-8, and metadata immutability.
- **Acceptance:** no translation is generated; every variant is explicit; query IDs are
  unique; output Top-K is between 1 and 100.

### Module 18 — Contest runner, artifacts, and reproducibility

- **Input:** corpus manifest, contest queries, official OpenAI CLIP text encoder,
  exact-retrieval and opt-in Weighted RRF configuration, output directory, and failure
  policy.
- **Output:** combined and isolated core JSONL, internal CSV/candidate inspection data,
  validation report, run manifest, timings, Markdown summaries, and an optional derived
  low-resolution contact sheet outside Git.
- **Intended source:** `src/system_tai/kis/contest_runner.py`,
  `src/system_tai/inspection/candidate_report.py`, and
  `src/system_tai/kis/contest.py`.
- **Status:** `BASELINE`; the contest CLI passed a real full-corpus technical run over
  873 videos, 177,321 feature rows, five queries, 15 variants, and 500 valid records.
  This is operational evidence, not semantic-quality or official-performance proof.
- **Dependencies:** Modules 1–17, `CheckpointExporter`, `CheckpointValidator`, and
  optional Pillow only when a contact sheet is explicitly requested.
- **Unit tests:** single/batch execution; registry/model loaded once; RRF output used;
  exact frame preservation; contiguous Top-100; no core-provenance leak; invalid-output
  failure; query-error isolation; timing/run-manifest fields; deterministic result
  artifacts; image-path resolution; and no source copying.
- **Acceptance:** exact NumPy remains the per-variant backend; canonical Top-100 is never
  temporally suppressed; model downloads require explicit authorization; invalid
  checkpoints return non-zero; failed queries receive no fabricated metrics; all
  generated files stay in the caller-selected output directory.

### Module 19 — Latency evidence gate

- **Input:** measured Phase 3 discovery, loading, encoding, retrieval, fusion, export,
  validation, per-query, and batch durations plus corpus sizes.
- **Output:** `timings.json` and the timing section of `run_summary.md`.
- **Intended source:** `src/system_tai/kis/contest_runner.py`.
- **Status:** `IMPLEMENTED`; the private full-corpus run measured about 1.1–1.3 seconds
  per exact retrieval, about 18 seconds across 15 variants, and about 185.8 seconds in
  pre-Phase-3.1 export/inspection. Exact retrieval was not the bottleneck.
- **Dependencies:** monotonic local clock and the immutable run configuration.
- **Unit tests:** required timing keys, non-negative values, per-variant records, corpus
  video/row counts, and failure-path timing.
- **Acceptance:** Phase 3 adds no FAISS; the real latency evidence supports optimizing
  inspection/export before considering a different retrieval backend.

## Phase 3.1 — Fast contest export and inspection

### Module 20 — Bounded thumbnail inspection and fast contest mode

- **Input:** canonical ranked `KISResult` records, inspection mode `none`, `top-n`, or
  `all`, optional contact-sheet request, and the existing corpus manifest.
- **Output:** unchanged core JSONL/CSV plus bounded candidate JSON/Markdown, optional
  derived contact sheet, detailed export timings, and run-manifest query summaries.
- **Intended source:** `src/system_tai/inspection/candidate_report.py`,
  `src/system_tai/kis/contest_runner.py`, and `src/system_tai/kis/contest.py`.
- **Status:** `IMPLEMENTED`; local synthetic tests cover inspection call bounds, lazy
  index reuse, fast/default JSONL equality, warning policy, timing fields, and manifest
  summaries. A real Kaggle fast-mode retry remains pending.
- **Dependencies:** unchanged Module 18 results, keyframe directories only when
  inspection is enabled, and optional Pillow only for explicitly requested contact
  sheets.
- **Unit tests:** zero/Top-N/all resolver calls; one directory scan across candidates and
  queries; numeric filenames; ambiguous and missing thumbnails; flag conflicts; full
  candidates JSON in mode none; canonical JSONL provenance isolation; timing fields;
  deterministic output; and Phase 3/duplicate-frame regressions.
- **Acceptance:** default behavior remains `top-n`; mode `none` performs no keyframe
  scan; mode `all` is explicit; `--fast-contest-mode` changes only inspection behavior;
  each directory is indexed at most once per run; no decoded image is cached; isolated
  records are reused for combined inspection; core Top-100, validation, RRF, exact
  retrieval, and `frame_idx`-derived `frame_id` remain unchanged.

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

## Phase 4 — opt-in exact-frame refinement

- **Input:** completed Phase 3 `candidates.json`, `run_manifest.json`,
  `feature_manifest.json`, canonical `top100.jsonl`, and original raw videos.
- **Output:** core `refined_top100.jsonl`; internal CSV, bounded candidate/trace JSON,
  timings, validation, run manifest, summary, and optional derived contact sheet.
- **Source:** `src/system_tai/refinement/` and `src/system_tai/kis/refine.py`.
- **Status:** `BASELINE`; implemented with synthetic fake-decoder/fake-model evidence.
  Private Kaggle acceptance is pending.
- **Coordinate contract:** candidate, decoded, refined, and exported IDs are absolute
  original-video frame indexes. Decoder output carries the absolute ID; a local list
  position is never exported.
- **Algorithm:** inclusive bounded window; deterministic coarse sampling including the
  candidate; batched image encoding; independent cosine ranking for every explicit
  Phase 3 variant; Weighted RRF; deduplicated fine neighborhoods; repeat local fusion;
  select the fine winner.
- **Final ranking:** preserve Phase 3 order, apply explicit keep/skip/fail policy,
  deduplicate `(video_id, final_frame_id)`, and assign contiguous ranks up to 100. A
  local refinement score never reorders the original candidate list.
- **Acceptance:** no whole-video decode, no implicit model download, model loaded once,
  core JSONL leaks no diagnostics, validator passes, and traces contain no image bytes
  or embedding vectors.

Phase 4 is opt-in and never mutates Phase 3 artifacts. Synthetic mechanics do not prove
semantic quality or official performance. Official BTC export, UI, Q&A, TRAKE, FAISS,
and advanced model modules remain outside this slice.

## Phase 4.1 — discovery/bootstrap optimization

- **Input:** bounded BTC dataset root containing mapping, CLIP, keyframe, and optional
  raw-video families.
- **Output:** deterministic schema-v1 runtime manifest or schema-v2 portable manifest,
  dataset identity, one-pass statistics, and discovery timing/call counts.
- **Source:** `src/system_tai/data/corpus_discovery.py`,
  `src/system_tai/kis/build_manifest.py`, and `src/system_tai/kis/contest.py`.
- **Status:** `IMPLEMENTED` locally; full-corpus Kaggle timing acceptance is pending.
- **Traversal contract:** each direct artifact-family root is walked at most once;
  keyframe directories and supported-image counts are indexed in that same pass; no
  image content is read, decoded, or copied.
- **Strict gate:** mapping columns/counts, memory-mapped NPY shape/dimension, row-count
  agreement, keyframe image presence, and raw-video ambiguity.
- **Fast gate:** unique paths and every correctness check needed for row/feature/frame
  compatibility remain; it consumes the same one-pass family statistics for a trusted
  layout and is not a skip-validation mode.
- **Portable boundary:** paths are relative to `dataset_root`, use POSIX separators, and
  are rebased to the current resolved Kaggle root. Schema v1 remains backward compatible.
- **Cache boundary:** hit loads/rebases without full family traversal; missing performs a
  strict build; invalid fails unless explicit rebuild is selected.
- **Acceptance:** instrumented walker proves one traversal per root and no per-video
  keyframe rescan, including 873 synthetic directories; process smoke proves portable
  rebase, contest cache hit, valid checkpoint, and retained Phase 4 raw-video path.

The 579.27-second pre-optimization discovery measurement is operational evidence, not
retrieval or semantic performance. `/kaggle/working` is ephemeral, generated manifests
remain outside Git, and Phase 4 exact-frame refinement semantics are unchanged.

## Phase 4.2 — long-lived contest operational session

- **Input:** stdin JSON-line protocol, portable corpus manifest, and a unified 
  CLIP text/image model instance.
- **Output:** isolated per-request directory artifacts, stdout JSON-line responses, 
  and a session metrics manifest.
- **Source:** `src/system_tai/kis/session_engine.py`, `src/system_tai/kis/session_schema.py`,
  and `src/system_tai/kis/session.py`.
- **Status:** `IMPLEMENTED` locally with process-level JSON-line IPC test evidence.
  Private Kaggle acceptance is pending.
- **Protocol contract:** Stdin takes `health`, `query`, or `shutdown` JSON lines. Stdout
  emits single-line JSON responses parsed by `json.loads`. Malformed JSON or unknown
  types generate explicit structured error responses, followed by continuation. Progress bars, 
  tracebacks, and model download logs are absent from stdout.
- **Shared Model Strategy:** One `SharedOpenAIClipEncoder` wraps the single loaded `Model`
  and `preprocess`. The model is loaded exactly once upon bootstrap. Text encoding
  and batched image refinement encoding share this exact model weight instance. If separate
  backends are configured, it gracefully falls back to a lazy dual-load strategy.
  However, in the canonical case, model load count remains exactly 1 across all queries.
- **Artifact contract:** Each query targets a unique digest-based isolated folder 
  (e.g., `requests/req-q1-abcdef12/...`). Top-100 JSONL bytes strictly equal
  standalone Phase 3 contest baseline bytes. Refined JSONL bytes strictly equal standalone
  Phase 4 JSONL bytes.
- **Acceptance:** Exact 242-test pytest pass rate; subprocess smoke tests parsing JSON-line IPC;
  verifiable unchanged exact retrieval logic; single memory-loaded registry and model instances.
  Private Kaggle run pending.

## Phase 4.3A — Batch Query Text Encoding
- **Input**: Query requests with single or multiple text variants.
- **Output**: Single batched CLIP text forward pass per request; exact same embeddings reused for refinement.
- **Status**: \IMPLEMENTED\ locally; private Kaggle acceptance is pending.
- **Contract**: Reuses precomputed text embeddings in `ExactFrameRefiner` if provided. No semantic change to retrieval ranks, RRF, exact retrieval, or refined frame selection. Canonical `top100.jsonl` bytes are perfectly preserved. No speedup claimed until real Kaggle verification.

## Phase 4.3B — Verified Sparse Coarse Decode
- **Input**: Sparse requested absolute frame IDs from coarse sampling.
- **Output**: Verified sparse seek with single-frame physical read.
- **Status**: [IMPLEMENTED] locally; Phase 4.3B private Kaggle/T4 A/B:
  - canonical refined records identical: PASS
  - decoded frames: 1231 -> 448
  - physical decode reduction: ~63.61%
  - refinement latency: 13.104s -> 16.323s
  - sparse refinement approximately 24.56% slower
  - total query: 16.900s -> 19.804s
  - performance gate: FAIL
- **Contract**: Sequential fine stage unchanged. Explicit opt-in via `--coarse-decode-strategy sparse-verified`, while `sequential` remains the default. Fallback to sequential protects correctness on seek/position failures. Reduces physical `read()` calls during coarse stage without modifying canonical frame semantics or reducing search space.

**Decision**:
SEQUENTIAL remains contest/default strategy.
SPARSE_VERIFIED remains experimental opt-in only.

Phase 4.3B proved that fewer physical frame reads do not necessarily mean lower latency because repeated H.264/OpenCV seeks can be expensive.

## Phase 4.3C0 — Refinement Stage Timings
- **Status**: [IMPLEMENTED] locally; Phase 4.3C0: instrumentation for stage-level Kaggle profiling.
- **Decision**: No optimization claim. PRIVATE KAGGLE STAGE PROFILE PENDING.
