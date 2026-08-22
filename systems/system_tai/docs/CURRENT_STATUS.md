# Current system_tai Status

## Status date

`2026-08-09`

## Summary

PRELIMINARY P0-A (official schemas, strict int, GT contracts, evaluation) is COMPLETE / FROZEN.
PRELIMINARY P0-B1 (evidence-grounded closed-set Q&A baseline core) is COMPLETE / FROZEN.
PRELIMINARY P0-B2 / P0-B2.1 (shared-runtime Q&A vertical slice & compatibility hotfix) is COMPLETE / FROZEN.
PRELIMINARY P0-C1 (deterministic same-video temporal-chain TRAKE core) is COMPLETE / FROZEN.
PRELIMINARY P0-C2 (shared-runtime TRAKE vertical slice) is COMPLETE / FROZEN (Kaggle/T4 technical acceptance PASS, commit `6849324769820dca27ae9b338b2bb5cfe178c436`).
PRELIMINARY P0-OPT1 (request-scoped raw-frame CLIP image embedding reuse) is COMPLETE / FROZEN (commit `977ccf55b3cdb782052f41cec1edd65259395473`):
- Request-scoped image embedding cache keyed by `(video_id, absolute_frame_id)`.
- Eliminates duplicate model image inference across coarse/fine stages and multiple event refinement calls within a TRAKE request.
- Preserves 100% exact output semantics and independent per-event text scoring.
- Kaggle/T4 STRONG PASS: full runtime reduced from ~41.629s to median ~23.715s; refinement from ~34.966s to median ~16.893s; physical CLIP image rows from 1057 to 382.
- Refinement reduction: 51.69%. Physical image work reduction: 63.86%.

PRELIMINARY P0-OPT2 (shared-scan exact multi-vector retrieval) is COMPLETE / FROZEN (commit `64841a63471ade21eebf7f22299420c5369a637c`):
- Shared-scan `ExactNumpyRetriever.search_vectors` scans each video feature chunk once for all query vectors in a single request.
- Computes exact per-query GEMV `(chunk @ query_unit) / norms` inside one shared corpus scan, reducing corpus memory reads by `num_queries`.
- Preserves 100% exact numerical and deterministic tie-break ranking equivalence with `search_vector`.
- Integrated into `TRAKERuntimePipeline.process_trake_query`.
- Kaggle/T4 PASS: same-session retrieval reduced from 7.024408s to median 4.834751s (31.17%); full wall time from 24.835049s to median 22.127217s.
- FULL Top-20 exact semantic equivalence PASS.

PRELIMINARY P0-D (Unified Top-100 Internal Checkpoint Boundary) is COMPLETE / FROZEN at commit `dfebb59deeb7834c0662825d03435e615c464823` (remote audit PASS):
- Wraps the frozen P0-A `KISPrediction`, `QAPrediction`, and `TRAKEPrediction` schemas directly.
- Provides strict task-specific UTF-8 JSONL parsing, deterministic serialization, per-query max-100 validation, optional TRAKE event-count validation, and exact semantic roundtrip.
- Preserves physical query/prediction order, non-contiguous positive ranks, Q&A Unicode text, and ordered TRAKE frame tuples without sorting or renumbering.
- This is the canonical preliminary internal checkpoint boundary, not a confirmed official BTC upload format.
- No Kaggle or semantic-accuracy claim is made for P0-D.

PRELIMINARY P0-E (Final Runtime-to-Canonical-Top100 Integration) is COMPLETE / FROZEN at code commit `efb4b3dd54cb632809483796b4b09713f750e884` (remote audit PASS; final Kaggle/T4 E2E PASS):
- Converts final KIS runtime results and directly wraps Q&A/TRAKE P0-A predictions as P0-D `RankedTop100Query` objects.
- Audits the existing `top100.jsonl` or `refined_top100.jsonl`, `qa_predictions.jsonl`, and `trake_predictions.jsonl`; it creates no new prediction artifact.
- Requires exact runtime-memory to artifact equality after strict P0-D loading, including physical prediction order, non-contiguous ranks, Q&A Unicode/whitespace, and TRAKE event-frame order.
- Handles zero-prediction queries explicitly: an empty or whitespace-only existing artifact is accepted as `EMPTY_QUERY_UNREPRESENTABLE` without weakening P0-D record-only JSONL.
- Canonical predictions are directly consumable by the existing P0-A evaluator without another translation layer.
- Final KIS real E2E: `SUCCESS`, wall `18.124886s`, 20 results, 3 refined, canonical `refined_top100.jsonl`, P0-E roundtrip `EXACT`, audit wall approximately `0.000305s`; the existing KIS artifact contract is unchanged.
- Final Q&A real E2E: `SUCCESS`, wall `17.799365s`, question type `COLOR`, 3 predictions, P0-E roundtrip `EXACT`, audit wall approximately `0.000215s`; the exact four artifacts remain `qa_predictions.jsonl`, `qa_evidence.json`, `qa_request_manifest.json`, and `qa_timings.json`.
- Final TRAKE real E2E medians: retrieval `4.731498s`, refinement `16.360532s`, full wall `21.120408s`, and P0-E audit approximately `0.000422s`. Both measured runs retained 382 physical image rows/request, 8 image encode calls/request, 1 `search_vectors` call/request, and 0 `search_vector` calls/request. FULL20 RUN1 equals RUN2 and the historical golden Top3 is preserved. Against the accepted OPT2 benchmark, retrieval changed by `-2.14%`, refinement by `-3.15%`, and full wall by `-4.55%`; no P0-E performance regression was observed.
- The accepted corpus had 873 videos, 177321 CLIP feature rows, 873 raw videos, and manifest fingerprint `b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4`; execution used official OpenAI CLIP ViT-B/32 on Tesla T4/CUDA with shared encoder identity PASS.
- Cold bootstrap remains known infrastructure debt: `619.136776s` total with `588.613069s` in family indexing (`CACHE_BUILT`). It is not a P0-E regression and is not solved here.
- Known KIS telemetry accounting debt includes per-variant `search_vector` work in text-encode timing while `retrieval_seconds` mostly reflects later fusion; this does not affect ranking correctness, and `0.0011s` must not be presented as true vector-retrieval latency.
- This is an internal fail-closed integration invariant, not a confirmed official BTC upload format or semantic-accuracy result. Competition semantic accuracy remains NOT ESTABLISHED.

**PRELIMINARY TECHNICAL BASELINE is COMPLETE / FROZEN at code commit `efb4b3dd54cb632809483796b4b09713f750e884`.**

The workbook and Canva design have been reviewed. Phase 1 input auditing, Phase 1.5C
compatibility calibration, the Phase 2 exact NumPy KIS implementation, Phase 2.5
ground-truth evaluation, and an opt-in Phase 2.6 Weighted RRF pilot implementation are
present. Phase 3 now provides a contest-ready Textual KIS CLI MVP with bounded corpus
discovery, manifest reuse, batch execution, inspection artifacts, validator gating, and
latency instrumentation. Real compatibility evidence covers `L21_V001`, `L21_V002`, and `L22_V001`.
The Phase 3 full-corpus technical acceptance also passed over 873 videos, 177,321
feature rows, five queries, 15 explicit variants, and 500 valid output records.
The first real semantic smoke passed the technical pipeline but did not pass semantic
quality. A small human-verified pilot now contains three intents and nine comparable
verified query variants; its scope and retrieval-selection bias prevent official or
dataset-wide claims.

No TRIAGE-EG source, tests, configs, documentation, or generated assets belong to
`system_tai`. All current implementation is isolated under `systems/system_tai`.

## Module status

| Area | Status | Evidence | Limitation |
|---|---|---|---|
| System design | PLANNED | Workbook and Canva design reviewed; only the minimal KIS slice is implemented. | Broader design is not runtime evidence. |
| Shared checkpoint boundary | IMPLEMENTED / FROZEN | P0-D strict unified Top100 internal boundary plus P0-E exact runtime/artifact integration | Official BTC upload artifact format is not confirmed. |
| Benchmark video catalog | IMPLEMENTED | `src/system_tai/data/video_catalog.py` and acceptance tests | Requires authoritative real catalog data. |
| Frame mapping | IMPLEMENTED | `frame_idx` is preserved exactly; three real mapping-policy gates pass | Dataset-wide behavior beyond three videos is pending. |
| BTC CLIP store | IMPLEMENTED | Temporary-NPY tests and real finite/shape/row audits on three videos | Compatibility evidence remains limited to three videos. |
| Data-audit CLI | IMPLEMENTED | CLI tests and three real valid input/feature-row audits | Broader dataset audit is pending. |
| Kaggle input discovery | IMPLEMENTED | Nested tests and real artifact resolution for all three calibration videos | Broader dataset coverage is pending. |
| Mapping-rounding audit | IMPLEMENTED | Decimal-exact, binary-float-truncation, and Decimal-nearest diagnostics with regression tests | Numeric generation rule remains inferred. |
| Raw-frame calibration | IMPLEMENTED | Three decoder agreements, 42 explained decisive samples, 3 ambiguous, 0 contradictory decisive | Dataset-wide reproduction is pending. |
| BTC CLIP identification | IMPLEMENTED | 867 real rows across three videos identify official OpenAI ViT-B/32 and OpenCLIP QuickGELU/openai as equivalent compatible candidates | Three-video scope; exact preprocessing is not claimed BTC-official. |
| Kaggle notebook/config | IMPLEMENTED | Bounded discovery and a real semantic smoke completed in Kaggle | Real-data cells cannot run locally. |
| Compatible query encoder | IMPLEMENTED | Optional official OpenAI CLIP public-API adapter with fake-module tests | Requires dependency and cached/explicitly downloadable weights at runtime. |
| Vector retrieval | BASELINE | Exact chunked NumPy cosine search passed the real technical smoke | Direct Vietnamese quality failed; translated-English quality is mixed; FAISS deferred. |
| Candidate construction | IMPLEMENTED | Immutable candidates copy `frame_idx` through the physical-row mapping | Corpus coverage remains limited to three videos. |
| Grouping and Top-100 ranking | BASELINE | Exact rank is canonical; optional suppression is tested and disabled by default | Suppression policy is not benchmarked. |
| Checkpoint exporter | IMPLEMENTED | Core-only UTF-8 JSONL and explicit internal mode tested | Accepted schema remains proposed. |
| Validator | IMPLEMENTED | Structured syntax/type/rank/duplicate/registry checks tested | This local validator is not yet the accepted shared validator. |
| Phase 2 CLI/notebook | IMPLEMENTED | Real smoke wrote 500 valid JSONL records with zero validator errors | Semantic quality did not pass. |
| Phase 2.5 benchmark schema/validator | IMPLEMENTED | Typed YAML/JSON schema, registry-aware validation, structured errors, draft exclusion tests, and a human-verified three-group pilot | Pilot labels were selected after retrieval inspection and are not an official benchmark. |
| Phase 2.5 exact evaluator | IMPLEMENTED | Binary query Recall@K, ground-truth coverage, hit count, first rank, reciprocal rank/MRR, relevant-video coverage, aggregates, paired VI/EN tests, and measured pilot ranks | Only three videos and three semantic intents are represented. |
| Phase 2.5 reports/annotation notebook | IMPLEMENTED | Deterministic JSON/CSV/Markdown serialization, validation-only CLI, bounded annotation helper, and clean Kaggle notebook | Generated reports remain external and Kaggle working storage is ephemeral. |
| Phase 2.6 explicit variants and Weighted RRF | BASELINE | Immutable validated variants, canonical per-variant exact retrieval, deterministic weighted rank fusion, provenance isolation, evaluator, CLI, and synthetic tests | Opt-in only; real RRF metrics have not yet been executed in Kaggle. |
| Phase 3 corpus discovery/manifest | IMPLEMENTED | Bounded discovery passed a real 873-video, 177,321-row corpus run with a reusable fingerprinted manifest | Private corpus remains unavailable locally. |
| Corpus/index readiness gate | IMPLEMENTED (LOCAL) | Fail-closed manifest/features/full levels; portable rebase, finite feature rows, original-frame bounds, structured reports, and deterministic CLI tests | Full 873-video readiness report is pending Kaggle; this is not semantic-quality evidence. |
| Phase 3 contest Textual KIS CLI | BASELINE | The full-corpus technical run completed 5/5 queries, 15 variants, 500 records, and validator PASS with no duplicate pairs or rank errors | This is not official performance or semantic-quality proof. |
| Phase 3.1 inspection/reproducibility | IMPLEMENTED | Explicit none/top-n/all modes, lazy per-video Path-only thumbnail indexes, isolated/combined record reuse, fast mode, detailed export timings, and run summary IDs | Phase 3.1 needs a real Kaggle retry; Pillow remains optional for contact sheets. |
| Legacy interval fixture evaluator | PLANNED | Interface specified | Superseded for Phase 2.5 by the implemented exact-label evaluator; not an official evaluator. |
| Official BTC exporter | DEFERRED | Separate boundary recognized | Official format is unresolved. |
| Q&A (P0-B1/B2) | BASELINE / FROZEN | `system_tai.qa` closed-set baseline, session runtime, and final Kaggle/T4 E2E PASS | Semantic QA accuracy on competition ground truth is not established. |
| TRAKE (P0-C1/C2) | BASELINE / FROZEN | `system_tai.trake` deterministic temporal core, session runtime, and final Kaggle/T4 E2E PASS | Semantic TRAKE event-grounding accuracy on competition ground truth is not established. |
| Backend and production frontend | DEFERRED | Personal design and untracked prototype | Outside the first slice. |
| Agent/GNN/Event Graph/VLM/OCR/ASR | DEFERRED | Optional design ideas | Explicitly excluded. |

## Resolved

- Work lives in the shared repository under `systems/system_tai` on a personal branch.
- BTC map-keyframe `frame_idx` is preserved exactly as `actual_frame_id`.
- Shared KIS checkpoint core fields are `query_id`, `rank`, `video_id`, and `frame_id`.

## Real Kaggle verified

- `L21_V001` artifacts were discovered in the nested Kaggle input layout.
- Raw video reports FPS `30.0`, 37,849 frames, and bounds `[0, 37848]`.
- The 307-row mapping and `[307, 512]` float16 CLIP array agree row-for-row; the
  feature array is finite and appears L2-normalized.
- Decimal floor matches 303 of 307 mapping rows. For the other four rows,
  `frame_idx - Decimal floor = -1`; they are valid mappings and binary-float truncation
  is a candidate explanation.
- The full Decimal-nearest offset distribution is `0` for 233 rows and `+1` for 74.
- Random and sequential decoding agree for all 15 sampled rows.
- Only 15 rows have been visually decoded: ten align best with offset `0` and five with
  `+1`. All 15 match the Decimal-nearest prediction. This is not a global `+1` error.
- Official/shared `actual_frame_id` remains the CSV `frame_idx` exactly.
- `L21_V001`, `L21_V002`, and `L22_V001` all pass input, mapping-policy, and decoder
  gates; all have binary-float-truncation ratio `1.0`.
- Across 45 visual samples, 42 are decisive and match the Decimal-nearest prediction.
  Three are ambiguous because margins `0.000007`, `0.000014`, and `0.000017` are below
  `superiority_margin = 0.0001`. There is no contradictory decisive sample.
- Full-corpus compatibility covers 867 rows: 307, 262, and 298 for the three videos.
- Official OpenAI CLIP `ViT-B/32` and OpenCLIP `ViT-B-32-quickgelu/openai` both report
  mean cosine approximately `0.999162`, minimum p05 `0.997200`, Top-1 `1.0`, mean
  self-match rank `1.0`, dimension 512, and numerical equivalence within about `1e-10`.

See `docs/KAGGLE_PHASE_1_5_REPORT.md` for the evidence boundary and exact commands.

## Pending real Kaggle work

- Retry the accepted full-corpus Phase 3 run with `--fast-contest-mode`; confirm identical
  canonical JSONL and measure the reduced export/inspection time.
- Run the measured Weighted RRF milestone against the checked-in three-group pilot and
  save generated reports only under Kaggle working output storage.
- Extend compatibility checks beyond the current three videos before dataset-wide claims.
- Investigate language handling and still-frame action/state ambiguity without changing
  the exact baseline to fit anecdotal results.

## Unverified

- BTC-official image preprocessing and dataset-wide CLIP compatibility.
- Robust multilingual text-query retrieval quality: direct Vietnamese failed and the
  first English-translated diagnostic was mixed.
- Binary-float truncation and nearest timestamp extraction remain inferred generation
  behavior even though the current three-video evidence is consistent with them.

## Still open

- Official BTC submission artifact.
- Optional JSONL envelope and version fields.

## Validation status

The technical integration smoke passed: exact retrieval completed, 500 canonical JSONL
records were written, and the validator passed with zero errors. Direct Vietnamese
semantic retrieval failed. English-translated retrieval improved but remained mixed:
`vi_03_en` was strong, while `vi_01_en` and `vi_02_en` were partially relevant. Gate B
and Phase 2 semantic quality have not passed.

Codex local execution cannot reproduce the BTC smoke because the private dataset and
model weights are attached only inside Kaggle. Corpus coverage and ambiguity between
action/state language and a single still frame remain important limitations.

Phase 2.5 evaluation infrastructure is implemented and synthetically tested. It does
not change the Phase 2 semantic result. The original benchmark template still contains
only drafts and returns `no_verified_queries`. The separate pilot fixture contains
three semantic groups, nine human-verified variants, and six drafts over the three
audited videos. Observed canonical unsuppressed ranks are:

- `city_pedestrians`: direct Vietnamese missed Top-100; English translation ranked 1;
  English expansion ranked 3.
- `conference_attendees`: direct Vietnamese missed Top-100; English translation and
  expansion both ranked 14.
- `landslide_warning_sign`: direct Vietnamese ranked 1; English translation ranked 4;
  English expansion ranked 6.

These are pilot observations, not official performance. Positives were chosen after
retrieval inspection, so retrieval-selection bias applies. The multilingual policy is
still open and Gate B is not fully passed. Phase 2.6 adds a separately invoked Weighted
RRF measurement path; it does not change the single-query baseline.

## Remaining blockers and decisions

### Blocks evidence-backed semantic KIS readiness

- A broader positive/distractor benchmark independent of retrieval-selection bias.
- Multilingual query policy supported by measured evidence.
- Broader corpus coverage and handling of still-frame action/state ambiguity.
- Independent semantic-quality evidence beyond the technically accepted full-corpus run.

### Blocks final shared checkpoint compatibility

- Optional JSONL envelope/version fields.
- Shared validator/evaluator interface and ownership.

### Blocks official submission

- Accepted official BTC submission format.

## Next milestone

Reuse the accepted 873-video manifest and rerun `system_tai.kis.contest` with
`--fast-contest-mode`. Confirm byte-identical canonical Top-100 content, validator PASS,
and reduced export timing. Exact NumPy retrieval is not the measured bottleneck, so do
not add FAISS in this milestone. Generated artifacts must remain outside Git. Do not
call the technical run official performance, semantic-quality proof, or a completed
Gate B benchmark.

## Phase 4 status

Raw-video exact-frame refinement is now an opt-in `BASELINE`. It consumes retained
Phase 3 artifacts, probes and decodes bounded original-video neighborhoods, uses one
canonical CLIP text/image model for the run, and performs separate per-variant cosine
ranking followed by local Weighted RRF. It replaces a frame while preserving Phase 3
rank order. Phase 3 canonical JSONL and `--fast-contest-mode` are unchanged.

Refined `frame_id` is the decoder-returned absolute original-video frame index. Local
sample index, keyframe order, CLIP row, and filename never cross the shared boundary.
Missing-video and decode-failure policies are explicit. Synthetic tests prove mechanics
only; private Kaggle acceptance, semantic quality, and official competition performance
remain pending. Generated refinement outputs must remain outside Git.

## Phase 4.1 status

One-pass corpus discovery and portable manifest reuse are `IMPLEMENTED` locally. The
pre-optimization fresh private-corpus measurement was 579.27 seconds, caused by separate
recursive mapping/CLIP/raw-video/keyframe scans plus one extra recursive count for every
keyframe directory. Each bounded family root is now traversed at most once, and keyframe
image counts are collected during its family traversal.

Absolute schema-v1 manifests remain loadable. Portable schema v2 uses dataset-root-
relative POSIX paths and shallow relative-artifact metadata identity, then rebases and
validates artifacts against the current `--input-root`. Cache hit bypasses full family
discovery; invalid cache replacement requires an explicit rebuild flag. `/kaggle/working`
is not persistent, so portable manifests must be retained as notebook output, a private
lightweight Kaggle Dataset, or a separately uploaded input artifact.

The real 873-video latency improvement is pending Kaggle rerun. Phase 4 refinement,
Exact NumPy retrieval, Weighted RRF, and `frame_id = frame_idx` semantics are unchanged.
No semantic-quality or official-performance claim follows from this optimization.

## Phase 4.2 status

Long-Lived Contest Operational Session Runtime is `IMPLEMENTED` locally. A fail-closed,
single-process JSON-line IPC session processes queries via stdin/stdout without
reloading the registry or model between requests. It handles retrieval and
opt-in refinement within a single continuous runtime.

The canonical Shared OpenAI CLIP model handles both text queries and image frames via the
`SharedOpenAIClipEncoder`. The model instance is loaded exactly once per session. 
A lazy fallback dual-load strategy is used if separate backends are needed, but the primary
design is one `SharedOpenAIClipEncoder` wrapping the single loaded `Model` and `preprocess`.
Text encodings and batched image refinement encodings share the same model weights.

Requests are isolated into unique digest-based directories. A process-level protocol smoke
test verifies exact retrieval, refinement, continue-on-error for malformed JSON, unknown request
types, and clean decoder/model resource shutdown. Byte equality against contest retrieval JSONL
and standalone refinement core JSONL is preserved. Private Kaggle acceptance is pending.

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

## KIS semantic video-first optimization — local experimental status

- **Status:** implemented locally behind `KISVideoFirstConfig.enabled`; private Kaggle
  and verified ground-truth evaluation are pending.
- **Input policy:** Vietnamese source only. The pinned VinAI v2 provider translates the
  full query and deterministic semantic clauses in one batch. Caller-supplied English is
  rejected in this mode and there is no Marian production path under `system_tai`.
- **Retrieval:** canonical OpenAI CLIP ViT-B/32 plus exact NumPy. One full-corpus store
  traversal scores all variants; rank-only weighted RRF selects videos; one restricted
  traversal per selected video scores all variants and produces deterministic frame-level
  RRF candidates.
- **Evidence weights:** action/scene clauses remain primary; counts, colors, clothing,
  and accessories are supporting evidence by default and do not dominate the action.
- **Compatibility:** feature disabled preserves the existing KIS path. Existing Q3
  keyframe conditioning and raw-video exact-frame refinement are accepted downstream;
  core JSONL and `frame_id = mapping frame_idx` remain unchanged.
- **Current claim boundary:** synthetic mechanics and regression safety only. The observed
  `query-p1-1-kis` gallery motivated the change but is not ground truth and does not prove
  Recall@K improvement.

## Phase 4.3C0 — Refinement Stage Timings
- **Status**: [IMPLEMENTED] locally; Phase 4.3C0: instrumentation for stage-level Kaggle profiling.
- **Decision**: No optimization claim. PRIVATE KAGGLE STAGE PROFILE PENDING.

## Preliminary P0-A — Official Task Contracts
- **Status**: [IMPLEMENTED] locally. Official published task structure and scoring implementation.
- **Decision**:
  - Phase 4.3C1 batch-size 32/64/128 sweep produced no material improvement. `image_batch_size=32` remains default. KIS performance optimization paused until preliminary P0 is complete.
  - Implemented strong typed schema predictions and ground truths for Textual KIS, Q&A, TRAKE.
  - Implemented `R-Score`, `R@1/5/20/50/100`, and Final Score calculation exactly following published preliminary rules.
  - Q&A semantic answer matching uses deterministic configured aliases as an explicit local approximation.
