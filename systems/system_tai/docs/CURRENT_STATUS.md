# Current system_tai Status

## Status date

`2026-08-06`

## Summary

The workbook and Canva design have been reviewed. Phase 1 input auditing, Phase 1.5C
compatibility calibration, the Phase 2 exact NumPy KIS implementation, Phase 2.5
ground-truth evaluation, and an opt-in Phase 2.6 Weighted RRF pilot implementation are
present. Phase 3 now provides a contest-ready Textual KIS CLI MVP with bounded corpus
discovery, manifest reuse, batch execution, inspection artifacts, validator gating, and
latency instrumentation. Real compatibility evidence covers `L21_V001`, `L21_V002`, and `L22_V001`.
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
| Shared checkpoint boundary | PLANNED | Core KIS fields resolved as `query_id`, `rank`, `video_id`, `frame_id`; UTF-8 JSONL proposed | Optional envelope/version fields remain open. |
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
| Phase 3 corpus discovery/manifest | IMPLEMENTED | Bounded artifact-family discovery, deterministic ordering/fingerprint, row alignment, ambiguity checks, reuse CLI, and synthetic multi-video tests | Full private corpus has not been discovered locally. |
| Phase 3 contest Textual KIS CLI | BASELINE | Single/batch schema, registry/model reuse, exact per-variant retrieval, Weighted RRF, isolated/combined artifacts, validator gate, failure isolation, and synthetic end-to-end tests | Real full-corpus Kaggle acceptance and latency evidence are pending. |
| Phase 3 inspection/reproducibility | IMPLEMENTED | Candidate JSON/Markdown, optional derived contact sheet, run manifest, timings, and summary outputs | Pillow is optional; missing keyframes produce warnings rather than fabricated images. |
| Legacy interval fixture evaluator | PLANNED | Interface specified | Superseded for Phase 2.5 by the implemented exact-label evaluator; not an official evaluator. |
| Official BTC exporter | DEFERRED | Separate boundary recognized | Official format is unresolved. |
| Q&A and TRAKE | DEFERRED | Personal design reference | Outside the first slice. |
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

- Run the Phase 3 CLI over the complete attached corpus, retain generated files only in
  `/kaggle/working`, and measure full-corpus latency.
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
- Full-corpus Phase 3 latency evidence before deciding whether FAISS is necessary.

### Blocks final shared checkpoint compatibility

- Optional JSONL envelope/version fields.
- Shared validator/evaluator interface and ownership.

### Blocks official submission

- Accepted official BTC submission format.

## Next milestone

Run `system_tai.kis.contest` on the full private Kaggle corpus using an explicit batch
query file. Confirm valid Top-100 output and inspect timings before deciding whether to
add FAISS. Generated artifacts must remain outside Git. Do not call the three-video
pilot official, dataset-wide, or a completed Gate B benchmark.
