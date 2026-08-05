# Current system_tai Status

## Status date

`2026-08-05`

## Summary

The workbook and Canva design have been reviewed. Phase 1 input auditing, Phase 1.5C
compatibility calibration, and the Phase 2 exact NumPy KIS implementation are present.
Real compatibility evidence covers `L21_V001`, `L21_V002`, and `L22_V001`; real
text-query retrieval quality has not yet been evaluated.

No TRIAGE-EG source, tests, configs, documentation, or generated assets belong to
`system_tai`. All current implementation is isolated under `systems/system_tai`.

## Module status

| Area | Status | Evidence | Limitation |
|---|---|---|---|
| System design | PLANNED | Workbook and Canva design reviewed; only the minimal KIS slice is implemented. | Broader design is not runtime evidence. |
| Shared checkpoint boundary | PLANNED | Core KIS fields resolved as `query_id`, `rank`, `video_id`, `frame_id`; UTF-8 JSONL proposed | Optional envelope/version fields remain open. |
| Benchmark video catalog | IMPLEMENTED | `src/system_tai/data/video_catalog.py` and acceptance tests | Requires authoritative real catalog data. |
| Frame mapping | IMPLEMENTED | `frame_idx` is preserved exactly; three real mapping-policy gates pass | Dataset-wide behavior beyond three videos is pending. |
| BTC CLIP store | IMPLEMENTED | Temporary-NPY tests and real finite/shape/row audits on three videos | Exact model pipeline remains unverified. |
| Data-audit CLI | IMPLEMENTED | CLI tests and three real valid input/feature-row audits | Broader dataset audit is pending. |
| Kaggle input discovery | IMPLEMENTED | Nested tests and real artifact resolution for all three calibration videos | Broader dataset coverage is pending. |
| Mapping-rounding audit | IMPLEMENTED | Decimal-exact, binary-float-truncation, and Decimal-nearest diagnostics with regression tests | Numeric generation rule remains inferred. |
| Raw-frame calibration | IMPLEMENTED | Three decoder agreements, 42 explained decisive samples, 3 ambiguous, 0 contradictory decisive | Dataset-wide reproduction is pending. |
| BTC CLIP identification | IMPLEMENTED | 867 real rows across three videos identify official OpenAI ViT-B/32 and OpenCLIP QuickGELU/openai as equivalent compatible candidates | Three-video scope; exact preprocessing is not claimed BTC-official. |
| Kaggle notebook/config | IMPLEMENTED | Notebook structure and example YAML are locally validated | Real-data cells have not run locally. |
| Compatible query encoder | IMPLEMENTED | Optional official OpenAI CLIP public-API adapter with fake-module tests | Requires dependency and cached/explicitly downloadable weights at runtime. |
| Vector retrieval | BASELINE | Exact chunked NumPy cosine search and deterministic multi-video Top-K tests | No real text-query quality result yet; FAISS deferred. |
| Candidate construction | IMPLEMENTED | Immutable candidates copy `frame_idx` through the physical-row mapping | Real Kaggle smoke is pending. |
| Grouping and Top-100 ranking | BASELINE | Exact rank is canonical; optional suppression is tested and disabled by default | Suppression policy is not benchmarked. |
| Checkpoint exporter | IMPLEMENTED | Core-only UTF-8 JSONL and explicit internal mode tested | Accepted schema remains proposed. |
| Validator | IMPLEMENTED | Structured syntax/type/rank/duplicate/registry checks tested | This local validator is not yet the accepted shared validator. |
| Phase 2 CLI/notebook | IMPLEMENTED | CLI composition and clean five-query notebook path | Notebook cells have not been executed locally or on Kaggle. |
| Fixture evaluator | PLANNED | Interface specified | Not an official evaluator. |
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

- Run the five-query Phase 2 smoke path and manually inspect Top-10 results.
- Build the small positive/distractor sanity benchmark and report Video Recall@K.
- Extend compatibility checks beyond the current three videos before dataset-wide claims.

## Unverified

- BTC-official image preprocessing and dataset-wide CLIP compatibility.
- Text-query retrieval quality on real queries.
- Binary-float truncation and nearest timestamp extraction remain inferred generation
  behavior even though the current three-video evidence is consistent with them.

## Still open

- Official BTC submission artifact.
- Optional JSONL envelope and version fields.

## Validation status

Gate A and Gate B have not run. Local Phase 2 tests use deterministic synthetic fixtures
and do not measure retrieval quality. Three-video encoder compatibility is not a
retrieval gate or official performance evidence.

Codex local execution cannot run the BTC smoke path because the private dataset and
model weights are attached only inside Kaggle. The query encoder, retrieval, JSONL
export, and local validator have test evidence; no real-query benchmark exists yet.

## Remaining blockers and decisions

### Blocks evidence-backed semantic KIS readiness

- Five-query manual Kaggle smoke execution.
- Positive/distractor sanity benchmark and expected intervals.

### Blocks final shared checkpoint compatibility

- Optional JSONL envelope/version fields.
- Shared validator/evaluator interface and ownership.

### Blocks official submission

- Accepted official BTC submission format.

## Next milestone

Run `notebooks/phase_2_kis_smoke.ipynb` against the three audited videos, inspect Top-10
for five Vietnamese/English queries, export/validate Top-100, and record observations
without calling them official performance.
