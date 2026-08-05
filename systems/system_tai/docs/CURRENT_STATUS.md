# Current system_tai Status

## Status date

`2026-08-04`

## Summary

The workbook and Canva design have been reviewed. The Phase 1 input-audit foundation
and Phase 1.5C compatibility diagnostics are implemented locally; semantic retrieval
has not started. Real Kaggle gate evidence covers `L21_V001`, `L21_V002`, and
`L22_V001`.

No TRIAGE-EG source, tests, configs, documentation, or generated assets belong to
`system_tai`. All current implementation is isolated under `systems/system_tai`.

## Module status

| Area | Status | Evidence | Limitation |
|---|---|---|---|
| System design | PLANNED | Workbook and Canva design reviewed; implementation has not started. | Design is not runtime evidence. |
| Shared checkpoint boundary | PLANNED | Core KIS fields resolved as `query_id`, `rank`, `video_id`, `frame_id`; UTF-8 JSONL proposed | Optional envelope/version fields remain open. |
| Benchmark video catalog | IMPLEMENTED | `src/system_tai/data/video_catalog.py` and acceptance tests | Requires authoritative real catalog data. |
| Frame mapping | IMPLEMENTED | `frame_idx` is preserved exactly; three real mapping-policy gates pass | Dataset-wide behavior beyond three videos is pending. |
| BTC CLIP store | IMPLEMENTED | Temporary-NPY tests and real finite/shape/row audits on three videos | Exact model pipeline remains unverified. |
| Data-audit CLI | IMPLEMENTED | CLI tests and three real valid input/feature-row audits | Broader dataset audit is pending. |
| Kaggle input discovery | IMPLEMENTED | Nested tests and real artifact resolution for all three calibration videos | Broader dataset coverage is pending. |
| Mapping-rounding audit | IMPLEMENTED | Decimal-exact, binary-float-truncation, and Decimal-nearest diagnostics with regression tests | Numeric generation rule remains inferred. |
| Raw-frame calibration | IMPLEMENTED | Three decoder agreements, 42 explained decisive samples, 3 ambiguous, 0 contradictory decisive | Dataset-wide reproduction is pending. |
| BTC CLIP identification | IMPLEMENTED | Public OpenAI API, Transformers 5 output handling, two OpenCLIP variants, dynamic summary | Corrected adapters require a CLIP-only Kaggle rerun; identity remains `UNVERIFIED`. |
| Kaggle notebook/config | IMPLEMENTED | Notebook structure and example YAML are locally validated | Real-data cells have not run locally. |
| Compatible query encoder | UNKNOWN | ViT-B/32-style compatibility is expected | Exact BTC encoder is unknown. |
| Vector retrieval | PLANNED | Interface specified | No search implementation. |
| Candidate construction | PLANNED | Interface specified | No mapping implementation. |
| Grouping and Top-100 ranking | PLANNED | Interface specified | Ranking policy is unbenchmarked. |
| Checkpoint exporter | PLANNED | Adapter boundary specified | Accepted schema is unresolved. |
| Validator | PLANNED | Interface specified | Shared ownership is unresolved. |
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
- Initial OpenCLIP `ViT-B-32` / `openai` measurement: mean cosine `0.957185`, minimum
  p05 `0.925354`, self-match Top-1 `1.0`, and mean rank `1.0`. This is not sufficient
  for identification.

See `docs/KAGGLE_PHASE_1_5_REPORT.md` for the evidence boundary and exact commands.

## Pending real Kaggle work

- Rerun CLIP-only identification with the corrected public OpenAI API adapter,
  Transformers 5 output extraction, and both OpenCLIP model variants.
- Extend calibration beyond the current three videos before making dataset-wide claims.

## Unverified

- Exact BTC-compatible CLIP implementation and image preprocessing.
- Text-query encoder compatibility.
- Binary-float truncation and nearest timestamp extraction remain inferred generation
  behavior even though the current three-video evidence is consistent with them.

## Still open

- Official BTC submission artifact.
- Optional JSONL envelope and version fields.

## Validation status

Gate A and Gate B have not run. Local Phase 1/1.5C unit and CLI tests use temporary or
synthetic fixtures only. Three-video input/calibration evidence is not a retrieval gate
or official performance evidence.

Codex local execution cannot reproduce real BTC calibration because the private dataset
is attached only inside Kaggle. No query encoder, retrieval, JSONL export, shared
validation, or fixture benchmark result currently exists for `system_tai`.

## Remaining blockers and decisions

### Blocks a semantic KIS run

- Exact BTC-compatible text encoder and preprocessing.
- Corrected multi-candidate CLIP identification rerun.

### Blocks final shared checkpoint compatibility

- Optional JSONL envelope/version fields.
- Shared validator/evaluator interface and ownership.

### Blocks official submission

- Accepted official BTC submission format.

## Next milestone

Run only Phase 1.5C CLIP identification in Kaggle with official OpenAI CLIP, OpenCLIP
`ViT-B-32`, OpenCLIP `ViT-B-32-quickgelu`, and Hugging Face CLIP as separate candidates.
Do not implement text retrieval until a compatible pipeline is identified from real
multi-video evidence.
