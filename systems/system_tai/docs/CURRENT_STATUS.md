# Current system_tai Status

## Status date

`2026-08-04`

## Summary

The workbook and Canva design have been reviewed. The Phase 1 input-audit foundation
and Phase 1.5B mapping diagnostics are implemented locally; semantic retrieval has
not started. Real Kaggle evidence is currently limited to `L21_V001`.

No TRIAGE-EG source, tests, configs, documentation, or generated assets belong to
`system_tai`. All current implementation is isolated under `systems/system_tai`.

## Module status

| Area | Status | Evidence | Limitation |
|---|---|---|---|
| System design | PLANNED | Workbook and Canva design reviewed; implementation has not started. | Design is not runtime evidence. |
| Shared checkpoint boundary | PLANNED | Core KIS fields resolved as `query_id`, `rank`, `video_id`, `frame_id`; UTF-8 JSONL proposed | Optional envelope/version fields remain open. |
| Benchmark video catalog | IMPLEMENTED | `src/system_tai/data/video_catalog.py` and acceptance tests | Requires authoritative real catalog data. |
| Frame mapping | IMPLEMENTED | `frame_idx` is preserved exactly as `actual_frame_id`; temporary-CSV tests pass locally | Zero-based bounds are verified for `L21_V001`; dataset-wide behavior is pending. |
| BTC CLIP store | IMPLEMENTED | `src/system_tai/features/btc_clip_store.py`, temporary-NPY tests, and real `L21_V001` shape/finite/norm audit | Exact model pipeline remains unverified. |
| Data-audit CLI | IMPLEMENTED | `scripts/audit_kis_inputs.py`, CLI tests, and real 307-row mapping/feature agreement for `L21_V001` | Multi-video reproduction is pending. |
| Kaggle input discovery | IMPLEMENTED | Nested discovery tests and real `L21_V001` artifact resolution | `L21_V002` and `L22_V001` runs are pending. |
| Mapping-rounding audit | IMPLEMENTED | Decimal-exact, binary-float-truncation, and Decimal-nearest diagnostics with regression tests | Numeric generation rule remains inferred. |
| Raw-frame calibration | IMPLEMENTED | Separate mapping, rounding, and visual statuses; real 15-sample `L21_V001` result | Multi-video reproduction is pending. |
| BTC CLIP identification | IMPLEMENTED | Optional adapters, metrics, and multi-video gate have synthetic tests | Pipeline identity and text compatibility remain `UNVERIFIED`. |
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

See `docs/KAGGLE_PHASE_1_5_REPORT.md` for the evidence boundary and exact commands.

## Pending real Kaggle calibration

- Reproduce discovery, input audit, mapping-rounding audit, and decoded-frame
  calibration on at least `L21_V002` and `L22_V001`.
- Establish whether binary-float truncation is the mapping-generation rule and nearest
  timestamp alignment is the JPEG extraction rule. Both are inferred, not verified
  implementations; do not generalize from `L21_V001`.

## Unverified

- Exact BTC-compatible CLIP implementation and image preprocessing.
- Text-query encoder compatibility.

## Still open

- Official BTC submission artifact.
- Optional JSONL envelope and version fields.

## Validation status

Gate A and Gate B have not run. Local Phase 1/1.5B unit and CLI tests use temporary or
synthetic fixtures only. Real input/calibration evidence exists for `L21_V001`, but it
is not a retrieval gate or official performance evidence.

Codex local execution cannot reproduce real BTC calibration because the private dataset
is attached only inside Kaggle. No query encoder, retrieval, JSONL export, shared
validation, or fixture benchmark result currently exists for `system_tai`.

## Remaining blockers and decisions

### Blocks a semantic KIS run

- Exact BTC-compatible text encoder and preprocessing.
- Multi-video decoded-frame calibration and dataset-wide mapping behavior.

### Blocks final shared checkpoint compatibility

- Optional JSONL envelope/version fields.
- Shared validator/evaluator interface and ownership.

### Blocks official submission

- Accepted official BTC submission format.

## Next milestone

Run Phase 1.5B in Kaggle for `L21_V001`, `L21_V002`, and `L22_V001`: audit mapping/NPY
alignment, Decimal timestamp rounding, and decoded visual agreement. Do not run CLIP
pipeline identification unless all three input/feature and mapping gates pass. Do not
implement text retrieval until compatibility is reported from real multi-video evidence.
