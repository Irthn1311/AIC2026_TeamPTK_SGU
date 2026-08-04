# Current system_tai Status

## Status date

`2026-08-04`

## Summary

The workbook and Canva design have been reviewed. The Phase 1 input-audit foundation
is implemented locally; semantic retrieval has not started.

No TRIAGE-EG source, tests, configs, documentation, or generated assets belong to
`system_tai`. All current implementation is isolated under `systems/system_tai`.

## Module status

| Area | Status | Evidence | Limitation |
|---|---|---|---|
| System design | PLANNED | Workbook and Canva design reviewed; implementation has not started. | Design is not runtime evidence. |
| Shared checkpoint boundary | PLANNED | Core KIS fields resolved as `query_id`, `rank`, `video_id`, `frame_id`; UTF-8 JSONL proposed | Optional envelope/version fields remain open. |
| Benchmark video catalog | IMPLEMENTED | `src/system_tai/data/video_catalog.py` and acceptance tests | Requires authoritative real catalog data. |
| Frame mapping | IMPLEMENTED | `frame_idx` is preserved exactly as `actual_frame_id`; temporary-CSV tests pass locally | Zero-based working interpretation needs raw-video Kaggle calibration. |
| BTC CLIP store | IMPLEMENTED | `src/system_tai/features/btc_clip_store.py` and temporary-NPY tests | Real BTC artifact compatibility has not been audited. |
| Data-audit CLI | IMPLEMENTED | `scripts/audit_kis_inputs.py` and CLI tests | No real L21 artifacts were available in the workspace. |
| Kaggle input discovery | IMPLEMENTED | `scripts/discover_kaggle_inputs.py` and temporary-tree tests | Real Dataset_AIC2026 execution is Kaggle-only. |
| Raw-frame calibration | IMPLEMENTED | `scripts/calibrate_frame_mapping.py`; synthetic offset tests | Tests prove mechanics only, not BTC frame agreement. |
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

## Pending real Kaggle calibration

- Zero-based indexing is the current working interpretation.
- Decoded raw-frame agreement is not yet established.
- Dataset-wide mapping behavior is not yet established.

## Unverified

- Exact BTC-compatible CLIP implementation and image preprocessing.
- Text-query encoder compatibility.

## Still open

- Official BTC submission artifact.
- Optional JSONL envelope and version fields.

## Validation status

Gate A and Gate B have not run. Phase 1/1.5A unit and CLI tests use temporary or
synthetic fixtures only.

Codex local execution cannot produce real BTC calibration results because the private
dataset is attached only inside Kaggle. No real-data calibration, query encoder,
retrieval, JSONL export, shared validation, or fixture benchmark result currently exists
for `system_tai`.

## Remaining blockers and decisions

### Blocks a semantic KIS run

- Exact BTC-compatible text encoder and preprocessing.
- Real sample data locations and authoritative catalog metadata.
- Decoded raw-frame calibration and dataset-wide mapping behavior.

### Blocks final shared checkpoint compatibility

- Optional JSONL envelope/version fields.
- Shared validator/evaluator interface and ownership.

### Blocks official submission

- Accepted official BTC submission format.

## Next milestone

Run Phase 1.5A in Kaggle: discover at least three real videos, audit their mapping/NPY
alignment, calibrate decoded frame coordinates, and compare optional ViT-B/32 image
pipelines. Do not implement text retrieval until compatibility is reported from real
multi-video evidence.
