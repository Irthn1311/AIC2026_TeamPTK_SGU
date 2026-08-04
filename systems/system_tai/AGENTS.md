# AGENTS.md — system_tai

## Scope

This file governs Nguyễn Tài's independent end-to-end system. Internal architecture,
models, retrieval, temporal processing, evidence, logging, UI, and deployment choices
are system-specific.

Do not modify TRIAGE-EG source, tests, configs, or generated assets. TRIAGE-EG documents
are reference material only and are not the architecture of `system_tai`.

## Team boundaries

The shared boundaries are benchmark inputs, original-video frame coordinates, selected
checkpoint outputs, the shared validator, and the shared evaluator.

UTF-8 JSONL is the current **proposed** team checkpoint format. Until the team accepts
the final schema, checkpoint serialization must remain isolated behind an exporter
adapter. The accepted team schema overrides this proposal.

The resolved shared KIS checkpoint core fields are `query_id`, `rank`, `video_id`, and
`frame_id`. Optional envelope and version fields remain open.

Official BTC submission is a separate export boundary. Do not assume that a team JSONL
checkpoint and an official submission are the same artifact or format.

## Architecture and implementation truth

The personal workbook and Canva document are design references. They do not prove that
runtime implementation exists.

Source under `systems/system_tai/src/`, scripts, configs, and corresponding tests is
authoritative for whether a `system_tai` implementation exists. Test source is not
test-pass evidence; verification results must be reported separately.

Use these status values: `IMPLEMENTED`, `BASELINE`, `TEMPLATE`, `EXPERIMENTAL`,
`PLANNED`, `DEFERRED`, and `UNKNOWN`.

## Frame invariants

- Raw BTC video is the final frame-coordinate source of truth.
- For BTC keyframes, preserve `frame_idx` from the BTC map-keyframes CSV exactly as
  `actual_frame_id`; never add or subtract one.
- The zero-based coordinate and raw-video bounds are verified for `L21_V001`; behavior
  across the dataset remains pending multi-video calibration.
- `keyframe_visual_frame_id = decimal_round_half_up(pts_time * fps)` is diagnostic only
  and must never replace or offset `actual_frame_id`.
- Decimal floor, binary-float truncation, and Decimal nearest are separate numeric
  diagnostics. Matching any proposed generation rule is not required for an in-bounds
  mapping to be valid.
- Internal `actual_frame_id` must equal the original frame index in the original BTC video.
- At an accepted shared boundary, `frame_id` must carry that same original-frame value.
- Keyframe order `n`, CLIP row, `local_frame_idx`, and filename number are never shared
  `frame_id` values.
- Resampled or clip-local coordinates must be mapped back to the original BTC video.
- Reject ambiguous mappings instead of guessing.
- For self-extracted frames, shared `frame_id` is the position in the original BTC video
  coordinate system.

## Repository placement

`system_tai` work lives in the shared repository on a personal branch. TRIAGE-EG must
remain untouched, and changes must not be committed, pushed, pulled, or merged unless a
later task explicitly authorizes that operation.

## Kaggle data rules

- Discover the private Dataset_AIC2026 attachment recursively within the bounded
  `/kaggle/input` runtime layout; do not hard-code its Kaggle slug or nesting depth.
- Never copy videos, keyframes, feature arrays, or dataset folders into the repository or
  `/kaggle/working`.
- Calibration output is restricted to
  `/kaggle/working/system_tai_outputs/calibration/`.
- Local synthetic tests prove mechanics only. They are not BTC calibration evidence.
- BTC currently confirms only `clip-ViT-B-32` and feature-row order matching keyframe
  order. Implementation, weights, preprocessing, normalization, and metric remain
  unverified until multi-video calibration succeeds.

## First baseline

Prioritize the real KIS vertical slice:

mapping CSV + BTC CLIP NPY
→ validated frame mapping
→ compatible query encoder
→ vector retrieval
→ `CandidateFrame(actual_frame_id)`
→ grouping and deduplication
→ ranked Top-100
→ proposed shared JSONL exporter
→ validator
→ fixture evaluation

The first slice must not use guessed or dummy semantic embeddings.

## Validation gates

Gate A — integration smoke test:

- one real video;
- one real mapping CSV;
- one real CLIP NPY;
- one query;
- valid end-to-end output.

Gate B — retrieval sanity benchmark:

- a small corpus with positive and distractor videos;
- at least five manually verified queries when data is available;
- expected video and/or interval for each query;
- Video Recall@K and observed ranking in the report;
- no claim of official performance.

## Excluded from the first slice

Do not add Agent, GNN, Event Graph, VLM, OCR, ASR, Q&A, TRAKE, an API server,
a backend, or a production frontend to the first vertical slice.

## Development rules

- Keep changes inside the isolated `system_tai` scope.
- Do not depend directly on TRIAGE-EG internal classes.
- Put checkpoint serialization behind an adapter.
- Preserve dataset, mapping, encoder, index, and schema-version information.
- Reject feature/query dimension or model incompatibility.
- Every unimplemented path must raise `NotImplementedError` or an equivalent clear error.
- Never fabricate results to make a skeleton appear runnable.
- Do not claim lint, tests, benchmarks, or demos passed unless executed.

Before implementing a module, document its input, output, intended path, status,
dependencies, tests, and acceptance criteria.
