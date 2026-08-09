# QUALITY Q1 — Unified Semantic Quality Benchmark

## Purpose

Q1 adds measurement infrastructure for KIS, Q&A, and TRAKE after the frozen
Preliminary P0 technical baseline. P0 proves runtime, artifact, deterministic, and
regression correctness. It does not prove semantic quality against competition ground
truth. Q1 changes no model, retrieval, refinement, runtime, or output behavior.

## Evidence boundary

The historical Phase 2.5 KIS pilot is not migrated or treated as authoritative ground
truth. It uses KIS-specific relevant frames/videos, covers only three intents, and has
retrieval-selection bias because some labels were chosen after inspecting retrieval
output. Q1 defines a new three-task contract and does not infer event intervals from the
old labels.

Synthetic labels exist only in tests. No guessed real ground truth is committed.

## Benchmark contract

One strict UTF-8 JSON object contains schema version 1, benchmark ID, description, and
physically ordered queries with globally unique IDs. Unknown fields, BOM, invalid UTF-8,
wrong task fields, invalid enums, bool frame IDs, duplicate tags/IDs, and malformed
ground truth fail closed. Raw JSON also rejects duplicate object keys at every nesting
level; no last-key-wins repair is permitted.

Task ground truths reuse the frozen P0-A types directly:

- KIS: query text plus one `KISGroundTruth` video interval.
- Q&A: event, question, optional English text, accepted answers, and one
  `QAGroundTruth` video interval.
- TRAKE: one or more physically ordered events and matching ordered
  `TRAKEGroundTruth.event_intervals`.

BTC raw video is the source of truth for manual labels. A `human_raw_video`
`source_reference` must identify the reviewed evidence in a human-readable way; it need
not name the annotator. Frame IDs remain original BTC video coordinates.

## Draft, verified, and label origins

Draft queries may be unlabeled and have null ground truth. They are never scored and the
report records how many were skipped. Verified queries require ground truth, a non-empty
source reference, and a label origin other than `unlabeled`.

Supported origins are `unlabeled`, `human_raw_video`, `official`, and `synthetic`.
Verified human/official queries are score-eligible by default. Synthetic queries are
excluded unless the caller explicitly sets `include_synthetic=True`.

## Evaluation and internal aggregates

The evaluator consumes one in-memory `RankedTop100Query` for every score-eligible query,
including an explicit empty tuple for a genuine zero-prediction result. It delegates to
the frozen P0-A evaluator and scorers, including the configured Q&A alias matcher.

Reports expose R@1, R@5, R@20, R@50, R@100, and P0-A final score per query. They also
provide task summaries and deterministic difficulty/tag breakdowns.

`overall_query_macro_score` is the mean final score across all scored queries.
`task_macro_score` is the mean of non-empty task mean final scores. These are internal
experiment-comparison aggregates. They are not claimed to be the official competition
cross-task weighting, which is unknown.

## Experiment comparison workflow

Run the same benchmark through baseline and candidate systems, evaluate both canonical
in-memory prediction sets, and compare reports. Q1 requires identical benchmark IDs,
scored query IDs, and task types. It reports per-query improved/tied/regressed outcomes,
per-task deltas, overall query-macro delta, and task-macro delta with numerical tolerance.
It does not make an automatic merge or acceptance decision.

JSON and CSV metric reports are deterministic UTF-8 outputs with no timestamps, images,
videos, or embeddings.

## Next steps

Q1-B bootstraps a human-verified gold set from raw BTC video without inspecting system
retrieval output first. Only after that benchmark is reviewed should Q2 attempt KIS
semantic retrieval optimization. No later optimization should be accepted merely because
examples look better.
