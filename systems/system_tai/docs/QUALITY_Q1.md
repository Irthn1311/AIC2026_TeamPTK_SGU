# QUALITY Q1 — Unified Semantic Quality Benchmark

## Purpose

Q1 adds measurement infrastructure for KIS, Q&A, and TRAKE after the frozen
Preliminary P0 technical baseline. P0 proves runtime, artifact, deterministic, and
regression correctness. It does not prove semantic quality against competition ground
truth. Q1 changes no model, retrieval, refinement, runtime, or output behavior.

## E0-A KIS DEV frozen-baseline evidence

E0-A is complete from recovered predictions produced by the frozen
`OperationalKISRuntime` on the internal L21-150 DEV KIS split. All 38 queries completed
inference on Tesla T4/CUDA with official OpenAI CLIP ViT-B/32. The output contains 3,789
rows: 27 queries have depth 100 and 11 have depth 99, with zero runtime query failures
and zero exact duplicate `(video_id, frame_id)` identities. Depth 99 is a diagnostic,
not a failure: the contract permits at most 100 results and no result is padded or
duplicated.

The recovered internal evaluation reports Final Score `0`, Video Recall@100 `0.052632`,
and Frame Recall@100 `0`. Mechanical categories are 36 `VIDEO_MISS` and two
`VIDEO_HIT_FRAME_MISS`; the latter are `KIS-01` and `KIS-50`. Runtime latency is
approximately 13.8597 seconds at P50 and 15.6969 seconds at P95.

These values are internal diagnostic evidence with
`semantic_gt_authority = SOURCE_PROPOSED_INTERNAL`. They are not official BTC ground
truth or leaderboard scores. Technical end-to-end completion does not establish
competition semantic accuracy. The observed result establishes a severe retrieval/video
ranking bottleneck for the frozen baseline on this internal DEV set, but it does not by
itself establish a cause. Language handling, query formulation, aggregation, diversity,
and evidence integration remain hypotheses until controlled experiments measure them.

## E0 roadmap before Q2

E0-A KIS DEV is complete through recovered evaluation. Next run E0-B Q&A DEV, then E0-C
TRAKE DEV, and only then produce the combined E0 diagnosis. Q2 one-change experiments
must not begin before that three-task baseline is complete.

Later KIS candidates, not implemented by this change, are: E1 Vietnamese normalization,
E2 Vietnamese-to-English query translation, E3 cue decomposition/query expansion, E4
multi-query fusion, E5 frame-to-video aggregation, E6 top-video allocation/diversity,
E7 temporal refinement, E8 OCR/Object soft evidence, and E9 reranking/Top-100 policy.

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
