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
ranking symptom for the frozen baseline on this internal DEV set, but it does not by
itself identify the stage or establish a cause. Language handling, query formulation,
aggregation, diversity, evidence integration, and task-specific finalization remain
hypotheses until controlled experiments measure them.

## Combined E0 internal DEV diagnosis

### A. Directly measured facts

Combined E0 is complete as an internal diagnostic. Measured final-output coverage is:

| Task | Output coverage | Zero output | Target video in final output | Target video among non-empty outputs |
|---|---:|---:|---:|---:|
| KIS | 38/38 | 0/38 | 2/38 | 2/38 |
| Q&A | 12/38 | 26/38 | 0/38 | 0/12 |
| TRAKE | 28/38 | 10/38 | 0/38 | 0/28 |

TRAKE `event_order_accuracy = 0.710526` and
`chain_completeness = 0.736842` are structural diagnostics. They can be non-zero even
when every predicted video is wrong; they are not semantic event-grounding success.
Likewise, `PARTIAL_CHAIN` means no evaluated candidate reached the required structural
frame count. It must not be described as partial semantic event success.

### B. Unresolved stage attribution

The largest directly measured issue on internal L21 DEV is low target-video inclusion
in final task outputs. This is a final-output observation, not a stage attribution.
Q&A can produce no final answer because of question classification, retrieval,
refinement, evidence decode/filtering, or answer generation. A TRAKE target video can
be present in one or more event pools and still fail to survive complete-video planning,
temporal chain construction, refinement, deduplication, or finalization. Therefore the
current evidence does not prove Vietnamese text, CLIP, exact retrieval, or the TRAKE
planner is the sole cause.

The relation between the ten `PARTIAL_CHAIN` queries and the ten zero-output queries is
not established unless the two query-ID sets are compared explicitly.

The runtime now records target-agnostic QA fused/refined/usable-evidence identities and
TRAKE event-pool/C1/final-path identities. The offline L21 analyzer compares these
artifacts to internal benchmark video IDs. Missing historical stage artifacts are
reported as `UNAVAILABLE`; they are never reconstructed by inference.

### C. Q2 hypothesis and KIS DEV Arm B preparation

The first controlled Q2 input is `TRANSLATION_AUGMENTED_RRF`: frozen benchmark
Vietnamese plus a separately reviewed English translation, fused by the existing
Weighted RRF policy. It is not an English-only arm and not a causal translation
experiment. The reviewed 38-record DEV-only sidecar is `REVIEWED_FROZEN`, contains no
HOLDOUT query, and was authored without retrieval feedback. Arm B is implementation
ready; its GPU experiment has not yet run.

The L21 runner remains VI-only by default. Arm B requires both
`--kis-query-policy translation_augmented_rrf` and `--kis-query-sidecar`; it keeps
Top-K 100, refine Top-N 3, equal existing variant weights, and no English expansion.
If Arm B improves primary target-video recall, an EN-only Arm C may be added later as an
ablation. No HOLDOUT run is authorized in Q2 preparation.

## Earlier Q2 roadmap

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
