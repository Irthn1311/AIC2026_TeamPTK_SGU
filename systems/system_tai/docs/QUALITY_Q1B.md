# QUALITY Q1-B - Human Gold-Set Protocol

## Purpose

Q1-B defines how to build an independent, human-verified semantic benchmark for
Textual KIS, Q&A, and TRAKE. Q1-B0 provides methodology, provenance, deterministic
sampling, and empty annotation scaffolds only. It contains no real verified ground
truth and does not measure semantic performance.

The future benchmark is intended to measure the frozen Preliminary P0 baseline,
diagnose failures, and compare later experiments without relying on subjective
examples. It does not change Q1 tooling, P0 scorers, retrieval, refinement, or runtime.

## Source-of-truth hierarchy

1. The BTC preliminary-round document defines task and scoring semantics.
2. Original BTC video is authoritative for semantic annotation and frame coordinates.
3. BTC mapping CSV maps provided keyframes to original-video `frame_idx` values.
4. Keyframes, CLIP features, Objects, and Metadata are supporting material only.
5. Q1 schema and evaluator source define the local benchmark representation and
   evaluation behavior.
6. Retrieval outputs, regression fixtures, and old Phase 2.5 labels are never sources
   of semantic ground truth.

Shared `frame_id` is always an original-video frame index. Keyframe order, CLIP row,
filename number, local decode index, and resampled index are never benchmark frame IDs.
For a provided BTC keyframe, preserve mapping CSV `frame_idx` exactly with no offset.

## Current diagnostic dataset identity

Q1-B0 is bound to the accepted current snapshot:

- Dataset scope: `CURRENT_873_VIDEO_SNAPSHOT`
- BTC batch scope: `BATCH_1_ONLY`
- Videos: 873
- CLIP rows: 177321
- Corpus fingerprint:
  `b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4`
- Full preliminary representativeness: `NOT_ESTABLISHED`

This benchmark is independently authored diagnostic GT. It is not BTC ground truth,
does not represent Batch 2, and must not be described as official or full AIC 2026 data.

## Target size and split

The planned target is 60 verified queries:

| Task | Development | Holdout | Total |
|---|---:|---:|---:|
| KIS | 17 | 8 | 25 |
| Q&A | 14 | 6 | 20 |
| TRAKE | 10 | 5 | 15 |
| Total | 41 | 19 | 60 |

These are planning slots, not placeholder queries. Split assignment must be frozen
before Q2 uses the benchmark. Development failures may be inspected. Holdout is for
periodic aggregate acceptance checks, not per-query rule design.

## Independent deterministic video sampling

Future sampling starts only from the authoritative validated raw-video inventory:

1. Sort canonical `video_id` values lexicographically.
2. For every video compute
   `SHA256("system_tai_q1b_v1|" + video_id)`.
3. Sort by the resulting digest, then by `video_id` as a deterministic safeguard.
4. Consume videos in that order while filling the annotation plan.
5. If no suitable event exists, record `SKIP_NO_SUITABLE_EVENT` and continue.

The fixed sampling seed is `system_tai_q1b_v1`. Q1-B0 does not execute this algorithm
against guessed IDs and does not commit a fabricated 873-video inventory. Retrieval is
never used to select an easier replacement video.

### Q1-B1A candidate-video sampling manifest

Q1-B1A materializes the global unbiased order as `candidate_video_manifest.csv` with
exactly `sample_rank`, `video_id`, and `selection_hash`. Rank is one-based physical CSV
order after sorting by the lowercase SHA-256 digest and then `video_id`; it is not a
retrieval rank or evidence that the video suits any planned category. The implementation
uses only UTF-8 `SHA256("system_tai_q1b_v1|" + video_id)` and rejects duplicate, empty,
or malformed inventories rather than silently repairing them.

Real generation is permitted only after the existing corpus loader validates the
accepted `CorpusManifest.fingerprint`/`manifest_fingerprint`, 873 videos, and 177321
feature rows for `CURRENT_873_VIDEO_SNAPSHOT` / `BATCH_1_ONLY`. In portable schema v2,
the existing loader additionally proves that `dataset_identity.fingerprint` equals that
same manifest fingerprint; the two fields are not independently substituted. Local
synthetic tests prove only sampler mechanics. When the private corpus is unavailable
locally, generation is deferred to bounded discovery under `/kaggle/input`, and only the
compact result may be written under `/kaggle/working/system_tai_outputs/quality_q1b/`.

`annotation_plan.csv` intentionally remains unassigned in Q1-B1A. A human must inspect
raw video in sampling order and apply `SKIP_NO_SUITABLE_EVENT` before a sampled video can
be associated with a semantic category slot; deterministic selection alone creates no
query, GT, or suitability claim.

#### Real-corpus acceptance

The Q1-B1A Kaggle real-corpus acceptance passed using sampler revision
`ae65395d91de5edb8d1a449ea35f3558a4a609d7`. The accepted identity is
`CURRENT_873_VIDEO_SNAPSHOT`, `BATCH_1_ONLY`, with 873 videos, 177321 feature rows,
873 raw videos, and corpus fingerprint
`b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4`.
Using sampling seed `system_tai_q1b_v1` produced 873 candidate rows; the byte-exact
`candidate_video_manifest.csv` SHA-256 is
`d4ef95e0fe51615a436de65760f99c588478f984f27a1cad25337af297aa4661`.

This sampling rank is neither retrieval rank nor task/category assignment. The manifest
contains no query and no GT, `annotation_plan.csv` remains unassigned, Batch 2 is not
represented, and full preliminary representativeness remains `NOT_ESTABLISHED`. This is
not BTC official GT.

## Annotation-before-retrieval rule

Before inspecting any `system_tai` prediction, the annotator must:

1. select a raw video independently;
2. watch the raw video;
3. author the query, question, or ordered TRAKE events;
4. define the semantic target;
5. record the GT interval or intervals in original-video coordinates; and
6. freeze the annotation candidate.

If retrieval output was inspected first, the record is contaminated and cannot become
verified. Retrieval-selected labels, Phase 2.5 pilot labels, and P0 golden regression
outputs are prohibited as semantic GT.

## KIS annotation policy

KIS GT contains `video_id`, `start_frame_id`, and `end_frame_id`. The inclusive interval
must cover the frames that satisfy the authored event. Do not widen it to make scoring
easier or collapse a persistent event to one frame. Record genuine boundary ambiguity
in annotation notes.

The 25-slot plan covers clear scenes, action/motion, attributes, relations, small
objects, brief actions, repeated actions, OCR-relevant events, crowded scenes, and short
transitions. A random video need not be forced into a category.

## Q&A annotation policy

First localize the evidence interval, then define genuinely equivalent accepted-answer
aliases supported by the raw video. Do not rewrite aliases to match current model output.
Ambiguous or incompatible answers require revision or rejection.

Coverage intentionally extends beyond the current four baseline families. It includes
color/attribute, count, yes/no, direction, object identity, action, spatial relation,
visible text/OCR, temporal before/after, and general open-ended questions. Current
implementation capability must not define benchmark difficulty or coverage.

## TRAKE annotation policy

TRAKE queries normally contain two to five physically ordered semantic events. Define
each event's semantic moment before selecting its inclusive GT interval. Keep precise
transition intervals narrow when the definition requires it. Never sort events
automatically or use technical keyframes as the event definition.

Coverage includes state sequences, transition onset/offset, first occurrence, contact,
separation, extrema, repeated actions, long gaps, and camera/scene transitions. Event
definitions and review notes belong in the TRAKE review sidecar; canonical intervals
remain in `TRAKEGroundTruth`.

## Difficulty and tags

Difficulty describes semantic and retrieval difficulty before baseline output is seen:

- `easy`: distinctive evidence with little ambiguity;
- `medium`: moderate distractors, temporal precision, or attribute ambiguity;
- `hard`: small, brief, repeated, crowded, OCR-dependent, or finely bounded evidence;
- `unknown`: not yet assessed.

Tags describe semantics and future failure slices, such as `person_action`,
`object_attribute`, `small_object`, `relation`, `motion`, `brief_event`,
`repeated_action`, `ocr`, `count`, `open_ended`, `transition`, `contact`, and
`camera_cut`. Model-result tags such as `rank_37` or `retrieval_bad` are forbidden before
baseline evaluation. Split tags may be `q1b_dev` and `q1b_holdout`.

## Two-pass verification

A record may become `verified` with label origin `human_raw_video` only after:

- Pass 1: an annotator reviews raw video and records query, GT, source reference, and
  notes.
- Pass 2: an independent reviewer rechecks raw video, semantic support, video ID,
  original-frame coordinates, intervals, Q&A answers, and TRAKE event order.

Use anonymous identifiers such as `A01` and `R01`. A single-person annotation remains
`DRAFT` / `REVIEW_PENDING` until independent review is complete.

Verified source references are human-readable and portable, for example
`raw_video:L21_V001;reviewed_frames:1200-1270` or
`raw_video:L21_V001;event_windows:100-105|200-207`. Never store machine-local absolute
paths as canonical source references.

## Rejection and revision

Reject or revise a candidate when raw-video evidence is ambiguous, boundaries cannot be
established, Q&A has incompatible answers, a TRAKE moment is underspecified, original
frame coordinates cannot be proven, reviewer disagreement is material, or retrieval was
seen before annotation freeze. Do not hide disagreement by widening GT intervals.

## Benchmark freeze procedure

Before freezing a later benchmark version:

1. complete two-pass review;
2. verify task counts, split counts, source references, and sidecars;
3. validate canonical JSON with the frozen Q1 loader;
4. confirm zero duplicate IDs and original-frame coordinates;
5. compute and record the benchmark SHA-256;
6. change the stable ID to `system_tai-q1b-b1-diagnostic-v1`; and
7. freeze corpus fingerprint, provenance, and split assignments together.

Never silently change the corpus behind an existing result.

## Batch 2 migration

When BTC officially releases Batch 2, validate release identity, rebuild the corpus
manifest, compute a new fingerprint, and rerun ingestion gates. Preserve clean Batch 1
annotations, add independently authored Batch 2 positives, and rerun old Batch 1 queries
with Batch 2 as additional distractors. Publish a new benchmark/provenance version rather
than rewriting the frozen Batch 1 diagnostic result.

## Q1-C handoff

Q1-C may measure the frozen P0 semantic baseline only after the canonical benchmark has
reviewed verified records and a frozen split. Q1-B0 itself has zero score-eligible
queries, does not run retrieval, and provides no semantic-quality claim. Q2 starts only
after the reviewed benchmark and frozen-baseline measurement are available.
