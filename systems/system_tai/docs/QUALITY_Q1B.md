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

## Q1-B1B0 human annotation queue

Q1-B1B0 adds annotation-operations infrastructure only. It freezes which semantic slot
is filled first, which independently sampled candidate is reviewed next, and how a
human records suitability without creating a query, frame interval, answer, event
chain, ground truth, or verified annotation.

### Slot-first deterministic review

The queue follows two independent deterministic sequences:

- Candidate order is the already frozen `candidate_video_manifest.csv` order generated
  with seed `system_tai_q1b_v1`.
- Slot order is generated before semantic video inspection with seed
  `system_tai_q1b_slot_v1`. For each `slot_id`, compute
  `SHA256("system_tai_q1b_slot_v1|" + slot_id)`, sort by `(slot_hash, slot_id)`, and
  assign one-based `assignment_rank` values.

The first slot without an `ASSIGN` decision is always the current target. The smallest
unreviewed candidate `sample_rank` is always the next video. Candidate ranks are
consumed strictly and never cherry-picked, revisited, wrapped, or resampled. A candidate
can be assigned to at most one slot, and a slot can receive at most one assignment.
Changing the target category after seeing a video is prohibited.

`category_codebook.csv` defines 30 categories: ten each for KIS, Q&A, and TRAKE. It
provides the category name, definition, acceptance and rejection guidance, and suggested
tags shown during raw-video review. `slot_assignment_manifest.csv` freezes the 60-slot
execution order. `candidate_review_log.csv` is header-only in Q1-B1B0 and will record
future decisions without query text, frame IDs, answer aliases, event intervals, GT, or
`planned_split`.

### Review decisions and split blindness

Only these decisions are permitted:

- `SKIP_NO_SUITABLE_EVENT`: the human reviewed the raw video before retrieval and found
  no suitable event for the current category. The candidate is permanently consumed;
  the slot remains open for the next candidate.
- `SKIP_TECHNICAL_UNREADABLE`: a concrete technical problem prevented review. The
  candidate is permanently consumed, the reason is recorded, and the slot remains open;
  `raw_video_reviewed` may be false because this is not a semantic rejection.
- `ASSIGN`: the human reviewed the raw video before retrieval and judged it suitable for
  the current category. The candidate and slot are permanently consumed, then both
  deterministic sequences advance.

The queue validates `planned_split` internally but never exposes development/holdout
status in its normal next-target output or human-facing review log. Split membership
must not influence suitability, future query wording, or future GT boundaries. Semantic
decisions require raw-original-video review and `reviewed_before_retrieval = true`; no
`system_tai` prediction may be inspected first.

`ASSIGN` means only candidate suitability for a planned category slot. It is not a
query, completed annotation, GT record, or verification decision, and it does not
populate `annotation_plan.csv`. Future Q1-B1B Pass 1 will separately author and freeze
the semantic query, definition, original-frame intervals, Q&A answers or TRAKE events,
and a draft benchmark record before independent verification.

### PILOT15 repaired human-suitability checkpoint

PILOT15 records 15 consecutive `ASSIGN` decisions as an annotation-workflow
checkpoint. It is not the final benchmark: the planned Q1-B target remains 60 verified
queries. PILOT15 created no semantic query or semantic GT, and no retrieval or model
output was inspected before the suitability decisions.

A post-Kaggle independent audit found stale duplicated notes in reviews 8 and 12. Both
raw videos were re-audited, and only the `notes` fields were repaired; all assignment
identities and other fields remained unchanged. The forensic pre-repair log SHA-256 is
`a595580b741ced0d03f4008b005659d7da99ce4d00c37802dfc05ca762fea27b`; the repaired
canonical log SHA-256 is
`41ee3117146ed446602b0a36422097173595fbf53f8fff07c7c22f46bc5f8a8e`.
Review 8 remains `L23_V013 / TRAKE-001 / TR-C1`; review 12 remains
`L25_V024 / KIS-018 / KIS-C8`. The next deterministic unreviewed target remains
`L23_V023 / KIS-011 / KIS-C1`.

The checkpoint remains `DIAGNOSTIC` for `CURRENT_873_VIDEO_SNAPSHOT` and
`BATCH_1_ONLY`; full preliminary representativeness remains `NOT_ESTABLISHED`.
PILOT15 is neither official BTC GT nor a semantic benchmark freeze. Q1-C and Q2 have
not started.

## Q1-B1C0 semantic annotation workstation

Q1-B1C0 is a local implementation under review for human-only Pass 1 authoring,
independent Pass 2 verification, revision, and cross-artifact audit. It does not run
retrieval, inspect model output, author text, translate, select boundaries, or infer
answers/events. A human must supply every semantic field after reviewing the original
raw video. Tooling implementation is not evidence that the semantic benchmark is
complete.

The two active queues have different purposes and positions:

- **Suitability queue:** asks whether the next independently sampled raw video suits
  the next frozen category slot. After PILOT15, its next target is review 16,
  `L23_V023 / KIS-011 / KIS-C1`.
- **Semantic Pass-1 queue:** consumes only existing `ASSIGN` records in ascending
  `assignment_rank`. Its initial target is assignment 1,
  `L26_V065 / KIS-015 / KIS-C5`, with derived query ID `q1b-kis-015`.

An `ASSIGN` decision therefore does not mean Pass 1 is complete. The suitability queue
may advance while the semantic queue begins at the earliest assigned slot without an
annotation-registry row. Core semantic logic is forward-compatible beyond PILOT15; the
PILOT15 packet is only the first 15 assignments (8 KIS, 5 Q&A, and 2 TRAKE).

### Strict input and identity contract

All human input is strict UTF-8 JSON without a BOM or duplicate keys. Unknown fields,
whitespace-only required strings, stale target expectations, and silent repair are
rejected. Each Pass-1 input includes `expect_assignment_rank`, `expect_slot_id`,
`annotator_id`, the four true confirmations (`raw_video_reviewed`,
`query_authored_before_retrieval`, `gt_authored_before_retrieval`, and
`original_frame_coordinates_verified`), positive `raw_video_frame_count`, an existing
Q1 difficulty enum, ordered unique semantic `tags`, `semantic_definition`,
`annotation_notes`, `boundary_notes`, and `answer_notes`.

The query ID is never accepted from Pass-1 human input. Frozen slot IDs must match the
exact uppercase `KIS-NNN`, `QA-NNN`, or `TRAKE-NNN` namespace and its frozen task range;
malformed, padded, or case-altered IDs are rejected rather than normalized. A valid ID
is permanently derived as `"q1b-" + slot_id.casefold()`, for example `KIS-015` becomes
`q1b-kis-015`. Assignment video, task, category, and slot identity come only from the
frozen suitability state. Annotator/reviewer IDs are exact, case-sensitive provenance
strings with no outer whitespace or control characters.

Task-specific Pass-1 fields are:

- **KIS:** non-empty `query_vi`; nullable `query_en` and `query_en_expansion`; inclusive
  `start_frame_id` and `end_frame_id` within the raw-video frame count.
- **Q&A:** non-empty `event_description` and `question`; nullable English counterparts;
  an inclusive evidence interval; and a non-empty, ordered, duplicate-free
  `accepted_answers` list.
- **TRAKE:** two to five ordered event objects containing exactly `description`,
  nullable `description_en`, non-empty `moment_definition`, `start_frame_id`, and
  `end_frame_id`. Event starts must be strictly increasing; the workstation never sorts
  them. `moment_definition` remains in the review sidecar because the frozen canonical
  Q1 schema stores only event descriptions and intervals.

Frame bounds are exact inclusive original-video coordinates:
`0 <= start_frame_id <= end_frame_id < raw_video_frame_count`. The tooling never widens
an interval or applies a plus/minus-one correction.

### Source references and synchronized state

Machine paths are not human inputs and never enter canonical semantic artifacts. The
workstation generates portable references:

- KIS/Q&A: `raw_video:<video_id>;reviewed_frames:<start>-<end>`
- TRAKE: `raw_video:<video_id>;event_windows:<s1>-<e1>|<s2>-<e2>|...`

A successful Pass 1 creates a `draft` query with label origin `human_raw_video`, writes
its GT to `benchmark.draft.json`, writes a registry row with
`COMPLETE / REVIEW_PENDING / benchmark_included=false`, and writes ordered
`REVIEW_PENDING` event-sidecar rows for TRAKE only. Draft human GT is not score-eligible.
Queries serialize in assignment order using deterministic UTF-8 JSON and must round-trip
through the frozen Q1 loader before publication.

Pass 2 is independent: `reviewer_id` must differ from `annotator_id`, and the reviewer
must recheck raw video, semantic support, video identity, coordinates, intervals, plus
Q&A answers or TRAKE event order as applicable. The state transitions are:

| Operation | Benchmark | Registry Pass 1 | Registry Pass 2 | Included |
|---|---|---|---|---:|
| Pass 1 | `draft` | `COMPLETE` | `REVIEW_PENDING` | `false` |
| Pass 2 `VERIFIED` | `verified` | `COMPLETE` | `VERIFIED` | `true` |
| Pass 2 `REVISION_REQUIRED` | `draft` | `COMPLETE` | `REVISION_REQUIRED` | `false` |
| Pass-1 revision | `draft` | `COMPLETE` | `REVIEW_PENDING` | `false` |

Verification changes status and reviewer fields only; it never rewrites authored
semantics. `REVISION_REQUIRED` requires non-empty review notes and also does not edit
semantic content automatically. A subsequent revision preserves query ID, slot, task,
and video, requires a complete new human-authored payload, clears reviewer state, and
replaces (rather than appends) that query's TRAKE sidecar rows.

### Transaction and audit guarantees

Every write operation acquires one exclusive same-directory writer lock, then loads and
audits the current benchmark, registry, sidecar, and frozen assignment state; builds the
complete next state in memory; serializes every affected artifact to collision-resistant
same-directory temporary files; flushes and `fsync`s those files; independently reloads
and audits them; and only then replaces canonical files. Pre-replacement failure leaves
all canonical bytes unchanged. A replacement failure attempts to restore every affected
file from preserved original bytes, audits the restored state, and reports rollback or
cleanup failures loudly. The lock prevents cooperating workstation processes from
silently overwriting a state loaded by another writer.

This is exception/process-level rollback, not a power-loss-atomic multi-file filesystem
transaction. Canonical-directory `fsync` and an OS journal are not provided. A hard kill
may leave dot-prefixed temp/rollback files, which are never treated as canonical. It may
also leave `.quality_q1b_semantic_annotation.lock`; after independently confirming that
no writer is active, a human must remove that stale lock before writing again. Manual
edits or writers that bypass this workstation are outside the concurrency guarantee.

The cross-artifact audit fails closed on duplicate or orphan identities, slot/task/video
mismatches, invalid state transitions or source references, missing/extra/reordered
TRAKE events, noncanonical UTF-8/order, and split leakage.

### Split blindness and commands

Normal human success and expected CLI error output never expose `planned_split`, development/holdout identity,
`q1b_dev`, or `q1b_holdout`. These two split tags, model-result tags such as
`retrieval_bad`, and rank-derived tags such as `rank_37` are rejected in semantic Pass
1. Frozen split information is reserved for a later benchmark freeze and cannot guide
query wording, GT, verification, status, or packet export.

The local command surface is:

```text
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py pass1-next
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py template --slot-id KIS-015
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py pass1-record --input pass1.json
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py pass2-next
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py pass2-record --input pass2.json
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py revision-next
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py pass1-revise --input revision.json
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py status
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py audit
python systems/system_tai/scripts/quality_q1b_semantic_annotation.py pilot15-export --output pilot15.json
```

`pass1-next`, `pass2-next`, `revision-next`, `status`, `template`, and `pilot15-export`
are split-blind. The export is a deterministic external work packet containing the first
15 `ASSIGN` records by `assignment_rank`, not the first 15 review rows. Its destination
must resolve and lexically reside outside the repository; it does not mutate repository
state and contains no GT or retrieval data.

At this Q1-B1C0 tooling milestone the repository remains deliberately semantic-data
empty: `benchmark.draft.json` has zero queries, `annotation_registry.csv` has zero data
rows, and `trake_event_review.csv` has zero data rows. There is no real query, real GT,
verified annotation, semantic score, Q1-C run, or Q2 experiment yet.

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
baseline evaluation. Split tags `q1b_dev` and `q1b_holdout` are also forbidden in
semantic Pass 1; split assignment remains hidden until the later benchmark-freeze
boundary.

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
