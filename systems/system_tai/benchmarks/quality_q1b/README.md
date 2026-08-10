# Q1-B Diagnostic Benchmark Workspace

This directory bootstraps independent human annotation for the current Batch 1
diagnostic corpus. Q1-B0 contains no real query, no verified ground truth, and no
competition-performance result.

## Files

- `benchmark.draft.json`: canonical Q1 schema input. It is intentionally empty in Q1-B0
  and is the only file here that will later contain scoreable query/GT records.
- `provenance.json`: deterministic corpus, revision, and benchmark identity. It binds
  future results to a specific dataset snapshot but is not itself a scoring input.
- `annotation_plan.csv`: 60 deterministic planning slots. It is not canonical GT.
- `annotation_registry.csv`: header-only audit sidecar for two-pass human verification.
- `trake_event_review.csv`: header-only event-level TRAKE review sidecar. It does not
  replace canonical `TRAKEGroundTruth`.

Original BTC raw video is the semantic and frame-coordinate source of truth. Keyframes,
CLIP features, Objects, and Metadata are support material. Mapping CSV `frame_idx` is
preserved exactly for provided keyframes; keyframe order, CLIP row, filename, and local
decode index are never shared frame IDs.

Annotation must be frozen before any `system_tai` retrieval output is inspected. Video
candidates will later be sampled independently by sorting validated `video_id` values by
`SHA256("system_tai_q1b_v1|" + video_id)`. Q1-B0 does not run that algorithm or include a
guessed inventory.

## Q1-B1A deterministic candidate order

After real-corpus validation, `candidate_video_manifest.csv` records only
`sample_rank,video_id,selection_hash`. Its physical row order is the one-based ascending
order of `SHA256("system_tai_q1b_v1|" + video_id)`, with `video_id` as the deterministic
tie-breaker. It contains no paths, retrieval scores, semantic assignments, timestamps,
or GT. The current-corpus gate requires 873 videos, 177321 CLIP rows, and the accepted
existing manifest fingerprint before a real artifact can be accepted.

The private corpus is not represented by repository fixtures. If it is not available
locally, run the isolated sampler on Kaggle using a validated reusable manifest (and
`--input-root /kaggle/input` when rebasing a portable manifest), for example:

```text
python systems/system_tai/scripts/build_quality_q1b_sampling_manifest.py \
  --manifest /kaggle/input/system-tai-manifest/feature_manifest.json \
  --input-root /kaggle/input \
  --require-current-q1b-corpus \
  --output /kaggle/working/system_tai_outputs/quality_q1b/candidate_video_manifest.csv
```

Dataset nesting is resolved by the existing bounded manifest loader/discovery code; no
slug is hard-coded and no source artifact is copied. `annotation_plan.csv` remains
unassigned until human raw-video inspection establishes category suitability. Sampling
rank is not a category assignment and creates no query or verified label.

### Accepted Kaggle artifact

Real-corpus acceptance passed with sampler revision
`ae65395d91de5edb8d1a449ea35f3558a4a609d7`, dataset scope
`CURRENT_873_VIDEO_SNAPSHOT`, batch scope `BATCH_1_ONLY`, 873 videos, 177321 feature
rows, 873 raw videos, and corpus fingerprint
`b0c5ea97a9d5e10dbb7e77dba18d153191218935e2a3275ef888e0a8a83ed6e4`.
Seed `system_tai_q1b_v1` produced 873 rows in the byte-frozen
`candidate_video_manifest.csv`; its SHA-256 is
`d4ef95e0fe51615a436de65760f99c588478f984f27a1cad25337af297aa4661`.

Sampling rank is not retrieval rank or task/category assignment. The manifest contains
no query and no GT, `annotation_plan.csv` remains unassigned, Batch 2 is not represented,
and full preliminary representativeness remains `NOT_ESTABLISHED`. It is not BTC
official GT.

## Q1-B1B0 annotation queue

Q1-B1B0 is human-review operations infrastructure, not semantic annotation. It adds a
30-row `category_codebook.csv` (ten KIS, ten Q&A, and ten TRAKE categories), a
deterministic 60-row `slot_assignment_manifest.csv`, a header-only
`candidate_review_log.csv`, and an isolated queue helper. No Q1-B1B0 artifact contains a
query, frame interval, answer alias, event chain, GT, or verified annotation.

The method is **slot first, video second** and freezes two independent orders before
review:

1. Candidate order remains `candidate_video_manifest.csv`, generated with seed
   `system_tai_q1b_v1` and consumed by ascending `sample_rank`.
2. Slot order uses seed `system_tai_q1b_slot_v1`; each slot hash is
   `SHA256("system_tai_q1b_slot_v1|" + slot_id)`, sorted by `(slot_hash, slot_id)` and
   assigned one-based `assignment_rank` values.

The current target is the first slot without an `ASSIGN`; the next candidate is the
smallest unreviewed candidate rank. Review proceeds sequentially without gaps,
cherry-picking, revisiting, wraparound, or resampling. One candidate can be assigned to
at most one slot, and one slot can receive at most one assignment. The category codebook
supplies the definition, acceptance/rejection guidance, and suggested tags for the
current target.

The only allowed decisions are:

- `SKIP_NO_SUITABLE_EVENT`: raw video was reviewed before retrieval; consume the
  candidate and keep the same slot open.
- `SKIP_TECHNICAL_UNREADABLE`: record an explicit technical reason; consume the
  candidate and keep the same slot open. Raw review may be false when the file could not
  be reviewed.
- `ASSIGN`: raw video was reviewed before retrieval; consume the candidate, fill the
  current slot, and advance to the next deterministic slot and candidate.

`planned_split` remains frozen in `annotation_plan.csv` and the slot manifest for
internal validation, but is deliberately absent from normal next-target output and the
human-facing review log. This split blindness prevents development/holdout identity
from influencing suitability or later annotation choices. Retrieval output must not be
viewed before a semantic decision.

`ASSIGN` records only that a candidate video suits the planned category. It is not GT,
does not verify an annotation, and does not modify `annotation_plan.csv`. Future Q1-B1B
Pass 1 will separately author the query and semantic definition, establish
original-video frame intervals and task-specific answers/events, and create a draft
benchmark record for later independent review.

### PILOT15 repaired checkpoint

The canonical `candidate_review_log.csv` now contains a 15-`ASSIGN` human-suitability
checkpoint. It is annotation-workflow state, not the final benchmark; the full Q1-B
target remains 60 verified queries. PILOT15 created no semantic query or semantic GT,
and no retrieval or model output was inspected before the suitability decisions.

Independent post-Kaggle audit found stale duplicated notes in reviews 8 and 12. Their
raw videos were re-audited, and only the two `notes` fields were repaired. The
pre-repair forensic SHA-256 is
`a595580b741ced0d03f4008b005659d7da99ce4d00c37802dfc05ca762fea27b`; the repaired
canonical SHA-256 is
`41ee3117146ed446602b0a36422097173595fbf53f8fff07c7c22f46bc5f8a8e`.
Review 8 remains `L23_V013 / TRAKE-001 / TR-C1`, review 12 remains
`L25_V024 / KIS-018 / KIS-C8`, and the next deterministic unreviewed target remains
`L23_V023 / KIS-011 / KIS-C1`.

Dataset scope remains `CURRENT_873_VIDEO_SNAPSHOT`, BTC batch scope remains
`BATCH_1_ONLY`, and benchmark role remains `DIAGNOSTIC`; full preliminary
representativeness remains `NOT_ESTABLISHED`. PILOT15 is not official BTC GT or a
semantic benchmark freeze. Q1-C and Q2 have not started.

## Q1-B1C0 semantic workstation (frozen tooling)

Q1-B1C0 tooling was frozen on `feat/system-tai-quality-q1b` at commit
`810dd6a9c71882e5f29cb70e5d8d558672614a1b` after adversarial review and regression
validation. It supplies a strict human-only Pass-1, independent Pass-2, revision,
status, and cross-artifact audit workflow. It does not generate or translate queries,
select frame boundaries, infer answers/events, run retrieval, or inspect model output.
The original raw video remains the semantic and coordinate source of truth.

Do not confuse the two queues:

- Suitability next is review 16: `L23_V023 / KIS-011 / KIS-C1`.
- Semantic Pass-1 next is assignment 1:
  `L26_V065 / KIS-015 / KIS-C5 / q1b-kis-015`.

Semantic Pass 1 consumes only `ASSIGN` records in ascending `assignment_rank`.
PILOT15 contains 15 such targets (8 KIS, 5 Q&A, and 2 TRAKE), but the state machine is
not capped at 15.

### Operations

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

All write inputs are exact-field, duplicate-key-rejecting UTF-8 JSON without BOM.
`expect_assignment_rank` and `expect_slot_id` protect against stale targets. Frozen slot
IDs must use their exact uppercase `TASK-NNN` namespace/range; malformed or padded IDs
are rejected. Query IDs are derived, never entered: `q1b-` plus the case-folded valid
slot ID. Annotator/reviewer IDs are exact, case-sensitive, unpadded provenance values.
Common confirmations
require raw-video review, query and GT authorship before retrieval, verified original
coordinates, a positive raw-video frame count, an existing Q1 difficulty, ordered
unique semantic tags, and human notes/definition fields.

KIS input provides Vietnamese query text, optional English variants, and one inclusive
interval. Q&A provides event/question text, optional English text, one evidence
interval, and ordered unique accepted answers. TRAKE provides two to five already
ordered events, each with descriptions, a moment definition, and an inclusive interval;
event starts must be strictly increasing. Frame bounds must satisfy
`0 <= start <= end < raw_video_frame_count` and are never shifted or widened.

Portable source references are generated automatically:

- KIS/Q&A: `raw_video:<video_id>;reviewed_frames:<start>-<end>`
- TRAKE: `raw_video:<video_id>;event_windows:<s1>-<e1>|...`

No absolute Windows, Kaggle, or home-directory path is accepted as canonical provenance.

Pass 1 transactionally writes a `draft`/`human_raw_video` query, a registry row in
`COMPLETE / REVIEW_PENDING` with `benchmark_included=false`, and ordered TRAKE review
rows when applicable. Pass 2 requires a different reviewer and either moves all state
to `VERIFIED`/included without rewriting semantics, or records
`REVISION_REQUIRED`/not included with review notes. Revision preserves query ID, slot,
task, and video; resets review state; and replaces rather than appends TRAKE rows.

Writes use one exclusive single-writer lock. Candidate state is serialized to unique
same-directory temporary files, flushed and `fsync`ed, reloaded through the frozen
schema, audited, then replaced. Pre-replace failures leave canonical files unchanged;
mid-replace failures attempt to restore original bytes and audit the restoration, while
rollback/cleanup failures are reported loudly. `audit` fails closed on
duplicate/orphan records, identity or state mismatches, malformed source references,
TRAKE event inconsistencies, ordering/encoding errors, and split leakage.

This is exception-level rollback, not power-loss atomicity across three files; no
directory `fsync` or journal is claimed. A hard kill may leave dot-prefixed temp files
(which loaders ignore) or `.quality_q1b_semantic_annotation.lock`. Confirm that no
writer is active before manually deleting a stale lock. Writers that bypass this tool
are outside the lock guarantee.

Normal success and expected error output from `next`, `status`, `template`, audit, and
PILOT15 export is split-blind.
`planned_split`, development/holdout labels, `q1b_dev`, and `q1b_holdout` are not
exposed; the two split tags and model/rank-derived tags are rejected in Pass 1. The
PILOT15 packet deterministically selects the first 15 assignments (not the first 15
review rows), contains no GT/retrieval data, and may only be written to a destination
lexically and physically outside the repository.

Current canonical semantic state remains empty after this tooling work:

- `benchmark.draft.json`: zero queries;
- `annotation_registry.csv`: header only, zero data rows;
- `trake_event_review.csv`: header only, zero data rows.

Therefore Q1-B1C0 currently establishes workflow mechanics only: there is no real
semantic query, GT, verified annotation, benchmark score, Q1-C run, or Q2 experiment.

The provenance explicitly covers `CURRENT_873_VIDEO_SNAPSHOT` and `BATCH_1_ONLY`.
Batch 2 is not represented, and full preliminary representativeness is not established.
No file in this directory currently contains real GT.
