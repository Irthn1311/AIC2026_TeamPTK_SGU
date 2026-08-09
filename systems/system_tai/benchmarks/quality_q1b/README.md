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

The provenance explicitly covers `CURRENT_873_VIDEO_SNAPSHOT` and `BATCH_1_ONLY`.
Batch 2 is not represented, and full preliminary representativeness is not established.
No file in this directory currently contains real GT.
