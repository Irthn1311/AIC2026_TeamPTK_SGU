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

The provenance explicitly covers `CURRENT_873_VIDEO_SNAPSHOT` and `BATCH_1_ONLY`.
Batch 2 is not represented, and full preliminary representativeness is not established.
No file in this directory currently contains real GT.
