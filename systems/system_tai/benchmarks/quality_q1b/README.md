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

The provenance explicitly covers `CURRENT_873_VIDEO_SNAPSHOT` and `BATCH_1_ONLY`.
Batch 2 is not represented, and full preliminary representativeness is not established.
No file in this directory currently contains real GT.
