# Kaggle Phase 1.5 Report

## Scope

This report records the Phase 1.5B correction derived from a real Kaggle run. It
separates verified evidence for `L21_V001` from hypotheses that still require
multi-video reproduction. It does not identify a BTC-compatible CLIP implementation
and is not an official retrieval benchmark.

## Verified facts

The private dataset was attached beneath the nested Kaggle input layout at
`/kaggle/input/datasets/nadkli/dataset-aic`. Dynamic discovery resolved the original
video, map-keyframes CSV, CLIP NPY, and keyframes for `L21_V001` without copying them.

For `L21_V001`:

- video FPS: `30.0`;
- total frames: `37849`, with zero-based bounds `[0, 37848]`;
- mapping rows: `307`, with `frame_idx` range `[0, 37716]`;
- CLIP array: shape `[307, 512]`, dtype `float16`, with no NaN or Infinity;
- vectors appear L2-normalized;
- mapping/feature row-count agreement is true and feature-row coverage is `1.0`;
- `clip_row = n - 1` was observed for this artifact.

The full 307-row mapping audit using Decimal-exact arithmetic reports:

- `frame_idx - decimal_floor(pts_time * fps)`: `-1` for 4 rows and `0` for 303;
- `decimal_round_half_up(pts_time * fps) - frame_idx`: `0` for 233 rows and `1`
  for 74;
- therefore Decimal floor matches 303 of 307 rows and must not be described as a
  universal rule.

The four Decimal-floor differences are:

| n | pts_time | fps | frame_idx | Decimal product |
|---:|---:|---:|---:|---:|
| 63 | 260.4 | 30 | 7811 | 7812 |
| 248 | 1024.1 | 30 | 30722 | 30723 |
| 249 | 1031.1 | 30 | 30932 | 30933 |
| 254 | 1058.6 | 30 | 31757 | 31758 |

For these values, binary floating-point multiplication lies slightly below the exact
integer and truncation can equal the stored `frame_idx`; for example,
`float(260.4) * 30.0` is represented as `7811.999999999999`, whose integer truncation
is `7811`. These four rows are valid mapping records, not malformed data.

Only 15 of the 307 rows have been visually decoded. Offset `0` was best for 10 samples,
offset `+1` for 5, and offset `-1` for none. All 15 visual best offsets equal
`decimal_round_half_up(pts_time * fps) - frame_idx`, and random-seek and sequential
decoding agree for all 15. Therefore this evidence does not indicate a global `+1`
indexing error.

Examples include keyframe order `67`, where `pts_time * fps = 8358.99`,
`frame_idx = 8358`, and the JPEG visually matches raw frame `8359`; and order `220`,
where `pts_time * fps = 26670.99`, `frame_idx = 26670`, and the JPEG visually matches
raw frame `26671`.

## Frame-coordinate rule

The official/shared coordinate remains:

```text
actual_frame_id = frame_idx
```

No addition or subtraction is permitted. The diagnostic coordinate is separate:

```text
keyframe_visual_frame_id = decimal_round_half_up(pts_time * fps)
keyframe_visual_offset = keyframe_visual_frame_id - actual_frame_id
```

The diagnostic never modifies `actual_frame_id`, shared `frame_id`, or official
output. Keyframe order `n`, physical CSV row, filename number, `local_frame_idx`, and
`clip_row` are not shared frame IDs.

## Inferred behavior

- Mapping generation may have used binary-float multiplication followed by integer
  truncation. The full numeric audit is consistent with this candidate for the four
  Decimal-floor exceptions, but the generation implementation is not known.
- Keyframe JPEG extraction may have used nearest timestamp alignment, producing a
  visual offset of `0` or `+1`; only 15 rows have been visually checked.
These are inferences from one video, not accepted dataset-wide rules.

## Pending reproduction

- Run the full mapping numeric audit for `L21_V002` and `L22_V001`.
- Run sampled random/sequential visual calibration for both videos.
- Determine whether binary-float truncation and nearest timestamp extraction remain
  consistent candidates across all three videos.

## Unresolved items

- multi-video mapping, rounding, and feature-row behavior;
- exact BTC CLIP library, checkpoint, tokenizer, preprocessing, normalization, and
  similarity metric;
- compatible text-query encoder;
- semantic retrieval quality;
- official BTC submission artifact;
- optional JSONL envelope and version fields.

## Exact Kaggle commands

Run from `/kaggle/working/AI_Challenge_HCM` after attaching Dataset_AIC2026. The
notebook runs these operations for `L21_V001`, `L21_V002`, and `L22_V001` and writes
reports only below `/kaggle/working/system_tai_outputs/calibration/`.

```bash
python systems/system_tai/scripts/discover_kaggle_inputs.py \
  --input-root /kaggle/input \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/discovery_L21_V001.json

python systems/system_tai/scripts/audit_mapping_rounding.py \
  --mapping-csv <discovered-map-keyframes.csv> \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/mapping_rounding_L21_V001.json

python systems/system_tai/scripts/calibrate_frame_mapping.py \
  --batch-manifest /kaggle/working/system_tai_outputs/calibration/frame_calibration_batch_manifest.json \
  --output /kaggle/working/system_tai_outputs/calibration/frame_mapping_calibration.json
```

Use `systems/system_tai/notebooks/phase_1_5_kaggle.ipynb` for the complete dynamic
three-video workflow, including discovery, real-input audit, rounding audit, visual
calibration, compact CSV summaries, and final status. CLIP pipeline identification
remains disabled until all three videos pass input/feature-row and mapping gates.

## Evidence boundary

Local tests validate only parsing, Decimal rounding, status transitions, and synthetic
image-comparison mechanics. The real facts above come from the reported Kaggle run;
the local Codex environment does not contain the private dataset and did not reproduce
those real-data results.
