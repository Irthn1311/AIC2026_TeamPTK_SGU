# system_tai

Isolated foundation for Nguyễn Tài's independent AIC 2026 system.

Phase 1 implements trustworthy BTC input auditing. Phase 1.5B adds nested Kaggle input
discovery, Decimal timestamp-rounding diagnostics, corrected raw-frame calibration,
and a gated optional image-encoder identification path:

mapping CSV + BTC CLIP NPY + authoritative video catalog
→ validated original-frame mapping
→ explicit feature-row-to-frame mapping.

It does not implement a CLIP text encoder, semantic retrieval, ranking, checkpoint
serialization, Q&A, TRAKE, an API server, or a frontend. Synthetic tests are mechanics
evidence only and are never BTC calibration evidence.

## Install

From the shared repository root:

```bash
python -m pip install -e "systems/system_tai[dev]"
```

## Verify

```bash
pytest systems/system_tai/tests -v
ruff check systems/system_tai
```

## Audit KIS inputs

```bash
python systems/system_tai/scripts/audit_kis_inputs.py \
  --video-catalog <catalog.csv> \
  --video-id <BTC_VIDEO_ID> \
  --mapping-csv <mapping.csv> \
  --clip-npy <features.npy> \
  --expected-dimension <dimension>
```

Add `--strict-video-path-check` when catalog paths must exist locally. The audit command
returns non-zero for invalid input.

UTF-8 JSONL remains a proposed team checkpoint format. It is not implemented in Phase 1.
Official BTC submission is a separate unresolved boundary.

## Kaggle Phase 1.5B

Attach the private Dataset_AIC2026 and clone the repository to
`/kaggle/working/AI_Challenge_HCM`. Do not hard-code or assume the runtime dataset slug.
Open and run `notebooks/phase_1_5_kaggle.ipynb` from top to bottom.

The discovery CLI scans bounded nested layouts beneath `/kaggle/input`, so it supports
`/kaggle/input/datasets/<owner-or-runtime-id>/<dataset-root>` without hard-coding a
dataset slug. It can also be run directly:

```bash
python systems/system_tai/scripts/discover_kaggle_inputs.py \
  --input-root /kaggle/input \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/discovery_L21_V001.json
```

Audit Decimal-exact, binary-float-truncation, and Decimal-nearest numeric models:

```bash
python systems/system_tai/scripts/audit_mapping_rounding.py \
  --mapping-csv <discovered-map-keyframes.csv> \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/mapping_rounding_L21_V001.json
```

The notebook repeats discovery, input audit, rounding audit, and visual calibration for
`L21_V001`, `L21_V002`, and `L22_V001`.

For multi-video calibration, create a JSON/YAML batch manifest from discovery reports:

```bash
python systems/system_tai/scripts/calibrate_frame_mapping.py \
  --batch-manifest /kaggle/working/system_tai_outputs/calibration/frame_calibration_batch_manifest.json \
  --output /kaggle/working/system_tai_outputs/calibration/frame_mapping_calibration.json
```

Frame calibration preserves every CSV `frame_idx` exactly. Numeric-generation-rule
identification is diagnostic and does not affect mapping validity. It separately reports
`keyframe_visual_frame_id = decimal_round_half_up(pts_time * fps)` and the visual
offset. An explained `+1` JPEG alignment never changes `actual_frame_id` or shared
`frame_id`.

Verified `L21_V001` evidence covers all 307 mapping rows: Decimal floor matches 303;
the other four have `frame_idx - Decimal floor = -1`; and Decimal-nearest offset is `0`
for 233 rows and `+1` for 74. Only 15 rows have been visually decoded. All 15 visual
best offsets match the Decimal-nearest prediction, and random/sequential decoding agrees
for all 15. Binary-float truncation as the mapping generation rule and nearest timestamp
alignment as the JPEG extraction rule remain inferred pending `L21_V002` and
`L22_V001`.

Optional CLIP candidates can be compared without implementing text retrieval only after
all three videos pass mapping and feature-row gates:

```bash
python systems/system_tai/scripts/identify_btc_clip_pipeline.py \
  --batch-manifest /kaggle/working/system_tai_outputs/calibration/clip_identification_batch_manifest.json \
  --minimum-identification-videos 3 \
  --output /kaggle/working/system_tai_outputs/calibration/clip_pipeline_identification.json
```

Unavailable packages or weights are reported as `SKIPPED`. Add
`--allow-model-download` only when Kaggle internet policy permits it. All generated
reports must stay under `/kaggle/working/system_tai_outputs/calibration/`; source dataset
artifacts remain under `/kaggle/input`.

Verified `L21_V001` evidence and its limits are recorded in
`docs/KAGGLE_PHASE_1_5_REPORT.md`. It is one-video calibration evidence, not an official
performance result or dataset-wide confirmation.

TRIAGE-EG is reference material only and must remain untouched.
