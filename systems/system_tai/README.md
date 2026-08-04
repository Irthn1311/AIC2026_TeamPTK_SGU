# system_tai

Isolated foundation for Nguyễn Tài's independent AIC 2026 system.

Phase 1 implements trustworthy BTC input auditing. Phase 1.5A adds Kaggle-native input
discovery, raw-frame coordinate calibration, and optional image-encoder identification:

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

## Kaggle Phase 1.5A

Attach the private Dataset_AIC2026 and clone the repository to
`/kaggle/working/AI_Challenge_HCM`. Do not hard-code or assume the runtime dataset slug.
Open and run `notebooks/phase_1_5_kaggle.ipynb` from top to bottom.

The discovery CLI can also be run directly:

```bash
python systems/system_tai/scripts/discover_kaggle_inputs.py \
  --input-root /kaggle/input \
  --video-id L21_V001 \
  --output /kaggle/working/system_tai_outputs/calibration/discovery_L21_V001.json
```

For multi-video calibration, create a JSON/YAML batch manifest from discovery reports:

```bash
python systems/system_tai/scripts/calibrate_frame_mapping.py \
  --batch-manifest /kaggle/working/system_tai_outputs/calibration/frame_calibration_batch_manifest.json \
  --output /kaggle/working/system_tai_outputs/calibration/frame_mapping_calibration.json
```

Optional CLIP candidates are compared without implementing text retrieval:

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

TRIAGE-EG is reference material only and must remain untouched.
