# Cross-Asset Survey v0.2

This targeted, read-only survey verifies ID-set equality and samples the contract between video,
mapping CSV, keyframe filenames, memory-mapped CLIP rows, Object JSON, and metadata. It scans only
known direct roots and inspects three to five complete video IDs. It does not decode video, read
image pixels, run models, or replace the full Data Audit.

```bash
python scripts/survey_cross_assets.py \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/cross_asset_survey_v02 \
  --max-videos 5 \
  --max-object-json-total 25 \
  --max-object-json-bytes 1048576 \
  --max-mapping-rows 10000 \
  --seed 2026 \
  --strict-root
```

Outputs are one JSON summary, one Markdown report, cross-asset records, bounded Object JSON schema
samples, and issue records. The report separates verified, inferred, and unknown contracts and ends
with `READY_FOR_STAGE_0_DESIGN` or `NOT_READY_FOR_STAGE_0_DESIGN`.
