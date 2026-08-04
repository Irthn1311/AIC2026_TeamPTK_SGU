# Targeted Cross-Asset Survey patch v0.2.1

This patch directly inspects at most 15 fixed Object JSON files and the five locked
duplicate `frame_idx` cases. It does not repeat the v0.1 or v0.2 surveys, traverse ID
sets, decode media, read image pixels, or run a model.

Run on Kaggle from the repository root:

```bash
python scripts/survey_cross_assets_v021.py \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/cross_asset_survey_v021 \
  --v02-summary /kaggle/working/cross_asset_survey_v02/cross_asset_survey_v02.json \
  --strict-root
```

`--v02-summary` is optional. The output ZIP contains only the five text artifacts and
is written to `/kaggle/working/cross_asset_survey_v021/cross_asset_survey_v021.zip`.
