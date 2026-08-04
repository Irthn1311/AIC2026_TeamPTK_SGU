# Object numeric-string contract v0.2.2

This final bounded patch validates fail-closed numeric conversion for bbox coordinates,
scores, and class labels in the same 15 Object JSON files locked by v0.2.1. It does not
scan IDs or directories, modify source JSON, decode media, or rerun an earlier survey.

```bash
python scripts/survey_object_numeric_contract_v022.py \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/object_numeric_contract_v022 \
  --max-object-json-total 15 \
  --max-object-json-bytes 1048576 \
  --strict-root
```

The ZIP contains exactly five text artifacts. If readiness is
`READY_FOR_STAGE_0_DATA_AUDIT`, the next phase is Stage 0 Data Audit—not another survey
patch.
