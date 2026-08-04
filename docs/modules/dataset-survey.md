# Dataset Survey

Dataset Survey is a bounded, read-only inspection of an attached dataset's layout, filenames, asset
groups, and small schema samples. It is not a complete Data Audit: it does not decode video, probe
media metadata, validate the full corpus, derive frame IDs, run models, or establish CLIP model
compatibility.

Default safeguards limit directory depth to 4, displayed entries to 20 per directory, retained
examples to 5 per asset group, CSV samples to 20 rows, JSON reads to 1 MB, NPY inspection to 5
memory-mapped rows, and each traversal pass to 5,000 stat operations. Symlinks are recorded but not
followed.

Run on Kaggle after the repository package is available:

```bash
python scripts/survey_dataset.py \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/dataset_survey \
  --max-depth 4 \
  --max-examples-per-group 5 \
  --max-listed-per-directory 20 \
  --strict-root
```

The command writes at most `dataset_survey.json`, `dataset_survey.md`, and
`sample_inventory.jsonl`. Use `--no-write` to print the JSON summary without creating artifacts.
After inspecting the report, confirm canonical video IDs, cross-asset joins, frame numbering,
map-keyframe semantics, metadata missing behavior, and feature partitioning before designing the
full Data Audit.
