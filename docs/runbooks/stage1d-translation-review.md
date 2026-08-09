# Stage 1D translation-ablation review runbook

## Run on Kaggle

Attach the raw dataset, Stage 0 audit, Stage 1A runtime/index, accepted Stage 1B
artifacts, frozen Stage 1C bundle, official OpenAI CLIP asset, and the OPUS-MT
asset `irthn1311/aic2026-opus-mt-vi-en`. Open
`notebooks/09_stage1d_vi_en_translation_ablation.ipynb`, confirm the environment
cell paths, and use **Run All**. Do not enable Internet and do not install or
download a model in the notebook.

The last cell creates:

`/kaggle/working/triage_eg_stage1d_translation_ablation_bundle.zip`

Download that single ZIP. A successful execution still reports human review
and language-bridge quality as `NOT_REVIEWED`.

## Inspect evidence

Start with `stage1d_report.md` and `stage1d_summary.json`. Then inspect
`comparisons/<pair_id>/comparison_top5.jpg` for three side-by-side conditions:
English direct, Vietnamese direct, and translated Vietnamese. Contact sheets
are diagnostics only; do not infer a winner from CLIP score or overlap alone.

Use `translations/translations.jsonl` to audit exact Vietnamese inputs,
translated outputs, model revision, generation settings, latency, and status.
Frozen Top-20 JSONLs provide a bounded audit trail without copying the full
Stage 1C bundle.

## Complete blinded review

1. Copy `review/review_template_blinded.csv` outside the immutable result bundle.
2. For every row, enter one allowed `review_label`: `RELEVANT`, `PARTIAL`,
   `IRRELEVANT`, or `UNCERTAIN`. Notes are optional.
3. Do not edit identity columns or consult `review_key.json` while judging.
4. Score the completed copy:

```bash
python scripts/score_stage1d_translation_ablation_review.py \
  --stage1d-root /path/to/triage_eg_stage1d_translation_ablation \
  --review-csv /path/to/completed_review.csv
```

The scorer rejects schema changes, identity mutations, duplicated rows,
unresolved condition codes, and invalid labels. It writes
`review/review_metrics.json` and `review/review_metrics.md`; partial reviews
remain `NOT_REVIEWED` for language-bridge quality.

Only a complete human review can move the qualitative state to evaluated. Even
then, the result does not establish competition Recall@K or authorize a
production fallback policy.
