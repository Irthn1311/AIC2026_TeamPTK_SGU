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

To patch an already-frozen v0.1.0 output without invoking translation, CLIP, or
search, run:

```bash
python scripts/patch_stage1d_blinded_review_visuals.py \
  --stage1d-root /path/to/frozen_stage1d_v0_1_0 \
  --dataset-root /path/to/dataset-aic \
  --output-root /path/to/stage1d_v0_1_1 \
  --zip-path /path/to/triage_eg_stage1d_translation_ablation_v0_1_1_bundle.zip
```

Omit `--output-root` only when the frozen output is writable and an in-place
review-presentation patch is intentional. The command validates every CSV
identity against the frozen arm record before rendering.

## Inspect evidence

Start with `stage1d_report.md` and `stage1d_summary.json` for engineering audit.
The files under `comparisons/<pair_id>/comparison_top5.jpg` expose the real arm
names and are never review inputs. Contact sheets and ranking diagnostics do not
establish semantic relevance.

Use `translations/translations.jsonl` to audit exact Vietnamese inputs,
translated outputs, model revision, generation settings, latency, and status.
Frozen Top-20 JSONLs provide a bounded audit trail without copying the full
Stage 1C bundle.

## Complete blinded review

1. Copy `review/review_template_blinded.csv` outside the immutable result bundle.
2. Keep `review/review_key.json` and every engineering comparison sheet closed.
3. For each pair, open only `review/blinded_sheets/<pair_id>_top5.jpg` and match
   each `Cxx`/rank tile to the corresponding CSV row.
4. Enter one allowed `review_label`: `RELEVANT`, `PARTIAL`,
   `IRRELEVANT`, or `UNCERTAIN`. Notes are optional.
5. Do not edit identity columns and do not use retrieval scores as labels.
6. Score the completed copy:

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
