# Stage 1C Human Review Runbook

## 1. Purpose

Stage 1C measures the observable behavior of the raw Stage 1A cosine search driven
by the verified Stage 1B official OpenAI CLIP encoder. It is not a competition
Recall benchmark and it does not optimize retrieval.

## 2. Run Notebook 08

Attach the saved Stage 0 bundle, complete Stage 1A input bundle, accepted Stage 1B
report bundle, BTC dataset, and the same offline OpenAI CLIP asset bundle. Run
`notebooks/08_stage1c_qualitative_text_retrieval.ipynb` from top to bottom. The
last cell creates:

```text
/kaggle/working/triage_eg_stage1c_qualitative_eval_bundle.zip
```

The notebook must report `ENCODER = VERIFIED`, `PIPELINE = WORKING`, and
`RETRIEVAL_QUALITY = NOT_REVIEWED` before manual labeling.

## 3. Read contact sheets

For each query, inspect `contact_sheet_top20.jpg` in raw rank order. Rank, video ID,
keyframe ordinal `n`, authoritative original frame index, and score are printed
under each tile. Use `contact_sheet_top12_videos.jpg` only to understand video
diversity; it is a grouped diagnostic, not a replacement ranking.

## 4. Fill the review CSV

Open `review/review_template.csv`. Do not reorder rows or edit query, rank, frame,
or score identity columns. Fill `review_label` with exactly one of:

- `RELEVANT`: the frame clearly satisfies the main semantic intent.
- `PARTIAL`: part of the intent matches, but an important component is missing or wrong.
- `IRRELEVANT`: the main semantic intent is not present.
- `UNCERTAIN`: the frame alone is insufficient to decide.

`review_notes` is optional. `failure_tags` may contain semicolon-separated tags
documented by the evaluation contract. Scores must not determine labels.

## 5. Score a completed review

The scoring utility needs only the extracted Stage 1C bundle and filled CSV; it
does not load the index or model:

```bash
python scripts/score_stage1c_human_review.py \
  --stage1c-root /path/to/triage_eg_stage1c_qualitative_eval \
  --review-csv /path/to/filled-review.csv
```

It writes `review/review_metrics.json` and `review/review_metrics.md`. Invalid
labels, missing/duplicate rows, or changed identity fields fail closed.

## 6. Interpret metrics

`RELEVANT` has graded utility 1.0, `PARTIAL` 0.5, and `IRRELEVANT` 0.0.
`UNCERTAIN` is counted separately and never silently treated as zero. Summaries are
grouped by language, category, and difficulty. English/Vietnamese paired differences
are `OBSERVED_DIFFERENCE`, not a language causal effect.

## 7. Structural diagnostics

- Initial-frame concentration identifies `n == 1` and `original_frame_idx == 0`.
- Same-video concentration shows whether raw Top-K is dominated by one video.
- Exact-vector duplication detects byte-identical stored vectors only within returned Top-K.
- Pair overlap describes result-set similarity; it does not establish relevance.

Do not choose translation, query expansion, diversification, temporal support, or
another optimization until the human review evidence has been scored and examined.

