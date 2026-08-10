# Reference Experiment RT2: AI-Curated Temporal Benchmark

RT2 is a bounded research experiment. It prepares real chronological BTC contact sheets,
accepts a strict AI-curated internal pseudo-GT file, and calibrates the unchanged RT1 DANTE
distance penalty. It does not create official competition ground truth and does not promote
DANTE to production.

## Frozen dependencies

RT2 reuses the Stage 1 exact BTC index and canonical catalog, Stage 1B verified OpenAI CLIP,
Stage 1E language path, Stage 2A runtime, RT1 full-score APIs, unordered aggregation, and the
RT1 strict monotonic DANTE recurrence. It does not change encoders, translation, mapping,
cosine scores, index order, or tie handling.

## Mode A: prepare candidates

The default notebook mode is `PREPARE_CANDIDATES`. Attach only:

- `/kaggle/input/datasets/irthn1311/triage-eg-stage1b-input-bundle`
- `/kaggle/input/datasets/nadkli/dataset-aic`

Run `notebooks/12_reference_rt2.ipynb` with `AIC_RT2_MODE=PREPARE_CANDIDATES`, or run:

```powershell
python scripts/prepare_reference_rt2_candidates.py `
  --stage1-root <stage1-root> `
  --dataset-root <dataset-root> `
  --output-root <output-root>
```

The selection is deterministic with seed 2026. Eligible videos have at least 12 canonical BTC
keyframes. Eighteen `TEMPORALLY_DIVERSE` videos are selected by mean adjacent cosine distance
over evenly sampled frozen BTC image features. Eighteen `GENERAL_ELIGIBLE` videos are selected
one per deterministic corpus/video-ID stratum from the remaining eligible set. This diversity
score is sampling metadata only and never affects retrieval ranking.

Each sheet is a chronological 4x4 grid with up to 16 unique, evenly spaced canonical frames.
For videos containing 12-15 keyframes, unused grid cells remain blank rather than duplicating
frames. Download:

`/kaggle/working/triage_eg_rt2_benchmark_candidates.zip`

Do not rerun candidate preparation after preserving this ZIP; the command fails closed when its
output root already exists.

## AI benchmark handoff

Give the ZIP to the AI reviewer and keep all semantic decisions grounded in the visible sheets.
The desired output is `rt2_ai_benchmark.jsonl` with 20-24 usable queries. Each query must have
2-4 events and exact identities copied from strictly increasing sheet slots. The loader rejects
unknown videos, non-monotonic event positions, mismatched global row, `n`, or original frame ID,
and any benchmark type other than `AI_CURATED_INTERNAL_PSEUDO_GT`.

AI labels are sparse internal pseudo-GT. `human_reviewed` must remain `false` for this phase.

## Mode B: evaluate benchmark

Upload `rt2_ai_benchmark.jsonl` as a Kaggle Dataset and set
`AIC_RT2_MODE=EVALUATE_BENCHMARK`. Attach the benchmark plus the six frozen inputs used by RT1:

- Stage 1 input bundle
- Stage 1B encoder-compatibility reports
- Stage 1E language-path freeze
- OpenAI CLIP ViT-B/32 offline asset
- OPUS-MT vi-en offline asset
- AIC dataset

The evaluator encodes each event once and computes one full score matrix per query. It reuses
that matrix for unordered event-max, the exact lambda grid `0, 0.0001, 0.0003, 0.001, 0.003,
0.01`, and every reversed-order control. No CLIP similarity is recomputed during the sweep.

Queries are deterministically split approximately 2/3 DEV and 1/3 HOLDOUT, stratified by event
count when possible. Lambda is selected from DEV only by:

1. maximum internal Video Recall@5;
2. maximum MRR;
3. maximum approximate AI-reference anchor hit within ±3 technical keyframes;
4. minimum approximate anchor MAE;
5. smaller absolute lambda.

If fewer than 18 queries survive strict validation, evaluation reports
`INSUFFICIENT_BENCHMARK` and does not select a lambda. HOLDOUT never influences selection.

For a sufficient benchmark, download:

`/kaggle/working/triage_eg_rt2_evaluation_bundle.zip`

The ZIP contains reports, compact per-query audit records, blinded HOLDOUT sheets, and the hidden
review key. It excludes vectors, models, indexes, raw media, cache, logs, and Stage 2 control
artifacts.

## Metric interpretation

`INTERNAL_VIDEO_RECALL_AT_K` and MRR use the sheet's known source video as internal pseudo-GT.
`AI_REFERENCE_ANCHOR_*` compares DANTE positions against sparse AI-selected sheet positions and
is not official frame-localization accuracy. `CHAIN_COLLAPSE_CANDIDATE` is a diagnostic when
predicted span is at most the number of events; it is not an automatic failure threshold.
`ORDER_DISCRIMINATION_RATE` measures how often the source video ranks better under correct event
order than reversed event order.

RT2 never emits an automatic KEEP, REDESIGN, or DROP decision.
