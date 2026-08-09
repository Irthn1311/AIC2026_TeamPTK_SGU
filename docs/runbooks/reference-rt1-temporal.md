# Reference Experiment RT1 — Controlled Temporal Retrieval

RT1 compares exactly three arms over the frozen Stage 2A model space:

- `WHOLE_QUERY`: unchanged Stage 2A control;
- `UNORDERED_EVENT_MAX`: mean independent best-event score per video;
- `DANTE_DP`: strict monotonic event alignment with `lambda=0.001`.

The DANTE-style recurrence is an independent adaptation of the temporal-DP
principle described in [Integrated Semantic and Temporal Alignment for
Interactive Video Retrieval](https://arxiv.org/abs/2512.13169). It is not a
claim of exact reproduction of the authors' full system. RT1 uses the verified
BTC CLIP space, canonical BTC `n`/catalog order, and the frozen Stage 2A
language path. `original_frame_idx` remains an output coordinate only.

The repository and available local artifacts contain no real archived AIC 2025
TRAKE ground truth. Therefore `reference_rt1_queries.jsonl` contains only the
two paper examples and the result must remain `EXPLORATORY_NO_GT` until visual
review; it cannot automatically emit KEEP or DROP.

Attach these Kaggle inputs:

- `triage-eg-stage1b-input-bundle`;
- `triage-eg-stage1b-encoder-compatibility-reports`;
- `triage-eg-stage1e-language-path-freeze`;
- `aic2026-openai-clip-vit-b32`;
- `aic2026-opus-mt-vi-en`;
- `dataset-aic` for canonical keyframe images.

Run:

```bash
python scripts/run_reference_rt1_temporal.py \
  --stage1-root /kaggle/input/datasets/irthn1311/triage-eg-stage1b-input-bundle \
  --stage1b-root /kaggle/input/datasets/irthn1311/triage-eg-stage1b-encoder-compatibility-reports \
  --stage1e-root /kaggle/input/datasets/irthn1311/triage-eg-stage1e-language-path-freeze/stage1e_language_path_freeze \
  --clip-asset-root /kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32 \
  --translator-asset-root /kaggle/input/datasets/irthn1311/aic2026-opus-mt-vi-en \
  --dataset-root /kaggle/input/datasets/nadkli/dataset-aic \
  --output-root /kaggle/working/triage_eg_reference_rt1
```

Notebook 11 performs the same run and writes:

`/kaggle/working/triage_eg_reference_rt1_bundle.zip`

The ZIP contains rankings, chains, structural diagnostics, visual sheets and
the blinded review key. It excludes models, index arrays, raw media, logs and
caches.
