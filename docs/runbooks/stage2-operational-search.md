# Stage 2A Operational Search Runbook

Attach these offline Kaggle inputs:

- Stage 1A full input bundle containing index arrays;
- Stage 1B encoder compatibility reports;
- Stage 1E language-path freeze artifacts;
- `aic2026-openai-clip-vit-b32`;
- `aic2026-opus-mt-vi-en`.

Run one explicit-language query:

```bash
python scripts/search_stage2_operational.py \
  --stage1-root /kaggle/input/datasets/irthn1311/triage-eg-stage1b-input-bundle \
  --stage1b-root /kaggle/input/datasets/irthn1311/triage-eg-stage1b-encoder-compatibility-reports \
  --stage1e-root /kaggle/input/datasets/irthn1311/triage-eg-stage1e-language-path-freeze \
  --clip-asset-root /kaggle/input/datasets/irthn1311/aic2026-openai-clip-vit-b32 \
  --translator-asset-root /kaggle/input/datasets/irthn1311/aic2026-opus-mt-vi-en \
  --text "một người đang nấu ăn trong bếp" \
  --language vi \
  --top-k 20 \
  --output-root /kaggle/working/triage_eg_stage2a_operational_runtime
```

Use explicit language for Vietnamese without diacritics. `auto` ambiguity is a
clean error and performs no translation, encoding, or retrieval for that query.

Each query directory contains the request, language resolution, encoding
provenance, raw frames, grouped videos, KIS CSV, and latency summary. Download
`/kaggle/working/triage_eg_stage2a_operational_runtime_reports.zip`; it excludes
models, vectors, raw data, images, caches, and logs.
