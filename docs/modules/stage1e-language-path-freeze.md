# Stage 1E AI Evaluation Gate and Language Path Freeze

Stage 1E v0.1 is an evaluation-ingest and decision-freeze step. It is not a
retrieval stage. It consumes frozen Stage 1D evidence plus an AI-judged blinded
review, validates identity against the frozen review template, unblinds only
for scoring, and emits the operational language-path contract for Stage 2.

The internal Vietnamese baseline is:

`Vietnamese → Helsinki-NLP/opus-mt-vi-en → official OpenAI CLIP ViT-B/32 → frozen Stage 1A exact BTC index`

The translator revision is locked to
`c8d2853e77f5fae31124d993e0b35176b1c8914e`. English remains direct through
the same verified CLIP encoder. This decision does not claim that OPUS-MT is
globally optimal or that final competition retrieval quality is proven.

AI and human evaluation states are separate:

- `AI_REVIEW_STATUS=COMPLETE`
- `HUMAN_REVIEW_STATUS=NOT_PERFORMED`
- `LANGUAGE_BRIDGE_QUALITY_STATUS=AI_EVALUATED_ACCEPTED`

`difficult_01` remains a semantic retrieval failure after translation, while
`obj_01` shows that language bridging alone does not repair every CLIP semantic
failure. Both must be carried into Stage 2 evaluation.

Run the bounded ingest without model or retrieval assets:

```bash
python scripts/run_stage1e_language_path_freeze.py \
  --stage1d-root outputs/triage_eg_stage1d_translation_ablation_bundle \
  --ai-review-root outputs/stage1d_ai_review/stage1d_ai_review_artifacts \
  --output-root outputs/stage1e_language_path_freeze
```
