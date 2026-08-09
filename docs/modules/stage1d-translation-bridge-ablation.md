# Stage 1D — Vietnamese Translation Bridge Ablation v0.1

## Purpose

Stage 1D tests one bounded hypothesis: translating an existing Vietnamese query
to English before encoding it with the accepted OpenAI CLIP text encoder may
repair the Vietnamese-direct retrieval failure observed during Stage 1C.

The comparison has exactly three arms:

- `EN_DIRECT`: immutable ranked artifacts read from Stage 1C.
- `VI_DIRECT`: immutable ranked artifacts read from Stage 1C.
- `VI_TRANSLATED_EN`: the only arm translated, encoded, and searched in Stage 1D.

Stage 1D is an ablation/evaluation stage. It does not rebuild the Stage 1A
index, rerun Stage 1B compatibility, regenerate Stage 1C baselines, rerank,
diversify, expand queries, or select a production route.

## Frozen baseline contract

The runner fails closed unless the Stage 1C summary is complete, contains 28
queries in 14 English/Vietnamese pairs, matches its query-suite fingerprint,
matches the loaded Stage 1A index fingerprint, and refers to the same verified
Stage 1B encoder contract. Every selected baseline query must have a valid raw
Top-50 ranking, ranked-video artifact, and retrieval diagnostics. Baseline
scores and order are copied exactly; they are never searched again.

## Offline translation contract

The only accepted translator is `Helsinki-NLP/opus-mt-vi-en` at exact revision
`c8d2853e77f5fae31124d993e0b35176b1c8914e`. Its manifest and streaming file
hashes are verified before model loading. `AutoTokenizer` and
`AutoModelForSeq2SeqLM` receive an absolute local model path and
`local_files_only=True`; `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set
before loading.

The required Python runtime dependencies are `torch`, `transformers`, and
`sentencepiece`. `sacremoses` is optional in Transformers' Marian tokenizer:
when absent, Transformers uses its documented identity punctuation-normalizer
fallback. Stage 1D records availability, version, and the effective normalizer
backend in runtime provenance instead of blocking the run.

Generation is deterministic: sampling is disabled, beam count is 4,
`max_new_tokens` is 64, length penalty is 1.0, and early stopping is enabled.
The input is the exact Stage 1C Vietnamese text. The generated text is only
trimmed at both ends before CLIP encoding—there is no rewriting, expansion, or
fallback API.

The Kaggle asset may be mounted directly at either
`/kaggle/input/aic2026-opus-mt-vi-en` or
`/kaggle/input/datasets/irthn1311/aic2026-opus-mt-vi-en`. A nested directory or
one matching ZIP is resolved and materialized deterministically when needed.

## Retrieval and diagnostics

The translated query is encoded through the verified Stage 1B OpenAI CLIP
adapter and searched against the existing Stage 1A exact-cosine backend. The
raw Top-50, internal Top-100 KIS candidates, ranked videos, Stage 1C-compatible
structural diagnostics, and contact sheets are emitted without reranking.
KIS exports use `original_frame_idx`, never local `n`, as the submitted frame
identifier.

For each pair the report includes frame/video overlap and Jaccard at Top
5/10/20/50, plus CLIP text-space cosine for English–Vietnamese,
English–translated, and Vietnamese–translated. These measurements describe
movement and concentration; they do not measure semantic relevance.

## Human review gate

Each pair gets a deterministic, seeded, blinded mapping from three opaque
condition codes to the three arms. The default review contains 14 × 3 × 5 =
210 judgments. The visible CSV deliberately omits the arm and translated text;
the separate review key resolves conditions for scoring.

Allowed labels are `RELEVANT`, `PARTIAL`, `IRRELEVANT`, and `UNCERTAIN`.
`PARTIAL` contributes 0.5 to graded relevance, while `UNCERTAIN` is reported
separately. Reported human Top-1/Top-5 rates are qualitative metrics and are
not competition Recall@K.

No language-bridge or production decision is valid before review is complete.
Fallback policy is intentionally outside Stage 1D.

## Outputs

The standalone bundle contains manifests, translations, translated rankings,
bounded frozen comparison evidence, contact sheets, issues, and the blinded
review package. It excludes translator/CLIP weights, Stage 1 matrices, `.npy`
files, the full keyframe corpus, raw videos, logs, and caches.
