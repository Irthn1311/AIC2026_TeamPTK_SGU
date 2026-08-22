# KIS semantic-clause and video-first retrieval

Status: **experimental, opt-in, local implementation**. This document defines the
engineering contract; it does not establish semantic quality or competition
accuracy.

## Input

- A UTF-8 Vietnamese KIS description.
- A validated BTC feature registry whose mapping CSV `frame_idx` is the shared
  frame coordinate.
- The pinned VinAI `vinai/vinai-translate-vi2en-v2` translation provider.
- The canonical OpenAI CLIP ViT-B/32 encoder and exact NumPy feature search.

No caller-supplied English prompt, automatic network translation service,
Marian fallback, object/OCR/ASR evidence, or generated image is part of this
path.

## Output

- A deterministic `KISResult` with at most 100 unique `(video_id, frame_id)`
  candidates.
- `frame_id` is copied exactly from mapping CSV `frame_idx`.
- Internal diagnostics record translated semantic units, video-level RRF
  provenance, restricted-search counts, and existing Q3/refinement provenance.
- The core JSONL remains `query_id`, `rank`, `video_id`, `frame_id` only.

## Pipeline

1. Deterministically split the Vietnamese description into a full-query unit
   plus semantic clauses. Attribute/count-only clauses are supporting evidence;
   action/scene clauses are primary evidence.
2. Translate all units with one VinAI provider instance. Each English unit is
   losslessly segmented to the CLIP token budget; no CLIP truncation is allowed.
3. Encode all segments in one text batch.
4. Scan the complete feature registry once for all variants and retain each
   video's exact maximum-cosine frame for every variant.
5. Rank videos independently per variant, then apply weighted reciprocal-rank
   fusion at `video_id` identity. Raw cosine values are never added across
   variants.
6. Search every keyframe of the selected videos once for all variants. Fuse
   restricted frame rankings by the existing `WeightedRRFRetriever` at
   `(video_id, frame_id)` identity.
7. Existing Q3 keyframe conditioning and bounded raw-video refinement remain
   optional downstream stages. Their off behavior is unchanged.

## Paths and dependencies

- Production modules live under `src/system_tai/retrieval/` and
  `src/system_tai/kis/`.
- Tests use synthetic mapping/NPY fixtures and fake translation/encoding
  providers; they require no BTC data, model download, or network.
- Source videos, keyframes, feature arrays, weights, and generated run artifacts
  are never copied into Git.

## Failure and compatibility rules

- Enabling this path requires dynamic VinAI translation. Configuration fails
  clearly instead of falling back to manual English or another translator.
- Missing/malformed feature artifacts retain existing registry failures.
- The feature search is exact and deterministic. Ties use video ID, frame ID,
  and physical `clip_row` only as internal deterministic keys.
- `clip_row`, keyframe order, filename number, and local decoded index are never
  exported as shared frame IDs.
- When the feature is disabled, the existing frame-level KIS path remains byte-
  compatible in candidate ordering and core output.

## Tests and acceptance

Acceptance requires synthetic evidence for:

- Vietnamese semantic clause splitting without hard-coded English prompts;
- one VinAI batch translation and lossless CLIP segmentation;
- video-level accumulation when different clauses match different frames of the
  same video;
- no raw-cosine mixing between variants;
- deterministic video selection and restricted full-keyframe fusion;
- exact frame-coordinate preservation and no duplicate output identity;
- a full Top-K when selected videos contain enough unique frames;
- Q3/refinement compatibility and unchanged disabled behavior;
- full `systems/system_tai` pytest, Ruff, compile, and diff checks.

Private Kaggle execution and human/gold-set evaluation remain required before
any semantic-quality claim.
