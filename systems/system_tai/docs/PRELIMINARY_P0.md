# PRELIMINARY P0

## PRELIMINARY P0-A: Official published task structure/scoring implementation

This phase implements the exact preliminary task schemas and evaluation semantics for the AIC 2026 Preliminary round.

### PUBLISHED BTC RULE
- **Tuple structure**: Each prediction requires specific fields (KIS: `video_id, frame_id`, QA: `video_id, frame_id, answer`, TRAKE: `video_id, frame_ids`).
- **Strict Integer Fields**: Prediction `rank` ($\ge 1$), prediction `frame_id`/`frame_ids` ($\ge 0$), and GT frame intervals are strict integers (`type(val) is int`). `bool` and `float` types are strictly rejected.
- **Max 100**: The system restricts predictions to a maximum of 100 per query.
- **Task R-Score**:
  - KIS: 1 if video matches and frame is inside GT interval, else 0.
  - QA: 1 if video matches, frame inside interval, and answer semantically matches, else 0.
  - TRAKE: If video matches, `sum(hit) / N` where hit=1 if event's frame is within GT interval.
- **R@1/5/20/50/100**: The maximum R-score among the top K rank.
- **Final Score**: The arithmetic mean of R@1, R@5, R@20, R@50, and R@100.
- **Dataset Evaluation**: Evaluates over the full set of GT queries. GT queries with 0 predictions score 0.0 across all R@K metrics and are included in the dataset mean final score.
- **Task and GT Type Contract**: Task type and ground-truth concrete schema must match.

### LOCAL APPROXIMATION (Evidence Boundary)
- Structural and scoring rules follow published BTC preliminary semantics.
- Q&A semantic answer matching uses deterministic configured aliases because the hidden BTC semantic judging behavior is not published.
- Our local implementation `NormalizedAliasAnswerMatcher` performs unicode-safe, case-folded string equality with optional harmless trailing punctuation stripping against known GT aliases. No LLM or VLM is used for judging equivalence at the metric level.

### PERFORMANCE DECISIONS
- **Phase 4.3C1**: Batch-size 32/64/128 sweep produced no material improvement on Kaggle/T4.
- **Decision**: `image_batch_size=32` remains the default.
- KIS performance optimization is paused until preliminary P0 is complete.

## PRELIMINARY P0-B1: Evidence-Grounded Closed-Set Q&A Baseline Core
- Implemented closed-set baseline Q&A core in `system_tai.qa`.
- Reuses existing KIS retrieval and exact-frame refiner stack without duplicating resources.
- Supports deterministic question families: `COLOR`, `COUNT`, `YES_NO`, `DIRECTION`.
- Open-ended / unsupported questions return zero predictions (`UNSUPPORTED`).
- Enforces evidence-first policy: no evidence candidate or unsupported query produces no predictions.
- Preserves declared evidence candidate rank in output predictions (`p.rank == cand.rank`).
- Optional candidate metadata (`evidence_score`, `timestamp_seconds`) defaults to `None` when unknown, rather than fabricating numeric `0.0`.
- Strict embedding validation in `CosineEvidenceAnswerScorer`: requires float32, 1D, finite, non-zero norm, L2-normalized vectors (`norm ~ 1.0`), and matching dimensions.
- YES_NO questions are scored by baseline CLIP prompts but explicitly flagged with `confidence_level: "EXPERIMENTAL"` in result diagnostics, as pure image CLIP lacks compositional logic.
- Status: COMPLETE.

## PRELIMINARY P0-B2: Shared-Runtime Q&A Vertical Slice
- Connected `system_tai.qa` baseline core to long-lived `OperationalKISRuntime` session engine via `QARuntimePipeline`.
- Reuses existing single instance of `SharedOpenAIClipEncoder`, `FeatureStoreRegistry`, `ExactNumpyRetriever`, `WeightedRRFRetriever`, `RawVideoRegistry`, `OpenCVVideoDecoder`, and `ExactFrameRefiner`.
- Strict **Event-Only Localization Retrieval**: retrieval generates query variants exclusively from `event_description` (and optional `event_description_en`). Question text is strictly excluded from retrieval text encoding to avoid hypothesis confirmation bias.
- Refined absolute frame ID (`refined_frame_id`) is used directly as the output Q&A `frame_id`.
- Single-frame exact decoding: evidence decoder requests strictly `(refined_frame_id,)` and verifies absolute decoded frame ID match.
- Prompts embedding cache (`get_prompt_embeddings`) prevents redundant text encoding for visual prompt banks.
- Unified single session `request_id` namespace shared across `health`, `query` (KIS), `qa_query` (QA), and `shutdown`.
- Independent request directory outputs keyed by `safe_request_directory_name(request.request_id)`. Writes per-request `qa_predictions.jsonl` (EXACT task schema), `qa_evidence.json`, `qa_request_manifest.json`, and `qa_timings.json`.
- Scope & Status: Local vertical slice complete & verified. No private Kaggle run executed yet. No VLM/OCR/ASR. No claim of official JSONL submission format.
