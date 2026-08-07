# PRELIMINARY P0

## PRELIMINARY P0-A: Official published task structure/scoring implementation

This phase implements the exact preliminary task schemas and evaluation semantics for the AIC 2026 Preliminary round.

### PUBLISHED BTC RULE
- **Tuple structure**: Each prediction requires specific fields (KIS: `video_id, frame_id`, QA: `video_id, frame_id, answer`, TRAKE: `video_id, frame_id_1..n`).
- **Max 100**: The system restricts predictions to a maximum of 100 per query.
- **Task R-Score**:
  - KIS: 1 if video matches and frame is inside GT interval, else 0.
  - QA: 1 if video matches, frame inside interval, and answer semantically matches, else 0.
  - TRAKE: If video matches, `sum(hit) / N` where hit=1 if event's frame is within GT interval.
- **R@1/5/20/50/100**: The maximum R-score among the top K rank.
- **Final Score**: The arithmetic mean of R@1, R@5, R@20, R@50, and R@100.

### LOCAL APPROXIMATION (Evidence Boundary)
- Q&A semantic answer matching uses deterministic configured aliases because the hidden BTC semantic judging behavior is not published.
- Our local implementation `NormalizedAliasAnswerMatcher` performs unicode-safe, case-folded string equality with optional punctuation stripping against known GT aliases. No LLM or VLM is used for judging equivalence at the metric level.

### PERFORMANCE DECISIONS
- **Phase 4.3C1**: Batch-size 32/64/128 sweep produced no material improvement on Kaggle/T4.
- **Decision**: `image_batch_size=32` remains the default.
- KIS performance optimization is paused until preliminary P0 is complete.
