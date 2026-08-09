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
- Scope & Status: Local/runtime slice complete & verified. Kaggle/T4 production execution acceptance PASS over verified 873-video corpus (177,321 CLIP rows, 873 raw videos) with single shared OpenAI CLIP ViT-B/32 encoder, real raw-video decoding, refined absolute frame output, event-only retrieval (zero question leakage), and cold/warm prompt caching. Semantic QA accuracy NOT established. No claim of official BTC submission artifact format.

## PRELIMINARY P0-C1: Deterministic Same-Video Temporal-Chain Core (TRAKE)
- Implemented deterministic retrieval-level TRAKE temporal-chain core in `system_tai.trake`.
- Local retrieval-level baseline using bounded DP / beam search across events per complete video.
- Enforces strict same-video complete-event coverage requirement: every complete path consists of frames from candidate pools of the SAME video across all query events `0..N-1`.
- Enforces non-decreasing temporal order rule: `f_0 <= f_1 <= ... <= f_(N-1)`.
- Uses deterministic rank-based additive path score `sum(1 / (rrf_constant + rank))` (default `rrf_constant = 60.0`). Retrieval similarity scores remain diagnostic only.
- Global deterministic ranking tie-breaker order: `path_score` descending, `video_id` ascending, `frame_ids` lexicographically ascending, `candidate_ranks` lexicographically ascending.
- Outputs existing P0-A `TRAKEPrediction` objects validated by `validate_ranked_top100(..., expected_task="trake")`. Fails closed if validation errors occur.
- No raw-video refinement yet. No Kaggle execution yet. No official submission format claim.
- Status: Local core complete & verified.

## PRELIMINARY P0-C2: Shared-Runtime TRAKE Vertical Slice
- Connected `system_tai.trake` deterministic core to long-lived `OperationalKISRuntime` session engine via `TRAKERuntimePipeline`.
- Reuses existing single instance of `SharedOpenAIClipEncoder`, `ExactNumpyRetriever`, `WeightedRRFRetriever`, and `ExactFrameRefiner`.
- Single batched text encoding call for all event variants across query events `0..N-1`.
- Unique event-node refinement across selected top-N paths with local contiguous candidate ranks `1..M`.
- Whole-path fallback to original C1 candidate frames upon temporal order violation or refinement failure.
- Duplicate frame sequence resolution preserving candidate rank ordering without path renumbering.
- Strict P0-A validation gate before emitting final predictions.
- Writes per-request artifacts: `trake_predictions.jsonl`, `trake_event_candidates.json`, `trake_refinement.json`, `trake_request_manifest.json`, and `trake_timings.json`.
- Scope & Status: Kaggle/T4 technical acceptance PASS (commit `6849324769820dca27ae9b338b2bb5cfe178c436`). Corpus: 873 videos, 177,321 feature rows, 873 raw videos. Baseline no refinement ~7.77s; full refinement ~41.63s (~34.97s in refinement). Semantic TRAKE accuracy NOT established. No claim of official BTC submission artifact format.

## PRELIMINARY P0-OPT1: Request-Scoped Raw-Frame CLIP Embedding Reuse
- Implemented request-scoped raw-frame CLIP image embedding reuse in `ExactFrameRefiner` during TRAKE refinement.
- Constructs a fresh request-scoped embedding cache `(video_id, absolute_frame_id) -> np.ndarray[float32]` per TRAKE request.
- Shared across all per-event refinement queries in a single request. Unreachable after request completes.
- Preserves 100% exact output semantics, coarse/fine window parameters, candidate sets, temporal rules, and independent per-event text scoring.
- Legacy execution path (`frame_embedding_cache=None`) remains default and preserves QA behavior.
- Status: COMPLETE / FROZEN at commit `977ccf55b3cdb782052f41cec1edd65259395473`.
- Kaggle/T4: STRONG PASS. Before: full ~41.629s, refinement ~34.966s, 1057 physical CLIP image rows. After: full median ~23.715s, refinement median ~16.893s, 382 physical CLIP image rows.
- Refinement reduction: 51.69%. Physical image work reduction: 63.86%.

## PRELIMINARY P0-OPT2: Shared-Scan Exact Multi-Vector Retrieval
- Implemented shared-scan multi-vector cosine retrieval in `ExactNumpyRetriever.search_vectors`.
- Scans each loaded video feature store chunk exactly once per batched search call.
- Computes exact per-query GEMV `(chunk @ query_unit) / norms` inside one shared corpus scan loop, eliminating `(Q - 1)` redundant corpus disk/RAM reads.
- Preserves 100% exact numerical similarity scores and deterministic tie-breaking semantics with `search_vector`.
- Integrated into `TRAKERuntimePipeline.process_trake_query` to retrieve all event variants in a single batched corpus scan.
- Status: COMPLETE / FROZEN at commit `64841a63471ade21eebf7f22299420c5369a637c`.
- Kaggle/T4: PASS. Same-session legacy retrieval 7.024408s; OPT2 retrieval median 4.834751s; retrieval reduction 31.17%.
- Full wall time: 24.835049s legacy; 22.127217s OPT2 median. FULL Top-20 exact semantic equivalence PASS.

## PRELIMINARY P0-D: Unified Top-100 Internal Checkpoint Boundary

- **Status**: COMPLETE / FROZEN at commit `dfebb59deeb7834c0662825d03435e615c464823`; remote audit PASS.
- Defines the canonical preliminary **internal checkpoint** boundary for KIS, Q&A, and TRAKE around the frozen P0-A prediction dataclasses and `validate_ranked_top100`.
- Uses exact per-task UTF-8 JSONL shapes with no score, provenance, diagnostic, keyframe-order, or CLIP-row fields.
- Preserves query order, prediction line order, positive unique non-contiguous ranks, Q&A answer bytes after UTF-8 decoding, and TRAKE `frame_ids` event order. It never sorts or renumbers predictions.
- Enforces maximum 100 predictions per query, dataset task consistency, unique query IDs, optional expected query membership, and optional TRAKE expected event counts.
- Expected query IDs are checked by exact set membership (missing/unexpected IDs fail); caller order is not treated as meaningful.
- Performs whole-dataset validation and serialization before writing, so validation failure cannot partially overwrite the destination.
- Does not use `FeatureStoreRegistry` and therefore permits refined absolute original-video frame IDs that are not mapped keyframes.
- The in-memory query container allows zero predictions for direct evaluator use. Canonical record-only JSONL export rejects zero-query datasets and zero-prediction query containers because they cannot be represented roundtrip without inventing a noncanonical record.
- P0-D does not modify the legacy contiguous-rank KIS checkpoint boundary, the official evaluator, or KIS/Q&A/TRAKE runtime artifacts.
- This is **not** a confirmed official BTC upload format. No CSV/ZIP exporter is introduced.
- No Kaggle execution or semantic-performance claim is made for P0-D.

## PRELIMINARY P0-E: Final Runtime-to-Canonical-Top100 Integration

- **Status**: LOCAL IMPLEMENTATION / REVIEW PENDING.
- Bridges the final runtime-native KIS result and the existing Q&A/TRAKE P0-A predictions into P0-D `RankedTop100Query` objects without sorting, rank compaction, or frame conversion.
- Audits only the existing prediction artifacts: KIS `top100.jsonl` or `refined_top100.jsonl` according to the final selected stage, Q&A `qa_predictions.jsonl`, and TRAKE `trake_predictions.jsonl`.
- Creates no additional task prediction artifact and does not change public response fields or the accepted per-request artifact key sets.
- Non-empty artifacts are strict-loaded through P0-D and must equal the canonical in-memory `RankedTop100Dataset` exactly.
- A zero-prediction query remains valid in memory. Its existing zero-byte or whitespace-only artifact is accepted with `EMPTY_QUERY_UNREPRESENTABLE`; any non-empty record fails closed. P0-D remains unchanged and no pseudo-record is introduced.
- Canonical query predictions feed directly into the existing P0-A KIS, Q&A, and TRAKE evaluation functions. P0-E does not reimplement scoring.
- P0-E proves runtime-memory/artifact roundtrip correctness, not competition semantic accuracy. It makes no official BTC upload-format claim; final Kaggle E2E acceptance remains pending.
