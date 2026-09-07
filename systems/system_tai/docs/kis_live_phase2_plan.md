# Implementation Plan: Phase 2 — KIS Live Pipeline & Ground-Truth Evaluator (Audited)

## 1. Git Provenance Verification

```bash
$ git branch --show-current
feat/system-tai-live-p2

$ git rev-parse HEAD
508f041ff642b99e17b389aa83686ab73af66b82

$ git merge-base HEAD "KIS_V2A_RC1_REPLAY_HARDENED^{commit}"
4156dc5af562157d6d2fb1b5d4ac86fc24caf7c6

$ git diff --name-only "KIS_V2A_RC1_REPLAY_HARDENED^{commit}"..HEAD
systems/system_tai/docs/kis_live_phase2_plan.md
```
- **Branch:** `feat/system-tai-live-p2`
- **Base Commit (`branch_base`):** `4156dc5af562157d6d2fb1b5d4ac86fc24caf7c6` (100% matched to `KIS_V2A_RC1_REPLAY_HARDENED`)
- **HEAD Commit:** `508f041ff642b99e17b389aa83686ab73af66b82` (Committed in-tree Phase 2 specification plan)
- **Guardrail:** Replay resources, frozen golden digests, and tag `KIS_V2A_RC1_REPLAY_HARDENED` are strictly preserved and untouched.

---

## 2. Mathematical Definition & Scoring Protocol (`KISFixtureEvaluator`)

### 2.1 Protocol ID & Thresholds
- **Protocol Identifier:** `scoring_protocol_id = "aichallenge-hcmc-2025-mean-topk-rscore"`
- **Evaluation Cutoffs:** $K = (1, 5, 20, 50, 100)$
- **Protocol Lock:** When `scoring_protocol_id` is the official `"aichallenge-hcmc-2025-mean-topk-rscore"`, cutoffs are strictly locked to `(1, 5, 20, 50, 100)` and `max_predictions_per_query = 100`. Custom thresholds require a custom diagnostic protocol ID.

### 2.2 Formal Formula: Mean of Top-$k$ R-Scores
$$R@k = \max_{1 \le i \le \min(k, N)} \text{RScore}(r_i), \quad k \in \{1, 5, 20, 50, 100\}$$
$$\text{FinalScore} = \frac{1}{5} \sum_{k \in \{1, 5, 20, 50, 100\}} R@k$$

For binary Textual-KIS RScore ($\text{RScore} \in \{0, 1\}$ where 1 indicates official localized hit):
| First Localized Hit Rank ($r^*$) | $R@1$ | $R@5$ | $R@20$ | $R@50$ | $R@100$ | Final Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rank 1** | 1 | 1 | 1 | 1 | 1 | **1.0** |
| **Rank 2 – 5** | 0 | 1 | 1 | 1 | 1 | **0.8** |
| **Rank 6 – 20** | 0 | 0 | 1 | 1 | 1 | **0.6** |
| **Rank 21 – 50** | 0 | 0 | 0 | 1 | 1 | **0.4** |
| **Rank 51 – 100** | 0 | 0 | 0 | 0 | 1 | **0.2** |
| **Not in Top 100 / Miss** | 0 | 0 | 0 | 0 | 0 | **0.0** |

*Note:* Implemented directly as the mean over $k \in \{1, 5, 20, 50, 100\}$. Default official mode evaluates binary Textual-KIS RScore ($1.0$ if localized hit, $0.0$ if miss). For experimental or fractional scoring tasks, `KISFixtureEvaluator` accepts an optional `scorer: RScoreScorer | Callable` returning continuous RScore in $[0.0, 1.0]$, which strictly requires specifying a custom `scoring_protocol_id` (the official `DEFAULT_SCORING_PROTOCOL_ID` is protocol-locked against custom scorers).

---

### 2.3 Official Ground Truth Condition: Original Video Frame Index Range (Zero `OR` Ambiguity)
The **official Textual-KIS hit condition** is strictly:
$$\text{Hit}(p, GT) \iff \exists gt \in GT : \left( p.\text{video\_id} == gt.\text{video\_id} \quad \land \quad gt.\text{start\_frame} \le p.\text{frame\_id} \le gt.\text{end\_frame} \right)$$

- `frame_id` **MUST** be the original video frame index used in competition submission (distinct from keyframe ordinal, feature store row, or image filename). Supports boundary `frame_id = 0`.
- **No `OR` between timestamp and frame.**
- If both `timestamp_s` (or runtime `pts_time`) and `frame_id` are provided, they are cross-verified against `FrameTimestampResolver` if provided. If contradictory beyond tolerance, the evaluator raises `ValueError` (fail-closed) rather than masking data corruption.
- **Provisional Timestamp Tolerance:** `DEFAULT_TIMESTAMP_TOLERANCE_S = 0.05` (50ms ~ 1.25 frames at 25 fps or 1.5 frames at 30 fps) as a provisional conservative policy. A 1.0s window is rejected as it permits up to 25-frame drift. Corpus-wide calibration across all competition videos remains pending.
- Timestamp and Segment IoU are strictly diagnostic metrics.
- `point_in_gt_interval`: boolean flag strictly indicating whether the rank 1 (top-1) prediction is a localized hit in any ground truth interval.

---

### 2.4 Data Schemas & Constraints

```python
@dataclass(frozen=True)
class GroundTruthInterval:
    video_id: str
    start_frame: int
    end_frame: int
    start_s: float | None = None
    end_s: float | None = None
    keyframe_id: int | None = None
    query_id: str | None = None

@dataclass(frozen=True)
class PredictionRecord:
    query_id: str
    video_id: str
    frame_id: int            # Original video frame index for submission
    rank: int                # 1-indexed: 1, 2, ..., N (N <= 100)
    score: float | None = None # Optional confidence / fusion score (not strictly monotonic)
    timestamp_s: float | None = None
    keyframe_id: int | None = None
    pred_segment_start_s: float | None = None
    pred_segment_end_s: float | None = None

@dataclass(frozen=True)
class QueryEvaluationResult:
    query_id: str
    scoring_protocol_id: str
    official_final_score: float   # Mean of top-k R-Scores across (1, 5, 20, 50, 100)
    official_first_hit_rank: int | None
    video_first_hit_rank: int | None
    video_mrr: float
    localized_mrr: float
    r_at_k: dict[int, float]
    localized_hit_at_k: dict[int, bool]
    video_hit_at_k: dict[int, bool]
    point_in_gt_interval: bool    # True if rank 1 prediction is a localized hit
    temporal_iou_at_1: float | None
    best_temporal_iou_at_k: dict[int, float | None]
    prediction_count: int
    temporal_cross_check_status: str = "unavailable" # "unavailable", "unresolved", "unresolved-partial", "verified"

@dataclass(frozen=True)
class EvaluationReport:
    scoring_protocol_id: str
    query_results: dict[str, QueryEvaluationResult]
    mean_official_final_score: float
    query_count: int
    missing_query_count: int = 0
    mean_video_mrr: float = 0.0
    mean_localized_mrr: float = 0.0
    mean_video_hit_at_k: dict[int, float] = field(default_factory=dict)
    mean_localized_hit_at_k: dict[int, float] = field(default_factory=dict)
```

#### Prediction Validation Guards:
1. `rank` must be continuous 1-indexed integers: $1, 2, \dots, N$ ($N \le 100$).
2. No duplicate `(video_id, frame_id)` pairs within the same query.
3. Scores optional (`score: float | None = None`); if present, must be finite. Not required to be monotonically non-increasing.
4. Multi-interval support: Localized hit if prediction falls into *any* valid interval for that video.
5. Missing query in predictions evaluated as 0.0; unknown query in predictions fails closed (`ValueError`).
6. Resolver cross-check: if `FrameTimestampResolver` is provided and disagrees outside tolerance, fail closed (`ValueError`). Resolver must implement `FrameTimestampResolver` or be callable; validated at initialization.
7. Alias Consistency: Multiple aliases for coordinates (`frame_id` / `actual_frame_id`, `start_frame` / `target_frame`, `score` / `confidence_score`, `timestamp_s` / `pts_time`, `keyframe_id` / `keyframe_order_diagnostic`) must have identical values; conflicting values raise `ValueError`.
8. Envelope Query ID: Top-level `query_id` in prediction or GT JSON envelope must match all contained records and mapping keys.

---

## 3. Translation Architecture: Official Google Cloud + Pinned VinAI Fallback

1. **Google Cloud Translation API Only (Zero Web Scraping, v3-Only)**:
   - Uses official `google-cloud-translate` v3 client (`v3.TranslationServiceClient`) exclusively; v2 and unofficial scrapers are strictly prohibited and eliminated.
   - ADC / service account credentials from Kaggle Secrets / environment (never committed or logged).
   - Strict fail-closed error sanitation (never leaks credentials, tokens, or absolute paths in exceptions or telemetry).

2. **Pinned VinAI Local Fallback with External Trust Anchor Model**:
   - Pinned canonical revision: `ae7baa85da07dbe8e23ac26a9f5ef560c17e2138` (all-zero or unapproved revisions are strictly rejected).
   - **External Trust Anchor Verification**: Manifest integrity is validated against an external trust anchor (source constant `CANONICAL_MANIFEST_SHA256` or explicit caller-provided `trusted_manifest_sha256`). A local snapshot's self-signed `manifest.json` alone is never trusted.
   - **Storage Layout & Containment**: Full support for Hugging Face cache layouts where snapshot files are symlinked to `../../blobs/<hash>`, while enforcing strict containment within repo bounds. Standalone mirrors require strict snapshot root containment.
   - **Snapshot File Audit**: Scans and rejects untrusted load-relevant files (`.py`, `.exe`, `.so`, `.dll`, unapproved weights) with `[vinai_untrusted_file]`.
   - `local_files_only=True` in production/air-gapped environments (`allow_model_download=False`).
   - Records revision, checksum, and device in provenance telemetry (AGPL-3.0 compliance).

3. **Transparent Cache with Two Explicit Integrity Modes**:
   - `checksum-only`: Detects file corruption only, makes no claims of anti-tampering/forgery protection.
   - `hmac-required`: Requires keyed HMAC authentication using at least 32 bytes of key material from Kaggle Secrets or environment. Fails fast if missing.
   - Strict rejection of conflicting configuration (`require_hmac=True` with `hmac_mode="checksum-only"`).
   - In `cache.get()`, enforces strict mode matching (`entry["integrity_mode"] == self.integrity_mode`).
   - Strict error message sanitation (no echoing of user queries or paths).

4. **Clause/Entity-Level Semantic Sanity Validator**:
   - Represents semantic mentions as `(clause_idx, entity, count, color, spatial, negation, temporal)` to validate fine-grained bindings:
     - Numerical counts and count swaps across entities (e.g. `Hai người... ba con chó` vs `Three people... two dogs`).
     - Dropped counts and cardinal vs ordinal mismatches.
     - Color-entity bindings, dropped colors, and swapped colors without false rejections on multi-color sentences.
     - Spatial orientation swaps across entities (left vs right, top vs bottom).
     - Misplaced negation across entities.
     - Generalized temporal sequence order of entities across clauses.
   - Strict two-tier diagnostic separation: `ERROR` (triggers fallback) vs `WARNING` (stylistic variation / proper noun retention).
   - All 18 canonical benchmark pairs verified passing (18/18).

---

## 4. Configurable Competition Submission Adapter

- `ConfigurableSubmissionAdapter` supporting selectable formats:
  - `csv_ranked_tuples`: `query_id,video_id,frame_id,rank,score`
  - `csv_provisional`: `query_id,video_id,frame_id,confidence_score` (provisional adapter pending official BTC 2026 guidelines)
  - `json_structured`: top-level dictionary per query
- Configurable constraints:
  - `max_rows_per_query = 100`
  - `require_exact_row_count = False` (at most 100 rows, empty queries score 0.0, short lists supported)
  - Validation: unique (video_id, frame_id) per query, valid video IDs, finite scores (not required to be monotonic).
- Fixtures deferred until official BTC 2026 guidelines are announced.

---

## 5. Staged Atomic Commit Roadmap

- **Commit 1 (`evaluator`):** Schema & `KISFixtureEvaluator` with Mean of Top-$k$ R-Scores, multi-interval matching, original frame interval logic, and comprehensive unit tests (boundary, NaN, duplicates, disagreement).
- **Commit 2 (`translation`):** Official `GoogleCloudTranslationProvider` + Pinned `VinAITranslateProvider` + Semantic Sanity Validator.
- **Commit 3 (`profile`):** Profile `kis-v2a-rc1-live` definition and CLI session wiring.
- **Commit 4 (`submission`):** `ConfigurableSubmissionAdapter` with validation.
- **Commit 5 (`benchmark`):** Unseen query ablation runner and SLA profiler (Google-only, VinAI-only, Fallback, Ensemble, VI direct).
