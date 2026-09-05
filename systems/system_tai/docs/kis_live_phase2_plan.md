# Implementation Plan: Phase 2 — KIS Live Pipeline & Ground-Truth Evaluator (Audited)

## 1. Git Provenance Verification

```bash
$ git branch --show-current
feat/system-tai-live-p2

$ git rev-parse HEAD
4156dc5af562157d6d2fb1b5d4ac86fc24caf7c6

$ git rev-parse "KIS_V2A_RC1_REPLAY_HARDENED^{commit}"
4156dc5af562157d6d2fb1b5d4ac86fc24caf7c6

$ git status --porcelain=v1
(Clean: zero modified tracked files)
```
- **Branch:** `feat/system-tai-live-p2`
- **Base Commit:** `4156dc5af562157d6d2fb1b5d4ac86fc24caf7c6` (100% matched to `KIS_V2A_RC1_REPLAY_HARDENED`)
- **Guardrail:** Replay resources, frozen golden digests, and tag `KIS_V2A_RC1_REPLAY_HARDENED` are strictly preserved and untouched.

---

## 2. Mathematical Definition & Scoring Protocol (`KISFixtureEvaluator`)

### 2.1 Protocol ID & Thresholds
- **Protocol Identifier:** `scoring_protocol_id = "aichallenge-hcmc-2025-mean-topk-rscore"`
- **Evaluation Cutoffs:** $K = (1, 5, 20, 50, 100)$

### 2.2 Formal Formula: Mean of Top-$k$ R-Scores
$$R@k = \max_{1 \le i \le k} \text{RScore}(r_i), \quad k \in \{1, 5, 20, 50, 100\}$$
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

*Note:* Implemented directly as the mean over $k \in \{1, 5, 20, 50, 100\}$ to naturally support fractional R-Score when needed for temporal localization tasks.

---

### 2.3 Official Ground Truth Condition: Original Video Frame Index Range (Zero `OR` Ambiguity)
The **official Textual-KIS hit condition** is strictly:
$$\text{Hit}(p, GT) \iff \exists gt \in GT : \left( p.\text{video\_id} == gt.\text{video\_id} \quad \land \quad gt.\text{start\_frame} \le p.\text{frame\_id} \le gt.\text{end\_frame} \right)$$

- `frame_id` **MUST** be the original video frame index used in competition submission (distinct from keyframe ordinal, feature store row, or image filename).
- **No `OR` between timestamp and frame.**
- If both `timestamp_s` and `frame_id` are provided, they are cross-verified against the corpus mapping. If contradictory, the evaluator raises `ValueError` (fail-closed data disagreement) rather than masking data corruption.
- Timestamp and Segment IoU are strictly diagnostic metrics.

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

@dataclass(frozen=True)
class PredictionRecord:
    query_id: str
    video_id: str
    frame_id: int            # Original video frame index for submission
    rank: int                # 1-indexed: 1, 2, ..., N (N <= 100)
    score: float             # Confidence / fusion score
    timestamp_s: float | None = None
    keyframe_id: int | None = None
    pred_segment_start_s: float | None = None
    pred_segment_end_s: float | None = None

@dataclass(frozen=True)
class QueryEvaluationResult:
    query_id: str
    scoring_protocol_id: str
    official_rscore: float   # Mean of top-k R-Scores across (1, 5, 20, 50, 100)
    official_first_hit_rank: int | None
    video_first_hit_rank: int | None
    r_at_k: dict[int, float]
    localized_hit_at_k: dict[int, bool]
    video_hit_at_k: dict[int, bool]
    temporal_iou_at_1: float
    best_temporal_iou_at_k: dict[int, float]
    prediction_count: int
```

#### Prediction Validation Guards:
1. `rank` must be continuous 1-indexed integers: $1, 2, \dots, N$ ($N \le 100$).
2. No duplicate `(video_id, frame_id)` pairs.
3. Scores must be finite and monotonically non-increasing.
4. Multi-interval support: Localized hit if prediction falls into *any* valid interval for that video.

---

## 3. Translation Architecture: Official Google Cloud + Pinned VinAI Fallback

1. **Google Cloud Translation API Only (Zero Web Scraping)**:
   - Uses `google-cloud-translate` (v3/v2).
   - ADC / service account credentials from Kaggle Secrets / environment (never committed or logged).
2. **Pinned VinAI Local Fallback**:
   - Pinned revision for `vinai/vinai-translate-vi2en-v2`.
   - `local_files_only=True` in production/air-gapped environments.
   - Records revision, checksum, and device in provenance telemetry (AGPL-3.0 compliance).
3. **Semantic Sanity Validator**:
   - Validates numerical counts, negation, left/right polarity, temporal markers ("bắt đầu -> sau đó -> kết thúc"), and entities.
   - Falls back to VinAI if Google fails semantic validation.

---

## 4. Configurable Competition Submission Adapter

- `ConfigurableSubmissionAdapter` supporting selectable formats:
  - `csv_ranked_tuples`: `query_id,video_id,frame_id,rank,score`
  - `csv_btc_standard`: `query_id,video_id,frame_id,confidence_score`
  - `json_structured`: top-level dictionary per query
- Strict constraints: exactly 100 rows per query, unique keyframes, valid video IDs, finite scores.
- Fixtures deferred until official BTC 2026 guidelines are announced.

---

## 5. Staged Atomic Commit Roadmap

- **Commit 1 (`evaluator`):** Schema & `KISFixtureEvaluator` with Mean of Top-$k$ R-Scores, multi-interval matching, original frame interval logic, and comprehensive unit tests (boundary, NaN, duplicates, disagreement).
- **Commit 2 (`translation`):** Official `GoogleCloudTranslationProvider` + Pinned `VinAITranslateProvider` + Semantic Sanity Validator.
- **Commit 3 (`profile`):** Profile `kis-v2a-rc1-live` definition and CLI session wiring.
- **Commit 4 (`submission`):** `ConfigurableSubmissionAdapter` with validation.
- **Commit 5 (`benchmark`):** Unseen query ablation runner and SLA profiler (Google-only, VinAI-only, Fallback, Ensemble, VI direct).
