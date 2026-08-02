# Retrieval

- **Owner:** Hồng Phúc
- **Status:** Baseline
- **Responsibility:** Vector search, candidate mapping, grouping, RRF và video ranking.
- **Non-responsibility:** Semantic localization cuối, Event Graph, Q&A generation.
- **Inputs:** Query text, vector matrix/IDs, frame mapping.
- **Outputs:** `CandidateFrame`, `CandidateVideo`.
- **Processing:** Exact NumPy cosine search; optional dedup, fusion và aggregation.
- **Configuration:** `configs/retrieval/numpy_baseline.yaml`.
- **Failure modes:** Index chưa build, dimension sai, unknown frame UID, invalid top-k.
- **Metrics:** Rank/score, recall diagnostics, candidates/video.
- **Artifact locations:** `<run>/retrieval_results.jsonl`.
- **Dependencies:** NumPy, `features` và `common`.
- **Open questions:** FAISS index type, fusion calibration, latency/memory targets.

