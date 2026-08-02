# Features

- **Owner:** Phát Lê
- **Status:** Template
- **Responsibility:** Encoder contract, feature extraction và row-to-frame manifest.
- **Non-responsibility:** Frame selection, retrieval ranking, model training.
- **Inputs:** `FrameRecord`, encoder config.
- **Outputs:** `vectors.npy`, `feature_manifest.jsonl`.
- **Processing:** Dummy hash deterministic, float32 và L2 normalization trong v0.1.
- **Configuration:** `configs/features/dummy_encoder.yaml`.
- **Failure modes:** Dimension/row count sai, zero dimension, store thiếu file.
- **Metrics:** Vector count, dimension, normalization flag.
- **Artifact locations:** `<run>/features/`.
- **Dependencies:** NumPy và `common`.
- **Open questions:** CLIP/OpenCLIP model, batching, precision và image preprocessing.

