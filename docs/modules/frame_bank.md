# Frame Bank

- **Owner:** Phát Lê
- **Status:** Baseline
- **Responsibility:** Chọn retrieval frame, tạo frame mapping và bảo đảm coverage sơ bộ.
- **Non-responsibility:** Chọn semantic submission frame cuối, xây Event Graph, trả lời Q&A.
- **Inputs:** `VideoRecord`, `ShotRecord`, detector/selector config.
- **Outputs:** `ShotRecord`, `FrameRecord`, coverage report.
- **Processing:** v0.1 dummy full-video shot và center-frame policy; không đọc ảnh.
- **Configuration:** `configs/frame_bank/*.yaml`.
- **Failure modes:** Shot khác video, boundary/frame ngoài video, policy chưa implement.
- **Metrics:** Total shots/frames, frames per video, average frames per video.
- **Artifact locations:** `<run>/shots.jsonl`, `<run>/frames.jsonl`.
- **Dependencies:** `common`, `data.frame_mapping`.
- **Open questions:** Ngưỡng shot, coverage benchmark và budget adaptive.

Hybrid Frame Bank gồm BTC Auxiliary Frame Bank làm baseline/fallback, Team Retrieval Frame Bank làm
nhánh nghiên cứu, và raw-video dense refinement ở giai đoạn sau. Semantic submission frame cuối
cùng phải quay lại raw video.

