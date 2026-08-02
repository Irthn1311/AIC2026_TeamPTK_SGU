# Data

- **Owner:** Nguyễn Đăng Khoa
- **Status:** Baseline
- **Responsibility:** CSV manifest, metadata audit, relative-path validation, unified frame mapping.
- **Non-responsibility:** Decode video, shot detection, feature extraction, data download.
- **Inputs:** `VideoRecord`, CSV manifest, configured data root.
- **Outputs:** Validated records và `DataAuditReport`.
- **Processing:** Standard-library CSV parsing; kiểm tra ID/path và tổng hợp metadata.
- **Configuration:** `configs/data/*.yaml`.
- **Failure modes:** Thiếu env/input, schema sai, duplicate ID, path thiếu hoặc không relative.
- **Metrics:** Số video/frame/duration, missing paths, duplicate IDs, invalid records.
- **Artifact locations:** Theo run directory; data thật nằm ngoài Git.
- **Dependencies:** `common` và standard library.
- **Open questions:** Dataset manifest chính thức, codec audit và checksum policy.

