# Offline pipeline

1. Đọc và audit CSV video manifest, không decode codec trong v0.1.
2. Shot detector tạo boundary; selector tạo retrieval frame metadata.
3. Encoder tạo feature matrix và manifest ánh xạ liên tục theo `feature_row`.
4. Index được build từ vector và `frame_uid`.

Dummy implementations khóa contract để TransNetV2, CLIP/OpenCLIP và FAISS có thể thay thế độc lập.

