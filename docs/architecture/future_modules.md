# Future modules

Các module sau chỉ là thiết kế, không có source code v0.1:

- **Event Graph:** graph tạm thời trên candidate events, quan hệ entity/action/time và evidence.
- **Agent:** route công cụ theo task KIS/Q&A/TRAKE và kiểm soát verification.
- **Semantic Moment Localizer:** quay lại raw video quanh candidate để chọn submission frame cuối.
- **VLM/OCR/ASR/Q&A:** các evidence branch có version/config riêng.
- **Serving/UI:** backend, frontend và observability sau khi offline contract ổn định.

TransNetV2 sẽ implement `ShotDetector`; CLIP/OpenCLIP implement `MultimodalEncoder`; FAISS implement
`VectorIndex`. Event Graph nhận `CandidateVideo`/`CandidateFrame`, không thay đổi retrieval core.

