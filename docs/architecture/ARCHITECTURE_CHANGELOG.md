# Architecture Changelog

## Version 1.1

- Tách criticality, maturity và source basis.
- Sửa trạng thái Semantic Moment Localizer.
- Sửa trạng thái Q&A Answer Extractor.
- Vẽ BTC và Team Frame Bank thành hai branch song song.
- Thiết kế lại Event Graph bằng actual nodes và edges.
- Bổ sung PARTICIPATES_IN.
- Làm rõ EvidenceRef không phải full graph node.
- Thu gọn module card.
- Giảm aspect ratio các page.
- Thiết kế Agent Control Bus.
- Bổ sung Application Backend.
- Bổ sung Interactive UI.
- Bổ sung Automatic Mode.
- Phân tách architecture owner, implementation owner và reviewer.
- Bổ sung layout validation.
- Thay placeholder input/output/implementation bằng 203 contract cụ thể và bám đúng edge.
- Bổ sung kiểm tra generated-asset drift và source hash để validator không đọc nhầm asset bundle.
- Giữ GNN ở trạng thái deferred.

## Version 1.0

- Chuyển kiến trúc TRIAGE-EG sang architecture-as-code.
- Bổ sung ba task KIS, Q&A và TRAKE.
- Bổ sung Hybrid Frame Bank.
- Giữ BTC Frame Bank làm baseline/fallback.
- Bổ sung Team Retrieval Frame Bank.
- Bổ sung Candidate Video Ranker.
- Bổ sung Semantic Moment Localizer.
- Bổ sung Ranked Answer List Constructor.
- Xác định rõ Agent Control Plane.
- Tách Event Graph khỏi toàn corpus retrieval.
- Giữ GNN ở trạng thái deferred.
