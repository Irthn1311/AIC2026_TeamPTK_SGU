# Tổng quan TRIAGE-EG

TRIAGE-EG tách offline preparation khỏi online query processing bằng contract ổn định. v0.1 chỉ
hiện thực Data → Frame Bank → Features → Retrieval → Evaluation; những khối còn lại là roadmap.

Mọi frame đi qua `FrameRecord.actual_frame_id`, giúp retrieval frame, raw video và submission dùng
cùng hệ tọa độ. Mọi artifact có exact config, data version và git commit để tái lập.

