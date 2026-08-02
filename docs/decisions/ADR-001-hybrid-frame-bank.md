# ADR-001: Hybrid Frame Bank

- **Status:** Accepted for architecture; partial v0.1 baseline.
- **Decision:** Không bỏ BTC keyframe. BTC là baseline/fallback; Team Frame Bank là nhánh nghiên
  cứu riêng; semantic frame cuối phải quay lại raw video qua unified actual-frame mapping.
- **Consequence:** Đánh giá được coverage từng nhánh và tránh khóa submission vào keyframe thưa.

