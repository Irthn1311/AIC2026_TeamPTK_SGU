# ADR-002: GitHub + Kaggle workflow

- **Status:** Accepted.
- **Decision:** Local dùng để code/test nhỏ; GitHub lưu code/config/docs; Kaggle giữ data lớn và
  compute. Notebook không chứa business logic. Mỗi run gắn exact git commit.
- **Consequence:** Run tái lập được mà không đưa binary lớn hoặc secret vào Git.

