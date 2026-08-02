# Đóng góp

Không push trực tiếp lên `main`. Tạo branch có phạm vi rõ, ví dụ `feature/data-audit`,
`feature/frame-bank`, `feature/retrieval-baseline`, `feature/event-graph`, hoặc `docs/ui-research`.

Commit nên nhỏ, mô tả bằng động từ và không trộn refactor ngoài phạm vi. Pull request phải giải
thích mục tiêu, contract bị tác động, cách xác minh và artifact/config liên quan. Trước khi mở PR:

```bash
ruff check .
pytest -q
```

Không commit data, video, ảnh trích xuất, model, checkpoint, feature, index hay secret. Mọi
experiment phải có config; mọi artifact phải có run manifest chứa git commit, config và version.
Notebook chỉ bootstrap/call script. `main` phải luôn cài đặt và chạy demo được.

