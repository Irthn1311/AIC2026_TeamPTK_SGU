# Mở TRIAGE-EG architecture trong draw.io

## Cách A — Mở file native nhiều trang

1. Mở [draw.io](https://app.diagrams.net/).
2. Chọn **File → Open From → Device**.
3. Chọn `docs/architecture/TRIAGE_EG_Complete_System.drawio`.
4. Dùng thanh page phía dưới để chuyển giữa chín architecture sections.

File `.drawio` là bản native hoàn chỉnh, XML không nén, có `mxfile`, `diagram`, `mxGraphModel`,
`mxCell` và `mxGeometry`. Có thể chỉnh sửa tiếp bằng giao diện draw.io, nhưng thay đổi kiến trúc
chính thức phải được phản ánh lại vào YAML trước khi generate lại.

## Cách B — Import một Mermaid page

1. Mở một diagram mới trong draw.io.
2. Chọn **Arrange → Insert → Advanced → Mermaid**.
3. Mở file tương ứng trong `docs/architecture/mermaid/` và copy toàn bộ nội dung `.mmd`.
4. Dán nội dung rồi chọn **Insert**.
5. Chỉnh layout cục bộ nếu cần cho mục đích trình bày.

`.mmd` dễ review và tái sinh; `.drawio` là bản editable nhiều trang. Trong mọi trường hợp,
`architecture-spec.yaml` mới là source of truth.

