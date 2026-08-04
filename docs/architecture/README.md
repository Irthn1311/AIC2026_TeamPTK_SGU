# TRIAGE-EG Architecture-as-Code

Bộ tài liệu này biến kiến trúc TRIAGE-EG thành artifact có thể review, version và kiểm tra tự động.
Mục tiêu là giữ một mô hình kỹ thuật duy nhất cho pipeline retrieval-first, Candidate Local Event
Graph, Semantic Moment grounding, Agent Control Bus, application interaction và ranked output của
KIS/Q&A/TRAKE. Phiên bản source of truth hiện tại là **1.1**.

## Source of truth

[`architecture-spec.yaml`](architecture-spec.yaml) là nguồn sự thật duy nhất. Chín file Mermaid,
file draw.io native nhiều trang, `generated/architecture_summary.json` và
`generated/architecture_quality_report.md` đều do generator sinh ra.
Không sửa trực tiếp generated assets để thay đổi kiến trúc; thay YAML rồi generate lại.

## Chín architecture pages

1. `00 — Legend & Reading Guide`
2. `01 — Complete System Overview`
3. `02 — Offline Data & Representation`
4. `03 — Online Query & Retrieval`
5. `04 — Event Graph Internals`
6. `05 — Fine Grounding & Verification`
7. `06 — KIS, Q&A & TRAKE Flows`
8. `07 — Agent, Interaction, Output, Evaluation & Deployment`
9. `08 — Contracts, Artifacts & Ownership`

## Generate

```bash
python scripts/generate_architecture_assets.py --spec docs/architecture/architecture-spec.yaml --output-root docs/architecture
```

Hoặc chạy `make architecture`. Generator chỉ cần Python standard library và PyYAML; không gọi
network, Node.js, draw.io CLI hoặc model runtime. Với cùng YAML và geometry, output là deterministic.

## Validate

```bash
python scripts/validate_architecture_assets.py --spec docs/architecture/architecture-spec.yaml --drawio docs/architecture/TRIAGE_EG_Complete_System.drawio
pytest -q tests/architecture
```

Hoặc chạy `make architecture-validate`. Validator kiểm tra taxonomy, ba trường ownership,
page/node/edge IDs, endpoint, model registry, actual Event Graph, layout/aspect/overlap/font/shape,
contract specificity, edge-aligned module interfaces, Mermaid/draw.io drift, XML tooltip, summary
hash và quality report.

## Cập nhật module

1. Sửa hoặc thêm node trong đúng `pages[].nodes`.
2. Cập nhật input, processing, implementations, output, artifact, metric, failure, fallback,
   dependency, next module, geometry và `detail_level`.
3. Khai báo độc lập `criticality`, `criticality_scope`, `maturity`, `source_basis`,
   `architecture_owner`, `implementation_owner` và `reviewers`.
4. Thêm edge với stable ID và `flow_type` hợp lệ.
5. Nếu model/algorithm mới xuất hiện, đăng ký trong `models` trước.
6. Generate lại, validate và review Git diff của YAML, Mermaid, draw.io XML, summary và report.

## Taxonomy v1.1 và model

- Criticality: `CORE`, `CONDITIONAL`, `OPTIONAL`, `DEFERRED`.
- Maturity: `CURRENT_TEMPLATE`, `PLANNED`, `BASELINE`, `EXPERIMENTAL`, `SELECTED`, `VALIDATED`.
- Source basis: `BTC_CONFIRMED`, `TEAM_DECISION`, `RESEARCH_CANDIDATE`, `SOFTWARE_TEMPLATE`.

Không ghi model chưa benchmark thành selected. Registry phân biệt `interface`, `current_template`,
`baseline`, `candidates`, `selected`, `validation_status` và `source_basis`. GNN vẫn deferred và
không thuộc baseline graph solver.

## Giữ các asset đồng bộ

CI hoặc reviewer phải chạy generate trước validate. Validator so sánh Mermaid, draw.io labels,
tooltips, endpoints, summary hash và quality report với YAML; asset stale hoặc bị sửa tay sẽ fail.
Geometry nằm trong YAML để layout draw.io có thể tái sinh ổn định.

## Review bằng GitHub

Ưu tiên review `architecture-spec.yaml` theo module ID, taxonomy, ownership và edge. Sau đó đọc
Mermaid để kiểm tra flow, summary/report để kiểm tra layout/count/status, và draw.io XML để xác
nhận page/native shape/tooltip. Snapshot v1.0 được giữ trong `docs/architecture/archive/`.
