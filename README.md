# AIC2026_TeamPTK_SGU

Repository nền tảng cho đội PTK - SGU tại AI Challenge Ho Chi Minh City 2026.

## TRIAGE-EG

**TRIAGE-EG — Tool-Routed Intelligent Agent with Event-Graph Grounding** là kiến trúc dài hạn
cho ba dạng truy vấn:

- Textual Known-Item Search (KIS): trả về `<video_id>, <frame_id>`.
- Question Answering (Q&A): trả về `<video_id>, <frame_id>, <answer>`.
- Temporal Retrieval and Alignment of Key Events (TRAKE): trả về một video và chuỗi frame sự kiện.

Luồng mục tiêu là: raw video → data audit → unified frame mapping → hybrid frame bank → feature
extraction → multimodal retrieval → video ranking → event graph → temporal alignment → semantic
moment localization → evidence verification → ranked answer list.

## Phạm vi v0.1

Phiên bản này khóa contract và chứng minh pipeline kết nối được cho `common`, `data`, `frame_bank`,
`features`, `retrieval`, và `evaluation`. Dummy shot detector, center-frame selector, deterministic
dummy encoder và NumPy brute-force cosine index đều chạy thật nhưng không đo chất lượng AI.

Event Graph, Agent, semantic moment localizer, Q&A model, VLM, OCR, ASR, backend, frontend, graph
database, microservices, Docker/Kubernetes và DVC đều **Deferred**. Adaptive multiframe chỉ có config
minh họa, chưa được triển khai trong v0.1.

## Cấu trúc

```text
configs/              Cấu hình data, frame bank, feature, retrieval và experiment
src/triage_eg/        Python package theo src-layout
scripts/              CLI dùng argparse và demo end-to-end
tests/                Unit/integration tests và fixture metadata nhỏ
docs/                 Kiến trúc, module contracts, ADR và ownership
notebooks/            Kaggle bootstrap; không chứa business logic
kaggle/               Hướng dẫn và shell bootstrap
```

Không có thư mục dữ liệu thật trong repository. `actual_frame_id` luôn là frame ID của raw video
dùng cho submission; thứ tự keyframe không được thay thế nó.

## Cài đặt và chạy

Yêu cầu Python 3.11 trở lên:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python scripts/demo_pipeline.py --config configs/experiments/exp001_template.yaml
python scripts/evaluate.py --task trake --ground-truth tests/fixtures/sample_trake_ground_truth.json --predictions tests/fixtures/sample_trake_predictions.json
```

Demo tạo artifact tại `artifacts/demo_pipeline/<run_id>/`, gồm manifest, exact config, frame
metadata, feature metadata/vectors và kết quả retrieval. `artifacts/` được Git bỏ qua.

## Workflow local → GitHub → Kaggle

1. Code, lint và test nhỏ ở local trên feature branch.
2. Push code/config/docs lên GitHub qua pull request; `main` phải luôn chạy được.
3. Trên Kaggle, bootstrap notebook clone đúng `COMMIT_SHA`, cài package rồi gọi script trong repo.
4. Data/compute lớn ở Kaggle; mỗi output phải gắn `RunManifest` với exact git commit và config.

Private repository phải lấy token từ Kaggle Secrets. Không in token, nhúng token vào URL, hoặc
commit `.env`, `kaggle.json`, data, video, ảnh, model, checkpoint, feature hay index. Nếu cần fixture
ảnh cực nhỏ, thêm ngoại lệ `.gitignore` thật hẹp và giải thích trong pull request.

## Trạng thái module

| Module | Trạng thái | Ghi chú |
|---|---|---|
| Common contracts, config, run manifest | Template | Contract v0.1 |
| Data audit và frame mapping | Baseline | Chỉ audit metadata |
| Dummy shot + shot center | Baseline | Không decode video |
| Deterministic dummy feature | Template | Không có semantic meaning |
| NumPy cosine retrieval/evaluation | Baseline | Dành cho tập nhỏ |
| Adaptive multiframe | Experimental | Config example, chưa implementation |
| Event Graph, semantic localization, Agent | Deferred | Chỉ mô tả kiến trúc |
| Stable | Chưa có | Chỉ gán sau benchmark và review |

## Roadmap

1. BTC baseline.
2. Team Frame Bank.
3. Feature extraction thật.
4. Retrieval benchmark.
5. Event Graph.
6. Semantic localization.
7. Agent.

Chi tiết extension point nằm trong [future_modules.md](docs/architecture/future_modules.md) và tài
liệu từng module.

