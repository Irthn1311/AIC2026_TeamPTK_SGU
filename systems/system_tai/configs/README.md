# System Configuration Profiles

Thư mục này chứa các cấu hình chuẩn cho việc vận hành và triển khai hệ thống:

- **`production.yaml`**: Cấu hình chính thức duy nhất dùng cho thi đấu thực tế và đánh giá Benchmark. Đóng băng các siêu tham số tối ưu nhất cho KIS, Q&A (Top-100 Constructor, Visual Ontology, OCR, Whisper ASR) và TRAKE (Video-First Beam Solver).
- **`kis_vertical_slice.example.yaml`**: Cấu hình mẫu cho nhánh KIS Vertical Slice.
- **`phase_1_5_kaggle.example.yaml`**: Cấu hình mẫu cho môi trường Kaggle.

*(Lưu ý: Thư mục `config/` chứa các file cấu hình benchmark / contest thử nghiệm của các giai đoạn phát triển trước).*
