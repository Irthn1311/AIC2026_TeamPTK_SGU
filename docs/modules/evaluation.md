# Evaluation

- **Owner:** Hồng Phúc
- **Status:** Baseline
- **Responsibility:** Mô phỏng R-Score KIS/Q&A/TRAKE và ranked Final Score.
- **Non-responsibility:** Khẳng định metric chính thức khi rule thay đổi, semantic answer judging.
- **Inputs:** Ranked JSON predictions và ground truth intervals.
- **Outputs:** R@1/5/20/50/100 và Final Score.
- **Processing:** Inclusive interval checks; Q&A exact normalized matcher có thể thay thế.
- **Configuration:** CLI task/file arguments.
- **Failure modes:** JSON/schema sai, TRAKE frame count khác event count.
- **Metrics:** Per-query R-Score at k và macro average.
- **Artifact locations:** CLI stdout hoặc report path do caller chọn.
- **Dependencies:** Standard library.
- **Open questions:** Official 2026 rules và semantic answer matcher.

