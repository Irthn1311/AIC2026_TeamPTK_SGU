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

## Internal semantic evaluation policy

For qualitative retrieval ablations, visual candidate comparison, failure
taxonomy, and internal architecture gates, the primary development evaluation
mode is `AI_JUDGED`. AI judgment provenance and status must remain explicit;
AI labels must never be renamed or reported as human labels.

Human review is `ESCALATION_ONLY` during internal development. Reserve it for
the final system freeze, low-confidence or ambiguous AI cases, materially
contradictory evaluations, and a final competition rehearsal when required.
Track `AI_REVIEW_STATUS` and `HUMAN_REVIEW_STATUS` independently.
