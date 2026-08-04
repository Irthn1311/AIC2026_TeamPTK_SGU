# TRIAGE-EG Architecture Quality Report

## Version 1.1 changes

- Split module status into criticality, maturity, and source basis.
- Rebuilt Page 04 as an actual event graph and made frame banks parallel.
- Added application backend, interactive UI, automatic mode, control bus, and three-axis ownership.
- Added deterministic layout, shape, graph, taxonomy, and ownership checks.
- Replaced generic contract placeholders with edge-aligned, module-specific interfaces.

## Page dimensions

| Page | Width | Height | Aspect | Modules | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| PAGE_00 | 2980 | 1040 | 2.865 | 3 | 10 | 5 |
| PAGE_01 | 3940 | 1490 | 2.644 | 21 | 24 | 28 |
| PAGE_02 | 3380 | 1480 | 2.284 | 15 | 28 | 27 |
| PAGE_03 | 3380 | 1410 | 2.397 | 19 | 23 | 29 |
| PAGE_04 | 3360 | 1360 | 2.471 | 4 | 25 | 25 |
| PAGE_05 | 3380 | 1220 | 2.770 | 9 | 14 | 14 |
| PAGE_06 | 3380 | 1720 | 1.965 | 24 | 29 | 27 |
| PAGE_07 | 3360 | 1420 | 2.366 | 15 | 25 | 25 |
| PAGE_08 | 3380 | 1470 | 2.299 | 10 | 25 | 31 |

## Layout warnings

- None.

## Status migration

- Legacy node `status` and single `owner` fields: removed.
- Criticality, maturity, source basis, architecture owner, implementation owner, and reviewers: validated.

## Contract quality

- Specific node contracts: 203.
- Placeholder node contracts: 0.
- Edge-aligned interfaces: 203.

## Event Graph checks

- QueryEvent, Video, SegmentEvent, Entity, SemanticMoment, and external EvidenceRef are present.
- BEFORE, CONTAINS, PARTICIPATES_IN, POSSIBLE_SAME_ENTITY, ANCHORS, SUPPORTS, and MATCH are present.
- Matching produces Event Match Matrix → solver → Top-M Event Chains.

## Task-criticality checks

- Semantic Moment Localizer: CORE for TRAKE, EXPERIMENTAL.
- Q&A Answer Extractor: CORE for Q&A, EXPERIMENTAL.
- Team Visual Encoder: CONDITIONAL; Bounded Agent Planner: OPTIONAL.

## Ownership distribution

- architecture_owner: KHOA=11, PHAT=20, PHUC=28, TRI=202
- implementation_owner: KHOA=13, PHAT=27, PHUC=39, TAI=39, TRI=88
- reviewers: PHAT=2, PHUC=200

## Remaining open questions

- Benchmark and select the Team Visual Encoder.
- Benchmark TransNetV2 and frame-selection strategies.
- Decide the production ANN implementation after benchmark.
- Decide whether optional OCR, ASR, caption, or VLM branches justify their cost.
- Validate Event Graph solver, Semantic Moment Localizer, and Q&A extractor on official task data.
