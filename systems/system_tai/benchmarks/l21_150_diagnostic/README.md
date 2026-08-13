# L21-150 Diagnostic Benchmark

This namespace contains a deterministic import of 150 internal development queries:

- 50 Textual KIS;
- 50 Q&A;
- 50 TRAKE;
- 16 L21 videos.

`benchmark_id` is `system_tai-l21-150-diagnostic-v1` and `benchmark_role` is
`DIAGNOSTIC_DEVELOPMENT`. This is not BTC official ground truth. It must not be merged
into the independent Q1-B human-verified benchmark.

## Ground-truth limitations

Raw video is the original-frame coordinate source of truth. A BTC mapping row describes
one sparsely supplied keyframe: its `frame_idx` is the raw/original frame ID of that
keyframe, while `n`, CLIP row, and filename are not frame IDs. Absence of a supplied
keyframe inside a source-proposed interval therefore does not invalidate the raw-frame
coordinate. The validator never adds or subtracts one and never rewrites
`benchmark.json`.

Schema-v2 validation separates raw-frame coordinate/resource status from supplied
keyframe proximity. `OUTSIDE_GT_INTERVAL` is diagnostic only. Coordinate validation
uses raw-video bounds when requested and a +/-1-second timestamp/center tolerance because
the source timestamps have whole-second precision. It establishes structural usability,
not semantic correctness or BTC official ground truth.

The source document was assembled from manually inspected video frames. Its KIS/Q&A
frame centers and ±1-second intervals are proposed references until mapping validation.
TRAKE uses ±4 frames around each source semantic keyframe. `frame_idx` from the BTC
mapping CSV remains the authoritative `actual_frame_id`; the validator never adds or
subtracts one and never rewrites `benchmark.json`. No ASR transcript ground truth is
fabricated.

Evaluation must select one explicit policy:

- `proposed`: diagnostic scoring over `SOURCE_PROPOSED_GT`, prominently unverified;
- `validated-only`: score only queries whose required raw-frame coordinate/resource
  records are all `VALIDATED`; keyframe overlap is not a gate. Schema-v1 mapping reports
  must be regenerated because they used the obsolete overlap-based meaning.

## Frozen video-level split

Videos are sorted by `sha256("system_tai_l21_150_v1|" + video_id)`. The first 12 are
DEV and the final four are HOLDOUT. Query text, task, GT, model output, and retrieval
results do not influence the split. Every query from a video inherits that video's
split. Do not tune on HOLDOUT.

## Scoring

K values are `[1, 5, 20, 50, 100]`. KIS requires the correct video and a frame inside
the inclusive interval. Q&A additionally requires an accepted answer after conservative
deterministic normalization; a correct answer with wrong grounding scores zero. TRAKE
uses corresponding event indexes and scores `matched events / required events`; it never
reorders predicted events. Its diagnostic chain-order check is strict
`F1 < F2 < ... < FN`. R@k is the maximum score in the ranked prefix, and Final Score is
the mean of the five R@k values.

## Workflow

1. Re-import the untracked source DOCX with `scripts/l21_150_import.py`.
2. Produce schema-v2 raw-coordinate and keyframe-proximity evidence with
   `scripts/l21_150_validate_mapping.py`.
3. Run the current baseline through `scripts/l21_150_run_baseline.py` on Kaggle/T4.
4. Evaluate with `scripts/l21_150_evaluate.py` and an explicit GT policy.
5. Generate mechanical error reports with `scripts/l21_150_error_analysis.py`.

Generated experiment outputs remain outside Git. The baseline runner calls existing
system runtime contracts and intentionally does not change retrieval, Q&A, TRAKE, or
ranking behavior. Current limitations, including restricted Q&A answer types, incomplete
design-level TRAKE gap constraints, and partial OCR/ASR/Object/BM25 runtime integration,
must remain visible in E0 rather than being hidden or repaired here.

E0 should use coordinate-validated source labels rather than the obsolete
nearest-keyframe-overlap gate. These remain source-proposed internal diagnostic labels,
not independently verified semantic labels or official BTC ground truth.

The DOCX exposes one combined `Mô tả sự kiện + câu hỏi` cell for each Q&A row rather
than two independently authored strings. `benchmark.json` preserves that cell verbatim
as `question_vi`. The E0 adapter passes the same source text to the current runtime's
event-description and question fields instead of inventing a missing event description;
this adapter limitation must be considered when interpreting Q&A diagnostics.

## TR-A2-D0 nomination diagnostic

`tr_a2_d0_trake_dev_en_translation.json` is a model-authored, frozen, literal English
translation of the 114 event descriptions belonging to the 38 DEV TRAKE queries. It was
authored from `source_vi` only, without target videos, temporal labels, or retrieval
feedback, and is not human-reviewed. The strict loader rejects HOLDOUT IDs, missing or
duplicate events, source-text drift, duplicate JSON keys, and benchmark SHA mismatch.

The opt-in `--trake-nomination-only` runner executes text encoding, full-corpus M=1 video
maxima, event-level rank fusion, and the existing multi-event nomination order, then
stops. It never runs selected-video keyframe search, the planner, or raw refinement.
The runtime artifact contains a complete target-agnostic video ranking. Target video IDs
are joined only later by `scripts/l21_150_compare_trake_d0.py`, whose report explicitly
marks `GT_USED_OFFLINE_ONLY=true` and `GT_USED_IN_RUNTIME=false`.

Run the three arms separately with `--trake-language-policy vi_only`,
`vi_plus_en_weighted_rrf`, and `en_only`. The two English-dependent arms require
`--trake-dev-en-sidecar`; VI_ONLY rejects that argument. The comparator reports
Recall@32/50/100, rank buckets, nearest-rank percentiles, per-event target-video ranks,
and transparent language/cap opportunity labels. It does not alter the production cap
or declare a production winner.

## QA-D0 localization-language diagnostic

`qa_dev_translations_en.json` contains one frozen literal English translation of
`question_vi` for each of the 38 DEV QA queries. Translation authoring used only that
Vietnamese question: no target video, answer, frame, interval, timestamp, object
artifact, or retrieval output. The artifact contains no HOLDOUT entries, is not
official ground truth, and is validated against the frozen benchmark bytes and exact
DEV query IDs.

`scripts/qa_d0_language_nomination.py` measures only whether localization language puts
the target video into the existing QA-A1 nomination ranking. The `vi_only`,
`vi_plus_en`, and `en_only` arms construct genuine language variants and reuse
`search_video_maxima()` plus `nominate_qa_videos()`; EN_ONLY never creates or retrieves
a fake Vietnamese variant. Because L21-150 has no separate QA event description, this
remains a `QUESTION_AS_LOCALIZATION_FALLBACK` diagnostic.

Retrieval receives only query ID and localization text. It finishes for all queries
before a separate offline evaluator joins DEV query IDs to target videos. The diagnostic
does not run answer classification/providers, selected-video frame search, raw decode,
refinement, image encoding, or answer scoring. It does not change the default QA
production language policy or establish official/semantic QA accuracy.
