# Shared Output Boundary

## Status

UTF-8 JSONL is the current proposed team checkpoint format. The shared KIS core fields
are resolved as `query_id`, `rank`, `video_id`, and `frame_id`. Optional envelope and
version fields remain open. Serialization must stay behind an exporter adapter, and an
accepted schema revision will override this proposal.

## Coordinate alias

Within `system_tai`:

`actual_frame_id = frame_idx from the BTC map-keyframes CSV`

At a shared boundary:

`frame_id = actual_frame_id = original_frame_idx`

Preserve `frame_idx` exactly without adding or subtracting one. Zero-based raw-video
bounds and the visual rounding explanation are verified for `L21_V001`; dataset-wide
confirmation remains pending. Decimal floor, binary-float truncation, and Decimal
nearest are diagnostics only. The diagnostic `keyframe_visual_frame_id` must not change
the numeric coordinate at export. Keyframe order `n`, CLIP row,
`local_frame_idx`, filename number, and resampled coordinates are forbidden substitutes.

For self-extracted frames, `frame_id` is the position in the original BTC video
coordinate system.

## Three output layers

1. Internal records may contain scores, confidence, timestamps, evidence, feature rows,
   and version metadata.
2. Team checkpoints use the accepted shared schema; the current proposal is UTF-8 JSONL.
3. Official BTC submission is produced by a separate task-specific exporter.

Do not assume layers 2 and 3 use the same fields or file format.

## Proposed KIS checkpoint record

```json
{"query_id":"Q001","rank":1,"video_id":"L21_V001","frame_id":411}
```

Proposed required fields:

| Field | Type | Meaning |
|---|---|---|
| `query_id` | string | Shared query identifier |
| `rank` | integer | One-based rank within the query |
| `video_id` | string | Exact BTC video identifier |
| `frame_id` | integer | Original frame index in the original BTC video |

`schema_version`, envelope shape, `system_id`, `run_id`, dataset, and mapping metadata
remain unresolved and must not be made mandatory without team approval.

## Proposed validation

- UTF-8, one JSON object per non-empty line.
- Required fields and types are valid.
- Query and video identifiers exist in authoritative catalogs.
- `frame_id` is within authoritative original-video bounds.
- Ranks are positive, ordered, and non-duplicated per query.
- `(query_id, video_id, frame_id)` is unique.
- No query has more than 100 predictions.

## Official boundary

The future flow is:

proposed/accepted team checkpoint
→ official-schema exporter
→ official BTC submission artifact.

Do not implement or document a fixed CSV shape until the official format is accepted.
Internal scores, evidence, timestamps, provenance, keyframe order, and CLIP row must not
leak into official output unless the accepted contract explicitly requires them.

## Open decisions

- Optional JSONL envelope and version fields.
- Required provenance metadata outside the core record.
- Official BTC format.
