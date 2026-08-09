# Stage 2A Unified Operational Retrieval Runtime

Stage 2A v0.1 integrates frozen components; it does not optimize ranking. The
runtime consumes the Stage 1E language-path contract and fails closed when its
Stage 1 index fingerprint, verified CLIP candidate, model-space status, or
translator revision disagrees with Stage 1A/1B assets.

English queries follow `DIRECT_CLIP`. Vietnamese queries follow
`VI_TO_EN_THEN_CLIP` using the frozen offline OPUS-MT revision, then the same
official OpenAI CLIP ViT-B/32 adapter. The resulting normalized 512-D vector is
searched by the already-existing NumPy exact cosine backend.

Explicit `en` or `vi` is authoritative. `auto` recognizes clear Vietnamese
Unicode and a deliberately small set of clear English lexical patterns.
Ambiguous ASCII—including Vietnamese without diacritics—returns
`LANGUAGE_AMBIGUOUS`; it never silently falls back to English.

The runtime owns lifecycle and orchestration only. CLIP and the Stage 1 index
load once. The translator is lazy-loaded on the first Vietnamese query and then
reused. CPU is the correctness baseline; GPU is optional.

Raw frame ranking remains exact Stage 1 order. The video-grouped output is a
secondary view ordered by each video's first raw frame. KIS exports use
`frame_id=original_frame_idx`; `n` remains diagnostic metadata only.

Known Stage 1E failure modes remain visible: `difficult_01` is a semantic
failure after translation, and `obj_01` shows that translation alone cannot fix
all CLIP semantic failures. Stage 2A does not repair either case.

Acceptance establishes `OPERATIONAL_RETRIEVAL_RUNTIME_READY`. It does not claim
improved Recall@K, KIS score, ranking, object/temporal understanding, or optimal
Vietnamese translation.

