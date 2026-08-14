# Final Acceptance Test Report

Date: 2026-08-02  
Target: AIC 2026 Retrieval Workspace  
Audit viewport: 1440 × 900  
Audit URL: `http://127.0.0.1:4173/`

## Package scripts

| Script | Command | Purpose |
|---|---|---|
| `dev` | `vite` | Start the Vite development server |
| `build` | `tsc -b && vite build` | Run TypeScript project checks and create the production bundle |
| `test` | `vitest run` | Run all automated tests once, without watch mode |
| `test:watch` | `vitest` | Run Vitest in watch mode |
| `lint` | `eslint .` | Lint the project |

Dependencies were already installed and `npm ls --depth=0` completed successfully, so no reinstall was required.

## Commands executed

| Command | Result | Evidence |
|---|---|---|
| `npm ls --depth=0` | PASS | Dependency tree resolved with exit code 0 |
| `npm run lint` | PASS | ESLint completed with 0 errors and 0 warnings |
| `npm test` | PASS | 1 test file passed; 13 of 13 tests passed |
| `npx tsc -b --pretty false` | PASS | TypeScript completed with exit code 0 and no diagnostics |
| `npm run build` | PASS | Vite transformed 35 modules and generated `dist/` successfully |
| Development server check | PASS | Vite listening on `127.0.0.1:4173` |

All four quality gates were rerun after the acceptance fix and remained green.

## Build result

- Result: PASS
- Output: `dist/index.html`, bundled CSS, and bundled JavaScript
- Final JavaScript bundle: approximately 223.11 kB before gzip / 68.98 kB gzip
- Final CSS bundle: approximately 18.72 kB before gzip / 4.57 kB gzip

## Issue found and fixed

- Fixed: the answer-drawer validation summary was shared across task switches. Validation summaries are now stored independently for KIS, Q&A, and TRAKE. A browser recheck confirmed that a validated KIS banner does not appear in Q&A and remains intact when returning to KIS.

## Manual acceptance checklist

### Shared

- [x] PASS — KIS, Q&A, and TRAKE preserved their own query, selected frame/chain, and ranked answer data while switching tabs.
- [x] PASS — At 1440 × 900, document width and scroll width were both 1440; no horizontal browser scrollbar was present.
- [x] PASS — The header remained within y=5–61 and the bottom answer drawer within y=719–895.
- [x] PASS — Search History opened, displayed the KIS search entry, and closed correctly.
- [x] PASS — Interactive mode supported manual candidate/frame/chain selection.
- [x] PASS — Automatic mode automatically selected and verified the best TRAKE chain; Q&A automatic retrieval also populated selected evidence.
- [x] PASS — Task Reset cleared only the active task and restored its expected defaults.
- [x] PASS — Browser console audit returned no warnings or errors.

### KIS

- [x] PASS — Search returned six candidate cards.
- [x] PASS — Selecting `L21_V001` updated the inspector.
- [x] PASS — Selecting filmstrip frame `4598` updated Actual Frame ID to `4598` and timestamp to `00:03:15`.
- [x] PASS — Visual, OCR, ASR, Object, and Metadata tabs each displayed distinct evidence content.
- [x] PASS — Add Answer created one ranked tuple.
- [x] PASS — Re-adding the same `video_id + frame_id` displayed the duplicate rejection and kept one row.
- [x] PASS — Validation reported a valid KIS output.
- [x] PASS — Export was enabled and invoked; automated schema assertion confirmed exactly `video_id,frame_id`.

### Q&A

- [x] PASS — Initial consistency status was `Not evaluated` and evidence confidence was `—`.
- [x] PASS — Add Answer and Export CSV were disabled before evidence selection.
- [x] PASS — Evidence workspace used `overflow-y: auto` and had a 560 px viewport over 873 px of content.
- [x] PASS — Representative frame `4592` fell between displayed neighboring evidence frames `4586` and `4598`.
- [x] PASS — `  A   RED Bus!!! ` was normalized to `a red bus` in the ranked table.
- [x] PASS — `a red bus.` for the same video/frame was rejected as a normalized duplicate.
- [x] PASS — Export was enabled and invoked; automated schema assertion confirmed exactly `video_id,frame_id,answer`.

### TRAKE

- [x] PASS — E1 was edited to `A traveler approaches the bus stop` and retained during retrieval.
- [x] PASS — Candidate ranking displayed `3/3` and `2/3` event coverage values.
- [x] PASS — An out-of-order chain was rejected and Add Chain remained disabled.
- [x] PASS — The complete ordered same-video chain `L21_V001 / 4240 / 4592 / 5012` was verified and accepted.
- [x] PASS — Re-adding the accepted chain displayed the duplicate rejection and kept one row.
- [x] PASS — Export was enabled and invoked; automated schema assertion confirmed exactly `video_id,frame_id_1,frame_id_2,frame_id_3`.

## Screenshots

- `test-artifacts/kis-acceptance.png`
- `test-artifacts/qa-acceptance.png`
- `test-artifacts/trake-acceptance.png`

![KIS acceptance view](test-artifacts/kis-acceptance.png)

![Q&A acceptance view](test-artifacts/qa-acceptance.png)

![TRAKE acceptance view](test-artifacts/trake-acceptance.png)

## Unresolved issues

None found. The in-app browser's download event hook did not surface Blob URL downloads, so CSV column contents were confirmed by the passing automated schema test in addition to manually invoking each enabled Export CSV control.
