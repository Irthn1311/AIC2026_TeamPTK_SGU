# AIC 2026 Retrieval Workspace

A desktop-first React prototype for the AI Challenge HCMC 2026 Textual KIS, Q&A, and TRAKE retrieval workflows. The interface is intentionally compact and low-fidelity so search, evidence inspection, exact-frame selection, validation, and submission preparation remain the focus.

## Features

- Independent KIS, Q&A, and TRAKE task state when switching tabs
- Interactive and automatic modes plus local search history
- Six realistic mock video candidates with Actual Frame IDs and multimodal evidence
- KIS exact-frame filmstrip and duplicate-safe ranked answer list
- Q&A anchor/evidence interval, representative-frame selection, answer normalization, and consistency gating
- TRAKE editable three-event chain, timeline, same-video/order/gap validation, and duplicate prevention
- Official CSV schemas only: `video_id,frame_id`; `video_id,frame_id,answer`; and `video_id,frame_id_1,frame_id_2,frame_id_3`
- Fixed application shell with independently scrolling evidence and result regions
- Keyboard support for KIS frame arrows and Ctrl+Enter add

## Exact setup and run instructions

Prerequisites: Node.js 20 or newer and npm 10 or newer.

From the `UI_Test` project directory, install the locked dependency versions:

```bash
npm ci
```

Start the development server on the audited local address:

```bash
npm run dev -- --host 127.0.0.1 --port 4173
```

Open `http://127.0.0.1:4173/` and use a 1440 × 900 desktop viewport.

## Quality commands

```bash
npm run lint
npm test
npx tsc -b --pretty false
npm run build
```

`npm test` runs Vitest once in non-watch mode. Use `npm run test:watch` only during active development.

All data is local mock data. There is no backend, authentication, database, or external API.
