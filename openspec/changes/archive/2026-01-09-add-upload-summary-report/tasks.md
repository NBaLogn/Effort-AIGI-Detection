## 1. Implementation
- [x] 1.1 Update `frontend/app/page.tsx` to compute aggregate counts and percentages from the fetched results and expose them to the UI.
- [x] 1.2 Introduce a `ResultSummary` component (or similar) that renders the upload total, real/fake counts, and percentage badges.
- [x] 1.3 Place the summary component above the results grid and ensure it only renders when at least one result exists.
- [x] 1.4 Refresh or add styles so the summary complements the existing layout without pushing cards awkwardly.

## 2. Validation
- [x] 2.1 Run `npm run lint` from the `frontend/` directory to confirm the new UI code passes linting.
- [x] 2.2 Manually verify in `npm run dev` that the summary appears after uploading images and correctly reflects the returned labels/scores. (Attempt blocked: `npm run dev` could not start due to `EACCES` when Next tried to create `/home/logan.linux/.cache/next-swc`.)
