# Change: Add upload summary report for judged images

## Why
- Users currently have to infer how many images they submitted and how many were labeled real or fake by scanning result cards, which makes the immediately available signal noisy and hard to digest.
- A lightweight summary will surface the total count and real/fake breakdown so analysts can quickly judge the effectiveness of the detection run before inspecting individual cards.

## What Changes
- Track the number of images processed, how many were labeled REAL vs FAKE, and compute their percentages relative to the latest batch.
- Render a new summary widget above the grid that displays the aggregate counts and percentages once any result is available, keeping the existing grid and hover interactions unchanged.
- Keep the backend API unchanged; all data comes from the existing prediction responses.

## Impact
- Affected specs: frontend-interface
- Affected code: `frontend/app/page.tsx`, `frontend/app/components/ResultsGrid.tsx`, new summary component (e.g., `ResultSummary`), and related styles.
