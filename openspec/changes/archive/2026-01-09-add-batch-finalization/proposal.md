# Change: Introduce batch finalization controls for uploads

## Why
- Users can currently only see a single summary for every uploaded image, which makes it hard to tell when a set of uploads should be considered separate work units or benchmarks.
- We need a lightweight way to mark the existing batch as complete, keep its results visible, and start tracking fresh metrics for the next set of uploads.

## What Changes
- Add an explicit control that lets the user mark the current batch as finished while keeping the existing cards and summary in view.
- Visually group each completed batch with its own summary and a divider so new batches and their metrics clearly start below.
- Reset the live metrics once a batch is sealed so subsequent uploads report counts/percentages independently from prior batches.
- Update the UI/layout (spacing, divider, summary positioning) so the completed batches are pushed up and the current batch feels separated.

## Impact
- Affected specs: frontend-interface
- Affected code: `frontend/app/page.tsx`, `frontend/app/components/ResultSummary.tsx`, `frontend/app/components/ResultsGrid.tsx`, `frontend/app/components/ResultCard.tsx`, new batch metadata helpers/styles as needed
