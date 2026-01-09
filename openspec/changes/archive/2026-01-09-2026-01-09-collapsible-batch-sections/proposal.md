# Proposal: Collapsible Batch Sections

## Why
As users process multiple batches of images, the results page grows long and difficult to navigate. Users need a way to minimize previous or current batch results to focus on specific data while keeping the metrics and results accessible.

## What Changes
- **State Management**: Add `isCollapsed` to the `Batch` interface in `page.tsx`.
- **UI Components**:
    - Update `ResultSummary` to include a toggle button in the heading.
    - Conditionally render the stats and grid based on the collapse state.
- **Interactions**:
    - Clicking the toggle flips `isCollapsed` state for that specific batch.

## Impact
- Affected specs: `frontend-interface`
- Affected code: `frontend/app/page.tsx`, `frontend/app/components/ResultSummary.tsx`, `frontend/app/components/ResultSummary.module.css`

## Verification Plan
### Manual Verification
- Upload a batch of images and verify it starts expanded with a "V" icon.
- Click the "V" icon and verify it changes to ">" and the content collapses.
- Click the ">" icon and verify it changes to "V" and the content expands.
- Finalize the batch and verify the toggle still works for the finalized batch.
- Upload a new batch and verify toggles work independently.
