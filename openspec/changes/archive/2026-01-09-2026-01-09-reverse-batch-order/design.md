# Design: Reverse Batch Display Order

## Overview
The goal is to surface the newest batch at the top of the interface. This will be achieved by modifying the rendering logic in `frontend/app/page.tsx`.

## Architectural Reasoning
The current `batches` state is an array where the active batch is the last element (or found by `!isFinalized`).
Instead of changing how batches are stored in the state (which might affect indexing logic in `handleFilesSelected` and `addResult`), we will change how they are rendered.

### Changes to Rendering Logic:
1. **Active Batch**: Move the rendering of the active batch above the finalized batches.
2. **Finalized Batches**: Reverse the array of finalized batches before mapping over it.
3. **Divider**: Reposition the divider to sit between the active batch (top) and the list of finalized batches (below).

## Alternatives Considered
### Prepending to State
We could modify `handleFinalizeBatch` to use `unshift` instead of `push`. However:
- `activeBatchIndex` calculation might become more complex or need to be hardcoded to `0`.
- Existing logic in `addResult` uses `activeBatchIndex` (which is currently dynamic but often points to the last or only unfinalized item).
- Changing rendering is "safer" and achieves the same visual result with less risk to state integrity.

## User Interface Changes
- **Current Layout**: 
  1. Finalized 1
  2. Finalized 2
  3. --- Divider ---
  4. Active Batch
- **New Layout**:
  1. Active Batch
  2. --- Divider ---
  3. Finalized 2 (Newest)
  4. Finalized 1 (Oldest)
