# Tasks: Reverse Batch Order

- [x] **Frontend: Update Active Batch Rendering Position**
  - Move the active batch rendering logic above the finalized batches section in `frontend/app/page.tsx`.
  - Validation: Ensure the active batch is the first thing visible under the dashboard controls.

- [x] **Frontend: Reverse Finalized Batches Order**
  - Modify the mapping logic for `finalizedBatches` to display them from newest to oldest.
  - Validation: Finalize two batches and verify the one finalized last appears above the one finalized first.

- [x] **Frontend: Reposition Divider**
  - Update the divider logic to appear between the top active batch and the list of finalized batches below it.
  - Validation: Verify the dashed divider only appears when both an active batch (with results) and finalized batches exist.
