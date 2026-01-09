## 1. Implementation
- [x] 1.1 Extend `Home` page state so uploads are tracked as batches with metadata for counts, timestamps, and completion state.
- [x] 1.2 Add a "Finalize Batch" control near the upload area that becomes enabled when an open batch has at least one result and, when clicked, marks it as complete and anchors it in the UI.
- [x] 1.3 Update the summary and grid rendering so completed batches remain visible above a visual divider and the active batch renders below with its own computed metrics.
- [x] 1.4 Adjust `ResultSummary`/`ResultsGrid` helpers or styles as needed so each batch can render its own summary header, divider, and spacing without affecting other batches.
- [x] 1.5 Add any small helpers or styles (e.g., `BatchDivider`, spacing utilities) required to keep the layout tidy.

## 2. Validation
- [x] 2.1 Run `npm run lint` inside `frontend` to ensure the new code passes static analysis.
- [x] 2.2 Manually test via `npm run dev` and the UI: upload images, finalize a batch, verify the divider appears, then upload more images to confirm the summary resets and new results appear below the divider with separate metrics.