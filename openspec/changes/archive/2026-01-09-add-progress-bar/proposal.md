# Proposal: Add Progress Bar for Upload and Inference

## Summary
Add a visual progress bar to the frontend to indicate the status of image uploads and inference processing. This improves user feedback during batch operations, replacing the static "Processing images..." text.

## Background
Currently, when a user uploads a batch of images:
1. The UI shows a simple "Processing images..." textual indicator.
2. The browser processes all requests in parallel (via `Promise.all`).
3. Results appear all at once only after *all* images in the batch are processed.
4. There is no indication of how many images have completed or are remaining.

This leads to a poor user experience, especially for larger batches or when the backend is slow, as the user might think the app has stalled.

## Solution
We will implement a progress monitoring system in the frontend:
1.  **Incremental Updates**: Refactor the upload logic to process requests concurrently but update the UI state as *each* request completes, rather than waiting for the entire batch.
2.  **Progress Bar**: Display a progress bar showing the percentage of completed items (Completed / Total) during the processing phase.
3.  **Status Feedback**: Show a distinct visual state for the batch processing (e.g., "Processing 3 of 5 images...").

## Considerations
-   **Granularity**: We will track progress by "items completed" rather than bytes uploaded, as we are using the `fetch` API which does not support upload progress monitoring easily, and `axios` is not a dependency. Given the typical image size and inference time, item-level granularity is sufficient.
-   **Concurrency**: We will maintain the current behavior of sending requests in parallel (browser-managed limit) but handle their resolution individually.

## Plan
1.  Modify `handleFilesSelected` in `page.tsx` to update state incrementally.
2.  Create a `ProgressBar` component.
3.  Integrate the progress bar into the main page.
