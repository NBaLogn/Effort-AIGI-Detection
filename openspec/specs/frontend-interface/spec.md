# frontend-interface Specification

## Purpose
TBD - created by archiving change add-upload-summary-report. Update Purpose after archive.
## Requirements
### Requirement: Upload analysis summary when results available
The frontend SHALL display a concise summary panel that reports how many images have been uploaded and how many were judged real versus fake along with their share of the total, immediately after the first successful analysis response. The panel SHALL include a stylish toggle button positioned before the title to collapse or expand the batch section.

#### Scenario: Summary panel includes collapse toggle
- **WHEN** the summary panel is rendered for a batch
- **THEN** it includes a stylish SVG icon toggle positioned before the "Upload summary" title
- **AND** the icon indicates "expanded" or "collapsed" state through rotation or shape change.

#### Scenario: Toggling collapse hides stats and results
- **GIVEN** a batch section is currently expanded
- **WHEN** the user clicks the toggle button
- **THEN** the icon rotates or changes to indicate collapsed state
- **AND** the summary statistics (counts and percentages) for that batch are hidden
- **AND** the results grid (images) for that batch is hidden
- **AND** the batch header (timestamp, total count) remains visible.

### Requirement: Batch finalization control
The frontend SHALL let the user seal the currently active batch of uploads with a dedicated "Finalize Batch" control so previous results stay visible while new uploads start a fresh batch. The interface shall prioritize the active batch by displaying it at the top of the view.

#### Scenario: finalizing a batch inserts a divider below the active area
- **WHEN** the user uploads images and receives the first prediction summary for that batch
- **AND** the user clicks the "Finalize Batch" button that becomes enabled as soon as the batch contains one or more results
- **THEN** the interface keeps the finalized batch results visible but moves them *below* a horizontal divider
- **AND** the summary below the divider is labeled with the batch completion timestamp
- **AND** the newest active batch area remains at the *top* of the results section, ready for new uploads
- **AND** finalized batches are displayed in reverse chronological order (most recently finalized first)

#### Scenario: new uploads create newer metrics after sealing a batch
- **GIVEN** there is at least one sealed batch whose results appear below a divider
- **WHEN** the user uploads one or more new images after sealing the previous batch
- **THEN** the interface treats those uploads as a separate, active batch rendered at the top of the results section
- **AND** the currently visible summary counts and percentages at the top are calculated only from the active batch
- **AND** the completed batches below retain their recorded metrics and are visually separated from the current batch by the divider

### Requirement: Upload and Inference Progress Feedback
The frontend SHALL display a visual progress indicator that reflects the completion status of the current batch of uploads/inferences.

#### Scenario: Throttled processing for large batches
- **GIVEN** a user drops a folder containing 100 images
- **WHEN** the processing starts
- **THEN** the application SHALL limit the number of simultaneous network requests to a maximum of 6 (or browser-safe limit)
- **AND** the progress bar SHALL accurately reflect the total count (e.g., 1/100, 2/100...) as each request completes
- **AND** the results SHALL continue to appear incrementally in the grid.

### Requirement: Incremental Result Display
The frontend SHALL display analysis results as they become available, rather than waiting for the entire batch to finish.

#### Scenario: Results appear one by one
- **GIVEN** a batch of images is processing
- **WHEN** an individual image analysis is finished
- **THEN** its result card is added to the active batch view immediately

