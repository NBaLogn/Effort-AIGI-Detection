# frontend-interface Specification

## Purpose
TBD - created by archiving change add-upload-summary-report. Update Purpose after archive.
## Requirements
### Requirement: Upload analysis summary when results available
The frontend SHALL display a concise summary panel that reports how many images have been uploaded and how many were judged real versus fake along with their share of the total, immediately after the first successful analysis response.

#### Scenario: Summary panel appears with counts
- **WHEN** the user uploads at least one image and receives a prediction
- **THEN** a summary panel becomes visible above the result grid
- **AND** the panel lists the total number of images processed and the counts of images labeled REAL and FAKE
- **AND** the panel highlights the number of real versus fake images as absolute values.

#### Scenario: Percentage mirrors counts
- **GIVEN** a summary panel showing five total results where two were labeled REAL and three were labeled FAKE
- **WHEN** the summary is rendered
- **THEN** it shows REAL = 40% and FAKE = 60% relative to the total, with percentages calculated to a single decimal place and rounded consistently.

### Requirement: Batch finalization control
The frontend SHALL let the user seal the currently active batch of uploads with a dedicated "Finalize Batch" control so previous results stay visible while new uploads start a fresh batch.

#### Scenario: finalizing a batch inserts a divider
- **WHEN** the user uploads images and receives the first prediction summary for that batch
- **AND** the user clicks the "Finalize Batch" button that becomes enabled as soon as the batch contains one or more results
- **THEN** the interface keeps the finalized batch cards visible but moves them above a horizontal divider (with an optional caption) so the rest of the page is pushed downward
- **AND** the summary above the divider is labeled with the batch number or timestamp to reflect that it is a completed batch
- **AND** the next batch area below the divider shows an empty summary slot ready for new uploads

#### Scenario: new uploads create newer metrics after sealing a batch
- **GIVEN** there is at least one sealed batch whose cards and metrics appear above a divider
- **WHEN** the user uploads one or more new images after sealing the previous batch
- **THEN** the interface treats those uploads as a separate, active batch
- **AND** the currently visible summary counts and percentages are calculated only from the active batch, not from the sealed batches above
- **AND** the completed batches above retain their recorded metrics and are visually separated from the current batch by the divider

### Requirement: Upload and Inference Progress Feedback
The frontend SHALL display a visual progress indicator that reflects the completion status of the current batch of uploads/inferences.

#### Scenario: Progress bar appears during processing
- **WHEN** the user selects files for upload
- **THEN** a progress bar appears indicating 0% progress
- **AND** the "Processing images..." text is replaced or augmented with the progress bar

#### Scenario: Progress updates incrementally
- **GIVEN** a batch of 5 images is being uploaded
- **WHEN** the first image analysis completes successfully or fails
- **THEN** the progress bar updates to show 20% completion (1/5)
- **AND** the result for that image appears in the grid immediately, without waiting for the others

#### Scenario: Progress bar completion
- **WHEN** all images in the batch have been processed (success or failure)
- **THEN** the progress bar disappears
- **AND** the final summary is updated

### Requirement: Incremental Result Display
The frontend SHALL display analysis results as they become available, rather than waiting for the entire batch to finish.

#### Scenario: Results appear one by one
- **GIVEN** a batch of images is processing
- **WHEN** an individual image analysis is finished
- **THEN** its result card is added to the active batch view immediately

