# frontend-interface Spec Delta: Reverse Batch Order

## MODIFIED Requirements

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
