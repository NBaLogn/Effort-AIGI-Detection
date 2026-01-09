## ADDED Requirements
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
