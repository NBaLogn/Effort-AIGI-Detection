# frontend-interface Specification

## MODIFIED Requirements

### Requirement: Upload and Inference Progress Feedback
The frontend SHALL display a visual progress indicator that reflects the completion status of the current batch of uploads/inferences.

#### Scenario: Throttled processing for large batches
- **GIVEN** a user drops a folder containing 100 images
- **WHEN** the processing starts
- **THEN** the application SHALL limit the number of simultaneous network requests to a maximum of 6 (or browser-safe limit)
- **AND** the progress bar SHALL accurately reflect the total count (e.g., 1/100, 2/100...) as each request completes
- **AND** the results SHALL continue to appear incrementally in the grid.
