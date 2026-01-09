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

