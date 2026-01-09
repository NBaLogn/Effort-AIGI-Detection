# Capability: batch-collapse

## MODIFIED Requirements

### Requirement: Upload analysis summary when results available
The frontend SHALL display a concise summary panel that reports how many images have been uploaded and how many were judged real versus fake along with their share of the total, immediately after the first successful analysis response. The panel SHALL include a toggle button to collapse or expand the batch section.

#### Scenario: Summary panel appears with counts
- **WHEN** the user uploads at least one image and receives a prediction
- **THEN** a summary panel becomes visible above the result grid
- **AND** the panel lists the total number of images processed and the counts of images labeled REAL and FAKE
- **AND** the panel highlights the number of real versus fake images as absolute values.

#### Scenario: Percentage mirrors counts
- **GIVEN** a summary panel showing five total results where two were labeled REAL and three were labeled FAKE
- **WHEN** the summary is rendered
- **THEN** it shows REAL = 40% and FAKE = 60% relative to the total, with percentages calculated to a single decimal place and rounded consistently.

#### Scenario: Summary panel includes collapse toggle
- **WHEN** the summary panel is rendered for a batch
- **THEN** it includes a button showing "V" when expanded and ">" when collapsed
- **AND** the button is positioned near the "Upload summary" title.

#### Scenario: Toggling collapse hides stats and results
- **GIVEN** a batch section is currently expanded (icon "V")
- **WHEN** the user clicks the toggle button
- **THEN** the icon changes to ">"
- **AND** the summary statistics (counts and percentages) for that batch are hidden
- **AND** the results grid (images) for that batch is hidden
- **AND** the batch header (timestamp, total count) remains visible.

#### Scenario: Toggling expand reveals stats and results
- **GIVEN** a batch section is currently collapsed (icon ">")
- **WHEN** the user clicks the toggle button
- **THEN** it reverts to the expanded state with icon "V"
- **AND** all stats and images are visible again.
