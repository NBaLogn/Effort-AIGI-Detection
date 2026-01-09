# Capability: frontend-interface

## MODIFIED Requirements

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
