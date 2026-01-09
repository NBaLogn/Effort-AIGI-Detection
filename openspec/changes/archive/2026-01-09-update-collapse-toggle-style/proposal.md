# Proposal: Update Collapse Toggle Style

## Why
The current "V" and ">" text based toggle is functional but doesn't meet the premium aesthetic goals of the project. Moving the icon before the title and using a stylish SVG will improve visual hierarchy and professional feel.

## What Changes
- **Icon Update**: Replace text "V" and ">" with modern, sleek SVG chevron icons.
- **Positioning**: Move the toggle button from after the "Upload summary" title to before it.
- **Styling**: Enhance the button animations (rotation) and hover states for a more "premium" feel.

## Impact
- Affected specs: `frontend-interface`
- Affected code: `frontend/app/components/ResultSummary.tsx`, `frontend/app/components/ResultSummary.module.css`

## Verification Plan

### Manual Verification
- Verify the chevron icon is displayed before the "Upload summary" title.
- Verify the icon rotates smoothly when toggling.
- Verify the hover effect is subtle and high-quality.
