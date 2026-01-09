# Design: Collapsible Batch Sections

## Architecture
The collapse state will be stored within the `Batch` objects in the main `Home` component's state (`batches`). This ensures that the state persists as long as the session lasts and is correctly associated with each batch.

### Component hierarchy and data flow:
1.  **Home (`page.tsx`)**:
    *   Manages `batches` state array.
    *   Passes `isCollapsed` and `onToggle` to `ResultSummary`.
    *   Conditionally renders `ResultsGrid` based on `isCollapsed`.
2.  **`ResultSummary.tsx`**:
    *   Displays the toggle button in the `heading` area.
    *   Conditionally renders the `stats` div based on the `isCollapsed` prop.
    *   Uses a simple text or SVG icon ("V" and ">") for the toggle.

## Data Structures
The `Batch` interface will be extended:
```typescript
interface Batch {
  id: string;
  timestamp: number;
  isFinalized: boolean;
  results: AnalysisResult[];
  isCollapsed: boolean; // New property
}
```

## UI/UX
- **Icon**: A small, subtle button next to the "Upload summary" title.
- **Transition**: (Optional but recommended) A smooth transition when collapsing/expanding content.
- **Initial State**: All batches start expanded (`isCollapsed: false`) to ensure visibility of new results.

## Alternatives Considered
- **Store collapse state in local component**: Rejected because we need to hide the `ResultsGrid` which is a sibling to `ResultSummary` in `page.tsx`. Lifting the state to `Home` is cleaner.
- **Only collapse finalized batches**: Rejected because users might want to collapse the active batch while it's processing if they focus on previous results.
