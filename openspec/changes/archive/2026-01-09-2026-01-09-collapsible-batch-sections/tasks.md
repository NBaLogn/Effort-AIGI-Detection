# Tasks: Collapsible Batch Sections

- [x] **Infrastructure Update**
    - [x] Update `Batch` interface in `frontend/app/page.tsx` to include `isCollapsed: boolean`.
    - [x] Update `Home` component to initialize `isCollapsed: false` for new batches.
    - [x] Implement `handleToggleCollapse` in `Home` component.

- [x] **Component Enhancements**
    - [x] Update `ResultSummaryProps` in `frontend/app/components/ResultSummary.tsx` to include `isCollapsed` and `onToggle`.
    - [x] Add the toggle button to `ResultSummary` UI.
    - [x] Update `ResultSummary` to conditionally render the `stats` section.
    - [x] Style the toggle button in `ResultSummary.module.css`.

- [x] **Integration**
    - [x] Pass `isCollapsed` and `onToggle` from `page.tsx` to `ResultSummary`.
    - [x] Wrap `ResultsGrid` in a conditional check for `!isCollapsed` in `page.tsx`.
    - [x] Ensure finalized batches also correctly handle their independent collapse states.

- [x] **Validation**
    - [x] Verify toggle functionality for active batch.
    - [x] Verify toggle functionality for finalized batches.
    - [x] Verify icon change from "V" to ">".
