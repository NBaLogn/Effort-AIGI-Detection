# Proposal: Reverse Batch Display Order

## Problem
Currently, new batches are appended to the end of the batch list. As users process multiple batches, the newest (and often most relevant) batch is placed at the bottom of the page, requiring scrolling to see results.

## Proposed Solution
Modify the frontend to display the newest batch at the top of the list. This includes:
1. Reversing the rendering order of finalized batches so the most recently finalized batch appears first.
2. Placing the active (current) batch at the very top of the list.

## Impact
- **User Experience**: Improved visibility of recent results. Users won't need to scroll to find the batch they just finalized or are currently working on.
- **Consistency**: Follows common feed patterns where newest content surfaces at the top.

## Constraints
- Minimal changes to the state management logic to avoid introducing bugs in batch tracking.
- Ensure the divider between active and finalized batches still makes sense in the new order.
