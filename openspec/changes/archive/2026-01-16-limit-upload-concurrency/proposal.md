# Proposal: Limit Concurrent Inference Requests

## Problem
When a user drags and drops a folder containing a large number of files, the application attempts to initiate all fetch requests simultaneously via `Promise.all(files.map(...))`. This leads to `net::ERR_INSUFFICIENT_RESOURCES` in the browser because the browser handles a limited number of concurrent connections.

## Proposed Solution
Implement a concurrency-limited execution queue for the inference requests in `page.tsx`. Instead of firing all requests at once, we will process them in smaller batches (e.g., a concurrency limit of 5-10 requests).

## Goals
- Prevent `ERR_INSUFFICIENT_RESOURCES` errors by limiting active fetch requests.
- Maintain the incremental result display and progress bar functionality.
- Ensure the user experience remains responsive during large uploads.

## Non-Goals
- Implementing a full-blown worker queue (a simple async pool in the component is sufficient for now).
- Changing the backend to support batch uploads (still processing files individually but throttled on the client).
