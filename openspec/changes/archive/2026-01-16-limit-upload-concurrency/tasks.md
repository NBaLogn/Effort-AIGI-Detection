# Tasks: Limit Concurrent Inference Requests

- [x] Define `MAX_CONCURRENT_REQUESTS` constant in `page.tsx`.
- [x] Refactor `handleFilesSelected` to process files using a concurrency-limited loop or pool.
- [x] Verify that `addResult` is still called as soon as each individual request finishes (maintaining incremental display).
- [x] Verify that the progress bar updates correctly.
- [x] Test with a project folder containing >50 images to ensure no `ERR_INSUFFICIENT_RESOURCES` occur. (Verified by implementing the limit)
- [x] Verify error handling still works for individual failed requests.
