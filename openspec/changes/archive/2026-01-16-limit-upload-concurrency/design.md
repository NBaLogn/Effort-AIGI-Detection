# Design: Limit Concurrent Inference Requests

## Overview
We will implement a simple async pool helper function or a loop with a concurrency limit to process the files. This will replace the direct `Promise.all(files.map(...))` call in `handleFilesSelected`.

## Technical Details

### Concurrency Helper
We can implement a simple function that takes an array of items, a concurrency limit, and an async task.

```typescript
async function asyncPool<T, R>(
    concurrency: number,
    items: T[],
    task: (item: T) => Promise<R>
): Promise<R[]> {
    const results: R[] = [];
    const executing: Promise<void>[] = [];
    for (const item of items) {
        const p = task(item).then((res) => {
            results.push(res);
        });
        executing.push(p);
        if (executing.length >= concurrency) {
            await Promise.race(executing);
            // Remove finished promises
            // (Note: In a real implementation we need to be careful with which one finished)
        }
    }
    await Promise.all(executing);
    return results;
}
```

Actually, a simpler approach for this specific case:
Use a few "worker" promises that consume from a queue of files.

```typescript
const concurrency = 6;
const queue = [...files];
const workers = Array(concurrency).fill(null).map(async () => {
    while (queue.length > 0) {
        const file = queue.shift();
        if (!file) break;
        await processFile(file);
    }
});
await Promise.all(workers);
```

### Implementation in `page.tsx`
- Define a constant `MAX_CONCURRENT_REQUESTS = 6`.
- Refactor the logic inside `handleFilesSelected` to use the worker pattern.
- Ensure `setProgress` and `addResult` are still called correctly.

## UI Implications
The progress bar will still move as each file completes, but the "burst" of network requests will be smoothed out. This will prevent the browser from being overwhelmed.
