# Design: Folder Drag and Drop Support

## Overview
The folder drag-and-drop feature requires switching from `DataTransfer.files` to `DataTransfer.items` in the `handleDrop` event. `DataTransfer.items` provides access to `DataTransferItem`, which can be converted to a `FileSystemEntry` via `webkitGetAsEntry()`. 

## Technical Details

### Recursive Scanning
We need a recursive function to traverse the `FileSystemEntry` tree:
- If it's a file (`isFile`), check the extension/type and add it to the list.
- If it's a directory (`isDirectory`), create a `FileSystemDirectoryReader` and read all its entries recursively.

### Implementation in `Dropzone.tsx`
The `handleDrop` function will be updated to:
1. Iterate over `e.dataTransfer.items`.
2. For each item, call `item.webkitGetAsEntry()`.
3. Collect all files from these entries (recursively for directories).
4. Filter files by supported mime types (`image/*`, `video/*`).
5. Call `onFilesSelected` with the flattened list of files.

### UI Changes
- Update the text in `Dropzone.tsx` to explicitly mention folder support (though it already says "or folders here").
- Ensure the progress bar (implemented in a previous change) correctly handles the total count of files found in folders.

## Considerations
- **Performance**: Recursive scanning of very large folders might take time. Since this is client-side, it should be fast enough for typical user folders.
- **Browser Compatibility**: `webkitGetAsEntry` is widely supported in modern browsers (Chrome, Firefox, Safari, Edge).
- **File Types**: We will continue to filter for `image/*` and `video/*`.

## Alternatives Considered
- **HTML5 `directory` attribute**: This only works for the file picker (`<input type="file" webkitdirectory />`), not for drag-and-drop. We may add this as well for consistency.
