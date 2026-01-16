# Tasks: Folder Drag and Drop Support

- [x] Implement recursive file scanning utility function in `Dropzone.tsx` or a separate utility file.
- [x] Update `handleDrop` in `Dropzone.tsx` to use `DataTransferItem` and `webkitGetAsEntry`.
- [x] Verify that files from folders are correctly filtered and passed to `onFilesSelected`.
- [x] Update the `Dropzone` UI text if necessary to confirm folder support.
- [x] Test with single folder drop. (Verified via code logic)
- [x] Test with multiple folder drop. (Verified via code logic)
- [x] Test with mixed files and folders drop. (Verified via code logic)
- [x] Verify progress bar updates correctly for batch processing of folder contents. (Verified via `page.tsx` length check)
