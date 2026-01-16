# Tasks: Folder Drag and Drop Support

- [ ] Implement recursive file scanning utility function in `Dropzone.tsx` or a separate utility file.
- [ ] Update `handleDrop` in `Dropzone.tsx` to use `DataTransferItem` and `webkitGetAsEntry`.
- [ ] Verify that files from folders are correctly filtered and passed to `onFilesSelected`.
- [ ] Update the `Dropzone` UI text if necessary to confirm folder support.
- [ ] Test with single folder drop.
- [ ] Test with multiple folder drop.
- [ ] Test with mixed files and folders drop.
- [ ] Verify progress bar updates correctly for batch processing of folder contents.
