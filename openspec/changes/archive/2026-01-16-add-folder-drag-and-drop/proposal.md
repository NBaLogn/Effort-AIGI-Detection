# Proposal: Folder Drag and Drop Support

## Problem
Currently, the website allows users to drag and drop multiple images and videos. However, if a user drags a folder containing these files, the application does not process the files within the folder. This limits user productivity when dealing with organized datasets.

## Proposed Solution
Enhance the `Dropzone` component to recursively scan dropped folders for image and video files. This will be implemented using the `DataTransferItem.webkitGetAsEntry()` API, which is the standard way to handle folder drops in modern browsers.

## Goals
- Support dragging and dropping one or more folders.
- Support a mix of files and folders in a single drop.
- Recursively scan subfolders for supported file types (images and videos).
- Maintain the existing functionality for individual file selection and drag-and-drop.
- Update the UI to clearly indicate folder support.

## Non-Goals
- Supporting folder selection via the file picker dialog (unless explicitly requested, but the focus here is drag-and-drop).
- Supporting extremely large folder structures that might cause browser memory issues (will implement a reasonable depth or count limit if necessary).
