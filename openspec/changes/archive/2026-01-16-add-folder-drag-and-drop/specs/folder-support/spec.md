# folder-support Specification

## ADDED Requirements

### Requirement: Recursive folder drag and drop support
The frontend SHALL allow users to drag and drop folders into the dropzone. The application SHALL recursively scan these folders for all supported image and video files.

#### Scenario: Dropping a folder adds all its images and videos
- **WHEN** the user drags and drops a folder onto the dropzone
- **THEN** the application recursively traverses the folder structure
- **AND** identifies all files with mime types `image/*` or `video/*`
- **AND** adds all those files to the current processing batch
- **AND** the UI displays them as individual items in the results grid.

#### Scenario: Dropping mixed files and folders
- **GIVEN** a selection of individual images and folders containing videos
- **WHEN** the user drags and drops this selection onto the dropzone
- **THEN** both the individual files and the contents of the folders are added to the processing batch.
