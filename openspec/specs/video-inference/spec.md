# video-inference Specification

## Purpose
TBD - created by archiving change enable-video-inference. Update Purpose after archive.
## Requirements
### Requirement: Support Video Upload and Analysis
The system MUST support uploading video files for analysis.

#### Scenario: User uploads a video
Given the system is running
When a user uploads a valid video file (e.g., MP4)
Then the system accepts the file
And extracts 60 frames from the video
And performs inference on each frame
And returns a single "REAL" or "FAKE" judgment based on the average score.

### Requirement: Grad-CAM for Video
The system MUST provide explainability for video judgments.

#### Scenario: Fake video explanation
Given a video is judged as FAKE
When the system returns the result
Then it includes the Grad-CAM visualization for the frame with the highest fake probability.

