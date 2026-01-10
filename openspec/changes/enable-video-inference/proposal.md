# Proposal: Enable Video Inference

**Change ID**: `enable-video-inference`

## Summary
Enable the AIGI detection system to process video files by extracting frames, performing inference on each frame, and aggregating the results to determine if the video is real or generated/manipulated.

## Motivation
Deepfakes are increasingly appearing in video format. The current system only supports image inputs. Extending support to video is a natural and necessary evolution to address the broader threat landscape.

## Design Details

### Frame Extraction Strategy
To balance performance and accuracy, we will implement a Uniform Sampling strategy:
- **Count**: Extract **60 frames** uniformly distributed across the video duration.
  - *Rationale*: 60 frames provide a good survey of the video content without imposing excessive computational overhead. 1 frame/sec is safer but can be too slow for long videos. 10 frames is a constant-time operation relative to inference cost (mostly).
- **Fallback**: If the video is shorter than 60 frames, use all frames.

### Judgment (Aggregation) Logic
The video-level decision will be based on the **Average Probability** of the extracted frames.
- **Score**: $S_{video} = \frac{1}{N} \sum_{i=1}^{N} S_{frame\_i}$
- **Label**: `FAKE` if $S_{video} > 0.5$ else `REAL`.
- **Reasoning**: We will also track the frame with the maximum fake probability. If the video is judged as fake, we will return the reasoning (Grad-CAM analysis) from this "most fake" frame to explain the decision.

### Backend Changes
- New endpoint `POST /predict_video`.
- Use `cv2.VideoCapture` to read and seek to specific timestamps.
- Reuse `perform_inference` and `FaceAlignment` logic.

### Frontend Changes
- Update `Dropzone` to accept `video/*` MIME types.
- Update results display to handle video responses (show overall score + "worst" frame Grad-CAM).

## Alternatives Considered
- **All-frame inference**: Too slow for backend processing of user uploads.
- **Max-pooling score**: Highly sensitive to false positives (one bad frame marks video as fake). Average is more robust for a general classification tool.
