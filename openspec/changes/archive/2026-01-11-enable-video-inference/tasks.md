# Tasks: Enable Video Inference

1.  [x] **Backend: Implement Video Processing Utility**
    -   Create `backend/video_utils.py`.
    -   Implement `extract_frames(video_bytes, num_frames=10) -> List[np.ndarray]`.
    -   Use `cv2` to decode video from memory or temp file.

2.  [x] **Backend: Add Video Prediction Endpoint**
    -   Update `backend/server.py`.
    -   Add `video_predict` function (refactor `predict` to reuse valid logic or create a shared service function).
    -   Add `POST /predict_video`.
    -   Implement aggregation logic (Average score).

3.  [x] **Frontend: Update Dropzone**
    -   Modify `frontend/components/Dropzone.tsx` (or equivalent) to accept `video/mp4`, `video/webm`, etc.

4.  [x] **Frontend: Update Results UI**
    -   Modiy `frontend/app/page.tsx` or result component.
    -   Handle "Video" result type.
    -   Display overall Score/Label.
    -   Show "Most Suspicious Frame" with Grad-CAM if available.

5.  [x] **Validation**
    -   Test with real/fake video samples.
    -   Verify performance latency < 5s for short videos.
