"""Video processing utilities for backend inference.

This module provides helpers for extracting frames from uploaded video bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractedFrame:
    """A video frame plus some lightweight metadata."""

    index: int
    bgr: np.ndarray


def _uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))

    # Uniformly spaced indices across [0, total_frames-1]
    lin = np.linspace(0, total_frames - 1, num=num_frames)
    return np.unique(lin.round().astype(int)).tolist()


def extract_frames(video_bytes: bytes, num_frames: int = 10) -> list[ExtractedFrame]:
    """Extract uniformly-sampled frames from a video file.

    Args:
        video_bytes: Raw video bytes.
        num_frames: Target number of frames to sample.

    Returns:
        List of extracted frames in BGR color space.

    Raises:
        ValueError: If the video cannot be decoded.
    """
    # OpenCV's VideoCapture requires a filename, so we decode via a temp file.
    # Note: caller is responsible for bounding upload size.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(video_bytes)
        f.flush()

        cap = cv2.VideoCapture(f.name)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = _uniform_indices(total_frames, num_frames)

        frames: list[ExtractedFrame] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(ExtractedFrame(index=idx, bgr=frame))

        cap.release()

    if not frames:
        raise ValueError("No frames extracted")

    return frames
