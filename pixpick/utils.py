from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from typing import Union

ImageSource = Union[str, Path, np.ndarray]

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def load_image(source: ImageSource, frame: int = 0) -> np.ndarray:
    """
    Accept an image/video file path (str / Path) or a raw numpy array.
    For videos, load frame (0-based); by default this is the first frame.
    Always returns a BGR uint8 np.ndarray, which is what cv2 expects.
    """
    if frame < 0:
        raise ValueError(f"frame must be >= 0, got {frame}")

    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"Array must be (H, W, 3) BGR, got shape {source.shape}")
        return source.copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"cv2 could not open video: {path}")

            if frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"cv2 could not read frame {frame} from video: {path}")

            return frame
        finally:
            cap.release()

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read image: {path}")

    return img


def image_size(img: np.ndarray) -> tuple[int, int]:
    """Return (width, height) from a cv2 BGR array."""
    h, w = img.shape[:2]
    return w, h
