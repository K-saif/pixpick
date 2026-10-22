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
    """Load an image or video frame as a BGR NumPy array.

    Accepts an image/video file path (``str`` or ``Path``) or a raw NumPy array.
    For videos, loads the specified zero-based frame; by default, loads the
    first frame. The returned array is in BGR ``uint8`` format, as expected by
    OpenCV.

    Args:
        source (ImageSource): Image/video file path or a BGR NumPy array.
        frame (int): Zero-based video frame index to load. Defaults to ``0``.

    Returns:
        np.ndarray: The image as a BGR NumPy array. For array input, returns a
            copy of the provided array.

    Raises:
        ValueError: If ``frame`` is negative, the input array is not shaped
            ``(H, W, 3)``, or OpenCV cannot open/read the specified file or frame.
        FileNotFoundError: If the specified file path does not exist.
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
    """Return the width and height of an OpenCV BGR image.

    Args:
        img (np.ndarray): Image array in BGR format.

    Returns:
        tuple[int, int]: Image dimensions as ``(width, height)``.
    """
    h, w = img.shape[:2]
    return w, h


class SelectionCancelled(Exception):
    """Raised when the user cancels an interactive selection with Esc."""
