from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from typing import Union

ImageSource = Union[str, Path, np.ndarray]


def load_image(source: ImageSource) -> np.ndarray:
    """
    Accept a file path (str / Path) or a raw numpy array.
    Always returns a BGR uint8 np.ndarray, which is what cv2 expects.
    """
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"Array must be (H, W, 3) BGR, got shape {source.shape}")
        return source.copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read image: {path}")

    return img


def image_size(img: np.ndarray) -> tuple[int, int]:
    """Return (width, height) from a cv2 BGR array."""
    h, w = img.shape[:2]
    return w, h