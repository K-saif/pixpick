from __future__ import annotations
import numpy as np
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.core.select_polygon import Polygon
from pixpick.utils import load_image, image_size, ImageSource


class PolygonSelector:
    """
    Orchestrates: load image → open backend → capture clicks → return Polygon.

    Mirrors BoxSelector exactly — same pattern, different backend method
    and different Selection type returned.

    Parameters
    ----------
    backend : BaseBackend | None
        Pass a backend instance to override auto-detection.
        None → CV2Backend.
    """

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend or CV2Backend()

    def select(self, source: ImageSource, title: str = "pixpick", frame: int = 0) -> Polygon:
        """
        Open an interactive window on `source`, let the user click polygon
        vertices, and return a Polygon.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Image file path or BGR numpy array.
        title : str
            Window title.
        frame : int
            0-based frame number to load when source is a video.

        Returns
        -------
        Polygon

        Raises
        ------
        SelectionCancelled
            If the user pressed Esc or closed the window.
        """
        image = load_image(source, frame=frame)
        w, h  = image_size(image)

        raw = self.backend.select_polygon(image, title=title)

        if raw is None:
            raise SelectionCancelled("Polygon selection was cancelled by the user.")

        # raw is list[tuple[int, int]] coming straight from the backend
        return Polygon(points=raw, image_width=w, image_height=h)


class SelectionCancelled(Exception):
    """Raised when the user cancels an interactive selection (Esc)."""
