from __future__ import annotations
import numpy as np
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.core.point import Point
from pixpick.utils import SelectionCancelled, load_image, image_size, ImageSource


class PointSelector:
    """
    Orchestrates: load image → open backend → capture clicks → return points.

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

    def select(self, source: ImageSource, title: str = "pixpick", frame: int = 0) -> Point:
        """
        Open an interactive window on `source`, let the user click points,
        and return a Point.

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
        Point
            The picked coordinates plus their foreground/background labels.

        Raises
        ------
        SelectionCancelled
            If the user pressed Esc or closed the window.
        """
        image = load_image(source, frame=frame)
        w, h  = image_size(image)

        result = self.backend.select_point(image, title=title)

        if result is None:
            raise SelectionCancelled("Point selection was cancelled by the user.")

        points, labels = result

        return Point(points=points, image_width=w, image_height=h, labels=labels)
