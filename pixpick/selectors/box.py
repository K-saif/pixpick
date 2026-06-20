from __future__ import annotations
import numpy as np
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.core.selection import Box
from pixpick.utils import load_image, image_size, ImageSource


class BoxSelector:
    """
    Orchestrates: load image → open backend → capture drag → return Box.

    This is the only class that knows about both the backend and the
    Box result object. Backends know nothing about Box; Box knows nothing
    about backends. BoxSelector is the glue.

    Parameters
    ----------
    backend : BaseBackend | None
        Pass a backend instance to override auto-detection.
        None → CV2Backend (the only backend implemented in v0.1).
    """

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend or CV2Backend()

    def select(self, source: ImageSource, title: str = "pixpick") -> Box:
        """
        Open an interactive window on `source` and return a Box.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path or BGR numpy array.
        title : str
            Window title.

        Returns
        -------
        Box
            A fully populated Box with all format properties and adapter methods.

        Raises
        ------
        SelectionCancelled
            If the user pressed Esc or closed the window.
        """
        image = load_image(source)
        w, h  = image_size(image)

        raw = self.backend.select_box(image, title=title)

        if raw is None:
            raise SelectionCancelled("Box selection was cancelled by the user.")

        x1, y1, x2, y2 = raw
        return Box(x1=x1, y1=y1, x2=x2, y2=y2, image_width=w, image_height=h)


class SelectionCancelled(Exception):
    """Raised when the user cancels an interactive selection (Esc)."""