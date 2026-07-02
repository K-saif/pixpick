from __future__ import annotations
import numpy as np
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.core.selection import Line
from pixpick.utils import load_image, image_size, ImageSource


class LineSelector:
    """
    Orchestrates: load image → open backend → capture drag → return Line.

    This is the only class that knows about both the backend and the
    Line result object. Backends know nothing about Line; Line knows nothing
    about backends. LineSelector is the glue.

    Parameters
    ----------
    backend : BaseBackend | None
        Pass a backend instance to override auto-detection.
        None → CV2Backend (the only backend implemented in v0.1).
    """
    
    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend or CV2Backend()


    def select(self, source: ImageSource, 
               title: str = "pixpick | drag to select | Enter=confirm | Backspace=clear | Esc=cancel") -> Line:
        """
        Open an interactive window on `source` and return a Line.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path or BGR numpy array.
        title : str
            Window title.

        Returns
        -------
        Line
            A fully populated Line with all format properties and adapter methods.

        Raises
        ------
        SelectionCancelled
            If the user pressed Esc or closed the window.
        """
        print(f"Opening interactive selection window for: {source}")
        image = load_image(source)
        w, h  = image_size(image)

        raw = self.backend.select_line(image, title=title)

        if raw is None:
            raise SelectionCancelled("Line selection was cancelled by the user.")

        # raw is list of two points [(x1, y1), (x2, y2)]
        return Line(points=raw, image_width=w, image_height=h)


class SelectionCancelled(Exception):
    """Raised when the user cancels an interactive selection (Esc)."""