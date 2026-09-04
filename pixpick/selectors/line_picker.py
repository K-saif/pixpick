from __future__ import annotations
import numpy as np
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.core.line import Line, MultiLine
from pixpick.utils import SelectionCancelled, load_image, image_size, ImageSource


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
               title: str = "pixpick | drag to select | Enter=confirm | Backspace=clear | Esc=cancel",
               frame: int = 0) -> Line:
        """
        Open an interactive window on `source` and return a Line.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path or BGR numpy array.
        title : str
            Window title.
        frame : int
            0-based frame number to load when source is a video.

        Returns
        -------
        Line
            A fully populated Line with all format properties and adapter methods.

        Raises
        ------
        SelectionCancelled
            If the user pressed Esc or closed the window.
        """
        image = load_image(source, frame=frame)
        w, h  = image_size(image)

        lines = self.backend.select_line(image, title=title)

        if lines is None:
            raise SelectionCancelled("Line selection was cancelled by the user.")

        picked = [
            Line(points=list(points), image_width=w, image_height=h)
            for points in lines
        ]

        if len(picked) == 1:
            return picked[0]

        return MultiLine(lines=picked, image_width=w, image_height=h)
