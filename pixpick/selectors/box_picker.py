from __future__ import annotations
import numpy as np
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.core.box import Box, Multibox
from pixpick.utils import SelectionCancelled, load_image, image_size, ImageSource


class BoxSelector:
    """Orchestrates: load image → open backend → capture drag → return Box.

    This is the only class that knows about both the backend and the
    Box result object. Backends know nothing about Box; Box knows nothing
    about backends. BoxSelector is the glue.

    Args:
        backend: Pass a backend instance to override auto-detection.
            None means CV2Backend (the only backend implemented in v0.1).
    """

    def __init__(self, backend: BaseBackend | None = None):
        self.backend = backend or CV2Backend()


    def select(self, source: ImageSource,
               title: str = "pixpick | drag to select | Enter=confirm | Backspace=clear | Esc=cancel",
               frame: int = 0) -> Box:
        """Open an interactive window on `source` and return a Box.

        Args:
            source: File path or BGR numpy array.
            title: Window title.
            frame: 0-based frame number to load when source is a video.

        Returns:
            A fully populated Box with all format properties and adapter
            methods. If the user drew several boxes, a Multibox instead.

        Raises:
            SelectionCancelled: If the user pressed Esc or closed the window.
        """
        image = load_image(source, frame=frame)
        w, h  = image_size(image)

        boxes = self.backend.select_box(image, title=title)

        if boxes is None:
            raise SelectionCancelled("Box selection was cancelled by the user.")

        picked = [
            Box(x1=x1, y1=y1, x2=x2, y2=y2, image_width=w, image_height=h)
            for x1, y1, x2, y2 in boxes
        ]

        if len(picked) == 1:
            return picked[0]

        return Multibox(boxes=picked, image_width=w, image_height=h)
        