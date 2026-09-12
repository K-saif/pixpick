from __future__ import annotations  
from abc import ABC, abstractmethod
import numpy as np


class BaseBackend(ABC):
    """
    A backend owns the UI layer — it opens a window (or widget),
    captures user interactions, and returns raw pixel coordinates.

    It knows nothing about Selection objects or adapters.
    The selector calls the backend and wraps the raw result in the
    appropriate Selection type.

    Adding a new environment (Jupyter, Gradio, …) means adding a new
    backend — zero changes to selectors or adapters.

    Every select_* method returns a **list**, one entry per selection the
    user made, even when that list holds a single entry. The selector
    decides whether to wrap the result in a singular or a Multi* type.
    All of them return None when the user cancels; selectors turn that
    into a SelectionCancelled exception.
    """

    @abstractmethod
    def select_box(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[list[int]] | None:
        """
        Let the user drag one or more rectangles on the image.

        Returns
        -------
        [[x1, y1, x2, y2], ...] in absolute pixels — one list per
        rectangle, or None if cancelled.
        """
        ...

    @abstractmethod
    def select_polygon(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[list[tuple[int, int]]] | None:
        """
        Let the user click the vertices of one or more polygons.

        Returns
        -------
        [[(x0, y0), (x1, y1), ...], ...] in absolute pixels — one vertex
        list per polygon, each holding at least 3 points, or None if
        cancelled.
        """
        ...

    @abstractmethod
    def select_line(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        """
        Let the user click the endpoints of one or more lines.

        Returns
        -------
        [((x0, y0), (x1, y1)), ...] in absolute pixels — one pair of
        endpoints per line, or None if cancelled.
        """
        ...

    @abstractmethod
    def select_point(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> tuple[list[tuple[int, int]], list[int]] | None:
        """
        Let the user click points on the image, each tagged foreground
        or background.

        Returns
        -------
        ([(x0, y0), ...], [label, ...]) in absolute pixels — two parallel
        lists, one label per point: 1 (foreground) or 0 (background),
        or None if cancelled.
        """
        ...