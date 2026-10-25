from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseBackend(ABC):
    """Abstract interface for UI backends that collect pixel coordinates.

    A backend owns the UI layer: it opens a window or widget, captures user
    interactions, and returns raw pixel coordinates.

    It knows nothing about ``Selection`` objects or adapters. The selector
    calls the backend and wraps the raw result in the appropriate ``Selection``
    type.

    Adding a new environment (Jupyter, Gradio, etc.) means adding a backend,
    without changing selectors or adapters.

    Every ``select_*`` method returns a list, with one entry per selection
    the user made, even when the list holds a single entry. The selector
    decides whether to wrap the result in a singular or a ``Multi*`` type.
    All methods return ``None`` when the user cancels; selectors convert that
    result into a ``SelectionCancelled`` exception.
    """

    @abstractmethod
    def select_box(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[list[int]] | None:
        """Let the user draw one or more rectangles on the image.

        Args:
            image: Image on which the rectangles will be selected.
            title: Title to display in the backend UI. Defaults to ``"pixpick"``.

        Returns:
            A list of rectangles in absolute pixel coordinates, each formatted
            as ``[x1, y1, x2, y2]``. Returns ``None`` if the selection is
            cancelled.
        """
        ...

    @abstractmethod
    def select_polygon(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[list[tuple[int, int]]] | None:
        """Let the user click the vertices of one or more polygons.

        Args:
            image: Image on which the polygons will be selected.
            title: Title to display in the backend UI. Defaults to ``"pixpick"``.

        Returns:
            A list containing one vertex list per polygon. Each vertex is an
            ``(x, y)`` tuple in absolute pixel coordinates, and each polygon
            must contain at least three points. Returns ``None`` if the
            selection is cancelled.
        """
        ...

    @abstractmethod
    def select_line(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        """Let the user click the endpoints of one or more lines.

        Args:
            image: Image on which the lines will be selected.
            title: Title to display in the backend UI. Defaults to ``"pixpick"``.

        Returns:
            A list containing one pair of endpoints per line. Each endpoint
            is an ``(x, y)`` tuple in absolute pixel coordinates. Returns
            ``None`` if the selection is cancelled.
        """
        ...

    @abstractmethod
    def select_point(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> tuple[list[tuple[int, int]], list[int]] | None:
        """Let the user click points and label each as foreground or background.

        Args:
            image: Image on which the points will be selected.
            title: Title to display in the backend UI. Defaults to ``"pixpick"``.

        Returns:
            A tuple containing two parallel lists: the selected points as
            ``(x, y)`` tuples in absolute pixel coordinates, and their labels.
            Each label is ``1`` for foreground or ``0`` for background.
            Returns ``None`` if the selection is cancelled.
        """
        ...