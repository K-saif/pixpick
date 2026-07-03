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
    """

    @abstractmethod
    def select_box(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> tuple[int, int, int, int] | None:
        """
        Let the user drag a rectangle on the image.

        Returns
        -------
        (x1, y1, x2, y2) in absolute pixels, or None if cancelled.
        """
        ...

    @abstractmethod
    def select_polygon(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[tuple[int, int]] | None:
        """
        Let the user click polygon vertices on the image.

        Returns
        -------
        List of (x, y) tuples (≥ 3 points), or None if cancelled.
        """
        ...

    @abstractmethod
    def select_line(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[tuple[int, int]] | None:
        """
        Let the user click line endpoints on the image.

        Returns
        -------
        List of (x, y) tuples (2 points), or None if cancelled.
        """
        ...