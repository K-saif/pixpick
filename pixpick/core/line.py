from __future__ import annotations
import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Line:
    """
    Immutable result of a line selection.

    Attributes
    ----------
    points : list[tuple[int, int]]
        Ordered list of (x, y) vertices in absolute pixels.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    """

    points: list[tuple[int, int]]
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError(
                f"Line needs at least 2 points, got {len(self.points)}"
            )
        for i, pt in enumerate(self.points):
            x, y = pt
            if not (0 <= x <= self.image_width and 0 <= y <= self.image_height):
                raise ValueError(
                    f"Point {i} ({x},{y}) is outside image "
                    f"({self.image_width}x{self.image_height})"
                )


    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (2, 2) int32 array — [[x1,y1], [x2,y2]]."""
        return np.array(self.points, dtype=np.int32)

    @property
    def norm(self) -> list[tuple[float, float]]:
        """Points normalised to [0, 1]."""
        return [
            (x / self.image_width, y / self.image_height)
            for x, y in self.points
        ]

    @property
    def norm_numpy(self) -> np.ndarray:
        """Shape (2, 2) float32 array of normalised points."""
        return np.array(self.norm, dtype=np.float32)

    @property
    def center(self) -> tuple[int, int]:
        """(cx, cy) in absolute pixels."""
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def length(self) -> float:
        """Length of the line in pixels."""
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # ------------------------------------------------------------------ #
    # Adapter shortcuts                                                    #
    # ------------------------------------------------------------------ #
    def raw(self) -> dict:
        """All formats at once."""
        return {
            "points":            self.points,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
            "center":            self.center,
            "length":            self.length,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise to JSON."""
        data = {
            "type": "line",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "points":     self.points,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Line":
        """Reconstruct from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        if data["type"] != "line":
            raise ValueError(f"Expected type 'line', got '{data['type']}'")
        w, h = data["image_size"]
        points = [tuple(p) for p in data["coordinates"]["points"]]
        return cls(points=points, image_width=w, image_height=h)

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        image: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw the line on a copy of image."""
        canvas = image.copy()
        pts = self.as_numpy.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)
        for i, (x, y) in enumerate(self.points):
            cv2.circle(canvas, (x, y), 4, color, -1)
            cv2.putText(
                canvas, str(i), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
        return canvas