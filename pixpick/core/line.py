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

    @property
    def start(self) -> tuple[int, int]:
        """First point of the line."""
        return self.points[0]

    @property
    def end(self) -> tuple[int, int]:
        """Last point of the line."""
        return self.points[-1]

    @property
    def vector(self) -> tuple[float, float]:
        x1, y1 = self.start
        x2, y2 = self.end
        return (x2 - x1, y2 - y1)

    @property
    def horizontal(self) -> list[tuple[float, float]]:
        """Return a new Line points that is horizontal."""
        cx, cy = self.center
        length = self.length
        half_length = length / 2
        new_start = (int(cx - half_length), cy)
        new_end = (int(cx + half_length), cy)
        return [new_start, new_end]

    @property
    def vertical(self) -> list[tuple[float, float]]:
        """Return a new Line points that is vertical."""
        cx, cy = self.center
        length = self.length
        half_length = length / 2
        new_start = (cx, int(cy - half_length))
        new_end = (cx, int(cy + half_length))
        return [new_start, new_end]

    @property
    def raw(self) -> dict:
        """All formats at once."""
        return {
            "points":            self.points,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
            "center":            self.center,
            "length":            self.length,
            "start":             self.start,
            "end":               self.end,
            "vector":            self.vector,
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

@dataclass
class MultiLine:
    """
    Immutable result of a multi-line selection.

    Attributes
    ----------
    lines : list[Line]
        List of Line objects — each one validates itself.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    """

    lines: list[Line]
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if not self.lines:
            raise ValueError("MultiLine must contain at least one Line.")

        for i, line in enumerate(self.lines):
            if not isinstance(line, Line):
                raise TypeError(
                    f"MultiLine expects Line objects, got {type(line).__name__} "
                    f"at index {i} — build one with "
                    f"Line(points=[(x1, y1), (x2, y2)], image_width=..., image_height=...)"
                )

            if line.image_width != self.image_width or line.image_height != self.image_height:
                raise ValueError(
                    f"Line {i} image size does not match MultiLine size"
                )

    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def points(self) -> list[list[tuple[int, int]]]:
        """[[(x1,y1), (x2,y2)], ...] — absolute pixels, one pair per line."""
        return [line.points for line in self.lines]

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (N, 2, 2) int32 array — [[[x1,y1], [x2,y2]], ...]."""
        return np.array([line.as_numpy for line in self.lines], dtype=np.int32)

    @property
    def norm(self) -> list[list[tuple[float, float]]]:
        """Points normalised to [0, 1], for all lines."""
        return [line.norm for line in self.lines]

    @property
    def norm_numpy(self) -> np.ndarray:
        """Shape (N, 2, 2) float32 array of normalised points."""
        return np.array(self.norm, dtype=np.float32)

    @property
    def nlines(self) -> int:
        return len(self.lines)

    @property
    def center(self) -> list[tuple[int, int]]:
        """Center point of each line in absolute pixels."""
        return [line.center for line in self.lines]

    @property
    def length(self) -> list[float]:
        """Length of each line in pixels."""
        return [line.length for line in self.lines]

    @property
    def start(self) -> list[tuple[int, int]]:
        """First point of each line."""
        return [line.start for line in self.lines]

    @property
    def end(self) -> list[tuple[int, int]]:
        """Last point of each line."""
        return [line.end for line in self.lines]

    @property
    def vector(self) -> list[tuple[float, float]]:
        return [line.vector for line in self.lines]

    @property
    def horizontal(self) -> list[list[tuple[int, int]]]:
        """Return new lines with the same lengths, aligned horizontally."""
        return [line.horizontal for line in self.lines]

    @property
    def vertical(self) -> list[list[tuple[int, int]]]:
        """Return new lines with the same lengths, aligned vertically."""
        return [line.vertical for line in self.lines]

    @property
    def raw(self) -> dict:
        """All formats at once."""
        return {
            "points":            self.points,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
            "center":            self.center,
            "length":            self.length,
            "start":             self.start,
            "end":               self.end,
            "vector":            self.vector,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise to JSON."""
        data = {
            "type": "multiline",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "lines":      self.points,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MultiLine":
        """Reconstruct from a saved JSON file."""
        data = json.loads(Path(path).read_text())

        if data["type"] != "multiline":
            raise ValueError(f"Expected type 'multiline', got '{data['type']}'")

        w, h = data["image_size"]

        lines = [
            Line(
                points=[tuple(point) for point in line],
                image_width=w,
                image_height=h,
            )
            for line in data["coordinates"]["lines"]
        ]

        return cls(lines=lines, image_width=w, image_height=h)

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        image: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw all lines on a copy of the image."""
        canvas = image.copy()

        for line in self.lines:
            canvas = line.visualize(canvas, color=color, thickness=thickness)

        return canvas

    def __repr__(self) -> str:
        return (
            f"MultiLine(nlines={len(self.lines)}, "
            f"size={self.image_width}x{self.image_height})"
        )
