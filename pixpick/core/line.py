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
    """Represent a selected line.

    Attributes:
        points: Ordered list of line endpoints as ``(x, y)`` coordinates
            in absolute pixels.
        image_width: Width of the source image, used for normalization.
        image_height: Height of the source image, used for normalization.
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
        """Return the line endpoints as a NumPy array.

        Returns:
            np.ndarray: An ``int32`` array of shape ``(2, 2)``.
        """
        return np.array(self.points, dtype=np.int32)

    @property
    def norm(self) -> list[tuple[float, float]]:
        """Return line endpoints normalized to the range ``[0, 1]``.

        Returns:
            list[tuple[float, float]]: Normalized endpoint coordinates.
        """
        return [
            (x / self.image_width, y / self.image_height)
            for x, y in self.points
        ]

    @property
    def norm_numpy(self) -> np.ndarray:
        """Return normalized endpoints as a NumPy array.

        Returns:
            np.ndarray: A ``float32`` array of shape ``(2, 2)``.
        """
        return np.array(self.norm, dtype=np.float32)

    @property
    def center(self) -> tuple[int, int]:
        """Return the midpoint of the line in absolute pixels.

        Returns:
            tuple[int, int]: Integer center coordinates ``(cx, cy)``.
        """
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def length(self) -> float:
        """Return the Euclidean length of the line.

        Returns:
            float: Line length in pixels.
        """
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    @property
    def start(self) -> tuple[int, int]:
        """Return the first endpoint of the line.

        Returns:
            tuple[int, int]: The starting point in absolute pixels.
        """
        return self.points[0]

    @property
    def end(self) -> tuple[int, int]:
        """Return the last endpoint of the line.

        Returns:
            tuple[int, int]: The ending point in absolute pixels.
        """
        return self.points[-1]

    @property
    def vector(self) -> tuple[float, float]:
        """Return the direction vector from the first to the last endpoint.

        Returns:
            tuple[float, float]: Horizontal and vertical displacement ``(dx, dy)``.
        """
        x1, y1 = self.start
        x2, y2 = self.end
        return (x2 - x1, y2 - y1)

    @property
    def horizontal(self) -> list[tuple[float, float]]:
        """Return endpoints for a horizontal line centered on the current line.

        The returned line retains the current line length where image bounds
        permit.

        Returns:
            list[tuple[int, int]]: New start and end points in absolute pixels.
        """
        cx, cy = self.center
        length = self.length
        half_length = length / 2
        new_start = (max(0, int(cx - half_length)), cy)
        new_end = (min(self.image_width, int(cx + half_length)), cy)
        return [new_start, new_end]

    @property
    def vertical(self) -> list[tuple[float, float]]:
        """Return endpoints for a vertical line centered on the current line.

        The returned line retains the current line length where image bounds
        permit.

        Returns:
            list[tuple[int, int]]: New start and end points in absolute pixels.
        """
        cx, cy = self.center
        length = self.length
        half_length = length / 2
        new_start = (cx, max(0, int(cy - half_length)))
        new_end = (cx, min(self.image_height, int(cy + half_length)))
        return [new_start, new_end]

    @property
    def raw(self) -> dict:
        """Return line geometry in all supported formats.

        Returns:
            dict: Coordinates and derived geometry useful for debugging.
        """
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
        """Save the line selection to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
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
        """Load a line selection from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Line: The reconstructed line.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            KeyError: If required fields are missing.
            ValueError: If the saved selection has the wrong type or invalid
                line coordinates.
        """
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
        """Draw the line and endpoint labels on a copy of the image.

        Args:
            image: Source image as a NumPy array.
            color: Line and marker color in BGR order.
            thickness: Line thickness in pixels.

        Returns:
            np.ndarray: Copy of the image with the line visualization.
        """
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
    """Represent a selection containing multiple lines.

    Attributes:
        lines: List of ``Line`` objects. Each line validates itself.
        image_width: Width of the source image, used for normalization.
        image_height: Height of the source image, used for normalization.
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
        """Return the endpoints of all lines in absolute pixels.

        Returns:
            list[list[tuple[int, int]]]: Start and end points for each line.
        """
        return [line.points for line in self.lines]

    @property
    def as_numpy(self) -> np.ndarray:
        """Return all line endpoints as a NumPy array.

        Returns:
            np.ndarray: An ``int32`` array of shape ``(N, 2, 2)``.
        """
        return np.array([line.as_numpy for line in self.lines], dtype=np.int32)

    @property
    def norm(self) -> list[list[tuple[float, float]]]:
        """Return normalized endpoints for all lines.

        Returns:
            list[list[tuple[float, float]]]: Normalized endpoint coordinates.
        """
        return [line.norm for line in self.lines]

    @property
    def norm_numpy(self) -> np.ndarray:
        """Return normalized endpoints as a NumPy array.

        Returns:
            np.ndarray: A ``float32`` array of shape ``(N, 2, 2)``.
        """
        return np.array(self.norm, dtype=np.float32)

    @property
    def nlines(self) -> int:
        """Return the number of selected lines.

        Returns:
            int: Number of lines in the selection.
        """
        return len(self.lines)

    @property
    def center(self) -> list[tuple[int, int]]:
        """Return the midpoint of each line in absolute pixels.

        Returns:
            list[tuple[int, int]]: Center coordinates for each line.
        """
        return [line.center for line in self.lines]

    @property
    def length(self) -> list[float]:
        """Return the length of each line in pixels.

        Returns:
            list[float]: Euclidean length for each line.
        """
        return [line.length for line in self.lines]

    @property
    def start(self) -> list[tuple[int, int]]:
        """Return the first endpoint of each line.

        Returns:
            list[tuple[int, int]]: Starting points in absolute pixels.
        """
        return [line.start for line in self.lines]

    @property
    def end(self) -> list[tuple[int, int]]:
        """Return the last endpoint of each line.

        Returns:
            list[tuple[int, int]]: Ending points in absolute pixels.
        """
        return [line.end for line in self.lines]

    @property
    def vector(self) -> list[tuple[float, float]]:
        """Return the direction vector of each line.

        Returns:
            list[tuple[float, float]]: Displacement vectors ``(dx, dy)``.
        """
        return [line.vector for line in self.lines]

    @property
    def horizontal(self) -> list[list[tuple[int, int]]]:
        """Return horizontal versions of all lines.

        Returns:
            list[list[tuple[int, int]]]: Endpoints for each horizontal line.
        """
        return [line.horizontal for line in self.lines]

    @property
    def vertical(self) -> list[list[tuple[int, int]]]:
        """Return vertical versions of all lines.

        Returns:
            list[list[tuple[int, int]]]: Endpoints for each vertical line.
        """
        return [line.vertical for line in self.lines]

    @property
    def raw(self) -> dict:
        """Return line geometry in all supported formats.

        Returns:
            dict: Coordinates and derived geometry useful for debugging.
        """
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
        """Save the multi-line selection to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
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
        """Load a multi-line selection from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            MultiLine: The reconstructed multi-line selection.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            KeyError: If required fields are missing.
            ValueError: If the saved selection has the wrong type or invalid
                line coordinates.
        """
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
        """Draw all lines on a copy of the image.

        Args:
            image: Source image as a NumPy array.
            color: Line and marker color in BGR order.
            thickness: Line thickness in pixels.

        Returns:
            np.ndarray: Copy of the image with all line visualizations.
        """
        canvas = image.copy()

        for line in self.lines:
            canvas = line.visualize(canvas, color=color, thickness=thickness)

        return canvas

    def __repr__(self) -> str:
        return (
            f"MultiLine(nlines={len(self.lines)}, "
            f"size={self.image_width}x{self.image_height})"
        )
