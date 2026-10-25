from __future__ import annotations
import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from pixpick.core.box import Box
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Polygon:
    """Represent a selected polygon.

    Attributes:
        points: Ordered list of polygon vertices as ``(x, y)`` coordinates
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
        if len(self.points) < 3:
            raise ValueError(
                f"Polygon needs at least 3 points, got {len(self.points)}"
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
        """Return the polygon vertices as a NumPy array.

        Returns:
            np.ndarray: An ``int32`` array of shape ``(N, 2)`` containing
                the absolute ``(x, y)`` coordinates.
        """
        return np.array(self.points, dtype=np.int32)

    @property
    def norm(self) -> list[tuple[float, float]]:
        """Return polygon vertices normalized to the range ``[0, 1]``.

        Returns:
            list[tuple[float, float]]: Normalized ``(x, y)`` coordinates.
        """
        return [
            (x / self.image_width, y / self.image_height)
            for x, y in self.points
        ]

    @property
    def norm_numpy(self) -> np.ndarray:
        """Return normalized polygon vertices as a NumPy array.

        Returns:
            np.ndarray: A ``float32`` array of shape ``(N, 2)``.
        """
        return np.array(self.norm, dtype=np.float32)

    @property
    def bbox(self) -> Box:
        """Return the tight axis-aligned bounding box around the polygon.

        Returns:
            Box: Bounding box enclosing all polygon vertices.
        """
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        return Box(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    @property
    def npoints(self) -> int:
        """Return the number of vertices in the polygon.

        Returns:
            int: Number of vertices.
        """
        return len(self.points)

    @property
    def yolo_region(self) -> list[tuple[int, int]]:
        """Return polygon vertices in absolute pixel coordinates.

        Returns:
            list[tuple[int, int]]: Ordered ``(x, y)`` vertex coordinates.
        """
        return self.points

    @property
    def supervision(self) -> dict:
        """Return polygon data for use with ``sv.PolygonZone``.

        Returns:
            dict: Dictionary containing the polygon NumPy array.
        """
        return {"polygon": self.as_numpy}

    @property
    def raw(self) -> dict:
        """Return polygon coordinates in all supported formats.

        Returns:
            dict: Coordinate representations useful for inspection and debugging.
        """
        return {
            "points":            self.points,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
            "bbox_xyxy":         self.bbox,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save the polygon selection to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        data = {
            "type": "polygon",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "points":     self.points,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Polygon":
        """Load a polygon selection from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Polygon: The reconstructed polygon.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            KeyError: If required fields are missing.
            ValueError: If the saved selection has the wrong type or invalid
                polygon coordinates.
        """
        data = json.loads(Path(path).read_text())
        if data["type"] != "polygon":
            raise ValueError(f"Expected type 'polygon', got '{data['type']}'")
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
        fill_alpha: float = 0.15,
    ) -> np.ndarray:
        """Draw the polygon on a copy of the image.

        The visualization includes an optional translucent fill, a border,
        vertex markers, and vertex index labels.

        Args:
            image: Source image as a NumPy array.
            color: Fill, border, and marker color in BGR order.
            thickness: Polygon border thickness in pixels.
            fill_alpha: Fill opacity, from ``0.0`` (transparent) to ``1.0``
                (opaque). A value of ``0.0`` disables the fill.

        Returns:
            np.ndarray: Copy of the image with the polygon visualization.
        """
        canvas = image.copy()
        pts = self.as_numpy.reshape((-1, 1, 2))

        # Semi-transparent fill
        if fill_alpha > 0:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [self.as_numpy], color)
            cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)

        # Border
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        # Vertex dots + index labels
        for i, (x, y) in enumerate(self.points):
            cv2.circle(canvas, (x, y), 4, color, -1)
            cv2.putText(
                canvas, str(i), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        return canvas

    def __repr__(self) -> str:
        return (
            f"Polygon(npoints={self.npoints}, "
            f"size={self.image_width}x{self.image_height})"
        )


@dataclass
class MultiPolygon:
    """Represent a selection containing multiple polygons.

    Attributes:
        polygons: List of ``Polygon`` objects.
        image_width: Width of the source image, used for normalization.
        image_height: Height of the source image, used for normalization.
    """

    polygons: list[Polygon]
    image_width: int
    image_height: int

    def __post_init__(self):
        if not self.polygons:
            raise ValueError("MultiPolygon must contain at least one Polygon.")

        for i, polygon in enumerate(self.polygons):
            if not isinstance(polygon, Polygon):
                raise TypeError(
                    f"MultiPolygon expects Polygon objects, got {type(polygon).__name__} "
                    f"at index {i} — build one with "
                    f"Polygon(points=[...], image_width=..., image_height=...)"
                )

            if polygon.image_width != self.image_width or polygon.image_height != self.image_height:
                raise ValueError(
                    f"Polygon {i} image size does not match MultiPolygon size"
                )

    @property
    def as_numpy(self) -> list[np.ndarray]:
        """Return each polygon as a NumPy array.

        Returns:
            list[np.ndarray]: ``int32`` arrays of shape ``(N, 2)``.
        """
        return [polygon.as_numpy for polygon in self.polygons]

    @property
    def norm(self) -> list[list[tuple[float, float]]]:
        """Return normalized vertices for every polygon.

        Returns:
            list[list[tuple[float, float]]]: Normalized vertex coordinates.
        """
        return [polygon.norm for polygon in self.polygons]

    @property
    def norm_numpy(self) -> list[np.ndarray]:
        """Return normalized vertices as NumPy arrays.

        Returns:
            list[np.ndarray]: ``float32`` arrays of shape ``(N, 2)``.
        """
        return [polygon.norm_numpy for polygon in self.polygons]

    @property
    def bbox(self) -> list[Box]:
        """Return the bounding box for each polygon.

        Returns:
            list[Box]: Tight axis-aligned bounding boxes.
        """
        return [polygon.bbox for polygon in self.polygons]
    
    @property
    def npoints(self) -> list[int]:
        """Return the number of vertices in each polygon.

        Returns:
            list[int]: Vertex count for each polygon.
        """
        return [len(polygon.points) for polygon in self.polygons]

    @property
    def yolo_region(self) -> list[list[tuple[int, int]]]:
        """Return the vertices of each polygon in absolute pixels.

        Returns:
            list[list[tuple[int, int]]]: Ordered vertex coordinates per polygon.
        """
        return [polygon.points for polygon in self.polygons]

    @property
    def supervision(self) -> list[dict]:
        """Return polygon data for use with ``sv.PolygonZone``.

        Returns:
            list[dict]: Polygon dictionaries, one per polygon.
        """
        return [polygon.supervision for polygon in self.polygons]
    
    @property
    def raw(self) -> dict:
        """Return all polygon coordinate formats for debugging.

        Returns:
            dict: Coordinate representations for all polygons.
        """
        return {
            "points": [polygon.npoints for polygon in self.polygons],
            "numpy": [polygon.as_numpy.tolist() for polygon in self.polygons],
            "normalized": [polygon.norm for polygon in self.polygons],
            "normalized_numpy": [polygon.norm_numpy.tolist() for polygon in self.polygons],
            "bbox_xyxy": [polygon.bbox for polygon in self.polygons],
        }

    def save(self, path: str | Path) -> None:
        """Save the multi-polygon selection to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        data = {
            "type": "multipolygon",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "polygons": [polygon.points for polygon in self.polygons],
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MultiPolygon":
        """Load a multi-polygon selection from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            MultiPolygon: The reconstructed multi-polygon selection.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            KeyError: If required fields are missing.
            ValueError: If the saved selection has the wrong type or invalid
                polygon coordinates.
        """
        data = json.loads(Path(path).read_text())
        if data["type"] != "multipolygon":
            raise ValueError(f"Expected type 'multipolygon', got '{data['type']}'")

        w, h = data["image_size"]
        polygons = [
            Polygon(points=[tuple(point) for point in polygon], image_width=w, image_height=h)
            for polygon in data["coordinates"]["polygons"]
        ]
        return cls(polygons=polygons, image_width=w, image_height=h)


    def visualize(
        self,
        image: np.ndarray,
        color: tuple = (0, 0, 255),
        thickness: int = 2,
        fill_alpha: float = 0.15,
    ) -> np.ndarray:
        """Draw all polygons on a copy of the image.

        Args:
            image: Source image as a NumPy array.
            color: Fill, border, and marker color in BGR order.
            thickness: Polygon border thickness in pixels.
            fill_alpha: Fill opacity, from ``0.0`` (transparent) to ``1.0``
                (opaque). A value of ``0.0`` disables the fill.

        Returns:
            np.ndarray: Copy of the image with all polygon visualizations.
        """
        canvas = image.copy()

        for polygon in self.polygons:
            canvas = polygon.visualize(
                canvas,
                color=color,
                thickness=thickness,
                fill_alpha=fill_alpha,
            )

        return canvas


    def __repr__(self) -> str:
        return (
            f"MultiPolygon(npolygons={len(self.polygons)}, "
            f"size={self.image_width}x{self.image_height})"
        )