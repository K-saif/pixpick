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
class Polygon:
    """
    Immutable result of a polygon selection.

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
        """Shape (N, 2) int32 array — [[x0,y0], [x1,y1], ...]."""
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
        """Shape (N, 2) float32 array of normalised points."""
        return np.array(self.norm, dtype=np.float32)

    @property
    def bbox(self) -> Box:
        """Tight axis-aligned Box that encloses this polygon."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        xyxy = [x1, y1, x2, y2]
        return xyxy

    @property
    def npoints(self) -> int:
        return len(self.points)

    @property
    def yolo_region(self) -> list[tuple[int, int]]:
        """[(point1), (point2), (point3), ...]"""
        return self.points

    @property
    def supervision(self) -> dict:
        """Ready to unpack into sv.PolygonZone()"""
        return {"polygon": self.as_numpy}

    @property
    def raw(self) -> dict:
        """All formats at once."""
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
        """Serialise to JSON."""
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
        """Reconstruct from a saved JSON file."""
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
        """Draw the polygon (filled + border + vertex dots) on a copy of image."""
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
    """
    Immutable result of a multi-polygon selection.

    Attributes
    ----------
    polygons : list[Polygon]
        List of Polygon objects.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
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
        """List of (N, 2) int32 arrays, one per polygon."""
        return [polygon.as_numpy for polygon in self.polygons]

    @property
    def norm(self) -> list[list[tuple[float, float]]]:
        """Points normalised to [0, 1] for each polygon."""
        return [polygon.norm for polygon in self.polygons]

    @property
    def norm_numpy(self) -> list[np.ndarray]:
        """List of (N, 2) float32 arrays, one per polygon."""
        return [polygon.norm_numpy for polygon in self.polygons]

    @property
    def box(self) -> Box:
        """List of tight axis-aligned Boxes that enclose each polygon."""
        return [polygon.bbox for polygon in self.polygons]
    
    @property
    def npoints(self) -> list[int]:
        """List of length of polygon points."""
        return [len(polygon.points) for polygon in self.polygons]

    @property
    def yolo_region(self) -> list[list[tuple[int, int]]]:
        """List of polygon point lists in absolute pixels."""
        return [polygon.points for polygon in self.polygons]

    @property
    def supervision(self) -> list[dict]:
        """List of dicts ready to unpack into sv.PolygonZone() for each polygon."""
        return [polygon.supervision for polygon in self.polygons]
    
    @property
    def raw(self) -> dict:
        """All formats at once — handy for debugging."""
        return {
            "points": [polygon.npoints for polygon in self.polygons],
            "numpy": [polygon.as_numpy.tolist() for polygon in self.polygons],
            "normalized": [polygon.norm for polygon in self.polygons],
            "normalized_numpy": [polygon.norm_numpy.tolist() for polygon in self.polygons],
            "bbox_xyxy": [polygon.bbox for polygon in self.polygons],
        }

    def save(self, path: str | Path) -> None:
        """Serialise to JSON."""
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
        """Reconstruct from a saved JSON file."""
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
        """Draw all polygons on a copy of image and return it."""
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