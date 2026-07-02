from __future__ import annotations
import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ======================================================================== #
# Box                                                                        #
# ======================================================================== #

@dataclass
class Box:
    """
    Immutable result of a box selection.

    All coordinate math lives here.
    Adapters are thin — they just re-package what this class already holds.

    Attributes
    ----------
    x1, y1, x2, y2 : int
        Absolute pixel coordinates (top-left → bottom-right).
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        # Normalise so x1 < x2 and y1 < y2 regardless of drag direction.
        self.x1, self.x2 = sorted([self.x1, self.x2])
        self.y1, self.y2 = sorted([self.y1, self.y2])

        if self.x1 == self.x2 or self.y1 == self.y2:
            raise ValueError("Box has zero area — did the drag complete?")

        if not (0 <= self.x1 < self.x2 <= self.image_width):
            raise ValueError(
                f"x coords out of bounds: {self.x1}, {self.x2} (image width={self.image_width})"
            )
        if not (0 <= self.y1 < self.y2 <= self.image_height):
            raise ValueError(
                f"y coords out of bounds: {self.y1}, {self.y2} (image height={self.image_height})"
            )

    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def xyxy(self) -> list[int]:
        """[x1, y1, x2, y2] — absolute pixels."""
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def xywh(self) -> list[int]:
        """[x, y, width, height] — top-left origin, absolute pixels."""
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]

    @property
    def cxcywh(self) -> list[float]:
        """[cx, cy, width, height] — centre + size, absolute pixels."""
        w, h = self.x2 - self.x1, self.y2 - self.y1
        return [self.x1 + w / 2, self.y1 + h / 2, float(w), float(h)]

    @property
    def norm(self) -> list[float]:
        """[x1, y1, x2, y2] — values in [0, 1]."""
        return [
            self.x1 / self.image_width,
            self.y1 / self.image_height,
            self.x2 / self.image_width,
            self.y2 / self.image_height,
        ]

    @property
    def norm_xywh(self) -> list[float]:
        """[x, y, w, h] normalised — YOLO label format."""
        x1n, y1n, x2n, y2n = self.norm
        return [x1n, y1n, x2n - x1n, y2n - y1n]

    @property
    def center(self) -> tuple[int, int]:
        """(cx, cy) in absolute pixels."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        """Area in pixels²."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (4,) int32 array — [x1, y1, x2, y2]."""
        return np.array(self.xyxy, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Adapter shortcuts                                                    #
    # ------------------------------------------------------------------ #

    def yolo_region(self) -> list[float]:
        """[(point1), (point2), (point3), (point4)] """
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2)
        ]

    def yolo_prompt(self) -> np.ndarray:
        """[(point1), (point2), (point3), (point4)] """
        return np.array([
            [self.x1, self.y1, self.x2, self.y2]
        ])

    def sam(self) -> np.ndarray:
        """[(point1), (point2), (point3), (point4)] """
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def raw(self) -> dict:
        """All formats at once — handy for debugging."""
        return {
            "xyxy":            self.xyxy,
            "xywh":            self.xywh,
            "cxcywh":          self.cxcywh,
            "normalized":      self.norm,
            "normalized_xywh": self.norm_xywh,
            "numpy":           self.as_numpy.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise to JSON."""
        data = {
            "type": "box",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "xyxy":       self.xyxy,
                "xywh":       self.xywh,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Box":
        """Reconstruct from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        if data["type"] != "box":
            raise ValueError(f"Expected type 'box', got '{data['type']}'")
        w, h = data["image_size"]
        x1, y1, x2, y2 = data["coordinates"]["xyxy"]
        return cls(x1=x1, y1=y1, x2=x2, y2=y2, image_width=w, image_height=h)

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        image: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw the box on a copy of image and return it."""
        canvas = image.copy()
        cv2.rectangle(canvas, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
        label = f"({self.x1},{self.y1}) -> ({self.x2},{self.y2})"
        cv2.putText(
            canvas, label, (self.x1, max(self.y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        return canvas

    def __repr__(self) -> str:
        return (
            f"Box(xyxy={self.xyxy}, "
            f"size={self.image_width}x{self.image_height}, "
            f"area={self.area}px²)"
        )


# ======================================================================== #
# Polygon                                                                    #
# ======================================================================== #

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

    # ------------------------------------------------------------------ #
    # Adapter shortcuts                                                    #
    # ------------------------------------------------------------------ #

    def supervision(self) -> dict:
        """
        Ready to unpack into sv.PolygonZone().

        Usage
        -----
        zone = sv.PolygonZone(**polygon.to_supervision())
        """
        return {"polygon": self.as_numpy}

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
