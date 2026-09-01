from __future__ import annotations
import cv2
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from pixpick.core.box import Box
from pixpick.core.polygon import Polygon

# SAM / SAM2 / SAM3 prompt label convention.
BACKGROUND = 0
FOREGROUND = 1


@dataclass
class Point:
    """
    Immutable result of a point selection.

    Attributes
    ----------
    points : list[tuple[int, int]]
        Ordered list of (x, y) points in absolute pixels.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    labels : list[int]
        One label per point — 1 = foreground, 0 = background.
        Defaults to all-foreground when omitted.
    """

    points: list[tuple[int, int]]
    image_width: int
    image_height: int
    labels: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if len(self.points) < 1:
            raise ValueError(
                f"Point needs at least 1 point, got {len(self.points)}"
            )
        for i, pt in enumerate(self.points):
            x, y = pt
            if not (0 <= x <= self.image_width and 0 <= y <= self.image_height):
                raise ValueError(
                    f"Point {i} ({x},{y}) is outside image "
                    f"({self.image_width}x{self.image_height})"
                )

        # No labels given → every point is a foreground prompt.
        if not self.labels:
            self.labels = [FOREGROUND] * len(self.points)

        if len(self.labels) != len(self.points):
            raise ValueError(
                f"Need one label per point: got {len(self.labels)} labels "
                f"for {len(self.points)} points"
            )

        for i, label in enumerate(self.labels):
            if label not in (BACKGROUND, FOREGROUND):
                raise ValueError(
                    f"Label {i} must be 1 (foreground) or 0 (background), got {label}"
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
    def npoints(self) -> int:
        return len(self.points)

    # ------------------------------------------------------------------ #
    # Geometry                                                             #
    # ------------------------------------------------------------------ #

    @property
    def centroid(self) -> tuple[int, int]:
        """(cx, cy) — mean of all points, absolute pixels."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return sum(xs) // len(xs), sum(ys) // len(ys)

    @property
    def bbox(self) -> Box:
        """Tight axis-aligned Box that encloses every point."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        if x1 == x2 or y1 == y2:
            raise ValueError(
                "Points do not enclose any area — bbox needs at least two "
                "points that differ in both x and y"
            )

        return Box(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    @property
    def as_polygon(self) -> Polygon:
        """The points as a Polygon — needs at least 3 of them."""
        return Polygon(
            points=list(self.points),
            image_width=self.image_width,
            image_height=self.image_height,
        )

    # ------------------------------------------------------------------ #
    # Foreground / background split                                        #
    # ------------------------------------------------------------------ #

    @property
    def foreground(self) -> list[tuple[int, int]]:
        """Points labelled 1."""
        return [
            pt for pt, label in zip(self.points, self.labels)
            if label == FOREGROUND
        ]

    @property
    def background(self) -> list[tuple[int, int]]:
        """Points labelled 0."""
        return [
            pt for pt, label in zip(self.points, self.labels)
            if label == BACKGROUND
        ]

    @property
    def n_foreground(self) -> int:
        return len(self.foreground)

    @property
    def n_background(self) -> int:
        return len(self.background)

    # ------------------------------------------------------------------ #
    # Framework adapters                                                   #
    # ------------------------------------------------------------------ #

    @property
    def sam_coords(self) -> np.ndarray:
        """Shape (N, 2) float32 array — SAM's `point_coords`."""
        return self.as_numpy.astype(np.float32)

    @property
    def sam_labels(self) -> np.ndarray:
        """Shape (N,) int32 array — SAM's `point_labels` (1=fg, 0=bg)."""
        return np.array(self.labels, dtype=np.int32)

    @property
    def sam(self) -> dict:
        """
        Ready to unpack into predictor.predict().

        Unlike Box.sam (a bare [x1,y1,x2,y2] list), point prompts need two
        parallel arrays, so this returns a dict:

            masks, scores, _ = predictor.predict(**picks.sam)
        """
        return {
            "point_coords": self.sam_coords,
            "point_labels": self.sam_labels,
        }

    @property
    def supervision(self) -> dict:
        """Ready to unpack into sv.KeyPoints() — xy is (1, N, 2) float32."""
        return {"xy": self.sam_coords[None, ...]}

    @property
    def raw(self) -> dict:
        """All formats at once."""
        return {
            "points":            self.points,
            "labels":            self.labels,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
            "centroid":          self.centroid,
            "foreground":        self.foreground,
            "background":        self.background,
        }

    # ------------------------------------------------------------------ #
    # Transforms                                                           #
    # ------------------------------------------------------------------ #

    def rescale(self, width: int, height: int) -> "Point":
        """
        Return a new Point remapped to a different image size.

        Handy when you pick on a full-resolution frame but run inference on a
        resized one.
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Target size must be positive, got {width}x{height}"
            )

        scale_x = width / self.image_width
        scale_y = height / self.image_height

        points = [
            (
                min(int(round(x * scale_x)), width),
                min(int(round(y * scale_y)), height),
            )
            for x, y in self.points
        ]

        return Point(
            points=points,
            image_width=width,
            image_height=height,
            labels=list(self.labels),
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise to JSON."""
        data = {
            "type": "point",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "points":     self.points,
                "labels":     self.labels,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Point":
        """Reconstruct from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        if data["type"] != "point":
            raise ValueError(f"Expected type 'point', got '{data['type']}'")
        w, h = data["image_size"]
        points = [tuple(p) for p in data["coordinates"]["points"]]
        labels = list(data["coordinates"].get("labels", []))
        return cls(points=points, image_width=w, image_height=h, labels=labels)

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        image: np.ndarray,
        foreground_color: tuple = (0, 255, 0),
        background_color: tuple = (0, 0, 255),
        radius: int = 5,
    ) -> np.ndarray:
        """Draw the points on a copy of image — green foreground, red background."""
        canvas = image.copy()

        for i, ((x, y), label) in enumerate(zip(self.points, self.labels)):
            color = foreground_color if label == FOREGROUND else background_color

            cv2.circle(canvas, (x, y), radius, color, -1)
            cv2.putText(
                canvas, str(i), (x + radius + 2, y - radius),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        return canvas

    def __repr__(self) -> str:
        return (
            f"Point(npoints={self.npoints}, "
            f"fg={self.n_foreground}, bg={self.n_background}, "
            f"size={self.image_width}x{self.image_height})"
        )
