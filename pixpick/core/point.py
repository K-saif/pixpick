from __future__ import annotations
import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from pixpick.core.box import Box
from pixpick.core.polygon import Polygon

# SAM / SAM2 / SAM3 prompt label convention.
BACKGROUND = 0
FOREGROUND = 1


@dataclass
class Point:
    """
    Immutable result of a single point selection.

    Attributes
    ----------
    x, y : int
        Absolute pixel coordinates.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    label : int
        1 = foreground, 0 = background. Defaults to foreground.
    """

    x: int
    y: int
    image_width: int
    image_height: int
    label: int = FOREGROUND

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if not (0 <= self.x <= self.image_width and 0 <= self.y <= self.image_height):
            raise ValueError(
                f"Point ({self.x},{self.y}) is outside image "
                f"({self.image_width}x{self.image_height})"
            )

        if self.label not in (BACKGROUND, FOREGROUND):
            raise ValueError(
                f"Label must be 1 (foreground) or 0 (background), got {self.label}"
            )

    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def xy(self) -> tuple[int, int]:
        """(x, y) — absolute pixels."""
        return self.x, self.y

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (2,) int32 array — [x, y]."""
        return np.array(self.xy, dtype=np.int32)

    @property
    def norm(self) -> tuple[float, float]:
        """(x, y) normalised to [0, 1]."""
        return self.x / self.image_width, self.y / self.image_height

    @property
    def norm_numpy(self) -> np.ndarray:
        """Shape (2,) float32 array of the normalised point."""
        return np.array(self.norm, dtype=np.float32)

    @property
    def is_foreground(self) -> bool:
        return self.label == FOREGROUND

    @property
    def is_background(self) -> bool:
        return self.label == BACKGROUND

    # ------------------------------------------------------------------ #
    # Framework adapters                                                   #
    # ------------------------------------------------------------------ #

    @property
    def sam_coords(self) -> np.ndarray:
        """Shape (1, 2) float32 array — SAM's `point_coords`."""
        return np.array([self.xy], dtype=np.float32)

    @property
    def sam_labels(self) -> np.ndarray:
        """Shape (1,) int32 array — SAM's `point_labels` (1=fg, 0=bg)."""
        return np.array([self.label], dtype=np.int32)

    @property
    def sam(self) -> dict:
        """
        Ready to unpack into predictor.predict().

        Unlike Box.sam (a bare [x1,y1,x2,y2] list), point prompts need two
        parallel arrays, so this returns a dict:

            masks, scores, _ = predictor.predict(**pick.sam)
        """
        return {
            "point_coords": self.sam_coords,
            "point_labels": self.sam_labels,
        }

    @property
    def supervision(self) -> dict:
        """Ready to unpack into sv.KeyPoints() — xy is (1, 1, 2) float32."""
        return {"xy": self.sam_coords[None, ...]}

    @property
    def raw(self) -> dict:
        """All formats at once."""
        return {
            "xy":                self.xy,
            "label":             self.label,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
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

        return Point(
            x=min(int(round(self.x * width / self.image_width)), width),
            y=min(int(round(self.y * height / self.image_height)), height),
            image_width=width,
            image_height=height,
            label=self.label,
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
                "xy":         list(self.xy),
                "label":      self.label,
                "normalized": list(self.norm),
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
        x, y = data["coordinates"]["xy"]
        label = data["coordinates"].get("label", FOREGROUND)
        return cls(x=x, y=y, image_width=w, image_height=h, label=label)

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
        """Draw the point on a copy of image — green foreground, red background."""
        canvas = image.copy()
        color = foreground_color if self.is_foreground else background_color

        cv2.circle(canvas, self.xy, radius, color, -1)

        return canvas

    def __repr__(self) -> str:
        kind = "fg" if self.is_foreground else "bg"
        return (
            f"Point(xy={self.xy}, {kind}, "
            f"size={self.image_width}x{self.image_height})"
        )


@dataclass
class MultiPoint:
    """
    Immutable result of a multi-point selection.

    Attributes
    ----------
    points : list[Point]
        List of Point objects — each one validates itself.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    """

    points: list[Point]
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if not self.points:
            raise ValueError("MultiPoint must contain at least one Point.")

        for i, point in enumerate(self.points):
            if not isinstance(point, Point):
                raise TypeError(
                    f"MultiPoint expects Point objects, got {type(point).__name__} "
                    f"at index {i} — build one with "
                    f"Point(x=..., y=..., image_width=..., image_height=..., label=...)"
                )

            if point.image_width != self.image_width or point.image_height != self.image_height:
                raise ValueError(
                    f"Point {i} image size does not match MultiPoint size"
                )

    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def xy(self) -> list[tuple[int, int]]:
        """[(x0, y0), (x1, y1), ...] — absolute pixels."""
        return [point.xy for point in self.points]

    @property
    def labels(self) -> list[int]:
        """One label per point — 1 = foreground, 0 = background."""
        return [point.label for point in self.points]

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (N, 2) int32 array — [[x0,y0], [x1,y1], ...]."""
        return np.array(self.xy, dtype=np.int32)

    @property
    def norm(self) -> list[tuple[float, float]]:
        """Points normalised to [0, 1]."""
        return [point.norm for point in self.points]

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
        xs = [point.x for point in self.points]
        ys = [point.y for point in self.points]
        return sum(xs) // len(xs), sum(ys) // len(ys)

    @property
    def bbox(self) -> Box:
        """Tight axis-aligned Box that encloses every point."""
        xs = [point.x for point in self.points]
        ys = [point.y for point in self.points]
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
            points=self.xy,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    # ------------------------------------------------------------------ #
    # Foreground / background split                                        #
    # ------------------------------------------------------------------ #

    @property
    def foreground(self) -> list[Point]:
        """The Point objects labelled 1."""
        return [point for point in self.points if point.is_foreground]

    @property
    def background(self) -> list[Point]:
        """The Point objects labelled 0."""
        return [point for point in self.points if point.is_background]

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
            "xy":                self.xy,
            "labels":            self.labels,
            "numpy":             self.as_numpy.tolist(),
            "normalized":        self.norm,
            "normalized_numpy":  self.norm_numpy.tolist(),
            "centroid":          self.centroid,
            "foreground":        [point.xy for point in self.foreground],
            "background":        [point.xy for point in self.background],
        }

    # ------------------------------------------------------------------ #
    # Transforms                                                           #
    # ------------------------------------------------------------------ #

    def rescale(self, width: int, height: int) -> "MultiPoint":
        """Return a new MultiPoint remapped to a different image size."""
        return MultiPoint(
            points=[point.rescale(width, height) for point in self.points],
            image_width=width,
            image_height=height,
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise to JSON."""
        data = {
            "type": "multipoint",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "points":     [list(xy) for xy in self.xy],
                "labels":     self.labels,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MultiPoint":
        """Reconstruct from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        if data["type"] != "multipoint":
            raise ValueError(f"Expected type 'multipoint', got '{data['type']}'")

        w, h = data["image_size"]
        coords = data["coordinates"]
        labels = coords.get("labels") or [FOREGROUND] * len(coords["points"])

        points = [
            Point(x=x, y=y, image_width=w, image_height=h, label=label)
            for (x, y), label in zip(coords["points"], labels)
        ]

        return cls(points=points, image_width=w, image_height=h)

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
        """Draw all points on a copy of image — green foreground, red background."""
        canvas = image.copy()

        for point in self.points:
            canvas = point.visualize(
                canvas,
                foreground_color=foreground_color,
                background_color=background_color,
                radius=radius,
            )

        return canvas

    def __repr__(self) -> str:
        return (
            f"MultiPoint(npoints={self.npoints}, "
            f"fg={self.n_foreground}, bg={self.n_background}, "
            f"size={self.image_width}x{self.image_height})"
        )
