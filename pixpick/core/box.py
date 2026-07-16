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

    def sam(self) -> list[int]:
        """[(point1), (point2), (point3), (point4)] """
        return [self.x1, self.y1, self.x2, self.y2]

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

@dataclass
class Multibox:
    """
    Immutable result of a multi-box selection.

    Attributes
    ----------
    boxes : list[Box]
        List of Box objects.
    image_width, image_height : int
        Dimensions of the source image — needed for normalisation.
    """

    boxes: list[Box]
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        if not self.boxes:
            raise ValueError("Multibox must contain at least two Box.")
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            # Normalise so x1 < x2 and y1 < y2 regardless of drag direction.
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            if x1 == x2 or y1 == y2:
                raise ValueError(f"Box {i} has zero area — did the drag complete?")

            if not (0 <= x1 < x2 <= self.image_width):
                raise ValueError(
                    f"for box {i} coords out of bounds: {x1}, {x2} (image width={self.image_width})"
                )
            if not (0 <= y1 < y2 <= self.image_height):
                raise ValueError(
                    f"for box {i} coords out of bounds: {y1}, {y2} (image height={self.image_height})"
                )


    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def xyxy(self) -> list[int]:
        """[x1, y1, x2, y2] — absolute pixels for each box in boxes."""
        return self.boxes

    @property
    def xywh(self) -> list[int]:
        """[x, y, width, height] — top-left origin, absolute pixels for each box in boxes."""
        return [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in self.boxes]

    @property
    def cxcywh(self) -> list[float]:
        """[cx, cy, width, height] — centre + size, absolute pixels for each box in boxes."""
        return [[box[0] + (box[2] - box[0]) / 2, box[1] + (box[3] - box[1]) / 2, float(box[2] - box[0]), float(box[3] - box[1])] for box in self.boxes]

    @property
    def norm(self) -> list[float]:
        """[x1, y1, x2, y2] — values in [0, 1] for each box in boxes."""
        return [[
            box[0] / self.image_width,
            box[1] / self.image_height,
            box[2] / self.image_width,
            box[3] / self.image_height,
        ] for box in self.boxes]

    @property
    def norm_xywh(self) -> list[list[float]]:
        """[[x, y, w, h], ...] normalised — YOLO label format for each box in boxes."""
        return [
            [
                box[0] / self.image_width,
                box[1] / self.image_height,
                (box[2] - box[0]) / self.image_width,
                (box[3] - box[1]) / self.image_height,
            ]
            for box in self.boxes
        ]

    @property
    def center(self) -> tuple[int, int]:
        """(cx, cy) in absolute pixels.for each box in boxes."""
        return [[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] for box in self.boxes]


    @property
    def area(self) -> int:
        """Area in pixels² for each box in boxes."""
        return [(box[2] - box[0]) * (box[3] - box[1]) for box in self.boxes]

    @property
    def as_numpy(self) -> np.ndarray:
        """Shape (N, 4) int32 array — [[x1, y1, x2, y2], ...]."""
        return np.array(self.boxes, dtype=np.int32)


    # ------------------------------------------------------------------ #
    # Adapter shortcuts                                                    #
    # ------------------------------------------------------------------ #

    def yolo_region(self) -> list[float]:
        """[(point1), (point2), (point3), (point4)] """
        return [[
            (box[0], box[1]),
            (box[2], box[1]),
            (box[2], box[3]),
            (box[0], box[3])
        ] for box in self.boxes ]

    def yolo_prompt(self) -> np.ndarray:
        """[(point1), (point2), (point3), (point4)] """
        return np.array([
            [box[0], box[1], box[2], box[3]]
            for box in self.boxes
        ])

    def sam(self) -> list[list[int]] :
        """[(point1), (point2), (point3), (point4)] """
        return self.boxes


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
            "type": "multibox",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "boxes":      self.boxes,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Multibox":
        """Reconstruct from a saved JSON file."""
        data = json.loads(Path(path).read_text())
        if data["type"] != "multibox":
            raise ValueError(f"Expected type 'multibox', got '{data['type']}'")
        w, h = data["image_size"]
        boxes = [list(box) for box in data["coordinates"]["boxes"]]
        return cls(boxes=boxes, image_width=w, image_height=h)

    # ------------------------------------------------------------------ #
    # Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        image: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw all boxes on a copy of image and return it."""
        canvas = image.copy()
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"{i}: ({x1},{y1}) -> ({x2},{y2})"
            cv2.putText(
                canvas, label, (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return canvas

    def __repr__(self) -> str:
        return (
            f"Multibox(nboxes={len(self.boxes)}, "
            f"size={self.image_width}x{self.image_height})"
        )

