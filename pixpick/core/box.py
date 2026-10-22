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
    """Represent a selected bounding box.

    Attributes:
        x1: Left coordinate in absolute image pixels.
        y1: Top coordinate in absolute image pixels.
        x2: Right coordinate in absolute image pixels.
        y2: Bottom coordinate in absolute image pixels.
        image_width: Width of the source image, used for normalization.
        image_height: Height of the source image, used for normalization.
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
        """Return coordinates in ``[x1, y1, x2, y2]`` format, in absolute pixels.

        Returns:
            list[int]: Absolute pixel coordinates.
        """
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def xywh(self) -> list[int]:
        """Return the box in top-left ``[x, y, width, height]`` format.

        Returns:
            list[int]: Top-left coordinates and dimensions in absolute pixels.
        """
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]

    @property
    def cxcywh(self) -> list[float]:
        """Return the box in center-and-size format used by YOLO labels.

        Returns:
            list[float]: Center coordinates and dimensions in absolute pixels.
        """
        w, h = self.x2 - self.x1, self.y2 - self.y1
        return [self.x1 + w / 2, self.y1 + h / 2, float(w), float(h)]

    @property
    def norm(self) -> list[float]:
        """Return normalized ``[x1, y1, x2, y2]`` coordinates.

        Returns:
            list[float]: Coordinates normalized to the range ``[0, 1]``.
        """
        return [
            self.x1 / self.image_width,
            self.y1 / self.image_height,
            self.x2 / self.image_width,
            self.y2 / self.image_height,
        ]

    @property
    def norm_xywh(self) -> list[float]:
        """Return normalized top-left coordinates and dimensions.

        Returns:
            list[float]: Normalized ``[x, y, width, height]`` values.
        """
        x1n, y1n, x2n, y2n = self.norm
        return [x1n, y1n, x2n - x1n, y2n - y1n]

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point in absolute pixels.

        Returns:
            tuple[int, int]: Integer center coordinates ``(cx, cy)``.
        """
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        """Return the area of the box.

        Returns:
            int: Box area in square pixels.
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def as_numpy(self) -> np.ndarray:
        """Return the box coordinates as a NumPy array.

        Returns:
            np.ndarray: An ``int32`` array of shape ``(4,)`` containing
                ``[x1, y1, x2, y2]``.
        """
        return np.array(self.xyxy, dtype=np.int32)

    @property
    def yolo_region(self) -> list[float]:
        """Return the four box corners for a YOLO prompt.

        Returns:
            list[tuple[int, int]]: Corners in clockwise order, starting at
                the top-left corner.
        """
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2)
        ]

    @property
    def yolo_prompt(self) -> np.ndarray:
        """Return the box coordinates as a NumPy prompt array.

        Returns:
            np.ndarray: An array containing the box corner coordinates.
        """
        return np.array([
            [self.x1, self.y1, self.x2, self.y2]
        ])

    @property
    def sam(self) -> list[int]:
        """Return coordinates in the format expected by Segment Anything Model.

        Returns:
            list[int]: Coordinates in ``[x1, y1, x2, y2]`` format.
        """
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def raw(self) -> dict:
        """Return the box coordinates in all supported formats.

        Returns:
            dict: A dictionary containing coordinate formats and a NumPy
                representation suitable for debugging.
        """
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
        """Save the box selection to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
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
        """Load a box selection from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Box: The reconstructed bounding box.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            KeyError: If required fields are missing.
            ValueError: If the saved selection has the wrong type or invalid
                coordinates.
        """
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
        """Draw the bounding box on a copy of the image.

        Args:
            image: Source image as a NumPy array.
            color: Rectangle and label color in BGR order. Defaults to green.
            thickness: Rectangle line thickness in pixels.

        Returns:
            np.ndarray: Copy of the image with the box and coordinate label drawn.
        """
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
    """Represent a selection containing multiple bounding boxes.

    Attributes:
        boxes: List of ``Box`` objects. Each box validates and normalizes itself.
        image_width: Width of the source image, used for normalization.
        image_height: Height of the source image, used for normalization.
    """

    boxes: list[Box]
    image_width: int
    image_height: int

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if not self.boxes:
            raise ValueError("Multibox must contain at least one Box.")

        for i, box in enumerate(self.boxes):
            if not isinstance(box, Box):
                raise TypeError(
                    f"Multibox expects Box objects, got {type(box).__name__} "
                    f"at index {i} — build one with "
                    f"Box(x1=..., y1=..., x2=..., y2=..., image_width=..., image_height=...)"
                )

            if box.image_width != self.image_width or box.image_height != self.image_height:
                raise ValueError(
                    f"Box {i} image size does not match Multibox size"
                )

    # ------------------------------------------------------------------ #
    # Core format properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def xyxy(self) -> list[list[int]]:
        """Return all boxes in ``[x1, y1, x2, y2]`` format.

        Returns:
            list[list[int]]: Absolute pixel coordinates for each box.
        """
        return [box.xyxy for box in self.boxes]

    @property
    def xywh(self) -> list[list[int]]:
        """Return all boxes in top-left ``[x, y, width, height]`` format.

        Returns:
            list[list[int]]: Coordinates and dimensions in absolute pixels.
        """
        return [box.xywh for box in self.boxes]

    @property
    def cxcywh(self) -> list[list[float]]:
        """Return all boxes in center-and-size format used by YOLO labels.

        Returns:
            list[list[float]]: Center coordinates and dimensions in absolute pixels.
        """
        return [box.cxcywh for box in self.boxes]

    @property
    def norm(self) -> list[list[float]]:
        """Return normalized coordinates for all boxes.

        Returns:
            list[list[float]]: Coordinates normalized to the range ``[0, 1]``.
        """
        return [box.norm for box in self.boxes]

    @property
    def norm_xywh(self) -> list[list[float]]:
        """Return normalized top-left coordinates and dimensions for all boxes.

        Returns:
            list[list[float]]: Normalized ``[x, y, width, height]`` values.
        """
        return [box.norm_xywh for box in self.boxes]

    @property
    def center(self) -> list[tuple[int, int]]:
        """Return the center point of each box in absolute pixels.

        Returns:
            list[tuple[int, int]]: Integer center coordinates for each box.
        """
        return [box.center for box in self.boxes]

    @property
    def area(self) -> list[int]:
        """Return the area of each box.

        Returns:
            list[int]: Box areas in square pixels.
        """
        return [box.area for box in self.boxes]

    @property
    def nboxes(self) -> int:
        """Return the number of selected boxes.

        Returns:
            int: Number of boxes in the selection.
        """
        return len(self.boxes)

    @property
    def as_numpy(self) -> np.ndarray:
        """Return all box coordinates as a NumPy array.

        Returns:
            np.ndarray: An ``int32`` array of shape ``(N, 4)``.
        """
        return np.array(self.xyxy, dtype=np.int32)

    @property
    def yolo_region(self) -> list[list[tuple[int, int]]]:
        """Return the four corners of each box for YOLO prompts.

        Returns:
            list[list[tuple[int, int]]]: Corner coordinates for each box.
        """
        return [box.yolo_region for box in self.boxes]

    @property
    def yolo_prompt(self) -> np.ndarray:
        """Return all box coordinates in a NumPy prompt array.

        Returns:
            np.ndarray: An array of shape ``(N, 4)`` containing box coordinates.
        """
        return np.vstack([box.yolo_prompt for box in self.boxes])

    @property
    def sam(self) -> list[list[int]]:
        """Return coordinates in the format expected by Segment Anything Model.

        Returns:
            list[list[int]]: Coordinates for each box.
        """
        return [box.sam for box in self.boxes]

    @property
    def raw(self) -> dict:
        """Return the box coordinates in all supported formats.

        Returns:
            dict: A dictionary containing coordinate formats and a NumPy
                representation suitable for debugging.
        """
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
        """Save the multi-box selection to a JSON file.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        data = {
            "type": "multibox",
            "image_size": [self.image_width, self.image_height],
            "coordinates": {
                "boxes":      self.xyxy,
                "normalized": self.norm,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Multibox":
        """Load a multi-box selection from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Multibox: The reconstructed multi-box selection.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file does not contain valid JSON.
            KeyError: If required fields are missing.
            ValueError: If the saved selection has the wrong type or invalid
                coordinates.
        """
        data = json.loads(Path(path).read_text())
        if data["type"] != "multibox":
            raise ValueError(f"Expected type 'multibox', got '{data['type']}'")
        w, h = data["image_size"]
        boxes = [
            Box(x1=x1, y1=y1, x2=x2, y2=y2, image_width=w, image_height=h)
            for x1, y1, x2, y2 in data["coordinates"]["boxes"]
        ]
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
        """Draw all bounding boxes on a copy of the image.

        Args:
            image: Source image as a NumPy array.
            color: Rectangle and label color in BGR order. Defaults to green.
            thickness: Rectangle line thickness in pixels.

        Returns:
            np.ndarray: Copy of the image with all boxes and coordinate labels drawn.
        """
        canvas = image.copy()

        for box in self.boxes:
            canvas = box.visualize(canvas, color=color, thickness=thickness)

        return canvas

    def __repr__(self) -> str:
        return (
            f"Multibox(nboxes={len(self.boxes)}, "
            f"size={self.image_width}x{self.image_height})"
        )
