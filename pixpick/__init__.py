"""
pixpick
-------
Interactive coordinate picker for Computer Vision frameworks.

Quick start
-----------
    import pixpick

    # Box
    region = pixpick.box("frame.jpg")
    model.predict("frame.jpg", region.yolo_region)
    print(region.xyxy)          # [x1, y1, x2, y2]
    print(region.norm)    # [0.12, 0.08, 0.64, 0.48]

    # Polygon
    zone = pixpick.polygon("frame.jpg")
    sv.PolygonZone(**zone.supervision())
    print(zone.points)          # [(x0,y0), (x1,y1), ...]

    # Save and reload either type
    region.save("zone.json")
    region = pixpick.load("zone.json")
"""
from __future__ import annotations
from pixpick.selectors.box_picker import BoxSelector
from pixpick.selectors.polygon_picker import PolygonSelector
from pixpick.selectors.line_picker import LineSelector
from pixpick.selectors.point_picker import PointSelector
from pixpick.core.box import Box, Multibox
from pixpick.core.polygon import Polygon, MultiPolygon
from pixpick.core.line import Line, MultiLine
from pixpick.core.point import Point, MultiPoint
from pixpick.utils import SelectionCancelled, ImageSource


def box(source: ImageSource, title: str = "pixpick", frame: int = 0) -> Box | Multibox:
    """Open an interactive window to select a rectangular region.

    Args:
        source (ImageSource): Image file path or BGR NumPy array.
        title (str): Window title shown to the user.
        frame (int): Zero-based frame number to load when the source is a video.

    Returns:
        Box | Multibox: The selected rectangle or multiple rectangles.

    Raises:
        SelectionCancelled: If the user presses Esc or closes the window.
    """
    return BoxSelector().select(source, title=title, frame=frame)


def polygon(source: ImageSource, title: str = "pixpick", frame: int = 0) -> Polygon | MultiPolygon:
    """Open an interactive window to select a polygon by clicking vertices.

    Controls:
        Left-click adds a point, right-click undoes the last point, Enter confirms,
        Z clears the selection, and Esc cancels.

    Args:
        source (ImageSource): Image file path or BGR NumPy array.
        title (str): Window title shown to the user.
        frame (int): Zero-based frame number to load when the source is a video.

    Returns:
        Polygon | MultiPolygon: The selected polygon or multiple polygons.

    Raises:
        SelectionCancelled: If the user presses Esc or closes the window.
    """
    return PolygonSelector().select(source, title=title, frame=frame)


def line(source: ImageSource, title: str = "pixpick", frame: int = 0) -> Line | MultiLine:
    """Open an interactive window to select a line by dragging.

    Args:
        source (ImageSource): Image file path or BGR NumPy array.
        title (str): Window title shown to the user.
        frame (int): Zero-based frame number to load when the source is a video.

    Returns:
        Line | MultiLine: The selected line or multiple lines.

    Raises:
        SelectionCancelled: If the user presses Esc or closes the window.
    """
    return LineSelector().select(source, title=title, frame=frame)


def point(source: ImageSource, title: str = "pixpick", frame: int = 0) -> Point | MultiPoint:
    """Open an interactive window to select foreground and background points.

    Controls:
        Left-click adds a foreground point, Shift+left-click adds a background point,
        right-click undoes the last point, Enter confirms, Z clears, and Esc cancels.

    Args:
        source (ImageSource): Image file path or BGR NumPy array.
        title (str): Window title shown to the user.
        frame (int): Zero-based frame number to load when the source is a video.

    Returns:
        Point | MultiPoint: Picked coordinates and foreground/background labels.
            The result can be passed directly to SAM with
            ``predictor.predict(**picks.sam)``.

    Raises:
        SelectionCancelled: If the user presses Esc or closes the window.
    """
    return PointSelector().select(source, title=title, frame=frame)


def load(path: str) -> Box | Multibox | Polygon | MultiPolygon | Line | MultiLine | Point | MultiPoint:
    """Load a previously saved selection from a JSON file.

    The selection class is determined by the JSON ``type`` field.

    Args:
        path (str): Path to the JSON file containing the saved selection.

    Returns:
        Box | Multibox | Polygon | MultiPolygon | Line | MultiLine | Point | MultiPoint:
            The loaded selection object.

    Raises:
        ValueError: If the JSON file contains an unknown selection type.
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    sel_type = data.get("type")
    if sel_type == "box":
        return Box.load(path)
    elif sel_type == "multibox":
        return Multibox.load(path)
    elif sel_type == "polygon":
        return Polygon.load(path)
    elif sel_type == "multipolygon":
        return MultiPolygon.load(path)
    elif sel_type == "line":
        return Line.load(path)
    elif sel_type == "multiline":
        return MultiLine.load(path)
    elif sel_type == "point":
        return Point.load(path)
    elif sel_type == "multipoint":
        return MultiPoint.load(path)
    else:
        raise ValueError(f"Unknown selection type in JSON: '{sel_type}'")


__all__ = [
    "box",
    "polygon",
    "line",
    "point",
    "load",
    "Box",
    "Multibox",
    "Polygon",
    "MultiPolygon",
    "Line",
    "MultiLine",
    "Point",
    "MultiPoint",
    "BoxSelector",
    "PolygonSelector",
    "LineSelector",
    "PointSelector",
    "SelectionCancelled",
]
