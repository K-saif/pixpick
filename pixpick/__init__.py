"""
pixpick
-------
Interactive coordinate picker for Computer Vision frameworks.

Quick start
-----------
    import pixpick

    # Box
    region = pixpick.box("frame.jpg")
    model.predict("frame.jpg", **region.yolo_region())
    print(region.xyxy)          # [x1, y1, x2, y2]
    print(region.norm)    # [0.12, 0.08, 0.64, 0.48]

    # Polygon
    zone = pixpick.polygon("frame.jpg")
    sv.PolygonZone(**zone.to_supervision())
    print(zone.points)          # [(x0,y0), (x1,y1), ...]

    # Save and reload either type
    region.save("zone.json")
    region = pixpick.load("zone.json")
"""
from __future__ import annotations
from pixpick.selectors.box import BoxSelector
from pixpick.selectors.polygon import PolygonSelector, SelectionCancelled
from pixpick.core.selection import Box, Multibox, Polygon, Line
from pixpick.utils import ImageSource


def box(source: ImageSource, title: str = "pixpick") -> Box:
    """
    Open an interactive window on `source`, drag a rectangle, return a Box.

    Parameters
    ----------
    source : str | Path | np.ndarray
        Image file path or BGR numpy array.
    title : str
        Window title shown to the user.

    Returns
    -------
    Box

    Raises
    ------
    SelectionCancelled
        If the user pressed Esc.
    """
    return BoxSelector().select(source, title=title)


def polygon(source: ImageSource, title: str = "pixpick") -> Polygon:
    """
    Open an interactive window on `source`, click vertices, return a Polygon.

    Controls: LMB=add point  RMB=undo last  Enter=confirm  Z=clear  Esc=cancel

    Parameters
    ----------
    source : str | Path | np.ndarray
        Image file path or BGR numpy array.
    title : str
        Window title shown to the user.

    Returns
    -------
    Polygon

    Raises
    ------
    SelectionCancelled
        If the user pressed Esc.
    """
    return PolygonSelector().select(source, title=title)

def line(source: ImageSource, title: str = "pixpick") -> Line:
    """
    Open an interactive window on `source`, drag a line, return a Line.

    Parameters
    ----------
    source : str | Path | np.ndarray
        Image file path or BGR numpy array.
    title : str
        Window title shown to the user.

    Returns
    -------
    Line

    Raises
    ------
    SelectionCancelled
        If the user pressed Esc.
    """
    from pixpick.selectors.line import LineSelector
    return LineSelector().select(source, title=title)

def load(path: str) -> Box | Multibox | Polygon | Line:
    """
    Load a previously saved selection from a JSON file.
    Dispatches to Box.load or Polygon.load based on the 'type' field.
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
    else:
        raise ValueError(f"Unknown selection type in JSON: '{sel_type}'")


__all__ = [
    "box",
    "polygon",
    "line",
    "load",
    "Box",
    "Polygon",
    "Line",
    "BoxSelector",
    "PolygonSelector",
    "LineSelector",
    "SelectionCancelled",
]