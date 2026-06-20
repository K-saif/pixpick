"""
pixpick
-------
Interactive coordinate picker for Computer Vision frameworks.

Quick start
-----------
    import pixpick

    region = pixpick.box("frame.jpg")

    # use immediately with any framework
    model.predict("frame.jpg", **region.to_yolo())

    # or inspect the raw coords
    print(region.xyxy)        # [x1, y1, x2, y2]
    print(region.normalized)  # [0.12, 0.08, 0.64, 0.48]

    # save and reload
    region.save("zone.json")
    region = pixpick.load("zone.json")
"""

from pixpick.selectors.box import BoxSelector, SelectionCancelled
from pixpick.core.selection import Box
from pixpick.utils import ImageSource


def box(source: ImageSource, title: str = "pixpick") -> Box:
    """
    Open an interactive window on `source`, let the user drag a box,
    and return a Box selection object.

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

    Example
    -------
    >>> region = pixpick.box("frame.jpg")
    >>> region.xyxy
    [120, 80, 640, 480]
    >>> model.predict("frame.jpg", **region.to_yolo())
    """
    return BoxSelector().select(source, title=title)


def load(path: str) -> Box:
    """Load a previously saved Box from a JSON file."""
    return Box.load(path)


__all__ = [
    "box",
    "load",
    "Box",
    "BoxSelector",
    "SelectionCancelled",
]