from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """
    Contract every adapter must satisfy.

    An adapter takes a Selection object and returns a dict that can be
    unpacked directly into the target framework's API call.

    Example
    -------
    adapter = YOLOAdapter()
    kwargs  = adapter.convert(box)          # → {"crop": [x1, y1, x2, y2]}
    model.predict("image.jpg", **kwargs)    # no extra work
    """

    @abstractmethod
    def convert(self, selection: Any) -> dict:
        """
        Convert a Selection object to a framework-ready dict.

        Parameters
        ----------
        selection : Selection
            Any selection object (Box, Polygon, Points, …).
            Adapters should type-check and raise TypeError for wrong types.

        Returns
        -------
        dict
            Ready to unpack into the target framework call.
        """
        ...

    def validate(self, selection: Any, expected_type: type) -> None:
        """
        Helper: raise TypeError if selection is not the expected type.
        Call this at the top of convert() in every concrete adapter.
        """
        if not isinstance(selection, expected_type):
            raise TypeError(
                f"{self.__class__.__name__} expects {expected_type.__name__}, "
                f"got {type(selection).__name__}"
            )