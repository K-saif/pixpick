"""Registry that maps backend names to backend instances.

Backends own the UI layer and return raw pixel coordinates; selectors stay
backend-agnostic. Importing backend classes is deferred to the moment a
backend is actually requested, so optional dependencies such as Matplotlib
are never imported unless needed.
"""
from __future__ import annotations

from typing import Union

from pixpick.backends.base import BaseBackend

BackendChoice = Union[str, BaseBackend]


def resolve_backend(backend: BackendChoice) -> BaseBackend:
    """Resolve a backend name or instance into a ready-to-use backend.

    Args:
        backend (BackendChoice): Either a ``BaseBackend`` instance (returned
            unchanged) or a backend name: ``"cv2"`` for the OpenCV desktop
            window backend, ``"notebook"`` for the Matplotlib notebook
            backend.

    Returns:
        BaseBackend: A backend instance the selector can call.

    Raises:
        ValueError: If ``backend`` is a string that matches no known backend.
        TypeError: If ``backend`` is neither a string nor a ``BaseBackend``.
    """

    if isinstance(backend, BaseBackend):
        return backend

    if isinstance(backend, str):
        name = backend.strip().lower()

        if name == "cv2":
            from pixpick.backends.cv2_backend import CV2Backend

            return CV2Backend()

        if name == "notebook":
            from pixpick.backends.notebook_backend import NotebookBackend

            return NotebookBackend()

        raise ValueError(
            f"Unknown backend {backend!r}. Available backends: 'cv2', 'notebook'."
        )

    raise TypeError(
        "backend must be a BaseBackend instance or one of 'cv2', 'notebook'; "
        f"got {type(backend).__name__}"
    )
