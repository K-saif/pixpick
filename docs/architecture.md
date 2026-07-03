# Architecture

## File structure

```
pixpick/
├── core/
│   └── selection.py      # Box, Multibox, Polygon — all properties and to_*() methods
├── selectors/
│   ├── box.py            # BoxSelector
│   └── polygon.py        # PolygonSelector
├── backends/
│   ├── base.py           # AbstractBackend — contract for all backends
│   └── cv2_backend.py    # CV2Backend (OpenCV window)
├── utils.py              # load_image(), image_size()
└── __init__.py           # box(), polygon(), load() — public API
```

## How the layers relate

```
pixpick.box("frame.jpg")
    │
    ▼
BoxSelector.select(source)
    ├── utils.load_image(source)     → np.ndarray
    ├── CV2Backend.select_box(image) → [(x1, y1, x2, y2), ...] | None
    └── Box(...) or Multibox(...)    → returned to caller
                │
                ├── .xyxy / .xywh / .norm / ...   (properties)
                ├── .yolo_region()                           (inline, no extra file)
                ├── .sam()                           (inline, no extra file)
                └── .save() / .load()                   (persistence)
```

## Design decisions

**Backends are the only abstraction.**
`BaseBackend` is the one interface worth keeping because adding a new environment (Jupyter, Gradio) means writing a new backend with zero changes to selectors or selection objects. Everything else is concrete.

**Selectors are thin glue.**
A selector does three things: load the image, call the backend, wrap the result. No logic of its own.

**`pixpick.load()` dispatches on the JSON `"type"` field.**
You save a `Box` or `Polygon` and load it back with the same call. The dispatcher reads `"type"` and returns the right object.





# Backends

A backend handles the UI — opening a window, capturing mouse input, and returning raw pixel coordinates. Backends know nothing about Selection objects or frameworks; that is the selector's job.

## Available backends

| Backend | Class | Environment | Status |
|---|---|---|---|
| OpenCV window | `CV2Backend` | Local scripts | ✅ v0.1 |
| Matplotlib | `NotebookBackend` | Jupyter / Colab | 🔜 v0.2 |
| Gradio | `GradioBackend` | Headless / SSH | 🔜 v0.2 |

## CV2Backend (default)

Used automatically when no backend is specified. Opens a native OpenCV window.

**Requirements:** a display must be available (`DISPLAY` set on Linux, native on Windows/macOS).

```python
region = pixpick.box("frame.jpg")               # CV2Backend used by default
```

If you draw multiple boxes, the selector returns a `Multibox` instead of a single `Box`.

## Swapping backends

Pass a backend instance to any selector.

```python
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.selectors.box import BoxSelector

selector = BoxSelector(backend=CV2Backend())
region   = selector.select("frame.jpg")
```

Once `NotebookBackend` and `GradioBackend` land in v0.2, swapping is the same:

```python
from pixpick.backends.notebook import NotebookBackend

region = BoxSelector(backend=NotebookBackend()).select("frame.jpg")
```

## Writing a custom backend

Subclass `BaseBackend` and implement both methods. The return types are strict — selectors rely on them.

```python
from pixpick.backends.base import BaseBackend
import numpy as np


class MyBackend(BaseBackend):

    def select_box(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> tuple[int, int, int, int] | None:
        # open your UI, capture drag
        # return (x1, y1, x2, y2) or None if cancelled
        ...

    def select_polygon(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[tuple[int, int]] | None:
        # open your UI, capture clicks
        # return [(x0,y0), (x1,y1), ...] or None if cancelled
        ...
```

Both methods must return `None` on cancellation — selectors convert that into a `SelectionCancelled` exception.
