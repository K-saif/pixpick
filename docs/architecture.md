# Architecture

## File structure

```
docs/
├── architecture.md             # How it's built and how to extend it
├── CONTRIBUTING.md             # Contribution guidelines
├── frameworks.md               # Framework integration (YOLO, SAM, etc.)
├── getting-started.md          # Installation, first selection, controls
├── index.md                    # Home page
├── persistence.md              # Save, load, JSON schema
├── roadmap.md                  # What's coming next
└── selectors.md                # All properties and methods for every selection type
pixpick/
├── backends/
│   ├── base.py                 # BaseBackend — contract for all backends
│   └── cv2_backend.py          # CV2Backend (OpenCV window)
├── core/
│   ├── box.py                  # Box, Multibox
│   ├── line.py                 # Line, MultiLine
│   ├── point.py                # Point, MultiPoint
│   └── polygon.py              # Polygon, MultiPolygon
├── selectors/
│   ├── box_picker.py           # BoxSelector
│   ├── line_picker.py          # LineSelector
│   ├── point_picker.py         # PointSelector
│   └── polygon_picker.py       # PolygonSelector
├── __init__.py                 # box(), polygon(), line(), point(), load() — public API
└── utils.py                    # load_image(), image_size(), SelectionCancelled
```

## How the layers relate

```
pixpick.box("frame.jpg")
    │
    ▼
BoxSelector.select(source)
    ├── utils.load_image(source)     → np.ndarray
    ├── CV2Backend.select_box(image) → [[x1, y1, x2, y2], ...] | None
    └── Box(...) or Multibox(...)    → returned to caller
                │
                ├── .xyxy / .xywh / .norm / ...   (properties)
                ├── .yolo_region                  (inline, no extra file)
                ├── .sam                          (inline, no extra file)
                └── .save() / .load()             (persistence)
```

## Design decisions

**Backends are the only abstraction.**
`BaseBackend` is the one interface worth keeping because adding a new environment (Jupyter, Gradio) means writing a new backend with zero changes to selectors or selection objects. Everything else is concrete.

**Selectors are thin glue.**
A selector does three things: load the image, call the backend, wrap the result. No logic of its own.

**`pixpick.load()` dispatches on the JSON `"type"` field.**
You save any selection and load it back with the same call. The dispatcher reads `"type"` and returns the right object — one of the eight types.

**A `Multi*` type holds the singular objects, never raw coordinates.**
`Multibox` holds `Box` objects, `MultiPolygon` holds `Polygon`, `MultiLine` holds `Line`, `MultiPoint` holds `Point`. Each wrapper validates only collection-level rules (non-empty, all items share the container's image size) and delegates every coordinate property to the items, so per-item validation lives in one place.





# Backends

A backend handles the UI — opening a window, capturing mouse input, and returning raw pixel coordinates. Backends know nothing about Selection objects or frameworks; that is the selector's job.

## Available backends

| Backend | Class | Environment | Status |
|---|---|---|---|
| OpenCV window | `CV2Backend` | Local scripts | ✅ v0.1.0 |
| Matplotlib | `NotebookBackend` | Jupyter / Colab | 🔜 v0.3.0 |
| Gradio | `GradioBackend` | Headless / SSH | 🔜 v0.3.0 |

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
from pixpick.selectors.box_picker import BoxSelector

selector = BoxSelector(backend=CV2Backend())
region   = selector.select("frame.jpg")
```

Once `NotebookBackend` and `GradioBackend` land in v0.3, swapping is the same:

```python
from pixpick.backends.notebook import NotebookBackend

region = BoxSelector(backend=NotebookBackend()).select("frame.jpg")
```

## Writing a custom backend

Subclass `BaseBackend` and implement every abstract method (`select_box`, `select_polygon`, `select_line`, `select_point`). The return types are strict — selectors rely on them.

```python
from pixpick.backends.base import BaseBackend
import numpy as np


class MyBackend(BaseBackend):

    def select_box(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[list[int]] | None:
        # open your UI, capture drags
        # return [[x1, y1, x2, y2], ...] or None if cancelled
        ...

    def select_polygon(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[list[tuple[int, int]]] | None:
        # open your UI, capture clicks
        # return [[(x0,y0), (x1,y1), ...], ...]  — one list per polygon
        # or None if cancelled
        ...

    def select_line(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        # open your UI, capture clicks
        # return [((x0,y0), (x1,y1)), ...]  — one tuple per line
        # or None if cancelled
        ...

    def select_point(
        self,
        image: np.ndarray,
        title: str = "pixpick",
    ) -> tuple[list[tuple[int, int]], list[int]] | None:
        # open your UI, capture clicks
        # return ([(x0,y0), ...], [label, ...])  — 1 = foreground, 0 = background
        # or None if cancelled
        ...
```

Every method returns a **list** even for a single selection — the selector decides whether to wrap the result in a `Multi*` type. All four must return `None` on cancellation; selectors convert that into a `SelectionCancelled` exception.
