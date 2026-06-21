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

### Box controls

| Action | Control |
|---|---|
| Draw | Left-click + drag |
| Reset | `R` or `Z` |
| Cancel | `Esc` |

### Polygon controls

| Action | Control |
|---|---|
| Add vertex | Left-click |
| Undo last | Right-click |
| Clear all | `Z` |
| Confirm | `Enter` (minimum 3 points) |
| Cancel | `Esc` |

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
