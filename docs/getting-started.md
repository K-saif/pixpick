# Getting Started

## Installation

```bash
pip install pixpick
```

Requires Python 3.9+ and OpenCV (`opencv-python` is pulled in automatically).

## Your first selection

```python
import pixpick

region = pixpick.box("image.jpg")
print(region.xyxy)   # [x1, y1, x2, y2]
```

A window opens on `image.jpg`. Drag a rectangle, release the mouse — done. The window closes and `region` is ready to use.

## Controls

**Box selector**

| Action | Control |
|---|---|
| Draw box | Left-click + drag |
| Undo last box | Right-click |
| Clear all boxes | `Z` or Backspace |
| Confirm selection | `Enter` |
| Cancel | `Esc` |

If you draw more than one box, `pixpick.box()` returns a `Multibox` holding a `Box` per rectangle.

**Polygon selector**

| Action | Control |
|---|---|
| Add vertex | Left-click |
| Undo last vertex | Right-click |
| Start a new polygon (≥ 3 points) | `Space` |
| Clear all | `Z` or Backspace |
| Confirm (≥ 3 points) | `Enter` |
| Cancel | `Esc` |

Bank a polygon with `Space` and keep drawing to get a `MultiPolygon`.

**Line selector**

| Action | Control |
|---|---|
| Add endpoint (two clicks make a line) | Left-click |
| Undo last endpoint or line | Right-click |
| Clear all | `Z` or Backspace |
| Confirm | `Enter` |
| Cancel | `Esc` |

Keep clicking pairs of endpoints to draw several lines and get a `MultiLine`.

**Point selector**

| Action | Control |
|---|---|
| Add foreground point | Left-click |
| Add background point | `Shift` + left-click |
| Undo last point | Right-click |
| Clear all | `Z` or Backspace |
| Confirm | `Enter` |
| Cancel | `Esc` |

One click returns a `Point`, several return a `MultiPoint`. The labels are what SAM uses to include or exclude a region.

## Passing an array, image, or video frame

`pixpick` accepts a file path, a numpy BGR array, or a video file with an explicit frame number.

```python
import cv2
import pixpick

region = pixpick.box(array_img)  # where array_img is a BGR numpy array

region = pixpick.box("image.jpg")

region = pixpick.box("video.mp4", frame=15)
```

## Running on a video frame

Pass `frame=` when the source is a video file to open a specific frame directly:

```python
import pixpick

region = pixpick.box("video.mp4", frame=15)
```

## Handling cancellation

If the user presses `Esc`, a `SelectionCancelled` exception is raised. Catch it if you need to handle that gracefully.

```python
from pixpick import SelectionCancelled

try:
    region = pixpick.box("image.jpg")
except SelectionCancelled:
    print("No selection made.")
```

Every selector raises the same `SelectionCancelled`, so one `except` covers all of them.
