# Getting Started

## Installation

```bash
pip install pixpick
```

Requires Python 3.8+ and OpenCV (`opencv-python` is pulled in automatically).

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
| Clear all boxes | `Z` or Backspace/Delete |
| Confirm selection | `Enter` or `Space` |
| Cancel | `Esc` |

If you draw more than one box, `pixpick.box()` returns a `Multibox` that holds all selected coordinates.

**Polygon selector**

| Action | Control |
|---|---|
| Add vertex | Left-click |
| Undo last vertex | Right-click |
| Clear all | `Z` |
| Confirm (≥ 3 points) | `Enter` |
| Cancel | `Esc` |

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
from pixpick.selectors.box import SelectionCancelled

try:
    region = pixpick.box("image.jpg")
except SelectionCancelled:
    print("No selection made.")
```
