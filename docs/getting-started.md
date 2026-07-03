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

## Passing an array instead of a path

`pixpick` accepts a file path or a numpy BGR array directly — whatever you already have in memory.

```python
import cv2
import pixpick

frame = cv2.imread("frame.jpg")
region = pixpick.box(frame)
```

## Running on a video frame

There is no video selector yet (roadmap v0.3), but pulling a frame manually is one line:

```python
import cv2
import pixpick

cap = cv2.VideoCapture("video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)   # jump to frame 150
_, frame = cap.read()
cap.release()

region = pixpick.box(frame)
```

## Handling cancellation

If the user presses `Esc`, a `SelectionCancelled` exception is raised. Catch it if you need to handle that gracefully.

```python
from pixpick.selectors.box import SelectionCancelled

try:
    region = pixpick.box("frame.jpg")
except SelectionCancelled:
    print("No selection made.")
```
