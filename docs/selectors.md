# Selectors

A selector opens a UI, captures user input, and returns a typed Selection object. The selection object holds all coordinate math and framework conversion methods.

Every selector accepts an image path, a video path, or a BGR numpy array, and takes `frame=` to choose the video frame.

Each selector returns a single object when you make one selection, and a `Multi*` wrapper when you make several. Every `Multi*` type holds a list of the singular objects and exposes the same properties, returning one entry per item.

| Selector | Returns (one) | Returns (several) |
|---|---|---|
| `pixpick.box()` | `Box` | `Multibox` |
| `pixpick.polygon()` | `Polygon` | `MultiPolygon` |
| `pixpick.line()` | `Line` | `MultiLine` |
| `pixpick.point()` | `Point` | `MultiPoint` |

---

## Box

```python
region = pixpick.box("video.mp4", frame=10)
```

Drag a rectangle. Drag again to add more — `pixpick.box()` returns a `Box` for one rectangle and a `Multibox` for several. Reversed drags are normalised, so `x1 < x2` and `y1 < y2` always hold.

### Properties

| Property | Description |
|---|---|
| `xyxy` | `[x1, y1, x2, y2]` absolute pixels |
| `xywh` | `[x, y, w, h]` absolute pixels |
| `cxcywh` | `[cx, cy, w, h]` absolute pixels |
| `norm` | `[x1, y1, x2, y2]` 0.0 – 1.0 |
| `norm_xywh` | `[x, y, w, h]` 0.0 – 1.0 (YOLO label format) |
| `center` | `(cx, cy)` absolute pixels |
| `area` | `int` pixels² |
| `as_numpy` | np.array shape (4,) int32 |

### Framework properties

| Property | Description |
|---|---|
| `yolo_region` | `[(x1,y1), (x2,y1), (x2,y2), (x1,y2)]` — the four corners, for YOLO region arguments |
| `yolo_prompt` | np.array shape (1, 4) — YOLOE `bboxes` visual prompt |
| `sam` | `[x1, y1, x2, y2]` — SAM box prompt |
| `raw` | dict with all coordinate formats in one place |

### Visualise

```python
canvas = region.visualize(image)                          # default green
canvas = region.visualize(image, color=(0,0,255), thickness=2)
cv2.imshow("result", canvas)
cv2.waitKey(0)
```

### Persistence

```python
region.save("selection.json")
region = pixpick.load("selection.json")   # or Box.load("selection.json")
```

### Multi-box results

```python
region = pixpick.box("image.jpg")

region.boxes        # [Box(...), Box(...), ...]   ← real Box objects
region.xyxy         # [[x1, y1, x2, y2], ...]
region.center       # [(cx, cy), ...]
region.as_numpy     # np.array shape (N, 4) int32
region.yolo_prompt  # np.array shape (N, 4)
region.nboxes       # int

region.boxes[0].area                # every Box property is available per box
region.boxes[0].visualize(image)
```

Build one from `Box` objects, not raw coordinates — raw lists raise `TypeError`:

```python
Multibox(
    boxes=[Box(x1=100, y1=50, x2=400, y2=300, image_width=1920, image_height=1080)],
    image_width=1920,
    image_height=1080,
)
```

---

## Polygon

```python
zone = pixpick.polygon("video.mp4", frame=10)
```

Click vertices in order. Minimum 3 points before `Enter` confirms. Press `Space` to bank the current polygon and start another — several polygons return a `MultiPolygon`.

### Properties

| Property | Description |
|---|---|
| `points` | list of `(x, y)` tuples, absolute pixels |
| `as_numpy` | np.array shape (N, 2) int32 |
| `norm` | list of `(x, y)` tuples, 0.0 – 1.0 |
| `norm_numpy` | np.array shape (N, 2) float32 |
| `npoints` | `int` |
| `bbox` | `[x1, y1, x2, y2]` — tight axis-aligned bounds |
| `image_width` / `image_height` | `int` — source image size |

### Framework properties

| Property | Description |
|---|---|
| `yolo_region` | the vertex list, for YOLO region arguments |
| `supervision` | `{"polygon": np.array}` — unpack into `sv.PolygonZone(**zone.supervision)` |
| `raw` | dict with all coordinate formats in one place |

### Visualise

```python
canvas = zone.visualize(image)                        # default green, 15% fill
canvas = zone.visualize(image, color=(0,0,255), fill_alpha=0.3)
```

### Persistence

```python
zone.save("zone.json")
zone = pixpick.load("zone.json")   # or Polygon.load("zone.json")
```

### Multi-polygon results

```python
zones = pixpick.polygon("frame.jpg")

zones.polygons      # [Polygon(...), Polygon(...), ...]  ← real Polygon objects
zones.as_numpy      # [np.array (N,2), np.array (N,2), ...]  one array per polygon
zones.norm          # [[(x0n,y0n), ...], ...]
zones.npoints       # [int, int, ...]
zones.box           # [[x1,y1,x2,y2], ...]   bounds of each polygon
zones.supervision   # [{"polygon": np.array}, ...]

zones.polygons[0].npoints           # every Polygon property is available per polygon
```

---

## Line

```python
line = pixpick.line("image.jpg")
```

Click the start point, then the end point — that is one line. Keep clicking pairs to add more; several lines return a `MultiLine`.

### Properties

| Property | Description |
|---|---|
| `points` | `[(x1, y1), (x2, y2)]` absolute pixels |
| `start` | `(x1, y1)` absolute pixels |
| `end` | `(x2, y2)` absolute pixels |
| `center` | `(cx, cy)` absolute pixels |
| `length` | `float` pixels |
| `vector` | `(dx, dy)` from start to end |
| `as_numpy` | np.array shape (2, 2) int32 |
| `norm` | `[(x1n, y1n), (x2n, y2n)]` 0.0 – 1.0 |
| `norm_numpy` | np.array shape (2, 2) float32 |
| `horizontal` | `[(x, y), (x, y)]` — the same line re-drawn horizontally through its centre |
| `vertical` | `[(x, y), (x, y)]` — the same line re-drawn vertically through its centre |
| `raw` | dict with all formats in one place |

> **Note:** `horizontal` and `vertical` return **coordinates**, not booleans. They give you a line of the same length, re-aligned through the original's centre — handy for snapping a hand-drawn counting line to an axis.

### Visualise

```python
canvas = line.visualize(image)                        # default green
canvas = line.visualize(image, color=(0,0,255), thickness=2)
```

### Persistence

```python
line.save("line.json")
line = pixpick.load("line.json")   # or Line.load("line.json")
```

### Multi-line results

```python
lines = pixpick.line("frame.jpg")

lines.lines          # [Line(...), Line(...), ...]   ← real Line objects
lines.points         # [[(x1,y1), (x2,y2)], ...]
lines.start          # [(x1,y1), ...]
lines.end            # [(x2,y2), ...]
lines.length         # [length1, length2, ...]
lines.center         # [(cx, cy), ...]
lines.as_numpy       # np.array shape (N, 2, 2) int32
lines.nlines         # int

lines.lines[0].center       # every Line property is available per line
```

`MultiLine` saves as `"type": "multiline"` and reloads through `pixpick.load()`.

---

## Point

```python
pick  = pixpick.point("image.jpg")     # one click  → Point
picks = pixpick.point("image.jpg")     # several    → MultiPoint
```

Left-click marks a **foreground** point, `Shift`+left-click marks a **background** point — the labels SAM uses to include or exclude a region.

### Point properties

| Property | Description |
|---|---|
| `xy` | `(x, y)` absolute pixels |
| `label` | `1` foreground, `0` background |
| `is_foreground` / `is_background` | `bool` |
| `norm` | `(xn, yn)` 0.0 – 1.0 |
| `as_numpy` | np.array shape (2,) int32 |
| `norm_numpy` | np.array shape (2,) float32 |
| `raw` | dict with all formats in one place |

### MultiPoint properties

| Property | Description |
|---|---|
| `points` | `[Point(...), ...]` — real `Point` objects |
| `xy` | `[(x0, y0), (x1, y1), ...]` absolute pixels |
| `labels` | `[1, 0, ...]` one per point |
| `foreground` / `background` | `[Point(...), ...]` filtered by label |
| `n_foreground` / `n_background` | `int` |
| `npoints` | `int` |
| `centroid` | `(cx, cy)` mean of all points |
| `bbox` | `Box` — tight box around every point |
| `as_polygon` | `Polygon` — the points as a shape (needs 3+) |
| `as_numpy` | np.array shape (N, 2) int32 |
| `norm` / `norm_numpy` | normalised, 0.0 – 1.0 |
| `raw` | dict with all formats in one place |

### Framework properties

Both types expose the same adapters, so code written against one works with the other.

| Property | Description |
|---|---|
| `sam` | `{"point_coords": (N,2) float32, "point_labels": (N,) int32}` — unpack with `**` |
| `sam_coords` | the coordinate array on its own |
| `sam_labels` | the label array on its own |
| `supervision` | `{"xy": (1, N, 2) float32}` — unpack into `sv.KeyPoints()` |

### Rescale

Pick on a full-resolution frame, run inference on a resized one:

```python
picks_640 = picks.rescale(640, 640)
```

### Visualise

```python
canvas = picks.visualize(image)   # green foreground, red background
canvas = picks.visualize(image, foreground_color=(255,0,0), radius=8)
```

### Persistence

```python
picks.save("picks.json")
picks = pixpick.load("picks.json")   # Point → "point", MultiPoint → "multipoint"
```

---

## Coming in future releases

| Selector | Interaction | Returns | Release |
|---|---|---|---|
| `pixpick.perspective()` | 4-corner click | `Perspective` | v0.3.0 |
