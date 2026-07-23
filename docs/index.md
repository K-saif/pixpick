# PixPick

**Interactive coordinate picker for Computer Vision.**

Draw boxes, polygons and lines on images or videos and instantly get coordinates for YOLO, SAM, YOLOE, OpenCV, and your own pipelines.


![Project Overview](./pixpick_main.png)

---
## Why PixPick?

Most computer vision frameworks require coordinates before inference.

Traditionally you have to:

1. Open CVAT or Roboflow
2. Draw a region
3. Copy the coordinates
4. Paste them back into your code

PixPick lets you draw directly from Python and immediately returns framework-ready coordinates.


```python
import pixpick

region = pixpick.box("video.mp4", frame=10)  # drag a box on a specific video frame
zone   = pixpick.polygon("image.jpg")        # click polygon vertices
```

---

## Install

```bash
pip install pixpick
```

---

## Selectors

| Selector | How to use | Returns |
|---|---|---|
| `pixpick.box()` | Left-click + drag | `Box` |
| `pixpick.polygon()` | Click vertices | `Polygon` |
| `pixpick.line()` | Click start → click end | `Line` |

**Box controls** — `drag` to draw · `R` to reset · `Enter` to confirm · `Esc` to cancel

**Polygon controls** — `LMB` add point · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

---

## Output formats

Every selection object carries all the formats you'll ever need.

```python
# ── Box ──────────────────────────────────────────────────────
region = pixpick.box("frame.jpg")

region.xyxy              # [x1, y1, x2, y2]            absolute pixels
region.xywh              # [x, y, w, h]                absolute pixels
region.norm_xywh         # [x, y, w, h]                0.0 – 1.0  ← YOLO label format
region.center            # (cx, cy)
region.area              # pixels²


# ── Polygon ───────────────────────────────────────────────────
zone = pixpick.polygon("frame.jpg")

zone.points              # [(x0,y0), (x1,y1), ...]     absolute pixels
zone.as_numpy            # np.array shape (N, 2)
zone.norm                # [(x0n,y0n), ...]             0.0 – 1.0
zone.bbox                # → Box   tight bbox around the polygon
zone.npoints             # int
```
For more details, see [Selectors](docs/selectors.md).

---

## Framework integration

| Framework | Selector | Method |
|---|---|---|
| Ultralytics YOLOE — visual prompt | `Box` | `region.yolo_prompt()` |
| Ultralytics YOLO — region | `Box`/`Polygon` | `region.yolo_region()` |
| SAM / SAM2 / SAM3 — box prompt | `Box` | `region.sam()` |
| Any other format | `Box` / `Polygon` | `region.raw()` |