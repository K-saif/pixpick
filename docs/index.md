# PixPick

**Interactive coordinate picker for Computer Vision.**

Draw boxes, polygons, lines and points on images or videos and instantly get coordinates for YOLO, SAM, YOLOE, OpenCV, and your own pipelines.


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
picks  = pixpick.point("image.jpg")          # click foreground / background points
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
| `pixpick.point()` | Click points (fg / bg) | `Point` |

Draw several and you get the matching wrapper instead — `Multibox`, `MultiPolygon`, `MultiLine` or `MultiPoint`.

**Box controls** — `LMB` drag to draw · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

**Polygon controls** — `LMB` add vertex · `RMB` undo · `Space` start a new polygon · `Z` clear · `Enter` confirm · `Esc` cancel

**Line controls** — `LMB` start → `LMB` end · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

**Point controls** — `LMB` foreground · `Shift`+`LMB` background · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

---

## Output formats

Every selection object carries all the formats you'll ever need.

```python
# ── Box ──────────────────────────────────────────────────────
region = pixpick.box("frame.jpg")

region.yolo_region       # coordinates in YOLO region format
region.yolo_prompt       # coordinates in YOLOE visual prompt format
region.sam               # coordinates in SAM box prompt format
region.center            # center point of the box (cx, cy)
region.area              # area of the box in pixels²


# ── Polygon ───────────────────────────────────────────────────
zone = pixpick.polygon("frame.jpg")

zone.supervision         # {"polygon": np.array} — unpack into sv.PolygonZone()
zone.yolo_region         # coordinates in YOLO region format
zone.bbox                # [x1, y1, x2, y2] tight bounds around the polygon
zone.npoints             # int
zone.norm                # normalized coordinates [(x0n,y0n), ...]  0.0 – 1.0


# ── Line ─────────────────────────────────────────────────────
line = pixpick.line("frame.jpg")

line.center               # center point of the line (cx, cy)
line.length               # length of the line in pixels
line.start                # start point of the line (x1, y1)
line.end                  # end point of the line (x2, y2)
line.vertical             # the same line re-drawn vertically   [(x,y), (x,y)]
line.horizontal           # the same line re-drawn horizontally [(x,y), (x,y)]


# ── Point ────────────────────────────────────────────────────
picks = pixpick.point("frame.jpg")

picks.sam                 # {"point_coords": ..., "point_labels": ...} for SAM
picks.xy                  # [(x0,y0), (x1,y1), ...]
picks.labels              # [1, 0, ...]  1 = foreground, 0 = background
picks.bbox                # Box — tight box around every point (only for MultiPoint)
picks.centroid            # (cx, cy) — (only for MultiPoint)
```
For more details, see [Selectors](selectors.md).

---

## Framework integration

| Framework | Selector | Method |
|---|---|---|
| Ultralytics YOLOE — visual prompt | `Box` | `region.yolo_prompt` |
| Ultralytics YOLO — region | `Box`/`Polygon` | `region.yolo_region` |
| SAM / SAM2 / SAM3 — box prompt | `Box` | `region.sam` |
| SAM / SAM2 / SAM3 — point prompt | `Point` / `MultiPoint` | `picks.sam` |
| Supervision PolygonZone — polygon | `Polygon` | `zone.supervision` |
| Supervision KeyPoints — points | `Point` / `MultiPoint` | `picks.supervision` |
| Any other format | any type | `.raw` |