<div align="center">

# PixPick 

**Interactive coordinate picker for Computer Vision — no external tools needed.**

[![PyPI version](https://badge.fury.io/py/pixpick.svg)](https://badge.fury.io/py/pixpick)
[![Downloads](https://static.pepy.tech/badge/pixpick/month)](https://pepy.tech/projects/pixpick)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://k-saif.github.io/pixpick/)

</div>

<img src="https://raw.githubusercontent.com/K-saif/pixpick/main/docs/pixpick_main.png" alt="Project Overview" width="100%">

---
## The problem

Every major CV framework needs coordinates before it can run.

```python
# YOLO 
regioncounter = RegionCounter(region=[120, 80, 640, 480])    # where does this region come from?

# SAM2/SAM3
predictor.predict(box=np.array([120, 80, 640, 480]))         # same problem
```

The standard workflow: open CVAT or Roboflow → grab coordinates → paste them back into code. Every. Single. Time.

## The fix

```python
import pixpick

region = pixpick.box("video.mp4", frame=10)  # drag a box on a specific video frame
zone   = pixpick.polygon("image.jpg")        # click polygon vertices

# coordinates are ready — unpack directly into any framework
# YOLO:
regioncounter = RegionCounter(
     region=zone.yolo_region,  # pass region points
     model="yolo26n.pt",
 )

# same for YOLOE
model.predict("image.jpg", visual_prompt= region.yolo_prompt)

# SAM/SAM2/SAM3:
predictor.predict(box=region.sam)
```

A window opens on your image, video, or a specific video frame. You interact. You get framework-ready coordinates back in Python. No round-trips.

`pixpick.box()` and `pixpick.polygon()` both accept a `frame=` argument when the source is a video file.

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

**Box controls** — `drag` to draw · `R` to reset · `Enter` to confirm · `Esc` to cancel

**Polygon controls** — `LMB` add point · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

**Line controls** — `LMB` start → `LMB` end · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

**Point controls** — `LMB` foreground · `Shift`+`LMB` background · `RMB` undo · `Z` clear · `Enter` confirm · `Esc` cancel

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


## ── Line ─────────────────────────────────────────────────────
line = pixpick.line("frame.jpg")

line.points              # [(x0,y0), (x1,y1)]           absolute pixels
line.as_numpy            # np.array shape (2, 2)
line.norm                # [(x0n,y0n), (x1n,y1n)]       0.0 – 1.0
line.center              # (cx, cy)
line.length              # pixels


## ── Point ────────────────────────────────────────────────────
point = pixpick.point("frame.jpg")

point.points             # [(x0,y0), (x1,y1), ...]      absolute pixels
point.labels             # [1, 0, ...]                  1 = foreground, 0 = background
point.foreground         # [(x,y), ...]                 label 1 only
point.background         # [(x,y), ...]                 label 0 only
point.norm               # [(x0n,y0n), ...]             0.0 – 1.0
point.centroid           # (cx, cy)
point.bbox               # → Box   tight bbox around the point
point.as_polygon         # → Polygon  (needs 3+ point)
point.rescale(640, 640)  # → Point remapped to another resolution
```
For more details, see [Selectors](docs/selectors.md).

---

## Framework integration

| Framework | Selector | Method |
|---|---|---|
| Ultralytics YOLOE — visual prompt | `Box` | `region.yolo_prompt` |
| Ultralytics YOLO — region | `Box`/`Polygon` | `region.yolo_region` |
| SAM / SAM2 / SAM3 — box prompt | `Box` | `region.sam` |
| SAM / SAM2 / SAM3 — point prompt | `Point` | `point.sam` |
| Supervision PolygonZone — polygon | `Polygon` | `region.supervision` |
| Any other format | `Box` / `Polygon` | `region.raw` |

---

## Persistence

Pick once, reuse forever.

```python
region.save("zone.json")
region = pixpick.load("zone.json")   # Box and Polygon both work
```

Production pattern — pick interactively the first time, load on every subsequent run:

```python
from pathlib import Path
import pixpick

ZONE = "config/count_zone.json"

zone = pixpick.load(ZONE) if Path(ZONE).exists() else pixpick.polygon("frame.jpg")
zone.save(ZONE)
```

---

## Docs

| | |
|---|---|
| 🚀 [Getting Started](https://github.com/K-saif/pixpick/blob/main/docs/getting-started.md) | Installation, first selection, controls |
| 🎯 [Selectors](https://github.com/K-saif/pixpick/blob/main/docs/selectors.md) | All properties and methods for Box and Polygon |
| 🔌 [Framework Integration](https://github.com/K-saif/pixpick/blob/main/docs/frameworks.md) | YOLO, SAM2/SAM3 and more |
| 💾 [Persistence](https://github.com/K-saif/pixpick/blob/main/docs/persistence.md) | Save, load, JSON schema |
| 🏗️ [Architecture](https://github.com/K-saif/pixpick/blob/main/docs/architecture.md) | How it's built and how to extend it |
| 🗺️ [Roadmap](https://github.com/K-saif/pixpick/blob/main/docs/roadmap.md) | What's coming next |


---

## Contributing

We welcome contributions! Please open a GitHub issue or submit a pull request. For more information, see [Contribution Guidelines](https://github.com/K-saif/pixpick/blob/main/CONTRIBUTING.md).