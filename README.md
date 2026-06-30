<div align="center">

# pixpick 🎯

**Interactive coordinate picker for Computer Vision — no external tools needed.**

[![PyPI version](https://badge.fury.io/py/pixpick.svg)](https://badge.fury.io/py/pixpick)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## The problem

Every major CV framework needs coordinates before it can run.

```python
regioncounter = RegionCounter(region=[120, 80, 640, 480])    # YOLO   — where does this region come from?
predictor.predict(box=np.array([120, 80, 640, 480]))         # SAM2   — same problem
```

The standard workflow: open CVAT or Roboflow → grab coordinates → paste them back into code. Every. Single. Time.

## The fix

```python
import pixpick

region = pixpick.box("frame.jpg")      # drag a box on the image
zone   = pixpick.polygon("frame.jpg")  # click polygon vertices

# coordinates are ready — unpack directly into any framework
# YOLO:
regioncounter = RegionCounter(
     region=region.yolo_region,  # pass region points
     model="yolo26n.pt",
 )

# same for YOLOE
model.predict("frame.jpg", visual_prompt= region.yolo_prompt())

# SAM1/SAM2:
predictor.predict(box=region.sam())
```

A window opens on your image. You interact. You get framework-ready coordinates back in Python. No round-trips.

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
| `pixpick.polygon()` | Click vertices → `Enter` to confirm | `Polygon` |

**Box controls** — `drag` to draw · `R` to reset · `Esc` to cancel

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
| Ultralytics YOLO — region | `Box` | `region.yolo_region()` |
| SAM / SAM2 — box prompt | `Box` | `region.sam()` |
| Any other format | `Box` / `Polygon` | `region.to_raw()` |

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
| 🔌 [Framework Integration](https://github.com/K-saif/pixpick/blob/main/docs/frameworks.md) | YOLO, SAM2 and more |
| 💾 [Persistence](https://github.com/K-saif/pixpick/blob/main/docs/persistence.md) | Save, load, JSON schema |
| 🏗️ [Architecture](https://github.com/K-saif/pixpick/blob/main/docs/architecture.md) | How it's built and how to extend it |
| 🗺️ [Roadmap](https://github.com/K-saif/pixpick/blob/main/docs/roadmap.md) | What's coming next |