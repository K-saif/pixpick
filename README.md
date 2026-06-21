# pixpick 🎯

> Interactive coordinate picker for Computer Vision — no external tools needed.

[![PyPI version](https://badge.fury.io/py/pixpick.svg)](https://badge.fury.io/py/pixpick)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Every major CV framework needs coordinates before it can run — a crop region, a zone polygon, a prompt box. The standard workflow is: open CVAT or Roboflow, grab coordinates, paste them back into code. Every single time.

pixpick eliminates that. Click on the image in Python, get framework-ready coordinates back immediately.

```python
import pixpick

region = pixpick.box("frame.jpg")
zone   = pixpick.polygon("frame.jpg")

model.predict("frame.jpg", **region.to_yolo())
predictor.predict(**region.to_sam2())
sv.PolygonZone(**zone.to_supervision())
```

## Install

```bash
pip install pixpick
```

## Selectors

| Function | Interaction | Returns |
|---|---|---|
| `pixpick.box()` | drag | `Box` |
| `pixpick.polygon()` | click vertices | `Polygon` |

## Output formats

```python
# Box
region.xyxy             # [x1, y1, x2, y2]
region.xywh             # [x, y, w, h]
region.normalized       # [0.12, 0.08, 0.64, 0.48]
region.to_yolo()        # {"crop": [...]}
region.to_sam2()        # {"box": np.array([...])}

# Polygon
zone.points             # [(x0,y0), (x1,y1), ...]
zone.as_numpy           # np.array shape (N, 2)
zone.to_supervision()   # {"polygon": np.array([...])}
zone.bounding_box       # → Box
```

## Persistence

```python
region.save("zone.json")
region = pixpick.load("zone.json")   # works for Box and Polygon
```

## Docs

- [Getting Started](docs/getting-started.md)
- [Selectors](docs/selectors.md)
- [Framework Integration](docs/frameworks.md)
- [Backends](docs/backends.md)
- [Persistence](docs/persistence.md)
- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
