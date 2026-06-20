# pixpick 🎯

> **Interactive coordinate picker for Computer Vision frameworks — no external tools needed.**

[![PyPI version](https://badge.fury.io/py/pixpick.svg)](https://badge.fury.io/py/pixpick)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Every major CV framework expects you to know coordinates *before* you can use it:

```python
# YOLO inference crop — where do you get [x1, y1, x2, y2]?
model.predict("frame.jpg", crop=[120, 80, 640, 480])

# YOLOE visual prompt — same problem
model.set_classes(bboxes=[[120, 80, 640, 480]])

# SAM2 box prompt
predictor.predict(box=np.array([120, 80, 640, 480]))

# Supervision PolygonZone — good luck eyeballing this
zone = sv.PolygonZone(polygon=np.array([[120,80],[640,80],[640,480],[120,480]]))
```

The standard workflow forces you to open CVAT, Roboflow, or some online tool — grab coordinates — paste them back. Every. Single. Time.

**pixpick eliminates that round-trip entirely.** Click on the image, get framework-ready coordinates back in Python, immediately.

---

## Features

- **5 selector types** — box, polygon, point, line, perspective (4-point)
- **3 rendering backends** — OpenCV window (scripts), Matplotlib (Jupyter/Colab), Gradio (headless servers)
- **10+ framework adapters** — YOLO, YOLOE, SAM2, Supervision, GroundingDINO, Grounded-SAM, PaddleOCR, EasyOCR, Kornia, raw formats
- **Persistence** — save/load selections as JSON, tied to image hash for reproducibility
- **Video support** — extract any frame from a video, pick on it, get coordinates back
- **Multi-selection** — pick multiple regions in one pass
- **Format utilities** — convert between xyxy, xywh, normalized, and pixel formats

---

## Installation

```bash
pip install pixpick
```

With optional Gradio backend (for headless / Colab environments):
```bash
pip install pixpick[gradio]
```

---

## Quick Start

```python
import pixpick

# Select a bounding box — opens an interactive window
region = pixpick.box("image.jpg")

# Use it directly with your framework
model.predict("image.jpg", **region.to_yolo())
predictor.predict(**region.to_sam2())
sv.PolygonZone(**region.to_supervision())

# Select a polygon zone
zone = pixpick.polygon("image.jpg")
zone.to_supervision()  # → {"polygon": np.array([[x,y], ...])}

# Select points (for SAM2 point prompts)
pts = pixpick.points("image.jpg", count=3)
pts.to_sam2()  # → {"point_coords": ..., "point_labels": ...}
```

---

## Architecture

```
pixpick/
│
├── core/
│   ├── selection.py        # Selection result objects (Box, Polygon, Points, Line, Perspective)
│   ├── selector.py         # Base Selector class — backend routing logic
│   └── registry.py         # Adapter + backend registration
│
├── selectors/
│   ├── box.py              # Single drag → xyxy box
│   ├── polygon.py          # Click vertices → closed polygon
│   ├── point.py            # Click → (x,y) point(s)
│   ├── line.py             # 2-click → line endpoints
│   └── perspective.py      # 4-click → homography source corners
│
├── backends/
│   ├── base.py             # AbstractBackend interface
│   ├── cv2_backend.py      # OpenCV imshow window (default for scripts)
│   ├── notebook.py         # Matplotlib / ipywidgets (Jupyter / Colab)
│   └── gradio_backend.py   # Gradio app server (headless / remote)
│
├── adapters/
│   ├── base.py             # AbstractAdapter interface
│   ├── yolo.py             # Ultralytics YOLO crop / stream inference
│   ├── yoloe.py            # YOLOE visual prompts
│   ├── sam2.py             # SAM2 box + point prompts
│   ├── grounded_sam.py     # GroundingDINO bbox → SAM pipeline
│   ├── supervision.py      # PolygonZone, LineZone, BEV transform
│   ├── paddleocr.py        # PaddleOCR region crop
│   ├── easyocr.py          # EasyOCR region crop
│   ├── kornia.py           # Patch extraction, homography
│   └── raw.py              # xyxy, xywh, normalized, JSON, numpy
│
├── io/
│   ├── persistence.py      # save() / load() JSON with image hash binding
│   └── schema.py           # JSON schema for selection files
│
├── video.py                # VideoFrame extractor + frame picker
├── utils.py                # Format converters, validators, image loaders
├── exceptions.py           # PixpickError, InvalidSelectionError, BackendError
└── __init__.py             # Public API surface
```

---

## Core Concepts

### 1. Selection Objects

Every selector returns a typed `Selection` object. The object holds raw coordinates and exposes `.to_<framework>()` adapter methods.

```python
from pixpick.core.selection import Box, Polygon, Points, Line, Perspective

# All selections share a common interface:
# .to_yolo()         → dict ready to unpack into model.predict()
# .to_sam2()         → dict ready to unpack into predictor.predict()
# .to_supervision()  → dict ready to unpack into sv.PolygonZone()
# .to_raw()          → {"xyxy": [...], "xywh": [...], "normalized": [...]}
# .save(path)        → write to JSON
# .visualize()       → draw selection on image and show it
```

#### Box

```python
region = pixpick.box("image.jpg")

region.xyxy          # [x1, y1, x2, y2]  — absolute pixels
region.xywh          # [x, y, w, h]
region.normalized    # [x1n, y1n, x2n, y2n]  — 0.0 to 1.0
region.center        # (cx, cy)
region.area          # pixel area
```

#### Polygon

```python
zone = pixpick.polygon("image.jpg")

zone.points          # np.array of shape (N, 2)
zone.normalized      # normalized version
zone.to_mask(h, w)   # binary mask np.array
```

#### Points

```python
pts = pixpick.points("image.jpg", count=3)

pts.coords           # [(x1,y1), (x2,y2), (x3,y3)]
pts.labels           # [1, 1, 1]  — foreground by default, toggle in UI
```

#### Line

```python
line = pixpick.line("image.jpg")

line.start           # (x1, y1)
line.end             # (x2, y2)
line.vector          # (dx, dy)
line.length          # pixel length
```

#### Perspective

```python
corners = pixpick.perspective("image.jpg")

corners.src_points   # np.array shape (4, 2) — top-left, top-right, bottom-right, bottom-left
corners.to_homography(dst_w, dst_h)   # → 3×3 H matrix
corners.to_supervision()              # → sv.ViewTransformer ready
```

---

### 2. Backends

The backend handles rendering — opening the window, capturing clicks, and returning raw pixel coordinates. pixpick auto-detects the right backend at runtime.

| Environment | Auto-selected backend | Override |
|---|---|---|
| Python script with display | `cv2` | `backend="cv2"` |
| Jupyter / Colab | `notebook` | `backend="notebook"` |
| Headless / SSH server | `gradio` | `backend="gradio"` |

```python
# Force a specific backend
region = pixpick.box("image.jpg", backend="notebook")

# Or set globally
pixpick.set_backend("gradio")
```

#### cv2 Backend Controls

| Action | Control |
|---|---|
| Box: drag | Left click + drag |
| Polygon: add vertex | Left click |
| Polygon: close | Right click or `Enter` |
| Point: place | Left click |
| Point: toggle label (fg/bg) | `f` / `b` keys |
| Undo last point | `z` |
| Confirm selection | `Enter` or `Space` |
| Cancel | `Esc` |
| Zoom in/out | Mouse wheel |

---

### 3. Adapters

Adapters convert a `Selection` object to the exact dict or object a target framework expects. No manual format wrangling.

#### YOLO

```python
region = pixpick.box("frame.jpg")

# Inference crop
model.predict("frame.jpg", **region.to_yolo())
# → model.predict("frame.jpg", crop=[x1, y1, x2, y2])

# Stream with persistent ROI
for result in model.track(source=0, **region.to_yolo()):
    ...
```

#### YOLOE Visual Prompt

```python
region = pixpick.box("frame.jpg")

# YOLOE expects normalized bboxes as visual prompt
model.set_classes(bboxes=region.to_yoloe()["bboxes"], image="frame.jpg")
```

#### SAM2 — Box Prompt

```python
region = pixpick.box("image.jpg")

predictor.set_image(image)
masks, scores, _ = predictor.predict(**region.to_sam2())
# → predictor.predict(box=np.array([x1,y1,x2,y2]))
```

#### SAM2 — Point Prompts

```python
pts = pixpick.points("image.jpg", count=2)

masks, scores, _ = predictor.predict(**pts.to_sam2())
# → predictor.predict(
#       point_coords=np.array([[x1,y1],[x2,y2]]),
#       point_labels=np.array([1, 1])
#   )
```

#### Supervision — PolygonZone

```python
zone = pixpick.polygon("frame.jpg")

polygon_zone = sv.PolygonZone(**zone.to_supervision())
# → sv.PolygonZone(polygon=np.array([[x1,y1],[x2,y2],...]))
```

#### Supervision — LineZone

```python
line = pixpick.line("frame.jpg")

line_zone = sv.LineZone(**line.to_supervision())
# → sv.LineZone(start=sv.Point(x1,y1), end=sv.Point(x2,y2))
```

#### Supervision — BEV ViewTransformer

```python
corners = pixpick.perspective("frame.jpg")

transformer = sv.ViewTransformer(**corners.to_supervision(dst_w=400, dst_h=800))
```

#### GroundingDINO / Grounded-SAM

```python
region = pixpick.box("image.jpg")

# Feed box prior into GroundingDINO
boxes, logits, phrases = predict(
    model=grounding_model,
    image=image,
    caption=text_prompt,
    **region.to_grounded_sam()    # → box_threshold, box_prior
)
```

#### PaddleOCR

```python
region = pixpick.box("doc.jpg")

result = ocr.ocr(**region.to_paddleocr(image="doc.jpg"))
# crops image to region before passing, returns only text within it
```

#### Raw Formats

```python
region = pixpick.box("image.jpg")

region.to_raw()
# {
#   "xyxy":       [x1, y1, x2, y2],
#   "xywh":       [x, y, w, h],
#   "cxcywh":     [cx, cy, w, h],
#   "normalized": [x1n, y1n, x2n, y2n],
#   "json":       '{"x1":..., "y1":..., "x2":..., "y2":...}'
# }
```

---

### 4. Multi-Selection

Pick multiple regions in a single interactive pass — essential for multi-zone Supervision pipelines.

```python
# Pick N boxes
regions = pixpick.multi_box("frame.jpg", count=3)
# → [Box, Box, Box]

zones = pixpick.multi_polygon("frame.jpg")  # press Esc to finish adding zones
# → [Polygon, Polygon, ...]

# Use them
for region in regions:
    model.predict("frame.jpg", **region.to_yolo())
```

---

### 5. Persistence

Save and reload selections so you don't re-pick on every run. Selections are bound to the source image via MD5 hash — pixpick warns you if the image has changed.

```python
# Save
region = pixpick.box("camera_feed_frame.jpg")
region.save("selections/entry_zone.json")

# Load and reuse
region = pixpick.load("selections/entry_zone.json")
zone = sv.PolygonZone(**region.to_supervision())
```

**JSON schema:**
```json
{
  "version": "1.0",
  "type": "box",
  "image_hash": "d41d8cd98f00b204e9800998ecf8427e",
  "image_size": [1920, 1080],
  "created_at": "2025-01-15T10:30:00Z",
  "coordinates": {
    "xyxy": [120, 80, 640, 480],
    "normalized": [0.0625, 0.074, 0.333, 0.444]
  }
}
```

---

### 6. Video Support

Pick coordinates from any video frame without external tools.

```python
# Open interactive frame picker, then box selector
region = pixpick.from_video("traffic.mp4").box()

# Jump to a specific frame
region = pixpick.from_video("traffic.mp4", frame=150).polygon()

# Scrub to best frame interactively
frame_picker = pixpick.from_video("traffic.mp4")
# → shows scrubbar UI, pick frame, then pick region

# Extract the frame itself
frame = pixpick.from_video("traffic.mp4", frame=150).get_frame()
# → np.ndarray (H, W, 3)
```

---

## Supported Frameworks

| Framework | Selector types | Adapter |
|---|---|---|
| Ultralytics YOLO | Box | `to_yolo()` |
| YOLOE | Box | `to_yoloe()` |
| SAM / SAM2 | Box, Points | `to_sam2()` |
| Grounded-SAM | Box | `to_grounded_sam()` |
| GroundingDINO | Box | `to_grounded_sam()` |
| Supervision PolygonZone | Polygon | `to_supervision()` |
| Supervision LineZone | Line | `to_supervision()` |
| Supervision ViewTransformer (BEV) | Perspective | `to_supervision(dst_w, dst_h)` |
| PaddleOCR | Box | `to_paddleocr()` |
| EasyOCR | Box | `to_easyocr()` |
| Kornia | Box, Perspective | `to_kornia()` |
| Detectron2 | Box | `to_raw()["xyxy"]` |
| MMDetection | Box | `to_raw()["xyxy"]` |
| Albumentations | Box | `to_raw()["xywh"]` |
| OpenCV ROI | Box | `to_raw()["xywh"]` |

---

## API Reference

### Top-Level Functions

```python
pixpick.box(source, backend=None, title=None)              → Box
pixpick.polygon(source, backend=None, title=None)          → Polygon
pixpick.points(source, count=None, backend=None)           → Points
pixpick.line(source, backend=None)                         → Line
pixpick.perspective(source, backend=None)                  → Perspective

pixpick.multi_box(source, count=None, backend=None)        → List[Box]
pixpick.multi_polygon(source, backend=None)                → List[Polygon]

pixpick.from_video(path, frame=0)                          → VideoFrameSelector
pixpick.load(path)                                         → Selection
pixpick.set_backend(backend_name)                          → None
```

`source` accepts: file path `str`, URL `str`, `np.ndarray`, `PIL.Image`, or `pathlib.Path`.

---

## Roadmap

### v0.1 — Core (MVP)
- [x] Box selector — cv2 backend
- [x] Polygon selector — cv2 backend
- [x] Point selector — cv2 backend
- [x] YOLO, SAM2, Supervision adapters
- [x] Raw format output
- [x] JSON persistence

### v0.2 — Environments
- [ ] Notebook backend (Matplotlib / ipywidgets)
- [ ] Gradio backend (headless/Colab)
- [ ] Line selector
- [ ] Perspective selector

### v0.3 — Adapters + Video
- [ ] YOLOE adapter
- [ ] PaddleOCR, EasyOCR adapters
- [ ] Kornia adapter
- [ ] Video frame extractor + scrubbar UI

### v0.4 — Polish
- [ ] Multi-selection UI
- [ ] Selection validation + warnings
- [ ] Zoom + pan in cv2 backend
- [ ] CLI: `pixpick image.jpg --type box --adapter yolo`

---

## Contributing

Contributions welcome — especially new adapters. Adding a framework is straightforward:

1. Create `pixpick/adapters/yourframework.py`
2. Implement the adapter method on the relevant `Selection` subclass
3. Add tests in `tests/adapters/test_yourframework.py`
4. Add a row to the Supported Frameworks table

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

---

## License

MIT License — see [LICENSE](LICENSE) for details.