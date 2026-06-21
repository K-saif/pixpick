# Architecture

## File structure

```
pixpick/
├── core/
│   └── selection.py      # Box, Polygon — all properties and to_*() methods
├── selectors/
│   ├── box.py            # BoxSelector
│   └── polygon.py        # PolygonSelector
├── backends/
│   ├── base.py           # AbstractBackend — contract for all backends
│   └── cv2_backend.py    # CV2Backend (OpenCV window)
├── utils.py              # load_image(), image_size()
└── __init__.py           # box(), polygon(), load() — public API
```

## How the layers relate

```
pixpick.box("frame.jpg")
    │
    ▼
BoxSelector.select(source)
    ├── utils.load_image(source)     → np.ndarray
    ├── CV2Backend.select_box(image) → (x1, y1, x2, y2)
    └── Box(x1, y1, x2, y2, w, h)   → returned to caller
                │
                ├── .xyxy / .xywh / .normalized / ...   (properties)
                ├── .to_yolo()                           (inline, no extra file)
                ├── .to_sam2()                           (inline, no extra file)
                └── .save() / .load()                   (persistence)
```

## Design decisions

**Selection objects own all conversion logic.**
There is no separate adapters layer. Every `to_*()` method lives directly on `Box` or `Polygon`. The conversion is always a few lines — no reason to put it in a separate class and file.

**Backends are the only abstraction.**
`BaseBackend` is the one interface worth keeping because adding a new environment (Jupyter, Gradio) means writing a new backend with zero changes to selectors or selection objects. Everything else is concrete.

**Selectors are thin glue.**
A selector does three things: load the image, call the backend, wrap the result. No logic of its own.

**`pixpick.load()` dispatches on the JSON `"type"` field.**
You save a `Box` or `Polygon` and load it back with the same call. The dispatcher reads `"type"` and returns the right object.

## Adding a new selector type (e.g. Line)

1. Add `Line` dataclass to `core/selection.py` with properties and `to_*()` methods.
2. Add `select_line()` to `BaseBackend` and implement it in `CV2Backend`.
3. Create `selectors/line.py` with `LineSelector` — mirrors `BoxSelector` exactly.
4. Add `pixpick.line()` to `__init__.py`.

No other files change.

## Adding a new framework method

Add a method directly to the relevant class in `core/selection.py`.

```python
# in Box
def to_detectron2(self) -> dict:
    return {"bbox": self.xyxy, "bbox_mode": BoxMode.XYXY_ABS}
```

That's it.
