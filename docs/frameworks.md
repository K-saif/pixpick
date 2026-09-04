# Framework Integration

Every selection object exposes named properties that map straight onto the shape each framework expects — no manual format conversion.

A note on shapes: most properties return a **list or array** you pass as a single argument (`region=`, `box=`). Two return a **dict** meant for `**` unpacking — `Point.sam` / `MultiPoint.sam`, and `Polygon.supervision`. The examples below show which is which.

---

## Ultralytics YOLO

**Inference on a region** — `yolo_region` gives the region outline as a list of corner points, which is what the Ultralytics solutions take for `region=`.

```python
from ultralytics import solutions

zone = pixpick.polygon("image.jpg")

counter = solutions.RegionCounter(
    region=zone.yolo_region,     # [(x0,y0), (x1,y1), ...]
    model="yolo26n.pt",
)
```

`Box.yolo_region` returns the box's four corners in the same shape, so a box and a polygon are interchangeable here:

```python
region = pixpick.box("image.jpg")
region.yolo_region           # [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
```

**Visual prompt for YOLOE** — `yolo_prompt` returns an `(N, 4)` array of `[x1, y1, x2, y2]` rows, one per box.

```python
region = pixpick.box("image.jpg")

visual_prompts = dict(
    bboxes=region.yolo_prompt,          # (N, 4) array
    cls=np.zeros(len(region.yolo_prompt)),
)

results = model.predict("image.jpg", visual_prompts=visual_prompts)
```

> Check `visual_prompts` against your installed Ultralytics version — YOLOE's prompt API has changed between releases, and some versions also want an explicit `predictor=`. `region.yolo_prompt` is the array either way.

---

## SAM / SAM2 / SAM3

**Box prompt** — `sam` returns `[x1, y1, x2, y2]`.

```python
region = pixpick.box("image.jpg")
predictor.set_image(image)
masks, scores, _ = predictor.predict(box=region.sam)
```

For a `Multibox`, `sam` returns one row per box — `[[x1,y1,x2,y2], ...]`.

**Point prompt**

Click the object to segment, `Shift`+click anything you want excluded.

```python
picks = pixpick.point("image.jpg")
predictor.set_image(image)
masks, scores, _ = predictor.predict(**picks.sam)
# expands to: predictor.predict(point_coords=(N,2) float32, point_labels=(N,) int32)
```

One click returns a `Point`, several return a `MultiPoint`. Both expose the same `.sam`, so the call above works either way.

`.sam` returns a **dict** here rather than a bare array — point prompts need two parallel arrays. Combine it with a box prompt by unpacking both:

```python
masks, scores, _ = predictor.predict(box=region.sam, **picks.sam)
```

The individual arrays are available too, if you would rather pass them yourself:

```python
picks.sam_coords    # (N, 2) float32
picks.sam_labels    # (N,)   int32  — 1 = foreground, 0 = background
```

---

## Supervision

**PolygonZone** — `supervision` returns `{"polygon": np.array}`, ready to unpack.

```python
import supervision as sv

zone = pixpick.polygon("image.jpg")
polygon_zone = sv.PolygonZone(**zone.supervision)
```

For a `MultiPolygon` you get one dict per polygon:

```python
zones = pixpick.polygon("image.jpg")
polygon_zones = [sv.PolygonZone(**z) for z in zones.supervision]
```

**KeyPoints** — `supervision` on a point selection returns `{"xy": (1, N, 2) float32}`.

```python
picks = pixpick.point("image.jpg")
keypoints = sv.KeyPoints(**picks.supervision)
```

> Verify these constructor signatures against your installed `supervision` version.

---

## Line crossing

`Line` gives you the two endpoints plus the geometry helpers most counting setups need.

```python
line = pixpick.line("image.jpg")

line.start          # (x1, y1)
line.end            # (x2, y2)
line.center         # (cx, cy)
line.length         # float, pixels
line.horizontal     # the same line re-drawn horizontally through its centre
```

`horizontal` and `vertical` return **coordinates**, not booleans — useful for snapping a hand-drawn counting line to an axis.

---

## Raw formats

When you need a format that isn't covered by a named property. `raw` is a **property**, not a method — no parentheses.

```python
region = pixpick.box("image.jpg")
raw = region.raw

raw["xyxy"]                # [x1, y1, x2, y2]     absolute pixels
raw["xywh"]                # [x, y, w, h]         absolute pixels
raw["cxcywh"]              # [cx, cy, w, h]       absolute pixels
raw["normalized"]          # [x1, y1, x2, y2]     0.0 – 1.0
raw["normalized_xywh"]     # [x, y, w, h]         0.0 – 1.0
raw["numpy"]               # [x1, y1, x2, y2]     as list (JSON serialisable)
```

Every selection type has `raw`, with keys suited to its geometry:

| Type | `raw` keys |
|---|---|
| `Box` / `Multibox` | `xyxy`, `xywh`, `cxcywh`, `normalized`, `normalized_xywh`, `numpy` |
| `Polygon` | `points`, `numpy`, `normalized`, `normalized_numpy`, `bbox_xyxy` |
| `MultiPolygon` | `points`*, `numpy`, `normalized`, `normalized_numpy`, `bbox_xyxy` |
| `Line` / `MultiLine` | `points`, `numpy`, `normalized`, `normalized_numpy`, `center`, `length`, `start`, `end`, `vector` |
| `Point` | `xy`, `label`, `numpy`, `normalized`, `normalized_numpy` |
| `MultiPoint` | `xy`, `labels`, `numpy`, `normalized`, `normalized_numpy`, `centroid`, `foreground`, `background` |

\* `MultiPolygon.raw["points"]` currently holds the vertex **count** of each polygon, not the vertices. Use `zones.yolo_region` for the vertex lists until that is fixed.
