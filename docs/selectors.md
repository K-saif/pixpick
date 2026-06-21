# Selectors

A selector opens a UI, captures user input, and returns a typed Selection object. The selection object holds all coordinate math and framework conversion methods.

## Box

```python
region = pixpick.box("image.jpg")
```

### Properties

```python
region.xyxy             # [x1, y1, x2, y2]          absolute pixels
region.xywh             # [x, y, w, h]               absolute pixels
region.cxcywh           # [cx, cy, w, h]             absolute pixels
region.normalized       # [x1, y1, x2, y2]           0.0 – 1.0
region.normalized_xywh  # [x, y, w, h]               0.0 – 1.0  (YOLO label format)
region.center           # (cx, cy)                   absolute pixels
region.area             # int                         pixels²
region.as_numpy         # np.array shape (4,)        int32
region.image_width      # int
region.image_height     # int
```

### Framework methods

```python
region.to_yolo()        # {"crop": [x1, y1, x2, y2]}
region.to_yolo_label(class_id=0)  # {"line": "0 cx cy w h", ...}
region.to_sam2()        # {"box": np.array([x1, y1, x2, y2])}
region.to_raw()         # all formats in one dict
```

### Visualise

```python
canvas = region.visualize(image)   # returns BGR array with box drawn
cv2.imshow("result", canvas)
cv2.waitKey(0)
```

### Persistence

```python
region.save("selection.json")
region = pixpick.load("selection.json")   # or Box.load("selection.json")
```

---

## Polygon

```python
zone = pixpick.polygon("image.jpg")
```

Minimum 3 points required before `Enter` confirms. Vertices are recorded in the order you click them.

### Properties

```python
zone.points             # [(x0,y0), (x1,y1), ...]   list of tuples, absolute pixels
zone.as_numpy           # np.array shape (N, 2)      int32
zone.normalized         # [(x0n,y0n), ...]           0.0 – 1.0
zone.normalized_numpy   # np.array shape (N, 2)      float32
zone.n_points           # int
zone.bounding_box       # Box  — tight axis-aligned bbox around the polygon
zone.image_width        # int
zone.image_height       # int
```

### Framework methods

```python
zone.to_supervision()   # {"polygon": np.array shape (N, 2)}
zone.to_raw()           # all formats in one dict
```

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

---

## Coming in future releases

| Selector | Interaction | Returns | Release |
|---|---|---|---|
| `pixpick.points()` | click (fg/bg toggle) | `Points` | v0.2 |
| `pixpick.line()` | 2-click | `Line` | v0.2 |
| `pixpick.perspective()` | 4-corner click | `Perspective` | v0.2 |
| `pixpick.multi_box()` | multiple drags | `list[Box]` | v0.3 |
| `pixpick.multi_polygon()` | multiple polygons | `list[Polygon]` | v0.3 |
