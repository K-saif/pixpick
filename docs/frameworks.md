# Framework Integration

All conversion methods return a dict that unpacks directly into the framework's API call — no manual format conversion.

## Ultralytics YOLO

**Inference on Region** — runs inference only inside the selected region.

```python
region = pixpick.box("image.jpg")
results = model.predict("image.jpg", **region.yolo_region())
# expands to: model.predict("image.jpg", crop=[x1, y1, x2, y2])
```

**Visual Prompt for YOLOE**

```python
region = pixpick.box("image.jpg")

visual_prompts = dict(
    bboxes= region.yolo_prompt()
)

# Run inference on an image, using the provided visual prompts as guidance
results = model.predict(visual_prompts=visual_prompts)

```

---


## SAM / SAM2

**Box prompt**

```python
region = pixpick.box("image.jpg")
predictor.set_image(image)
masks, scores, _ = predictor.predict(box=region.sam())
```

---



## Raw formats

When you need a specific format that isn't covered by a named method:

```python
region = pixpick.box("image.jpg")
raw = region.to_raw()

raw["xyxy"]             # [x1, y1, x2, y2]       absolute pixels
raw["xywh"]             # [x, y, w, h]            absolute pixels
raw["cxcywh"]           # [cx, cy, w, h]          absolute pixels
raw["norm"]             # [x1, y1, x2, y2]        0.0 – 1.0
raw["norm_xywh"]        # [x, y, w, h]            0.0 – 1.0
raw["numpy"]            # [x1, y1, x2, y2]        as list (JSON serialisable)
```
