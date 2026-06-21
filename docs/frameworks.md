# Framework Integration

All conversion methods return a dict that unpacks directly into the framework's API call — no manual format conversion.

## Ultralytics YOLO

**Inference crop** — runs inference only inside the selected region.

```python
region = pixpick.box("frame.jpg")
results = model.predict("frame.jpg", **region.to_yolo())
# expands to: model.predict("frame.jpg", crop=[x1, y1, x2, y2])
```

**Persistent ROI on a stream**

```python
region = pixpick.box("frame.jpg")
for result in model.track(source=0, **region.to_yolo()):
    ...
```

**Generate a YOLO label file**

```python
region = pixpick.box("image.jpg")
label = region.to_yolo_label(class_id=0)
with open("image.txt", "w") as f:
    f.write(label["line"] + "\n")
# writes:  0 0.3854 0.2593 0.2708 0.3704
```

---

## YOLOE

YOLOE expects normalised bounding boxes as visual prompts.

```python
region = pixpick.box("frame.jpg")
model.set_classes(bboxes=[region.normalized], image="frame.jpg")
```

---

## SAM / SAM2

**Box prompt**

```python
region = pixpick.box("image.jpg")
predictor.set_image(image)
masks, scores, _ = predictor.predict(**region.to_sam2())
# expands to: predictor.predict(box=np.array([x1, y1, x2, y2]))
```

**Point prompts** — coming in v0.2 with the `Points` selector.

---

## Supervision

**PolygonZone** — count or filter detections inside a region.

```python
zone = pixpick.polygon("frame.jpg")
polygon_zone = sv.PolygonZone(**zone.to_supervision())
# expands to: sv.PolygonZone(polygon=np.array([[x,y], ...]))
```

**LineZone** — count objects crossing a line. Coming in v0.2 with the `Line` selector.

**ViewTransformer (BEV)** — bird's-eye view transform. Coming in v0.2 with the `Perspective` selector.

---

## PaddleOCR

Crop the image to your selected region before passing to OCR.

```python
import cv2
import pixpick

image  = cv2.imread("doc.jpg")
region = pixpick.box(image)

x1, y1, x2, y2 = region.xyxy
crop   = image[y1:y2, x1:x2]
result = ocr.ocr(crop, cls=True)
```

---

## EasyOCR

Same crop approach.

```python
import cv2
import pixpick

image  = cv2.imread("doc.jpg")
region = pixpick.box(image)

x1, y1, x2, y2 = region.xyxy
crop   = image[y1:y2, x1:x2]
result = reader.readtext(crop)
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
raw["normalized"]       # [x1, y1, x2, y2]        0.0 – 1.0
raw["normalized_xywh"]  # [x, y, w, h]            0.0 – 1.0
raw["numpy"]            # [x1, y1, x2, y2]        as list (JSON serialisable)
```
