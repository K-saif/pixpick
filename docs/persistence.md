# Persistence

Save selections to JSON and reload them. Useful when you pick a zone once and reuse it across runs — common in production pipelines that run against the same camera feed.

## Save

Every selection type has a `.save()` method: `Box`, `Multibox`, `Polygon`, `MultiPolygon`, `Line`, `MultiLine`, `Point` and `MultiPoint`.

```python
region = pixpick.box("frame.jpg")
region.save("selections/entry_zone.json")

zone = pixpick.polygon("frame.jpg")
zone.save("selections/count_zone.json")

line = pixpick.line("frame.jpg")
line.save("selections/line_zone.json")

picks = pixpick.point("frame.jpg")
picks.save("selections/sam_prompt.json")
```

## Load

`pixpick.load()` reads the `"type"` field from the JSON and returns the correct object — you don't need to know what was saved.

```python
selection = pixpick.load("selections/entry_zone.json")
```

| `"type"` in JSON | Returned object |
|---|---|
| `box` | `Box` |
| `multibox` | `Multibox` |
| `polygon` | `Polygon` |
| `multipolygon` | `MultiPolygon` |
| `line` | `Line` |
| `multiline` | `MultiLine` |
| `point` | `Point` |
| `multipoint` | `MultiPoint` |

If you know the type, you can load directly from the class:

```python
from pixpick import Box, Polygon, Line, MultiPoint

region = Box.load("entry_zone.json")
zone   = Polygon.load("count_zone.json")
line   = Line.load("line_zone.json")
picks  = MultiPoint.load("sam_prompt.json")
```

Loading a file whose `"type"` does not match the class raises `ValueError`.

## JSON schema

**Box**

```json
{
  "type": "box",
  "image_size": [1920, 1080],
  "coordinates": {
    "xyxy":       [120, 80, 640, 480],
    "xywh":       [120, 80, 520, 400],
    "normalized": [0.0625, 0.074, 0.333, 0.444]
  }
}
```

**Multibox**

```json
{
  "type": "multibox",
  "image_size": [1920, 1080],
  "coordinates": {
    "boxes":      [[120, 80, 640, 480], [700, 200, 900, 500]],
    "normalized": [[0.0625, 0.074, 0.333, 0.444], [0.364, 0.185, 0.468, 0.463]]
  }
}
```

**Polygon**

```json
{
  "type": "polygon",
  "image_size": [1920, 1080],
  "coordinates": {
    "points":     [[100, 50], [400, 50], [400, 300], [100, 300]],
    "normalized": [[0.052, 0.046], [0.208, 0.046], [0.208, 0.278], [0.052, 0.278]]
  }
}
```

**MultiPolygon**

```json
{
  "type": "multipolygon",
  "image_size": [1920, 1080],
  "coordinates": {
    "polygons":   [[[100, 50], [400, 50], [400, 300]]],
    "normalized": [[[0.052, 0.046], [0.208, 0.046], [0.208, 0.278]]]
  }
}
```

**Line**

```json
{
  "type": "line",
  "image_size": [1920, 1080],
  "coordinates": {
    "points":     [[100, 50], [400, 300]],
    "normalized": [[0.052, 0.046], [0.208, 0.278]]
  }
}
```

**MultiLine**

```json
{
  "type": "multiline",
  "image_size": [1920, 1080],
  "coordinates": {
    "lines":      [[[100, 50], [400, 300]], [[500, 200], [800, 600]]],
    "normalized": [[[0.052, 0.046], [0.208, 0.278]], [[0.260, 0.185], [0.417, 0.556]]]
  }
}
```

**Point**

```json
{
  "type": "point",
  "image_size": [1920, 1080],
  "coordinates": {
    "xy":         [100, 50],
    "label":      1,
    "normalized": [0.052, 0.046]
  }
}
```

**MultiPoint**

```json
{
  "type": "multipoint",
  "image_size": [1920, 1080],
  "coordinates": {
    "points":     [[100, 50], [400, 300]],
    "labels":     [1, 0],
    "normalized": [[0.052, 0.046], [0.208, 0.278]]
  }
}
```

## Typical production pattern

Pick once interactively, save, then load on every subsequent run.

```python
import pixpick
from pathlib import Path

ZONE_FILE = "config/count_zone.json"

if Path(ZONE_FILE).exists():
    zone = pixpick.load(ZONE_FILE)
    print("Loaded saved zone.")
else:
    zone = pixpick.polygon("reference_frame.jpg")
    zone.save(ZONE_FILE)
    print("Zone saved.")
```
