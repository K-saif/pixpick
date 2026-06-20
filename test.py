
import sys
sys.path.insert(0, '/home/claude')
import pixpick
from pixpick.core.selection import Box
from pixpick.adapters.base import BaseAdapter
from pixpick.adapters.yolo import YOLOAdapter, YOLOLabelAdapter
from pixpick.backends.base import BaseBackend
from pixpick.backends.cv2_backend import CV2Backend
from pixpick.selectors.box import BoxSelector, SelectionCancelled

# Instantiate a Box directly (no UI needed for testing)
b = Box(x1=120, y1=80, x2=640, y2=480, image_width=1920, image_height=1080)
print('Box:', b)
print('xyxy:', b.xyxy)
print('xywh:', b.xywh)
print('normalized:', [round(v,4) for v in b.normalized])
print('center:', b.center)
print('area:', b.area)
print()

# YOLO adapter
yolo = YOLOAdapter()
print('to_yolo():', yolo.convert(b))

# YOLO label adapter
label = YOLOLabelAdapter(class_id=2)
print('to_yolo_label():', label.convert(b))
print()

# Box shortcut
print('box.to_yolo():', b.to_yolo())
print('box.to_raw() keys:', list(b.to_raw().keys()))

# Persistence round-trip
import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    tmp = f.name
b.save(tmp)
b2 = Box.load(tmp)
os.unlink(tmp)
print()
print('Saved and reloaded:', b2)
assert b.xyxy == b2.xyxy, 'Round-trip failed'
print('Persistence round-trip: OK')

# Wrong type guard
try:
    YOLOAdapter().convert('not a box')
except TypeError as e:
    print('Type guard:', e)
