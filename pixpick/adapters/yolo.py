from pixpick.adapters.base import BaseAdapter
from pixpick.core.selection import Box


class YOLOAdapter(BaseAdapter):
    """
    Converts a Box selection to kwargs for Ultralytics YOLO.

    Inference crop
    --------------
    model.predict("image.jpg", **box.to_yolo())
    # → model.predict("image.jpg", crop=[x1, y1, x2, y2])

    Stream with persistent ROI
    --------------------------
    for result in model.track(source=0, **box.to_yolo()):
        ...

    The "crop" kwarg tells Ultralytics to run inference only inside the
    specified region — it slices the frame before the forward pass.
    """

    def convert(self, selection: Box) -> dict:
        self.validate(selection, Box)

        return {
            # Ultralytics expects a flat [x1, y1, x2, y2] list.
            "crop": selection.xyxy,
        }


class YOLOLabelAdapter(BaseAdapter):
    """
    Converts a Box to YOLO label file format (normalised cxcywh).

    Usage
    -----
    adapter = YOLOLabelAdapter(class_id=0)
    line    = adapter.convert(box)   # → {"class_id": 0, "line": "0 0.5 0.5 0.25 0.25"}

    Write to file
    -------------
    with open("label.txt", "w") as f:
        f.write(line["line"] + "\\n")
    """

    def __init__(self, class_id: int = 0):
        self.class_id = class_id

    def convert(self, selection: Box) -> dict:
        self.validate(selection, Box)

        cx, cy, w, h = selection.cxcywh
        # Normalise all four values
        cx_n = cx / selection.image_width
        cy_n = cy / selection.image_height
        w_n  = w  / selection.image_width
        h_n  = h  / selection.image_height

        line = f"{self.class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"

        return {
            "class_id": self.class_id,
            "cx_n": cx_n,
            "cy_n": cy_n,
            "w_n":  w_n,
            "h_n":  h_n,
            "line": line,   # ready to write directly to a .txt label file
        }