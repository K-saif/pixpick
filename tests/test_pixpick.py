"""
pixpick test suite
------------------
Tests everything that doesn't require a display (no cv2.imshow).
The interactive selectors (BoxSelector, PolygonSelector) are excluded
because CI environments have no display — that logic is covered by the
backend itself which is independently testable.

Run locally:
    pip install pytest
    pytest tests/ -v
"""

import json
import os
import tempfile

import numpy as np
import pytest

from pixpick.core.box import Box, Multibox
from pixpick.core.polygon import Polygon, MultiPolygon
from pixpick.core.line import Line, MultiLine
from pixpick.core.point import Point, MultiPoint, FOREGROUND, BACKGROUND
from pixpick import load


# ======================================================================== #
# Fixtures                                                                   #
# ======================================================================== #

@pytest.fixture
def make_box():
    def _make_box(**kwargs):
        defaults = dict(
            x1=100,
            y1=50,
            x2=400,
            y2=300,
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return Box(**defaults)
    return _make_box

@pytest.fixture
def make_multibox():
    def _make_multibox(**kwargs):
        defaults = dict(
            boxes=[
                Box(
                    x1=100, y1=50, x2=400, y2=300,
                    image_width=1920, image_height=1080,
                ),
                Box(
                    x1=500, y1=200, x2=800, y2=600,
                    image_width=1920, image_height=1080,
                ),
            ],
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return Multibox(**defaults)
    return _make_multibox

@pytest.fixture
def make_polygon():
    def _make_polygon(**kwargs):
        defaults = dict(
            points=[(100, 50), (400, 50), (400, 300), (100, 300)],
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return Polygon(**defaults)
    return _make_polygon

@pytest.fixture
def make_multipolygon():
    def _make_multipolygon(**kwargs):
        defaults = dict(
            polygons=[
                Polygon(
                    points=[(100, 50), (400, 50), (400, 300), (100, 300)],
                    image_width=1920,
                    image_height=1080,
                ),
                Polygon(
                    points=[(500, 200), (800, 200), (800, 600), (500, 600)],
                    image_width=1920,
                    image_height=1080,
                ),
            ],
            image_width=1920,
            image_height=1080,
        )
        defaults.update(kwargs)
        return MultiPolygon(**defaults)
    return _make_multipolygon

@pytest.fixture
def line():
    return Line(
        points=[(100, 50), (400, 300)],
        image_width=1920, image_height=1080
    )

@pytest.fixture
def multiline():
    return MultiLine(
        lines=[
            Line(points=[(100, 50), (400, 300)],
                 image_width=1920, image_height=1080),
            Line(points=[(500, 200), (800, 600)],
                 image_width=1920, image_height=1080),
        ],
        image_width=1920,
        image_height=1080,
    )

@pytest.fixture
def point():
    return Point(x=100, y=50, image_width=1920, image_height=1080)

@pytest.fixture
def multipoint():
    return MultiPoint(
        points=[
            Point(x=100, y=50, image_width=1920, image_height=1080,
                  label=FOREGROUND),
            Point(x=400, y=300, image_width=1920, image_height=1080,
                  label=BACKGROUND),
            Point(x=800, y=600, image_width=1920, image_height=1080,
                  label=FOREGROUND),
        ],
        image_width=1920,
        image_height=1080,
    )

@pytest.fixture
def sample_image():
    """Synthetic BGR image — no file needed."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ======================================================================== #
# Box — construction and validation                                          #
# ======================================================================== #

class TestBoxConstruction:

    def test_basic(self, make_box):
        box = make_box()
        assert box.x1 == 100
        assert box.y1 == 50
        assert box.x2 == 400
        assert box.y2 == 300

    def test_auto_sort_coords(self, make_box):
        """Drag direction shouldn't matter — x1 < x2 and y1 < y2 always."""
        box = make_box()
        assert box.x1 == 100 and box.x2 == 400
        assert box.y1 == 50  and box.y2 == 300

    def test_zero_area_raises(self, make_box):
        with pytest.raises(ValueError, match="zero area"):
            make_box(x2=100)

    def test_x_out_of_bounds_raises(self, make_box):
        with pytest.raises(ValueError, match="x coords"):
            make_box(x1=-1)

    def test_y_out_of_bounds_raises(self, make_box):
        with pytest.raises(ValueError, match="y coords"):
            make_box(y2=-1)


# ======================================================================== #
# Box — coordinate properties                                               #
# ======================================================================== #

class TestBoxProperties:

    def test_xyxy(self, make_box):
        box = make_box()
        assert box.xyxy == [100, 50, 400, 300]

    def test_xywh(self, make_box):
        box = make_box()
        assert box.xywh == [100, 50, 300, 250]

    def test_cxcywh(self, make_box):
        box = make_box()
        cx, cy, w, h = box.cxcywh
        assert cx == 250.0
        assert cy == 175.0
        assert w  == 300.0
        assert h  == 250.0

    def test_norm(self, make_box):
        box = make_box()
        n = box.norm
        assert len(n) == 4
        assert all(0.0 <= v <= 1.0 for v in n)
        assert pytest.approx(n[0], abs=1e-4) == 100 / 1920
        assert pytest.approx(n[1], abs=1e-4) == 50  / 1080

    def test_norm_xywh(self, make_box):
        box = make_box()
        n = box.norm_xywh
        assert len(n) == 4
        assert all(0.0 <= v <= 1.0 for v in n)

    def test_center(self, make_box):
        box = make_box()
        assert box.center == (250, 175)

    def test_area(self, make_box):
        box = make_box()
        assert box.area == 300 * 250

    def test_as_numpy_shape_and_dtype(self, make_box):
        box = make_box()
        arr = box.as_numpy
        assert arr.shape == (4,)
        assert arr.dtype == np.int32
        assert arr.tolist() == [100, 50, 400, 300]

    def test_yolo_region(self, make_box):
        box = make_box()
        assert box.yolo_region == [
            (100, 50),
            (400, 50),
            (400, 300),
            (100, 300),
        ]

    def test_yolo_prompt(self, make_box):
        box = make_box()
        np.testing.assert_array_equal(
            box.yolo_prompt,
            np.array([[100, 50, 400, 300]]),
        )

    def test_sam(self, make_box):
        box = make_box()
        np.testing.assert_array_equal(box.sam, np.array([100, 50, 400, 300]))

    def test_raw_keys(self, make_box):
        box = make_box()
        raw = box.raw
        expected = {"xyxy", "xywh", "cxcywh", "normalized", "normalized_xywh", "numpy"}
        assert expected.issubset(raw.keys())

    def test_raw_xyxy_matches(self, make_box):
        box = make_box()
        assert box.raw["xyxy"] == box.xyxy


# ======================================================================== #
# Box — persistence                                                         #
# ======================================================================== #

class TestBoxPersistence:

    def test_save_creates_file(self, make_box):
        box = make_box()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_save_json_schema(self, make_box):
        box = make_box()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "box"
            assert "image_size" in data
            assert "coordinates" in data
            assert "xyxy" in data["coordinates"]
        finally:
            os.unlink(path)

    def test_round_trip(self, make_box):
        box = make_box()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box.save(path)
            reloaded = Box.load(path)
            assert reloaded.xyxy == box.xyxy
            assert reloaded.image_width  == box.image_width
            assert reloaded.image_height == box.image_height
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, make_polygon):
        polygon = make_polygon()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon.save(path)
            with pytest.raises(ValueError, match="Expected type 'box'"):
                Box.load(path)
        finally:
            os.unlink(path)


# ======================================================================== #
# Box — visualize                                                           #
# ======================================================================== #

class TestBoxVisualize:

    def test_returns_same_shape(self, make_box, sample_image):
        box = make_box()
        vis = box.visualize(sample_image)
        assert vis.shape == sample_image.shape

    def test_does_not_mutate_original(self, make_box, sample_image):
        box = make_box()
        original = sample_image.copy()
        box.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, original)



# ======================================================================== #
# Multibox — construction and properties                                    #
# ======================================================================== #

class TestMultiboxConstruction:

    def test_basic(self, make_multibox):
        multibox = make_multibox()

        assert multibox.xyxy == [
            [100, 50, 400, 300],
            [500, 200, 800, 600],
        ]

    def test_holds_box_objects(self, make_multibox):
        multibox = make_multibox()

        assert all(isinstance(box, Box) for box in multibox.boxes)
        assert multibox.nboxes == 2

    def test_empty_boxes_raises(self, make_multibox):
        with pytest.raises(ValueError, match="at least one Box"):
            make_multibox(boxes=[])

    def test_raw_coordinates_raise(self, make_multibox):
        with pytest.raises(TypeError, match="expects Box objects"):
            make_multibox(boxes=[[100, 50, 400, 300]])

    def test_mismatched_image_size_raises(self, make_multibox):
        with pytest.raises(ValueError, match="does not match Multibox size"):
            make_multibox(boxes=[
                Box(x1=10, y1=10, x2=50, y2=50,
                    image_width=640, image_height=480),
            ])

    def test_boxes_normalise_reversed_drags(self):
        multibox = Multibox(
            boxes=[
                Box(x1=400, y1=300, x2=100, y2=50,
                    image_width=1920, image_height=1080),
            ],
            image_width=1920,
            image_height=1080,
        )

        assert multibox.xyxy == [[100, 50, 400, 300]]
        assert multibox.xywh == [[100, 50, 300, 250]]


# ======================================================================== #
# Multibox — coordinate properties                                               #
# ======================================================================== #

class TestMultiboxProperties:

    def test_xyxy(self, make_multibox):
        multibox = make_multibox()
        assert multibox.xyxy == [
            [100, 50, 400, 300],
            [500, 200, 800, 600],
        ]

    def test_xywh(self, make_multibox):
        multibox = make_multibox()
        assert multibox.xywh == [
            [100, 50, 300, 250],
            [500, 200, 300, 400],
        ]

    def test_cxcywh(self, make_multibox):
        multibox = make_multibox()
        assert multibox.cxcywh == [
            [250.0, 175.0, 300.0, 250.0],
            [650.0, 400.0, 300.0, 400.0],
        ]

    def test_norm(self, make_multibox):
        multibox = make_multibox()
        norm = multibox.norm
        assert len(norm) == 2
        assert all(all(0 <= v <= 1 for v in box) for box in norm)

    def test_norm_xywh(self, make_multibox):
        multibox = make_multibox()
        norm = multibox.norm_xywh
        assert len(norm) == 2

    def test_center(self, make_multibox):
        multibox = make_multibox()
        assert multibox.center == [
            (250, 175),
            (650, 400),
        ]

    def test_area(self, make_multibox):
        multibox = make_multibox()
        assert multibox.area == [
            300 * 250,
            300 * 400,
        ]

    def test_as_numpy(self, make_multibox):
        multibox = make_multibox()
        arr = multibox.as_numpy
        assert arr.shape == (2, 4)
        assert arr.dtype == np.int32

    def test_yolo_region(self, make_multibox):
        multibox = make_multibox()
        assert len(multibox.yolo_region) == 2

    def test_yolo_prompt(self, make_multibox):
        multibox = make_multibox()
        np.testing.assert_array_equal(
            multibox.yolo_prompt,
            np.array([
                [100, 50, 400, 300],
                [500, 200, 800, 600],
            ]),
        )

    def test_sam(self, make_multibox):
        multibox = make_multibox()
        np.testing.assert_array_equal(
            multibox.sam,
            np.array([
                [100, 50, 400, 300],
                [500, 200, 800, 600],
            ]),
        )

    def test_raw_keys(self, make_multibox):
        multibox = make_multibox()
        raw = multibox.raw
        expected = {
            "xyxy",
            "xywh",
            "cxcywh",
            "normalized",
            "normalized_xywh",
            "numpy",
        }
        assert expected.issubset(raw.keys())

# ======================================================================== #
# Multibox — persistence                                                   #
# ======================================================================== #

class TestMultiboxPersistence:

    def test_round_trip(self, make_multibox):
        multibox = make_multibox()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            multibox.save(path)

            reloaded = Multibox.load(path)

            assert reloaded.xyxy == multibox.xyxy
            assert reloaded.image_width == multibox.image_width
            assert reloaded.image_height == multibox.image_height
        finally:
            os.unlink(path)


# ======================================================================== #
# Multibox — visualize                                                     #
# ======================================================================== #

class TestMultiboxVisualize:

    def test_returns_same_shape(self, make_multibox, sample_image):
        multibox = make_multibox()
        vis = multibox.visualize(sample_image)
        assert vis.shape == sample_image.shape

    def test_does_not_mutate_original(self, make_multibox, sample_image):
        multibox = make_multibox()
        original = sample_image.copy()
        multibox.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, original)



# ======================================================================== #
# Polygon — construction and properties                         #
# ======================================================================== #

class TestPolygonConstruction:

    def test_basic(self, make_polygon):
        polygon = make_polygon()
        assert polygon.npoints == 4

    def test_too_few_points_raises(self, make_polygon):
        with pytest.raises(ValueError, match="at least 3"):
            make_polygon(
                points=[(0, 0), (100, 100)],
                image_width=1920,
                image_height=1080,)

    def test_point_out_of_bounds_raises(self, make_polygon):
        with pytest.raises(ValueError, match="outside image"):
            make_polygon(
                points=[(0, 0), (100, 100), (2000, 500)],
                image_width=1920,
                image_height=1080,
            )


# ======================================================================== #
# Polygon — properties                                                     #
# ======================================================================== #

class TestPolygonProperties:

    def test_as_numpy_shape(self, make_polygon):
        polygon = make_polygon()
        arr = polygon.as_numpy
        assert arr.shape == (4, 2)
        assert arr.dtype == np.int32

    def test_norm_range(self, make_polygon):
        polygon = make_polygon()
        for x, y in polygon.norm:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_norm_numpy_shape(self, make_polygon):
        polygon = make_polygon()
        arr = polygon.norm_numpy
        assert arr.shape == (4, 2)
        assert arr.dtype == np.float32

    def test_bbox_type(self, make_polygon):
        polygon = make_polygon()
        bbox = polygon.bbox
        assert isinstance(bbox, list)

    def test_bbox_values(self, make_polygon):
        polygon = make_polygon()
        # polygon is a rectangle (100,50)→(400,300)
        bbox = polygon.bbox
        assert bbox == [100, 50, 400, 300]

    def test_n_points(self, make_polygon):
        polygon = make_polygon()
        assert polygon.npoints == 4

    def test_to_supervision_key(self, make_polygon):
        polygon = make_polygon()
        result = polygon.supervision
        assert "polygon" in result

    def test_to_supervision_numpy(self, make_polygon):
        polygon = make_polygon()
        arr = polygon.supervision["polygon"]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 2)

    def test_raw_keys(self, make_polygon):
        polygon = make_polygon()
        raw = polygon.raw
        expected = {"points", "numpy", "normalized", "normalized_numpy", "bbox_xyxy"}
        assert expected.issubset(raw.keys())


# ======================================================================== #
# Polygon — persistence                                                     #
# ======================================================================== #

class TestPolygonPersistence:

    def test_round_trip(self, make_polygon):
        polygon = make_polygon()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon.save(path)
            reloaded = Polygon.load(path)
            assert reloaded.points == polygon.points
            assert reloaded.image_width  == polygon.image_width
            assert reloaded.image_height == polygon.image_height
        finally:
            os.unlink(path)

    def test_save_json_schema(self, make_polygon):
        polygon = make_polygon()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "polygon"
            assert "image_size" in data
            assert "points" in data["coordinates"]
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, make_box):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box = make_box()
            box.save(path)
            with pytest.raises(ValueError, match="Expected type 'polygon'"):
                Polygon.load(path)
        finally:
            os.unlink(path)


# ======================================================================== #
# Polygon — visualize                                                       #
# ======================================================================== #

class TestPolygonVisualize:

    def test_returns_same_shape(self, make_polygon, sample_image):
        polygon = make_polygon()
        vis = polygon.visualize(sample_image)
        assert vis.shape == sample_image.shape

    def test_does_not_mutate_original(self, make_polygon, sample_image):
        polygon = make_polygon()
        original = sample_image.copy()
        polygon.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, original)


# ======================================================================== #
# Polygon — construction and properties                         #
# ======================================================================== #
# need to fix the polygon npoint for multipolygon then  only construction can be implemented same as polygon

# ======================================================================== #
# MultiPolygon — properties                                                     #
# ======================================================================== #
class TestMultiPolygonProperties:

    def test_as_numpy_shape(self, make_multipolygon):
        multipolygon = make_multipolygon()

        arr = multipolygon.as_numpy

        assert isinstance(arr, list)
        assert len(arr) == 2
        assert arr[0].shape == (4, 2)
        assert arr[1].shape == (4, 2)
        assert arr[0].dtype == np.int32
        assert arr[1].dtype == np.int32

    def test_norm_range(self, make_multipolygon):
        multipolygon = make_multipolygon()

        for polygon_norm in multipolygon.norm:
            for x, y in polygon_norm:
                assert 0.0 <= x <= 1.0
                assert 0.0 <= y <= 1.0

    def test_norm_numpy(self, make_multipolygon):
        multipolygon = make_multipolygon()

        result = multipolygon.norm_numpy

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (4, 2)
        assert result[1].shape == (4, 2)
        assert result[0].dtype == np.float32
        assert result[1].dtype == np.float32

    def test_bbox(self, make_multipolygon):
        multipolygon = make_multipolygon()

        assert multipolygon.box == [
            [100, 50, 400, 300],
            [500, 200, 800, 600],
        ]


    def test_to_supervision(self, make_multipolygon):
        multipolygon = make_multipolygon()

        result = multipolygon.supervision

        assert isinstance(result, list)
        assert len(result) == 2

        for item in result:
            assert "polygon" in item

# ======================================================================== #
# MultiPolygon — persistence                                                     #
# ======================================================================== #
# after fxing the multipolygon and multibox issue #64, persistence can be implemented same as polygon


# ======================================================================== #
# MultiPolygon — visualize                                                       #
# ======================================================================== #
# after fxing the multipolygon and multibox issue #64, visualize can be implemented same as polygon

# ======================================================================== #
# pixpick.load() dispatcher                                                 #
# ======================================================================== #

class TestLoadDispatcher:

    def test_dispatches_box(self, make_box):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            box = make_box()
            box.save(path)
            result = load(path)
            assert isinstance(result, Box)
            assert result.xyxy == box.xyxy
        finally:
            os.unlink(path)

    def test_dispatches_multibox(self, make_multibox):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multibox = make_multibox()
            multibox.save(path)
            result = load(path)
            assert isinstance(result, Multibox)
            assert result.xyxy == multibox.xyxy
        finally:
            os.unlink(path)

    def test_dispatches_polygon(self, make_polygon):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            polygon = make_polygon()
            polygon.save(path)
            result = load(path)
            assert isinstance(result, Polygon)
            assert result.points == polygon.points
        finally:
            os.unlink(path)

    def test_dispatches_line(self, line):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            line.save(path)
            result = load(path)
            assert isinstance(result, Line)
            assert result.center == line.center
        finally:
            os.unlink(path)

# ======================================================================== #
# Point — construction and validation                                       #
# ======================================================================== #

class TestPointConstruction:

    def test_basic(self, point):
        assert point.xy == (100, 50)
        assert point.label == FOREGROUND

    def test_label_defaults_to_foreground(self):
        assert Point(x=10, y=10, image_width=100, image_height=100).label == FOREGROUND

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="outside image"):
            Point(x=150, y=10, image_width=100, image_height=100)

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="foreground"):
            Point(x=10, y=10, image_width=100, image_height=100, label=7)


# ======================================================================== #
# Point — format properties                                                 #
# ======================================================================== #

class TestPointFormats:

    def test_as_numpy(self, point):
        arr = point.as_numpy
        assert arr.shape == (2,)
        assert arr.dtype == np.int32
        assert arr.tolist() == [100, 50]

    def test_norm(self, point):
        xn, yn = point.norm
        assert xn == pytest.approx(100 / 1920)
        assert yn == pytest.approx(50 / 1080)

    def test_norm_numpy(self, point):
        arr = point.norm_numpy
        assert arr.shape == (2,)
        assert arr.dtype == np.float32

    def test_foreground_flags(self, point):
        assert point.is_foreground
        assert not point.is_background

    def test_background_flags(self):
        pt = Point(x=10, y=10, image_width=100, image_height=100, label=BACKGROUND)
        assert pt.is_background
        assert not pt.is_foreground

    def test_raw_keys(self, point):
        assert set(point.raw) == {
            "xy",
            "label",
            "numpy",
            "normalized",
            "normalized_numpy",
        }

    def test_repr(self, point):
        assert "xy=(100, 50)" in repr(point)
        assert "fg" in repr(point)


# ======================================================================== #
# Point — framework adapters                                                #
# ======================================================================== #

class TestPointAdapters:

    def test_sam_shapes(self, point):
        prompt = point.sam
        assert prompt["point_coords"].shape == (1, 2)
        assert prompt["point_coords"].dtype == np.float32
        assert prompt["point_labels"].shape == (1,)
        assert prompt["point_labels"].dtype == np.int32

    def test_sam_values(self, point):
        prompt = point.sam
        assert prompt["point_coords"].tolist() == [[100.0, 50.0]]
        assert prompt["point_labels"].tolist() == [1]

    def test_supervision_shape(self, point):
        assert point.supervision["xy"].shape == (1, 1, 2)


# ======================================================================== #
# Point — rescale                                                           #
# ======================================================================== #

class TestPointRescale:

    def test_halves_coordinates(self):
        pt = Point(x=100, y=200, image_width=1000, image_height=1000)
        smaller = pt.rescale(500, 500)
        assert smaller.xy == (50, 100)
        assert smaller.image_width == 500
        assert smaller.image_height == 500

    def test_keeps_label(self):
        pt = Point(x=10, y=10, image_width=100, image_height=100, label=BACKGROUND)
        assert pt.rescale(50, 50).label == BACKGROUND

    def test_does_not_mutate_original(self, point):
        point.rescale(96, 54)
        assert point.xy == (100, 50)

    def test_stays_in_bounds(self):
        pt = Point(x=1000, y=1000, image_width=1000, image_height=1000)
        assert pt.rescale(640, 640).xy == (640, 640)

    def test_invalid_size_raises(self, point):
        with pytest.raises(ValueError, match="must be positive"):
            point.rescale(0, 100)


# ======================================================================== #
# Point — persistence                                                       #
# ======================================================================== #

class TestPointPersistence:

    def test_save_json_schema(self, point):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            point.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "point"
            assert data["image_size"] == [1920, 1080]
            assert set(data["coordinates"]) == {"xy", "label", "normalized"}
        finally:
            os.unlink(path)

    def test_round_trip(self, point):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            point.save(path)
            restored = Point.load(path)
            assert restored.xy == point.xy
            assert restored.label == point.label
            assert restored.image_width == point.image_width
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, make_box):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            make_box().save(path)
            with pytest.raises(ValueError, match="Expected type 'point'"):
                Point.load(path)
        finally:
            os.unlink(path)

    def test_dispatches_through_pixpick_load(self, point):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            point.save(path)
            result = load(path)
            assert isinstance(result, Point)
            assert result.xy == point.xy
        finally:
            os.unlink(path)


# ======================================================================== #
# Point — visualize                                                         #
# ======================================================================== #

class TestPointVisualize:

    def test_returns_same_shape(self, point, sample_image):
        assert point.visualize(sample_image).shape == sample_image.shape

    def test_does_not_mutate_original(self, point, sample_image):
        before = sample_image.copy()
        point.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, before)

    def test_foreground_draws_green(self, point, sample_image):
        canvas = point.visualize(sample_image)
        assert tuple(canvas[50, 100]) == (0, 255, 0)

    def test_background_draws_red(self, sample_image):
        pt = Point(x=100, y=50, image_width=1920, image_height=1080, label=BACKGROUND)
        canvas = pt.visualize(sample_image)
        assert tuple(canvas[50, 100]) == (0, 0, 255)


# ======================================================================== #
# MultiPoint — construction and validation                                  #
# ======================================================================== #

class TestMultiPointConstruction:

    def test_holds_point_objects(self, multipoint):
        assert all(isinstance(pt, Point) for pt in multipoint.points)
        assert multipoint.npoints == 3

    def test_empty_points_raises(self):
        with pytest.raises(ValueError, match="at least one Point"):
            MultiPoint(points=[], image_width=1920, image_height=1080)

    def test_raw_coordinates_raise(self):
        with pytest.raises(TypeError, match="expects Point objects"):
            MultiPoint(
                points=[(100, 50)],
                image_width=1920,
                image_height=1080,
            )

    def test_mismatched_image_size_raises(self):
        with pytest.raises(ValueError, match="does not match MultiPoint size"):
            MultiPoint(
                points=[Point(x=10, y=10, image_width=640, image_height=480)],
                image_width=1920,
                image_height=1080,
            )


# ======================================================================== #
# MultiPoint — coordinate properties                                        #
# ======================================================================== #

class TestMultiPointProperties:

    def test_xy(self, multipoint):
        assert multipoint.xy == [(100, 50), (400, 300), (800, 600)]

    def test_labels(self, multipoint):
        assert multipoint.labels == [1, 0, 1]

    def test_as_numpy(self, multipoint):
        arr = multipoint.as_numpy
        assert arr.shape == (3, 2)
        assert arr.dtype == np.int32

    def test_norm_range(self, multipoint):
        for xn, yn in multipoint.norm:
            assert 0.0 <= xn <= 1.0
            assert 0.0 <= yn <= 1.0

    def test_norm_numpy(self, multipoint):
        arr = multipoint.norm_numpy
        assert arr.shape == (3, 2)
        assert arr.dtype == np.float32

    def test_raw_keys(self, multipoint):
        assert set(multipoint.raw) == {
            "xy",
            "labels",
            "numpy",
            "normalized",
            "normalized_numpy",
            "centroid",
            "foreground",
            "background",
        }

    def test_repr(self, multipoint):
        text = repr(multipoint)
        assert "npoints=3" in text
        assert "fg=2" in text
        assert "bg=1" in text


# ======================================================================== #
# MultiPoint — geometry                                                     #
# ======================================================================== #

class TestMultiPointGeometry:

    def test_centroid(self):
        mp = MultiPoint(
            points=[
                Point(x=0, y=0, image_width=100, image_height=100),
                Point(x=10, y=20, image_width=100, image_height=100),
            ],
            image_width=100,
            image_height=100,
        )
        assert mp.centroid == (5, 10)

    def test_bbox_is_a_box(self, multipoint):
        bbox = multipoint.bbox
        assert isinstance(bbox, Box)
        assert bbox.xyxy == [100, 50, 800, 600]
        assert bbox.image_width == 1920

    def test_bbox_of_collinear_points_raises(self):
        mp = MultiPoint(
            points=[
                Point(x=10, y=10, image_width=100, image_height=100),
                Point(x=50, y=10, image_width=100, image_height=100),
            ],
            image_width=100,
            image_height=100,
        )
        with pytest.raises(ValueError, match="do not enclose any area"):
            mp.bbox

    def test_as_polygon(self, multipoint):
        polygon = multipoint.as_polygon
        assert isinstance(polygon, Polygon)
        assert polygon.points == multipoint.xy

    def test_as_polygon_needs_three_points(self):
        mp = MultiPoint(
            points=[
                Point(x=10, y=10, image_width=100, image_height=100),
                Point(x=50, y=50, image_width=100, image_height=100),
            ],
            image_width=100,
            image_height=100,
        )
        with pytest.raises(ValueError, match="at least 3 points"):
            mp.as_polygon


# ======================================================================== #
# MultiPoint — foreground / background split                                #
# ======================================================================== #

class TestMultiPointLabels:

    def test_foreground_returns_points(self, multipoint):
        fg = multipoint.foreground
        assert all(isinstance(pt, Point) for pt in fg)
        assert [pt.xy for pt in fg] == [(100, 50), (800, 600)]

    def test_background_returns_points(self, multipoint):
        bg = multipoint.background
        assert [pt.xy for pt in bg] == [(400, 300)]

    def test_counts(self, multipoint):
        assert multipoint.n_foreground == 2
        assert multipoint.n_background == 1
        assert multipoint.n_foreground + multipoint.n_background == multipoint.npoints


# ======================================================================== #
# MultiPoint — framework adapters                                           #
# ======================================================================== #

class TestMultiPointAdapters:

    def test_sam_shapes(self, multipoint):
        prompt = multipoint.sam
        assert prompt["point_coords"].shape == (3, 2)
        assert prompt["point_coords"].dtype == np.float32
        assert prompt["point_labels"].shape == (3,)
        assert prompt["point_labels"].dtype == np.int32

    def test_sam_labels_values(self, multipoint):
        assert multipoint.sam_labels.tolist() == [1, 0, 1]

    def test_sam_arrays_stay_parallel(self, multipoint):
        prompt = multipoint.sam
        assert len(prompt["point_coords"]) == len(prompt["point_labels"])

    def test_supervision_shape(self, multipoint):
        assert multipoint.supervision["xy"].shape == (1, 3, 2)


# ======================================================================== #
# MultiPoint — rescale                                                      #
# ======================================================================== #

class TestMultiPointRescale:

    def test_halves_coordinates(self, multipoint):
        smaller = multipoint.rescale(960, 540)
        assert smaller.xy == [(50, 25), (200, 150), (400, 300)]
        assert smaller.image_width == 960

    def test_keeps_labels(self, multipoint):
        assert multipoint.rescale(960, 540).labels == [1, 0, 1]

    def test_does_not_mutate_original(self, multipoint):
        before = multipoint.xy
        multipoint.rescale(96, 54)
        assert multipoint.xy == before


# ======================================================================== #
# MultiPoint — persistence                                                  #
# ======================================================================== #

class TestMultiPointPersistence:

    def test_save_json_schema(self, multipoint):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multipoint.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "multipoint"
            assert set(data["coordinates"]) == {"points", "labels", "normalized"}
        finally:
            os.unlink(path)

    def test_round_trip(self, multipoint):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multipoint.save(path)
            restored = MultiPoint.load(path)
            assert restored.xy == multipoint.xy
            assert restored.labels == multipoint.labels
            assert all(isinstance(pt, Point) for pt in restored.points)
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, point):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            point.save(path)
            with pytest.raises(ValueError, match="Expected type 'multipoint'"):
                MultiPoint.load(path)
        finally:
            os.unlink(path)

    def test_dispatches_through_pixpick_load(self, multipoint):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multipoint.save(path)
            result = load(path)
            assert isinstance(result, MultiPoint)
            assert result.labels == multipoint.labels
        finally:
            os.unlink(path)


# ======================================================================== #
# MultiPoint — visualize                                                    #
# ======================================================================== #

class TestMultiPointVisualize:

    def test_returns_same_shape(self, multipoint, sample_image):
        assert multipoint.visualize(sample_image).shape == sample_image.shape

    def test_does_not_mutate_original(self, multipoint, sample_image):
        before = sample_image.copy()
        multipoint.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, before)

    def test_draws_foreground_and_background_colors(self, multipoint, sample_image):
        canvas = multipoint.visualize(sample_image)
        assert tuple(canvas[50, 100]) == (0, 255, 0)     # foreground → green
        assert tuple(canvas[300, 400]) == (0, 0, 255)    # background → red


# ======================================================================== #
# MultiLine — construction and validation                                   #
# ======================================================================== #

class TestMultiLineConstruction:

    def test_holds_line_objects(self, multiline):
        assert all(isinstance(line, Line) for line in multiline.lines)
        assert multiline.nlines == 2

    def test_empty_lines_raises(self):
        with pytest.raises(ValueError, match="at least one Line"):
            MultiLine(lines=[], image_width=1920, image_height=1080)

    def test_raw_coordinates_raise(self):
        with pytest.raises(TypeError, match="expects Line objects"):
            MultiLine(
                lines=[[(100, 50), (400, 300)]],
                image_width=1920,
                image_height=1080,
            )

    def test_mismatched_image_size_raises(self):
        with pytest.raises(ValueError, match="does not match MultiLine size"):
            MultiLine(
                lines=[Line(points=[(10, 10), (50, 50)],
                            image_width=640, image_height=480)],
                image_width=1920,
                image_height=1080,
            )


# ======================================================================== #
# MultiLine — coordinate properties                                         #
# ======================================================================== #

class TestMultiLineProperties:

    def test_points(self, multiline):
        assert multiline.points == [
            [(100, 50), (400, 300)],
            [(500, 200), (800, 600)],
        ]

    def test_as_numpy(self, multiline):
        arr = multiline.as_numpy
        assert arr.shape == (2, 2, 2)
        assert arr.dtype == np.int32

    def test_norm_numpy(self, multiline):
        arr = multiline.norm_numpy
        assert arr.shape == (2, 2, 2)
        assert arr.dtype == np.float32

    def test_norm_range(self, multiline):
        for line in multiline.norm:
            for xn, yn in line:
                assert 0.0 <= xn <= 1.0
                assert 0.0 <= yn <= 1.0

    def test_center(self, multiline):
        assert multiline.center == [(250, 175), (650, 400)]

    def test_start_and_end(self, multiline):
        assert multiline.start == [(100, 50), (500, 200)]
        assert multiline.end == [(400, 300), (800, 600)]

    def test_length_matches_each_line(self, multiline):
        assert multiline.length == [line.length for line in multiline.lines]

    def test_vector(self, multiline):
        assert multiline.vector == [(300, 250), (300, 400)]

    def test_horizontal_and_vertical(self, multiline):
        assert len(multiline.horizontal) == 2
        assert len(multiline.vertical) == 2

    def test_raw_keys(self, multiline):
        assert set(multiline.raw) == {
            "points",
            "numpy",
            "normalized",
            "normalized_numpy",
            "center",
            "length",
            "start",
            "end",
            "vector",
        }


# ======================================================================== #
# MultiLine — persistence                                                   #
# ======================================================================== #

class TestMultiLinePersistence:

    def test_save_json_schema(self, multiline):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multiline.save(path)
            data = json.loads(open(path).read())
            assert data["type"] == "multiline"
            assert data["image_size"] == [1920, 1080]
        finally:
            os.unlink(path)

    def test_round_trip(self, multiline):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multiline.save(path)
            restored = MultiLine.load(path)
            assert restored.points == multiline.points
            assert all(isinstance(line, Line) for line in restored.lines)
        finally:
            os.unlink(path)

    def test_load_wrong_type_raises(self, line):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            line.save(path)
            with pytest.raises(ValueError, match="Expected type 'multiline'"):
                MultiLine.load(path)
        finally:
            os.unlink(path)

    def test_dispatches_through_pixpick_load(self, multiline):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            multiline.save(path)
            result = load(path)
            assert isinstance(result, MultiLine)
            assert result.points == multiline.points
        finally:
            os.unlink(path)


# ======================================================================== #
# MultiLine — visualize                                                     #
# ======================================================================== #

class TestMultiLineVisualize:

    def test_returns_same_shape(self, multiline, sample_image):
        assert multiline.visualize(sample_image).shape == sample_image.shape

    def test_does_not_mutate_original(self, multiline, sample_image):
        before = sample_image.copy()
        multiline.visualize(sample_image)
        np.testing.assert_array_equal(sample_image, before)


# ======================================================================== #
# MultiPolygon — rejects raw coordinates                                    #
# ======================================================================== #

class TestMultiPolygonConstruction:

    def test_raw_coordinates_raise(self):
        with pytest.raises(TypeError, match="expects Polygon objects"):
            MultiPolygon(
                polygons=[[(100, 50), (400, 50), (400, 300)]],
                image_width=1920,
                image_height=1080,
            )
