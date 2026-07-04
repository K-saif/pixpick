from __future__ import annotations

import cv2
import numpy as np

from pixpick.backends.base import BaseBackend


class CV2Backend(BaseBackend):
    """
    OpenCV imshow backend.

    Controls
    --------
    Left-click + drag  : draw the box
    Enter / Space      : confirm selection
    R                  : reset and redraw
    Esc                : cancel - returns None

    The live rubber-band rect is drawn on a scratch copy of the image
    so the original is never mutated.
    """

    _drawing: bool
    _start: tuple[int, int]
    _end: tuple[int, int]
    _confirmed: bool
    _cancelled: bool
    _image_width: int
    _image_height: int
    _display_width: int
    _display_height: int
    _scale_x: float
    _scale_y: float

    def select_box(
        self,
        image: np.ndarray,
        title: str = (
            "pixpick | boxes | "
            "Drag=LMB  RMB=undo  Enter=confirm  Z=clear  Esc=cancel"
        ),
    ):

        self._reset_state()
        self._boxes = []
        display_image = self._prepare_display_image(image)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._box_callback)

        while True:
            canvas = self._draw_box(display_image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                cv2.destroyWindow(title)
                return None

            if key in (ord("z"), 8, 127):
                self._reset_state()
                self._boxes.clear()
                continue

            if key in (13, 32):
                if not self._boxes:
                    continue

                cv2.destroyWindow(title)
                return self._boxes

    def select_polygon(
        self,
        image: np.ndarray,
        title: str = "pixpick | polygon | LMB=add RMB=undo Enter=confirm Z=clear Esc=cancel",
    ) -> list[tuple[int, int]] | None:

        self._points = []
        display_image = self._prepare_display_image(image)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._polygon_callback)

        while True:
            canvas = self._draw_polygon(display_image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                cv2.destroyWindow(title)
                return None

            if key in (ord("z"), 8, 127):
                self._points.clear()
                continue

            if key in (13, 32):
                if len(self._points) < 3:
                    continue

                cv2.destroyWindow(title)
                return [self._display_to_image_point(point) for point in self._points]

    def select_line(
        self,
        image: np.ndarray,
        title: str = (
            "pixpick | line | "
            "LMB=select endpoints  Enter=confirm  Z=reset  Esc=cancel"
        ),
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:

        self._line_points = []
        self._mouse_pos = None
        display_image = self._prepare_display_image(image)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._line_callback)

        while True:
            canvas = self._draw_line(display_image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                cv2.destroyWindow(title)
                return None

            if key in (ord("z"), 8, 127):
                self._line_points.clear()
                self._mouse_pos = None
                continue

            if key in (13, 32):
                if len(self._line_points) != 2:
                    continue

                cv2.destroyWindow(title)
                return tuple(
                    self._display_to_image_point(point) for point in self._line_points
                )

    # ------------------------------------------------------------------ #
    # Mouse callback                                                      #
    # ------------------------------------------------------------------ #

    def _box_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param,
    ) -> None:

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = self._display_to_image_point((x, y))
            self._end = self._start

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = self._display_to_image_point((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._end = self._display_to_image_point((x, y))

            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])

            if x1 != x2 and y1 != y2:
                self._boxes.append([x1, y1, x2, y2])

            self._start = (0, 0)
            self._end = (0, 0)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._boxes:
                self._boxes.pop()

    def _polygon_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param,
    ) -> None:

        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._points:
                self._points.pop()

    def _line_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param,
    ) -> None:

        self._mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._line_points) < 2:
                self._line_points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._line_points:
                self._line_points.pop()

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _reset_state(self) -> None:
        self._drawing = False
        self._start = (0, 0)
        self._end = (0, 0)
        self._confirmed = False
        self._cancelled = False
        self._points = []

    def _set_image_bounds(self, image: np.ndarray) -> None:
        self._image_height, self._image_width = image.shape[:2]

    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray:
        self._set_image_bounds(image)

        max_width = 1280
        max_height = 720

        if self._image_width <= max_width and self._image_height <= max_height:
            self._display_width = self._image_width
            self._display_height = self._image_height
            self._scale_x = 1.0
            self._scale_y = 1.0
            return image

        scale = min(max_width / self._image_width, max_height / self._image_height)
        self._display_width = max(1, int(round(self._image_width * scale)))
        self._display_height = max(1, int(round(self._image_height * scale)))
        self._scale_x = self._display_width / self._image_width
        self._scale_y = self._display_height / self._image_height

        return cv2.resize(
            image,
            (self._display_width, self._display_height),
            interpolation=cv2.INTER_AREA,
        )

    def _display_to_image_point(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = self._clamp_point_to_display(point)

        image_x = int(round(x / self._scale_x))
        image_y = int(round(y / self._scale_y))

        return self._clamp_point_to_image((image_x, image_y))

    def _image_to_display_point(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = self._clamp_point_to_image(point)

        display_x = int(round(x * self._scale_x))
        display_y = int(round(y * self._scale_y))

        return self._clamp_point_to_display((display_x, display_y))

    def _clamp_point_to_display(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point

        if self._display_width <= 0 or self._display_height <= 0:
            return point

        x = max(0, min(x, self._display_width - 1))
        y = max(0, min(y, self._display_height - 1))

        return (x, y)

    def _clamp_point_to_image(self, point: tuple[int, int]) -> tuple[int, int]:
        x, y = point

        if self._image_width <= 0 or self._image_height <= 0:
            return point

        x = max(0, min(x, self._image_width - 1))
        y = max(0, min(y, self._image_height - 1))

        return (x, y)

    def _draw_box(self, image: np.ndarray) -> np.ndarray:
        """Return image with completed boxes and current rubber-band box."""

        canvas = image.copy()

        for x1, y1, x2, y2 in self._boxes:
            x1, y1 = self._image_to_display_point((x1, y1))
            x2, y2 = self._image_to_display_point((x2, y2))

            overlay = canvas.copy()

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self._drawing:
            start = self._image_to_display_point(self._start)
            end = self._image_to_display_point(self._end)

            overlay = canvas.copy()

            cv2.rectangle(overlay, start, end, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            cv2.rectangle(canvas, start, end, (0, 255, 0), 2)

            x1 = min(start[0], end[0])
            y1 = min(start[1], end[1])
            x2 = max(start[0], end[0])
            y2 = max(start[1], end[1])

            label = f"({x1},{y1}) ({x2},{y2})"

            cv2.putText(
                canvas,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def _draw_polygon(self, image: np.ndarray) -> np.ndarray:
        """Draw current polygon preview."""

        canvas = image.copy()

        if not self._points:
            return canvas

        points = self._points

        if len(points) >= 3:
            overlay = canvas.copy()
            pts = np.array(points, dtype=np.int32)

            cv2.fillPoly(
                overlay,
                [pts],
                (0, 255, 0),
            )

            cv2.addWeighted(
                overlay,
                0.15,
                canvas,
                0.85,
                0,
                canvas,
            )

        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)

            cv2.polylines(
                canvas,
                [pts],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )

        if len(points) >= 3:
            cv2.line(
                canvas,
                points[-1],
                points[0],
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        for idx, pt in enumerate(points):
            cv2.circle(
                canvas,
                pt,
                4,
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                canvas,
                str(idx),
                (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"Points: {len(self._points)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def _draw_line(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of image with the current line drawn."""

        canvas = image.copy()

        line_points = self._line_points
        mouse_pos = self._mouse_pos

        for point in line_points:
            cv2.circle(canvas, point, 4, (0, 255, 0), -1)

        if len(line_points) == 1 and mouse_pos is not None:
            cv2.line(
                canvas,
                line_points[0],
                mouse_pos,
                (0, 255, 0),
                2,
            )

        elif len(line_points) == 2:
            p1, p2 = line_points

            cv2.line(canvas, p1, p2, (0, 255, 0), 2)

            label = f"{p1} to {p2}"

            cv2.putText(
                canvas,
                label,
                (min(p1[0], p2[0]), max(min(p1[1], p2[1]) - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return canvas
