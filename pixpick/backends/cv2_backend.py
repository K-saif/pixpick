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
    Esc                : cancel — returns None

    The live rubber-band rect is drawn on a scratch copy of the image
    so the original is never mutated.
    """

    # Internal mouse state — kept on the instance so the callback has access.
    _drawing: bool
    _start: tuple[int, int]
    _end: tuple[int, int]
    _confirmed: bool
    _cancelled: bool
    _image_width: int
    _image_height: int

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
        self._set_image_bounds(image)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._box_callback)

        while True:

            canvas = self._draw_box(image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            # Esc
            if key == 27:
                cv2.destroyWindow(title)
                return None

            # Clear all
            if key in (ord("z"), 8, 127):
                self._reset_state()
                self._boxes.clear()
                continue

            # Enter / Space
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
        self._set_image_bounds(image)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._polygon_callback)

        while True:
            canvas = self._draw_polygon(image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            # Esc
            if key == 27:
                cv2.destroyWindow(title)
                return None

            # Z / Backspace / Delete
            if key in (ord("z"), 8, 127):
                self._points.clear()
                continue

            # Enter / Space
            if key in (13, 32):

                if len(self._points) < 3:
                    continue

                cv2.destroyWindow(title)

                return self._points

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
        self._set_image_bounds(image)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._line_callback)

        while True:
            canvas = self._draw_line(image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            # Esc
            if key == 27:
                cv2.destroyWindow(title)
                return None

            # Z / Backspace / Delete
            if key in (ord("z"), 8, 127):
                self._line_points.clear()
                self._mouse_pos = None
                continue

            # Enter / Space
            if key in (13, 32):

                if len(self._line_points) != 2:
                    continue

                cv2.destroyWindow(title)
                return tuple(self._line_points)

    # ------------------------------------------------------------------ #
    # Mouse callback                                                       #
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
            self._start = self._clamp_point_to_image((x, y))
            self._end = self._start

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:

            self._end = self._clamp_point_to_image((x, y))

        elif event == cv2.EVENT_LBUTTONUP:

            self._drawing = False
            self._end = self._clamp_point_to_image((x, y))

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
            self._points.append(self._clamp_point_to_image((x, y)))

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

        self._mouse_pos = self._clamp_point_to_image((x, y))

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(self._line_points) < 2:
                self._line_points.append(self._clamp_point_to_image((x, y)))

        elif event == cv2.EVENT_RBUTTONDOWN:

            if self._line_points:
                self._line_points.pop()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
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

        # Draw completed boxes
        for x1, y1, x2, y2 in self._boxes:

            overlay = canvas.copy()

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw current rubber-band box
        if self._drawing:

            overlay = canvas.copy()

            cv2.rectangle(overlay, self._start, self._end, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            cv2.rectangle(canvas, self._start, self._end, (0, 255, 0), 2)

            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])

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
        """
        Draw current polygon preview.
        """

        canvas = image.copy()

        if not self._points:
            return canvas

        if len(self._points) >= 3:

            overlay = canvas.copy()

            pts = np.array(self._points, dtype=np.int32)

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

        if len(self._points) > 1:

            pts = np.array(self._points, dtype=np.int32)

            cv2.polylines(
                canvas,
                [pts],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )

        if len(self._points) >= 3:

            cv2.line(
                canvas,
                self._points[-1],
                self._points[0],
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        for idx, pt in enumerate(self._points):

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

        for point in self._line_points:
            cv2.circle(canvas, point, 4, (0, 255, 0), -1)

        if len(self._line_points) == 1 and self._mouse_pos is not None:
            cv2.line(
                canvas,
                self._line_points[0],
                self._mouse_pos,
                (0, 255, 0),
                2,
            )

        elif len(self._line_points) == 2:
            p1, p2 = self._line_points

            cv2.line(canvas, p1, p2, (0, 255, 0), 2)

            label = f"{p1} → {p2}"

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