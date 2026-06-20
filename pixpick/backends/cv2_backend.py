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
    _start:   tuple[int, int]
    _end:     tuple[int, int]
    _confirmed: bool
    _cancelled: bool

    def select_box(
        self,
        image: np.ndarray,
        title: str = "pixpick  |  drag to select  |  Enter=confirm  R=reset  Esc=cancel",
    ) -> tuple[int, int, int, int] | None:

        self._reset_state()
        canvas = image.copy()   # scratch copy for live drawing

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._mouse_callback, param={"image": image, "title": title})
        cv2.imshow(title, canvas)

        while True:
            # Redraw every frame so the rubber-band rect updates.
            canvas = self._draw_rect(image)
            cv2.imshow(title, canvas)

            key = cv2.waitKey(20) & 0xFF

            if self._cancelled or key == 27:        # Esc
                cv2.destroyWindow(title)
                return None

            if key in (13, 32) or self._confirmed:  # Enter / Space
                if self._start == self._end:
                    # User hit confirm without completing a drag — ignore.
                    continue
                cv2.destroyWindow(title)
                x1 = min(self._start[0], self._end[0])
                y1 = min(self._start[1], self._end[1])
                x2 = max(self._start[0], self._end[0])
                y2 = max(self._start[1], self._end[1])
                return x1, y1, x2, y2

            if key == ord("r"):                     # Reset
                self._reset_state()

    # ------------------------------------------------------------------ #
    # Mouse callback                                                       #
    # ------------------------------------------------------------------ #

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: dict) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start   = (x, y)
            self._end     = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing   = False
            self._end       = (x, y)
            self._confirmed = True   # auto-confirm on mouse release

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _reset_state(self) -> None:
        self._drawing   = False
        self._start     = (0, 0)
        self._end       = (0, 0)
        self._confirmed = False
        self._cancelled = False

    def _draw_rect(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of image with the current rubber-band rect drawn."""
        canvas = image.copy()
        if self._start != self._end:
            # Semi-transparent fill
            overlay = canvas.copy()
            cv2.rectangle(overlay, self._start, self._end, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
            # Solid border
            cv2.rectangle(canvas, self._start, self._end, (0, 255, 0), 2)
            # Corner coords label
            x1 = min(self._start[0], self._end[0])
            y1 = min(self._start[1], self._end[1])
            x2 = max(self._start[0], self._end[0])
            y2 = max(self._start[1], self._end[1])
            label = f"({x1},{y1}) ({x2},{y2})  w={x2-x1} h={y2-y1}"
            cv2.putText(canvas, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas