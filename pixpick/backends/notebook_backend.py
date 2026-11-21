from __future__ import annotations

import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle

from pixpick.backends.base import BaseBackend
from pixpick.core.point import BACKGROUND, FOREGROUND


# RGB twins of the BGR overlay colours used by the cv2 backend.
_BOX_COLOR = "#8000ff"
_POLYGON_COLOR = "#ff0000"
_LINE_COLOR = "#00ff00"
_FOREGROUND_COLOR = "#00ff00"
_BACKGROUND_COLOR = "#ff0000"
_FILL_ALPHA = 0.15

_MAX_FIG_WIDTH_IN = 12.8
_MAX_FIG_HEIGHT_IN = 7.2
_MIN_FIG_WIDTH_IN = 6.0
_MIN_FIG_HEIGHT_IN = 4.0


def _running_in_colab() -> bool:
    """Check whether the current process runs inside Google Colab.

    Returns:
        bool: True when the ``google.colab`` package is available.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


class NotebookBackend(BaseBackend):
    """Matplotlib-based backend for interactive selections inside notebooks.

    Renders the image inline in a Jupyter or VS Code notebook and returns
    selections in absolute image pixels, mirroring the controls of
    :class:`~pixpick.backends.cv2_backend.CV2Backend`.

    Requirements:
        The interactive ipympl backend must be active. Install it with
        ``pip install ipympl`` and enable it by running ``%matplotlib widget``
        in the notebook before calling pixpick. Google Colab is not supported
        because it does not provide the ipympl backend; a ``RuntimeError``
        with instructions is raised there.

    Controls:
        Left-click and drag: Draw a bounding box.
        Left-click: Add a polygon vertex, a line endpoint, or a point.
        Shift+left-click: Add a background point (point selector only).
        Right-click: Undo the last point, vertex, box, or line.
        Space: Commit the current polygon and start a new one.
        Enter: Confirm the selection.
        Z or Backspace: Clear everything drawn so far.
        Esc: Cancel and return ``None``.

    Keyboard shortcuts only reach the figure once it has keyboard focus;
    click the image once if the keys do not respond. The image is shown in
    Matplotlib data coordinates that map one-to-one to source pixels, so
    clicks convert to image pixels without any manual rescaling; the figure
    is only sized down for display on very large images. The original image
    array is never mutated.
    """

    _image_width: int
    _image_height: int
    _rgb: np.ndarray
    _title: str
    _result: object
    _done: threading.Event

    def select_box(
        self,
        image: np.ndarray,
        title: str = (
            "pixpick | boxes | "
            "Drag=LMB  RMB=undo  Enter=confirm  Z=clear  Esc=cancel"
        ),
    ) -> list[list[int]] | None:

        """Let the user draw one or more bounding boxes in the notebook.

        Args:
            image: BGR image on which to select boxes.
            title: Figure title and interaction instructions.

        Returns:
            A list of boxes in absolute pixel coordinates, each formatted as
            ``[x1, y1, x2, y2]``. Returns ``None`` if the user cancels.

        Raises:
            RuntimeError: If the active Matplotlib backend is not interactive
                ipympl, or if running on Google Colab.
        """

        self._check_environment()
        self._reset(image, title)

        self._boxes: list[list[int]] = []
        self._box_artists: list[Rectangle] = []
        self._drag_start: tuple[int, int] | None = None
        self._drag_rect: Rectangle | None = None

        self._open_figure()

        return self._interact(
            press=self._on_box_press,
            motion=self._on_box_motion,
            release=self._on_box_release,
            key=self._on_box_key,
        )

    def select_polygon(
        self,
        image: np.ndarray,
        title: str = (
            "pixpick | polygon | "
            "LMB=add  RMB=undo  Space=new polygon  "
            "Enter=confirm  Z=clear  Esc=cancel"
        ),
    ) -> list[list[tuple[int, int]]] | None:

        """Let the user select one or more polygons in the notebook.

        Args:
            image: BGR image on which to select polygons.
            title: Figure title and interaction instructions.

        Returns:
            A list containing one list of absolute pixel-coordinate vertices
            per polygon. Each polygon contains at least three points.
            Returns ``None`` if the user cancels.

        Raises:
            RuntimeError: If the active Matplotlib backend is not interactive
                ipympl, or if running on Google Colab.
        """

        self._check_environment()
        self._reset(image, title)

        self._verts: list[tuple[int, int]] = []
        self._vert_labels = []
        self._polys: list[list[tuple[int, int]]] = []
        self._poly_artists = []

        self._open_figure()

        ax = self._ax
        self._preview_dots = ax.scatter([], [], s=25, color=_POLYGON_COLOR, zorder=5)
        self._preview_line = Line2D([], [], color=_POLYGON_COLOR, linewidth=2, zorder=4)
        self._preview_line.set_visible(False)
        ax.add_line(self._preview_line)
        self._preview_fill = PolygonPatch(
            np.empty((0, 2)),
            closed=True,
            facecolor=to_rgba(_POLYGON_COLOR, _FILL_ALPHA),
            edgecolor="none",
            zorder=2,
        )
        self._preview_fill.set_visible(False)
        ax.add_patch(self._preview_fill)
        self._rubber = Line2D([], [], color=_POLYGON_COLOR, linewidth=1, zorder=4)
        self._rubber.set_visible(False)
        ax.add_line(self._rubber)

        return self._interact(
            press=self._on_polygon_press,
            motion=self._on_polygon_motion,
            key=self._on_polygon_key,
        )

    def select_line(
        self,
        image: np.ndarray,
        title: str = (
            "pixpick | lines | "
            "LMB=select endpoints  RMB=undo  Enter=confirm  Z=reset  Esc=cancel"
        ),
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:

        """Let the user select one or more lines in the notebook.

        Args:
            image: BGR image on which to select lines.
            title: Figure title and interaction instructions.

        Returns:
            A list of lines, each represented by two absolute pixel-coordinate
            endpoints: ``((x1, y1), (x2, y2))``. Returns ``None`` if the user
            cancels.

        Raises:
            RuntimeError: If the active Matplotlib backend is not interactive
                ipympl, or if running on Google Colab.
        """

        self._check_environment()
        self._reset(image, title)

        self._lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._line_artists = []
        self._pending: list[tuple[int, int]] = []

        self._open_figure()

        self._rubber = Line2D([], [], color=_LINE_COLOR, linewidth=2, zorder=4)
        self._rubber.set_visible(False)
        self._ax.add_line(self._rubber)

        return self._interact(
            press=self._on_line_press,
            motion=self._on_line_motion,
            key=self._on_line_key,
        )

    def select_point(
        self,
        image: np.ndarray,
        title: str = (
            "pixpick | points | "
            "LMB=foreground  Shift+LMB=background  RMB=undo  "
            "Enter=confirm  Z=reset  Esc=cancel"
        ),
    ) -> tuple[list[tuple[int, int]], list[int]] | None:

        """Let the user select foreground and background points in the notebook.

        Args:
            image: BGR image on which to select points.
            title: Figure title and interaction instructions.

        Returns:
            A tuple containing a list of absolute pixel-coordinate points and
            a parallel list of labels. Labels are ``1`` for foreground and
            ``0`` for background. Returns ``None`` if the user cancels.

        Raises:
            RuntimeError: If the active Matplotlib backend is not interactive
                ipympl, or if running on Google Colab.
        """

        self._check_environment()
        self._reset(image, title)

        self._points: list[tuple[int, int]] = []
        self._labels: list[int] = []
        self._point_artists = []

        self._open_figure()

        return self._interact(
            press=self._on_point_press,
            key=self._on_point_key,
        )

    # ------------------------------------------------------------------ #
    # Session plumbing                                                   #
    # ------------------------------------------------------------------ #

    def _check_environment(self) -> None:
        """Verify that an interactive notebook Matplotlib backend is active.

        Raises:
            RuntimeError: On Google Colab, or when the active Matplotlib
                backend is not the ipympl ``widget`` backend.
        """

        if _running_in_colab():
            raise RuntimeError(
                "NotebookBackend does not work on Google Colab because Colab "
                "does not support the ipympl ('%matplotlib widget') backend. "
                "Use Jupyter, JupyterLab, or a VS Code notebook instead, or "
                "keep the default cv2 backend."
            )

        backend_name = matplotlib.get_backend().lower()

        if backend_name != "widget" and "ipympl" not in backend_name:
            raise RuntimeError(
                "NotebookBackend requires the interactive ipympl Matplotlib "
                "backend. Install it with `pip install ipympl`, then run "
                "`%matplotlib widget` in the notebook before calling pixpick."
            )

    def _reset(self, image: np.ndarray, title: str) -> None:
        """Reset per-selection state and prepare the image for display.

        Args:
            image: BGR source image whose height and width are recorded.
            title: Figure title shown above the image.
        """

        self._title = title
        self._image_height, self._image_width = image.shape[:2]
        self._rgb = np.ascontiguousarray(image[:, :, ::-1])
        self._done = threading.Event()
        self._result = None

    def _open_figure(self) -> None:
        """Create the notebook figure showing the source image.

        The figure is capped at roughly 1280x720 logical pixels and enlarged
        to a comfortable minimum for tiny images. Display scaling never
        affects the returned coordinates: Matplotlib data coordinates map
        one-to-one to source image pixels.
        """

        dpi = float(plt.rcParams.get("figure.dpi", 100.0))
        width_in = self._image_width / dpi
        height_in = self._image_height / dpi
        scale = min(
            1.0,
            _MAX_FIG_WIDTH_IN / max(width_in, 1e-6),
            _MAX_FIG_HEIGHT_IN / max(height_in, 1e-6),
        )
        figsize = (
            max(width_in * scale, _MIN_FIG_WIDTH_IN),
            max(height_in * scale, _MIN_FIG_HEIGHT_IN),
        )

        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._ax.imshow(self._rgb)
        self._ax.set_axis_off()
        self._set_status()
        self._disconnect_default_keymap()

    def _disconnect_default_keymap(self) -> None:
        """Disconnect Matplotlib's default key bindings for this figure.

        Prevents built-in shortcuts (``q`` closes the figure, ``s`` saves,
        and so on) from interfering with pixpick's selection keys.
        """

        manager = getattr(self._fig.canvas, "manager", None)
        handler_id = getattr(manager, "key_press_handler_id", None)

        if handler_id is not None:
            self._fig.canvas.mpl_disconnect(handler_id)

    def _interact(
        self,
        press=None,
        motion=None,
        release=None,
        key=None,
    ):
        """Wire Matplotlib event handlers and block until confirm or cancel.

        Args:
            press: Handler for ``button_press_event``.
            motion: Handler for ``motion_notify_event``.
            release: Handler for ``button_release_event``.
            key: Handler for ``key_press_event``.

        Returns:
            The result stored by the mode handlers; ``None`` on cancel.
        """

        canvas = self._fig.canvas
        events = {
            "button_press_event": press,
            "motion_notify_event": motion,
            "button_release_event": release,
            "key_press_event": key,
            "close_event": self._on_close,
        }
        cids = {
            name: canvas.mpl_connect(name, handler)
            for name, handler in events.items()
            if handler is not None
        }

        plt.show()

        try:
            self._pump()
        finally:
            for cid in cids.values():
                canvas.mpl_disconnect(cid)
            plt.close(self._fig)

        return self._result

    def _pump(self) -> None:
        """Block while pumping notebook events until confirm or cancel.

        Runs the IPython kernel's single-iteration message loop so that
        Matplotlib events generated in the browser reach the handlers
        connected by this backend, then sleeps briefly to avoid spinning
        the CPU.
        """

        kernel = None

        try:
            from IPython import get_ipython

            kernel = getattr(get_ipython(), "kernel", None)
        except ImportError:
            kernel = None

        while not self._done.is_set():
            if kernel is not None:
                kernel.do_one_iteration()

            try:
                self._fig.canvas.flush_events()
            except NotImplementedError:
                pass

            time.sleep(0.01)

    def _on_close(self, event) -> None:
        """Treat a closed figure as a cancellation.

        Args:
            event: Matplotlib close event.
        """

        if not self._done.is_set():
            self._cancel()

    def _finish(self, result) -> None:
        """Store the selection result and unblock the caller.

        Args:
            result: Value to return from the ``select_*`` method.
        """

        self._result = result
        self._done.set()

    def _cancel(self) -> None:
        """Abort the selection so the caller receives ``None``."""
        self._finish(None)

    def _key_action(self, event) -> str | None:
        """Map a Matplotlib key event to a pixpick action name.

        Args:
            event: Matplotlib key-press event.

        Returns:
            One of ``"confirm"``, ``"cancel"``, ``"clear"``, or ``"next"``
            (new polygon), or ``None`` when the key is not bound.
        """

        key = (event.key or "").lower()

        if key in ("enter", "return"):
            return "confirm"

        if key == "escape":
            return "cancel"

        if key in ("z", "backspace"):
            return "clear"

        if key in (" ", "space"):
            return "next"

        return None

    def _event_point(self, event) -> tuple[int, int] | None:
        """Extract the image-pixel point under a Matplotlib event.

        Args:
            event: Matplotlib mouse event.

        Returns:
            The event position in absolute image pixels, or ``None`` when
            the event happened outside the image axes.
        """

        if event.inaxes is not self._ax:
            return None

        if event.xdata is None or event.ydata is None:
            return None

        return self._clamp_point((int(round(event.xdata)), int(round(event.ydata))))

    def _clamp_point(self, point: tuple[int, int]) -> tuple[int, int]:
        """Clamp a point to the source-image bounds.

        Args:
            point: Point in image pixel coordinates.

        Returns:
            The point clamped to the valid image bounds.
        """

        x, y = point
        x = max(0, min(x, self._image_width - 1))
        y = max(0, min(y, self._image_height - 1))
        return (x, y)

    def _set_status(self, status: str = "") -> None:
        """Set the figure title, appending an optional live status suffix.

        Args:
            status: Short live feedback (for example point counts) appended
                to the base title. Empty by default.
        """

        title = f"{self._title}  |  {status}" if status else self._title
        self._ax.set_title(title, fontsize=9)

    def _redraw(self) -> None:
        """Request an asynchronous redraw of the notebook figure."""
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Box handlers                                                       #
    # ------------------------------------------------------------------ #

    def _on_box_press(self, event) -> None:
        """Handle button presses during box selection.

        Args:
            event: Matplotlib button-press event.
        """

        point = self._event_point(event)
        if point is None:
            return

        if event.button == 1:
            self._drag_start = point
            rect = Rectangle(
                (point[0], point[1]),
                0,
                0,
                facecolor=to_rgba(_BOX_COLOR, _FILL_ALPHA),
                edgecolor=_BOX_COLOR,
                linewidth=2,
                zorder=4,
            )
            self._ax.add_patch(rect)
            self._drag_rect = rect

        elif event.button == 3:
            self._box_undo()

    def _on_box_motion(self, event) -> None:
        """Update the rubber-band rectangle while the left button is held.

        Args:
            event: Matplotlib motion-notify event.
        """

        if self._drag_rect is None or self._drag_start is None:
            return

        point = self._event_point(event)
        if point is None:
            return

        x1 = min(self._drag_start[0], point[0])
        y1 = min(self._drag_start[1], point[1])
        x2 = max(self._drag_start[0], point[0])
        y2 = max(self._drag_start[1], point[1])

        self._drag_rect.set_xy((x1, y1))
        self._drag_rect.set_width(x2 - x1)
        self._drag_rect.set_height(y2 - y1)

        self._set_status(f"({x1},{y1}) ({x2},{y2})")
        self._redraw()

    def _on_box_release(self, event) -> None:
        """Finalize the rubber-band rectangle when the left button is released.

        Args:
            event: Matplotlib button-release event.
        """

        if self._drag_rect is None or self._drag_start is None:
            return

        if event.button != 1:
            return

        point = self._event_point(event) or self._drag_start

        x1 = min(self._drag_start[0], point[0])
        y1 = min(self._drag_start[1], point[1])
        x2 = max(self._drag_start[0], point[0])
        y2 = max(self._drag_start[1], point[1])

        self._drag_start = None

        if x1 == x2 or y1 == y2:
            self._drag_rect.remove()
        else:
            self._boxes.append([x1, y1, x2, y2])
            self._box_artists.append(self._drag_rect)

        self._drag_rect = None
        self._set_status()
        self._redraw()

    def _on_box_key(self, event) -> None:
        """Handle keyboard shortcuts during box selection.

        Args:
            event: Matplotlib key-press event.
        """

        action = self._key_action(event)

        if action == "cancel":
            self._cancel()
        elif action == "clear":
            self._box_clear_all()
        elif action == "confirm":
            if self._boxes and self._drag_rect is None:
                self._finish([list(box) for box in self._boxes])

    def _box_undo(self) -> None:
        """Remove the box currently being drawn, or the last committed box."""

        if self._drag_rect is not None:
            self._drag_rect.remove()
            self._drag_rect = None
            self._drag_start = None
            self._set_status()
            self._redraw()
            return

        if self._box_artists:
            self._box_artists.pop().remove()
            self._boxes.pop()
            self._redraw()

    def _box_clear_all(self) -> None:
        """Remove every box drawn in the current selection session."""

        for artist in self._box_artists:
            artist.remove()

        self._box_artists = []
        self._boxes = []

        if self._drag_rect is not None:
            self._drag_rect.remove()
            self._drag_rect = None
            self._drag_start = None

        self._set_status()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Polygon handlers                                                   #
    # ------------------------------------------------------------------ #

    def _on_polygon_press(self, event) -> None:
        """Handle button presses during polygon selection.

        Args:
            event: Matplotlib button-press event.
        """

        point = self._event_point(event)
        if point is None:
            return

        if event.button == 1:
            self._verts.append(point)
            label = self._ax.text(
                point[0] + 5,
                point[1] - 5,
                str(len(self._verts) - 1),
                color=_POLYGON_COLOR,
                fontsize=8,
                zorder=6,
            )
            self._vert_labels.append(label)
            self._update_polygon_preview()

        elif event.button == 3:
            self._polygon_undo()

    def _on_polygon_motion(self, event) -> None:
        """Stretch a rubber line from the last vertex to the cursor.

        Args:
            event: Matplotlib motion-notify event.
        """

        if not self._verts:
            return

        point = self._event_point(event)
        if point is None:
            return

        last = self._verts[-1]
        self._rubber.set_data([last[0], point[0]], [last[1], point[1]])
        self._rubber.set_visible(True)
        self._redraw()

    def _on_polygon_key(self, event) -> None:
        """Handle keyboard shortcuts during polygon selection.

        Args:
            event: Matplotlib key-press event.
        """

        action = self._key_action(event)

        if action == "cancel":
            self._cancel()
        elif action == "clear":
            self._polygon_clear_all()
        elif action == "next":
            self._polygon_commit()
        elif action == "confirm":
            # Mirror the cv2 backend: an in-progress polygon with fewer
            # than three vertices blocks confirmation instead of being
            # silently discarded.
            if self._verts and len(self._verts) < 3:
                return

            if self._verts:
                self._polygon_commit()

            if self._polys:
                self._finish([list(polygon) for polygon in self._polys])

    def _update_polygon_preview(self) -> None:
        """Refresh the in-progress polygon preview artists and status line."""

        xs = [point[0] for point in self._verts]
        ys = [point[1] for point in self._verts]

        offsets = np.column_stack([xs, ys]) if self._verts else np.empty((0, 2))
        self._preview_dots.set_offsets(offsets)

        if len(self._verts) >= 2:
            self._preview_line.set_data(xs, ys)
            self._preview_line.set_visible(True)
        else:
            self._preview_line.set_visible(False)

        if len(self._verts) >= 3:
            self._preview_fill.set_xy(offsets)
            self._preview_fill.set_visible(True)
        else:
            self._preview_fill.set_visible(False)

        # Re-shown by the next motion event.
        self._rubber.set_visible(False)

        if self._verts:
            self._set_status(f"Points: {len(self._verts)}")
        else:
            self._set_status()

        self._redraw()

    def _polygon_commit(self) -> bool:
        """Commit the in-progress polygon as a closed overlay.

        Returns:
            bool: True when a polygon with at least three vertices was
            committed; False when there were too few vertices.
        """

        if len(self._verts) < 3:
            return False

        polygon = list(self._verts)
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]

        fill = PolygonPatch(
            np.column_stack([xs, ys]),
            closed=True,
            facecolor=to_rgba(_POLYGON_COLOR, _FILL_ALPHA),
            edgecolor="none",
            zorder=2,
        )
        self._ax.add_patch(fill)

        outline = Line2D(xs, ys, color=_POLYGON_COLOR, linewidth=2, zorder=4)
        self._ax.add_line(outline)

        dots = self._ax.scatter(xs, ys, s=25, color=_POLYGON_COLOR, zorder=5)

        self._polys.append(polygon)
        self._poly_artists.append([fill, outline, dots])

        for label in self._vert_labels:
            label.remove()

        self._vert_labels = []
        self._verts = []
        self._update_polygon_preview()

        return True

    def _polygon_undo(self) -> None:
        """Remove the last vertex, or the last committed polygon."""

        if self._verts:
            self._verts.pop()
            self._vert_labels.pop().remove()
            self._update_polygon_preview()
            return

        if self._poly_artists:
            self._polys.pop()
            for artist in self._poly_artists.pop():
                artist.remove()
            self._redraw()

    def _polygon_clear_all(self) -> None:
        """Remove every polygon and vertex drawn in the current session."""

        for label in self._vert_labels:
            label.remove()

        self._vert_labels = []
        self._verts = []

        for artists in self._poly_artists:
            for artist in artists:
                artist.remove()

        self._poly_artists = []
        self._polys = []
        self._update_polygon_preview()

    # ------------------------------------------------------------------ #
    # Line handlers                                                      #
    # ------------------------------------------------------------------ #

    def _on_line_press(self, event) -> None:
        """Handle button presses during line selection.

        Args:
            event: Matplotlib button-press event.
        """

        point = self._event_point(event)
        if point is None:
            return

        if event.button == 1:
            if len(self._pending) >= 2:
                return

            self._pending.append(point)

            if len(self._pending) == 2:
                self._line_commit()

            self._redraw()

        elif event.button == 3:
            self._line_undo()

    def _on_line_motion(self, event) -> None:
        """Stretch a rubber line from the first endpoint to the cursor.

        Args:
            event: Matplotlib motion-notify event.
        """

        if len(self._pending) != 1:
            return

        point = self._event_point(event)
        if point is None:
            return

        first = self._pending[0]
        self._rubber.set_data([first[0], point[0]], [first[1], point[1]])
        self._rubber.set_visible(True)
        self._redraw()

    def _on_line_key(self, event) -> None:
        """Handle keyboard shortcuts during line selection.

        Args:
            event: Matplotlib key-press event.
        """

        action = self._key_action(event)

        if action == "cancel":
            self._cancel()
        elif action == "clear":
            self._line_clear_all()
        elif action == "confirm":
            if self._lines and not self._pending:
                self._finish(list(self._lines))

    def _line_commit(self) -> None:
        """Commit two pending endpoints as a finished line overlay."""

        if len(self._pending) != 2:
            return

        p1, p2 = self._pending
        self._pending = []

        line = Line2D(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=_LINE_COLOR,
            linewidth=2,
            zorder=4,
        )
        self._ax.add_line(line)

        dots = self._ax.scatter(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            s=25,
            color=_LINE_COLOR,
            zorder=5,
        )

        label = self._ax.text(
            min(p1[0], p2[0]),
            max(min(p1[1], p2[1]) - 8, 12),
            f"{p1} to {p2}",
            color=_LINE_COLOR,
            fontsize=8,
            zorder=6,
        )

        self._lines.append((p1, p2))
        self._line_artists.append([line, dots, label])
        self._rubber.set_visible(False)

    def _line_undo(self) -> None:
        """Remove the endpoint being placed, or the last committed line."""

        if self._pending:
            self._pending = []
            self._rubber.set_visible(False)
            self._redraw()
            return

        if self._line_artists:
            self._lines.pop()
            for artist in self._line_artists.pop():
                artist.remove()
            self._redraw()

    def _line_clear_all(self) -> None:
        """Remove every line drawn in the current selection session."""

        for artists in self._line_artists:
            for artist in artists:
                artist.remove()

        self._line_artists = []
        self._lines = []
        self._pending = []
        self._rubber.set_visible(False)
        self._redraw()

    # ------------------------------------------------------------------ #
    # Point handlers                                                     #
    # ------------------------------------------------------------------ #

    def _on_point_press(self, event) -> None:
        """Handle button presses during labeled-point selection.

        Args:
            event: Matplotlib button-press event.
        """

        point = self._event_point(event)
        if point is None:
            return

        if event.button == 1:
            # Shift held → background prompt, plain click → foreground.
            label = BACKGROUND if (event.key or "").lower() == "shift" else FOREGROUND
            color = (
                _FOREGROUND_COLOR if label == FOREGROUND else _BACKGROUND_COLOR
            )

            artist = self._ax.scatter(
                [point[0]],
                [point[1]],
                s=60,
                color=color,
                zorder=5,
            )

            self._points.append(point)
            self._labels.append(label)
            self._point_artists.append(artist)
            self._update_point_status()
            self._redraw()

        elif event.button == 3:
            self._point_undo()

    def _on_point_key(self, event) -> None:
        """Handle keyboard shortcuts during labeled-point selection.

        Args:
            event: Matplotlib key-press event.
        """

        action = self._key_action(event)

        if action == "cancel":
            self._cancel()
        elif action == "clear":
            self._point_clear_all()
        elif action == "confirm":
            if self._points:
                self._finish((list(self._points), list(self._labels)))

    def _update_point_status(self) -> None:
        """Refresh the foreground/background counts in the status line."""

        self._set_status(
            f"fg: {self._labels.count(FOREGROUND)}  "
            f"bg: {self._labels.count(BACKGROUND)}"
        )

    def _point_undo(self) -> None:
        """Remove the last labeled point."""

        if not self._point_artists:
            return

        self._point_artists.pop().remove()
        self._points.pop()
        self._labels.pop()

        if self._points:
            self._update_point_status()
        else:
            self._set_status()

        self._redraw()

    def _point_clear_all(self) -> None:
        """Remove every labeled point drawn in the current session."""

        for artist in self._point_artists:
            artist.remove()

        self._point_artists = []
        self._points = []
        self._labels = []
        self._set_status()
        self._redraw()
