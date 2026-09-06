# Contributing to pixpick

Welcome! We are excited that you are interested in contributing to `pixpick`. Whether you are fixing a bug, improving documentation, or adding a new visual selector, your contributions help make interactive computer vision workflows faster and easier for everyone.

---

## 🚀 Quickstart Development Guide

Follow these minimal architecture rules when extending `pixpick`:

### Adding a New Selector Type (e.g. Line, Point)

1. **Core Data Structure**: Add a geometry dataclass to `core/<type>.py` with the coordinate properties (`.xyxy`, `.norm`, `.sam`, … — all properties, not methods) and persistence methods (`save()`, `load()`). Add a matching `Multi<Type>` that holds a `list[<Type>]` and delegates every property to its items.
2. **Backend Engine**: Add `select_<type>()` to `BaseBackend` and implement its interactive drawing behavior in `CV2Backend`.
3. **Selector Interface**: Create `selectors/<type>_picker.py` with a `<Type>Selector` class — load the image, call the backend, wrap the result, and raise `SelectionCancelled` (from `pixpick.utils`) when the backend returns `None`.
4. **Top-Level API**: Export the selector helper function (e.g., `pixpick.point()`) in `__init__.py`.

> **Note**: No other foundational files should require changes.

### Adding Properties or Persistence Methods

Add properties or serialization methods directly to the targeted target class inside `core/<class>.py`:

```python
# Example in core/box.py
@property
def xyxy(self) -> list[int]:
    """[x1, y1, x2, y2] — absolute pixels."""
    return [self.x1, self.y1, self.x2, self.y2]
```

On a `Multi*` class, delegate to the items rather than reimplementing the maths:

```python
# Example in core/box.py, on Multibox
@property
def xyxy(self) -> list[list[int]]:
    """[[x1, y1, x2, y2], ...] — absolute pixels, one per box."""
    return [box.xyxy for box in self.boxes]
```

---

## 🐞 Reporting Bugs & Opening Issues

Before creating a new issue, please search existing [GitHub Issues](https://github.com/K-saif/pixpick/issues) to verify it hasn't already been reported.

When opening an issue, please include:

* **Clear Description**: A short title and detailed description of the bug or feature request.
* **Minimum Reproducible Example**: A concise Python code snippet that reproduces the problem.
* **Environment Details**: Your OS, Python version, OpenCV version, and hardware (if GPU/display windowing issues occur).
* **Expected vs. Actual Behavior**: What you expected to happen versus what actually occurred (including tracebacks/error logs).

---

## 🤝 How to Submit a Pull Request

1. **Fork & Branch**: Fork the repository and create a descriptive feature branch (`git checkout -b feat-line-selector`).
2. **Keep PRs Scope-Focused**: Small, modular PRs that address a single bug or component are reviewed and merged much faster.
3. **Test Local Changes**: Ensure your changes run cleanly without breaking existing OpenCV interactive workflows.
4. **Submit PR**: Open a Pull Request into the `main` branch with a concise summary of what was added or changed.

---

## ✍️ Docstring Guidelines

Use **Google-style docstrings** with type hints for new functions, classes, and selector methods:

```python
def select_box(image_path: str, color: tuple[int, int, int] = (0, 255, 0)) -> list[int]:
    """Interactively select a bounding box region on an image.

    Args:
        image_path: Path to the target image file.
        color: BGR tuple for rendering the bounding box overlay.

    Returns:
        Bounding box coordinates as absolute pixel values [x1, y1, x2, y2].

    Examples:
        >>> box = select_box("frame.jpg")
    """
    ...

```

For smaller helper methods, a single-line docstring is sufficient:

```python
def xyxy(self) -> list[int]:
    """Returns bounding box coordinates in [x1, y1, x2, y2] format."""
    ...

```

---

## ✨ Best Practices

* **Minimize Code Duplication**: Reuse shared backend utilities across selector types.
* **Keep Dependencies Light**: `pixpick` is built to be a fast, lightweight toolkit—avoid adding heavy external dependencies unless strictly necessary.
* **Preserve Output Standards**: Ensure bounding box and geometry outputs map cleanly into standard computer vision formats (`YOLO`, `SAM2`, `Supervision`).

---