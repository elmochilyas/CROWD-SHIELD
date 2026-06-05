# SPEC_frame_reader.md — `pipeline/frame_reader.py`

## Purpose

Abstracts video input (file or RTSP stream) behind a simple iterator interface.
Handles open/close, resize, and frame-rate control.
The rest of the pipeline never touches `cv2.VideoCapture` directly.

---

## Class: `FrameReader`

### Constructor

```python
def __init__(self, source: str | int, target_fps: int = 20) -> None:
```

**Parameters:**
- `source`: file path (e.g. `"video.mp4"`) or RTSP URL (e.g. `"rtsp://..."`) or webcam index (`0`)
- `target_fps`: target processing frame rate. If source fps > target_fps, frames are skipped evenly.

**Behavior:**
- Opens `cv2.VideoCapture(source)`
- Raises `RuntimeError` if the capture fails to open
- Reads native fps from the capture (`cv2.CAP_PROP_FPS`)
- Computes `self._skip_interval = max(1, round(native_fps / target_fps))`
- Stores original frame dimensions: `self.width`, `self.height`

### Properties

```python
@property
def frame_width(self) -> int: ...

@property
def frame_height(self) -> int: ...

@property
def native_fps(self) -> float: ...

@property
def is_open(self) -> bool: ...
```

### Method: `read()`

```python
def read(self) -> tuple[bool, np.ndarray | None, int]:
    """
    Returns:
        (success: bool, frame: np.ndarray | None, frame_index: int)

    - Reads the next frame, respecting skip_interval
    - frame is BGR, shape (H, W, 3), dtype uint8
    - Returns (False, None, frame_index) on end-of-stream or read error
    - frame_index increments on every call, including skipped frames
    """
```

**Skip logic:**
```
frame_index % skip_interval != 0  →  grab (advance) but do not decode
frame_index % skip_interval == 0  →  retrieve (decode) and return
```

### Method: `release()`

```python
def release(self) -> None:
    """Releases the VideoCapture. Safe to call multiple times."""
```

### Context manager

```python
def __enter__(self) -> "FrameReader": ...
def __exit__(self, *args) -> None: ...
```

Usage:
```python
with FrameReader("stadium_zone_a.mp4", target_fps=20) as reader:
    while True:
        ok, frame, idx = reader.read()
        if not ok:
            break
        # process frame
```

---

## Acceptance criteria

1. `FrameReader("nonexistent.mp4")` raises `RuntimeError` with a descriptive message.
2. On a 60fps source with `target_fps=20`, every 3rd frame is decoded (skip_interval=3). The other frames are grabbed but not decoded.
3. `read()` returns `(False, None, N)` at end of file — never raises an exception.
4. `frame_index` increments monotonically regardless of skipping.
5. Two consecutive calls to `release()` do not raise.
6. Frame returned is always dtype `uint8` and shape `(H, W, 3)`.
