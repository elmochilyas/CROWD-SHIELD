# SPEC_detector.md — `pipeline/detector.py`

## Purpose

Wraps YOLOv8 detection + pose + ByteTrack into a single `.run(frame)` call.
Returns a typed list of `Detection` objects ready for the behavior engine.
Handles both person detection and dangerous-object detection in one pass.

---

## Class: `Detector`

### Constructor

```python
def __init__(
    self,
    det_model_path: str = config.DETECTION_MODEL,
    pose_model_path: str = config.POSE_MODEL,
    device: str = config.DEVICE,
) -> None:
```

**Behavior:**
- Loads detection model: `self._det = YOLO(det_model_path).to(device)`
- Loads pose model: `self._pose = YOLO(pose_model_path).to(device)`
- Sets both models to eval mode
- Logs model load confirmation to stdout

### Method: `run(frame, frame_index)`

```python
def run(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
    """
    Parameters:
        frame:       BGR image, shape (H, W, 3), uint8
        frame_index: current frame index (used for tracker state)

    Returns:
        List of Detection objects for all tracked persons + all flagged objects.
    """
```

**Internal steps:**

**Step 1 — Object detection with tracking (person + objects)**
```python
det_results = self._det.track(
    frame,
    persist=True,
    tracker=config.TRACKER_CONFIG,
    conf=config.CONF_THRESHOLD,
    iou=config.IOU_THRESHOLD,
    imgsz=config.INPUT_SIZE,
    verbose=False,
)
```
- From `det_results[0].boxes`, extract all boxes where `class_id == 0` (person).
- Also extract all boxes where `class_id in config.DANGEROUS_CLASS_IDS`.
- For each person box: note bbox, confidence, track_id (from `.id`). track_id may be None if tracker hasn't assigned one yet — assign a temporary negative ID in that case.

**Step 2 — Thin-rod heuristic**
For every detected bounding box (any class), check:
```python
if bbox.aspect_ratio >= config.THIN_ROD_ASPECT_RATIO \
   and bbox.width <= config.THIN_ROD_MAX_WIDTH_PX:
    flag as dangerous object, class_name = "thin_rod"
```

**Step 3 — Pose estimation (persons only)**
```python
pose_results = self._pose(
    frame,
    conf=config.CONF_THRESHOLD,
    imgsz=config.INPUT_SIZE,
    verbose=False,
)
```
- Extract keypoints from `pose_results[0].keypoints.data` → shape `(N, 17, 3)`
- Match each pose detection to a person detection by IoU of bounding boxes.
- If a match is found (IoU ≥ 0.50), attach the `(17, 3)` keypoint array to the Detection.
- If no match, set `keypoints = None`.

**Step 4 — Object-person association**
For each detected dangerous object:
- Find the person Detection whose bbox overlaps most with the object bbox (IoU or containment check).
- If overlap found, set `person.signals_hint_object = True` and `person.signals_hint_object_type = class_name`.
- Store object detections in a separate list `self._last_objects` for renderer access.

**Step 5 — Build and return Detection list**
Return only person Detections (not object boxes). The renderer fetches `self._last_objects` separately.

### Property: `last_objects`

```python
@property
def last_objects(self) -> list[Detection]:
    """Returns the last set of detected dangerous objects (for renderer)."""
```

---

## IoU helper (module-level function)

```python
def compute_iou(a: BBox, b: BBox) -> float:
    """Standard axis-aligned IoU."""
```

---

## Acceptance criteria

1. On a frame with 0 persons, `run()` returns `[]` and does not raise.
2. On a frame with a person, every returned Detection has `track_id >= 0`.
3. On two consecutive frames with the same person visible, `track_id` must be identical (ByteTrack continuity).
4. On a frame containing a baseball bat near a person, that person's Detection must have `signals_hint_object = True`.
5. A tall thin bounding box (aspect_ratio ≥ 5, width ≤ 30px) triggers the thin-rod flag regardless of COCO class.
6. `keypoints` shape is `(17, 3)` when pose is matched, `None` otherwise.
7. `run()` must complete in < 80ms per frame on GPU (1080p input downscaled to 640).
