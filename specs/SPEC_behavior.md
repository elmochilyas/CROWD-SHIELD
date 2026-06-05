# SPEC_behavior.md — `pipeline/behavior.py`

## Purpose

Maintains per-person track history across frames.
Extracts 5 behavioral signals per person per frame.
Computes a weighted risk score [0, 1] for each tracked person.

This is the core algorithmic module. All signal logic lives here.

---

## Class: `BehaviorEngine`

### Constructor

```python
def __init__(self) -> None:
```

**Internal state:**
```python
self._tracks: dict[int, TrackRecord]
```
Where `TrackRecord` is a dataclass:
```python
@dataclass
class TrackRecord:
    track_id: int
    centroid_history: deque  # deque of (cx, cy) tuples, maxlen=config.TRACK_HISTORY_LEN
    velocity_history: deque  # deque of float (normalized velocity), maxlen=config.ACCEL_WINDOW
    last_seen_frame: int
```

### Method: `update(detections, frame_index, frame_height)`

```python
def update(
    self,
    detections: list[Detection],
    frame_index: int,
    frame_height: int,
) -> list[PersonState]:
    """
    Parameters:
        detections:   output of Detector.run()
        frame_index:  current frame index
        frame_height: used to normalize velocity

    Returns:
        List of PersonState, one per detection, with computed risk scores.
    """
```

**Step 1 — Update centroid history**
For each detection:
- If `track_id` not in `self._tracks`, create a new `TrackRecord`.
- Append `(bbox.cx, bbox.cy)` to `centroid_history`.
- Set `last_seen_frame = frame_index`.

**Step 2 — Purge stale tracks**
Remove any `track_id` from `self._tracks` where `frame_index - last_seen_frame > 30`.

**Step 3 — Compute signals per person**
Call the 5 private signal methods below for each detection.
Assemble into `SignalVector`.

**Step 4 — Compute risk score**
```python
risk_score = (
    config.WEIGHT_VELOCITY  * signals.velocity +
    config.WEIGHT_ACCEL     * signals.acceleration +
    config.WEIGHT_PROXIMITY * min(signals.proximity_count / config.PROXIMITY_HIGH, 1.0) +
    config.WEIGHT_POSE      * signals.pose_anomaly +
    config.WEIGHT_OBJECT    * (1.0 if signals.object_flag else 0.0)
)
risk_score = float(np.clip(risk_score, 0.0, 1.0))
```

**Step 5 — Assign risk tier**
```python
if risk_score < config.THRESH_LOW:     tier = "low"
elif risk_score < config.THRESH_MEDIUM: tier = "medium"
elif risk_score < config.THRESH_HIGH:   tier = "high"
else:                                   tier = "critical"
```

**Step 6 — Build PersonState and return list**

---

## Private signal methods

### `_compute_velocity(track_id, frame_height) -> float`

```
If centroid_history has < 2 entries: return 0.0
raw_speed = euclidean_distance(history[-1], history[-2])  # pixels
normalized = raw_speed / (frame_height * config.VELOCITY_NORM_FACTOR / 100)
return float(np.clip(normalized, 0.0, 1.0))
```

Note: normalize against frame_height so signal is resolution-independent.

### `_compute_acceleration(track_id) -> float`

```
Append current velocity to velocity_history.
If velocity_history has < 2 entries: return 0.0
accel = abs(velocity_history[-1] - velocity_history[-2])
normalized = accel / (config.ACCEL_NORM_FACTOR / 100)
return float(np.clip(normalized, 0.0, 1.0))
```

### `_compute_proximity(detection, all_detections) -> int`

```
count = 0
for other in all_detections:
    if other.track_id == detection.track_id: continue
    dist = euclidean_distance(detection.bbox.cx/cy, other.bbox.cx/cy)
    if dist <= config.PROXIMITY_RADIUS_PX:
        count += 1
return count
```

Returns raw integer count. Normalization happens in the risk score formula.

### `_compute_pose_anomaly(keypoints) -> float`

```
If keypoints is None: return 0.0

COCO keypoint indices:
  5 = left_shoulder,  6 = right_shoulder
  7 = left_elbow,     8 = right_elbow
  9 = left_wrist,    10 = right_wrist

For each wrist (left: idx 9, right: idx 10):
  wrist_kp  = keypoints[wrist_idx]   # (x, y, conf)
  shoulder_kp = keypoints[shoulder_idx]
  If both conf >= config.POSE_CONF_MIN:
    if wrist_kp.y < (shoulder_kp.y - config.ARM_RAISE_ANGLE_THRESH):
      # In image coords, smaller y = higher in frame
      # wrist above shoulder → arm raised
      raised = True

Score:
  0.0 → no arms raised
  0.5 → one arm raised
  1.0 → both arms raised

Return score.
```

### `_compute_object_flag(detection) -> tuple[bool, str]`

```
If detection.signals_hint_object is True (set by Detector):
    return (True, detection.signals_hint_object_type)
return (False, "")
```

---

## Acceptance criteria

1. A person running fast (velocity signal ≥ 0.7) + both arms raised (pose signal = 1.0) + no object → risk_score ≥ 0.60.
2. A person with an object flag = True and pose anomaly = 1.0 → risk_score ≥ 0.40 (object+pose weights alone = 0.40).
3. A stationary person seated with no arm raise and no object → risk_score ≤ 0.25.
4. Stale tracks (not seen for 30+ frames) are removed from `_tracks`.
5. With 0 detections, `update()` returns `[]` and does not raise.
6. `risk_score` is always in [0.0, 1.0] — never negative, never > 1.
7. Proximity count for a person alone in frame (no others) = 0.
