# SPEC_config.md — `config.py`

## Purpose

Single source of truth for all numeric thresholds, model paths, and display constants.
No magic numbers anywhere else in the codebase — everything references `config.py`.

---

## File: `config.py`

### Model settings

```python
DETECTION_MODEL   = "yolov8m.pt"         # swap to yolov8n.pt for CPU-only
POSE_MODEL        = "yolov8m-pose.pt"     # swap to yolov8n-pose.pt for CPU-only
DEVICE            = "cuda:0"              # or "cpu"
CONF_THRESHOLD    = 0.40                  # minimum detection confidence
IOU_THRESHOLD     = 0.45                  # NMS IoU threshold
INPUT_SIZE        = 640                   # YOLOv8 input resolution
```

### Tracking settings

```python
TRACKER_CONFIG    = "bytetrack.yaml"      # Ultralytics built-in
TRACK_HISTORY_LEN = 15                    # frames of centroid history per track_id
```

### Behavior signal thresholds

```python
# Velocity
VELOCITY_NORM_FACTOR   = 50.0   # pixels/frame → divide raw pixel speed by this to normalize to 0–1
VELOCITY_HIGH          = 0.70   # normalized velocity above this = high signal

# Acceleration
ACCEL_WINDOW           = 5      # frames over which to compute delta-velocity
ACCEL_NORM_FACTOR      = 20.0   # normalization divisor

# Proximity
PROXIMITY_RADIUS_PX    = 80     # pixel radius to count neighbors
PROXIMITY_HIGH         = 5      # neighbor count above this = high signal

# Pose anomaly — arm raise detection
ARM_RAISE_ANGLE_THRESH = 30.0   # degrees: wrist above shoulder by this margin = raised
POSE_CONF_MIN          = 0.50   # ignore keypoints below this confidence

# Object detection — dangerous object COCO class IDs
DANGEROUS_CLASS_IDS    = {38, 49}    # 38 = baseball bat, 49 = knife
THIN_ROD_ASPECT_RATIO  = 5.0         # height/width ratio to flag as thin rod/weapon
THIN_ROD_MAX_WIDTH_PX  = 30          # max bbox width to qualify as thin object
```

### Risk scorer weights (must sum to 1.0)

```python
WEIGHT_VELOCITY    = 0.25
WEIGHT_ACCEL       = 0.15
WEIGHT_PROXIMITY   = 0.20
WEIGHT_POSE        = 0.25
WEIGHT_OBJECT      = 0.15
```

### Risk tier boundaries

```python
THRESH_LOW       = 0.25   # below this → "low"      (green)
THRESH_MEDIUM    = 0.45   # below this → "medium"   (yellow)
THRESH_HIGH      = 0.65   # below this → "high"     (orange)
# above THRESH_HIGH       → "critical"              (red)
```

### Zone state boundaries

```python
ZONE_CALM        = 0.25
ZONE_WATCH       = 0.45
ZONE_WARNING     = 0.65
# above ZONE_WARNING → CRITICAL
```

### Zone score formula weights

```python
ZONE_WEIGHT_MAX  = 0.60   # contribution of the top-3 average risk score
ZONE_WEIGHT_MEAN = 0.40   # contribution of population mean risk score
ZONE_TOP_K       = 3      # number of highest-risk persons for the max component
```

### Display constants

```python
# HUD bar
HUD_HEIGHT_PX       = 60
HUD_BG_COLOR        = (0, 0, 0)          # BGR
HUD_BG_ALPHA        = 0.75
HUD_FONT            = cv2.FONT_HERSHEY_DUPLEX
HUD_FONT_SCALE      = 1.4
HUD_FONT_THICKNESS  = 2

# Per-person bbox
BBOX_THICKNESS      = 2
LABEL_FONT          = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE    = 0.55
LABEL_FONT_THICK    = 1

# Risk tier → BGR color
TIER_COLORS = {
    "low":      (60,  200, 60),    # green
    "medium":   (0,   210, 255),   # yellow
    "high":     (0,   140, 255),   # orange
    "critical": (0,    40, 220),   # red
}

# Zone state → (BGR color, display text)
ZONE_DISPLAY = {
    "CALM":     {"color": (60,  200, 60),  "text": "● ZONE CALM"},
    "WATCH":    {"color": (0,   210, 255), "text": "◉ ZONE WATCH"},
    "WARNING":  {"color": (0,   140, 255), "text": "▲ ZONE WARNING"},
    "CRITICAL": {"color": (0,    40, 220), "text": "⬛ ZONE CRITICAL"},
}
```

---

## Acceptance criteria

1. All other modules import from `config` — no numeric literals in business logic.
2. Swapping `DETECTION_MODEL` to the nano variant must change inference speed without touching any other file.
3. `WEIGHT_VELOCITY + WEIGHT_ACCEL + WEIGHT_PROXIMITY + WEIGHT_POSE + WEIGHT_OBJECT == 1.0` — add an assertion at module load time.
