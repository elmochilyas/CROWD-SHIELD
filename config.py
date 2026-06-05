import cv2

# ── Model settings ──────────────────────────────────────────
DETECTION_MODEL = "yolov8n.pt"
POSE_MODEL      = "yolov8n-pose.pt"
DEVICE          = "cpu"
CONF_THRESHOLD  = 0.20
IOU_THRESHOLD   = 0.30
INPUT_SIZE      = 416

# ── Tracking settings ───────────────────────────────────────
TRACKER_CONFIG    = "bytetrack.yaml"
TRACK_HISTORY_LEN = 6
TARGET_FPS        = 60

# ── Behavior signal thresholds ──────────────────────────────
VELOCITY_NORM_FACTOR   = 5.0
VELOCITY_HIGH          = 0.70
ACCEL_WINDOW           = 3
ACCEL_NORM_FACTOR      = 3.0
PROXIMITY_RADIUS_PX    = 120
PROXIMITY_HIGH         = 3
ARM_RAISE_ANGLE_THRESH = 30.0
POSE_CONF_MIN          = 0.50
DANGEROUS_CLASS_IDS    = {38, 49}
THIN_ROD_ASPECT_RATIO  = 5.0
THIN_ROD_MAX_WIDTH_PX  = 30

# ── Risk scorer weights ─────────────────────────────────────
WEIGHT_VELOCITY  = 0.35
WEIGHT_ACCEL     = 0.20
WEIGHT_PROXIMITY = 0.15
WEIGHT_POSE      = 0.20
WEIGHT_OBJECT    = 0.10

_assert_sum = WEIGHT_VELOCITY + WEIGHT_ACCEL + WEIGHT_PROXIMITY + WEIGHT_POSE + WEIGHT_OBJECT
assert abs(_assert_sum - 1.0) < 1e-6, (
    f"Risk weights must sum to 1.0, got {_assert_sum}"
)

# ── Risk tier boundaries ────────────────────────────────────
THRESH_LOW    = 0.12
THRESH_MEDIUM = 0.22
THRESH_HIGH   = 0.35

# ── Zone state boundaries ───────────────────────────────────
ZONE_CALM    = 0.12
ZONE_WATCH   = 0.22
ZONE_WARNING = 0.35

# ── Zone score formula weights ──────────────────────────────
ZONE_WEIGHT_MAX  = 0.60
ZONE_WEIGHT_MEAN = 0.40
ZONE_TOP_K       = 3

# ── Display constants ───────────────────────────────────────
HUD_HEIGHT_PX      = 60
HUD_BG_COLOR       = (0, 0, 0)
HUD_BG_ALPHA       = 0.75
HUD_FONT           = cv2.FONT_HERSHEY_DUPLEX
HUD_FONT_SCALE     = 1.4
HUD_FONT_THICKNESS = 2

BBOX_THICKNESS     = 2
LABEL_FONT         = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE   = 0.55
LABEL_FONT_THICK   = 1

TIER_COLORS = {
    "low":      (60,  200, 60),
    "medium":   (0,   210, 255),
    "high":     (0,   140, 255),
    "critical": (0,    40, 220),
}

# ASCII-safe zone display (cv2.putText does not render Unicode)
ZONE_DISPLAY = {
    "CALM":     {"color": (60,  200, 60),  "text": "[O] ZONE CALM"},
    "WATCH":    {"color": (0,   210, 255), "text": "[O] ZONE WATCH"},
    "WARNING":  {"color": (0,   140, 255), "text": "[!] ZONE WARNING"},
    "CRITICAL": {"color": (0,    40, 220), "text": "[X] ZONE CRITICAL"},
}
