# CrowdGuard — Master Specification
> Early riot & quarrel detection in stadium zones using computer vision
> Hackathon: Football Tech Morocco
> Stack: Python 3.11 · YOLOv8 (Ultralytics) · ByteTrack · OpenCV · NumPy · PyTorch

---

## Project overview

CrowdGuard analyzes a live video feed (or recorded video) from a single stadium stand zone.
It detects abnormal crowd behavior and dangerous objects **early**, before a situation escalates.

**Input:** MP4 / RTSP stream of one stadium zone (one camera angle = one zone)
**Output:** Annotated video with:
- Per-person colored bounding boxes (green → yellow → orange → red by risk level)
- Risk score label on each box: `ID:42 · 0.78`
- Top-of-screen HUD band showing zone state: `CALM / WATCH / WARNING / CRITICAL`
- Real-time zone score graph (secondary window or embedded)

---

## Repository structure

```
crowdguard/
├── SPEC.md                  ← you are here
├── main.py                  ← entry point
├── config.py                ← all thresholds and constants
├── pipeline/
│   ├── __init__.py
│   ├── frame_reader.py      ← video capture loop
│   ├── detector.py          ← YOLOv8 + ByteTrack wrapper
│   ├── behavior.py          ← signal extraction + per-person risk scorer
│   ├── alert.py             ← zone state classifier
│   └── renderer.py          ← OpenCV overlay drawing
└── specs/
    ├── SPEC.md              ← this file
    ├── SPEC_frame_reader.md
    ├── SPEC_detector.md
    ├── SPEC_behavior.md
    ├── SPEC_alert.md
    ├── SPEC_renderer.md
    ├── SPEC_config.md
    └── SPEC_main.md
```

---

## Data flow (per frame)

```
Video source
    │
    ▼
FrameReader.read() → np.ndarray (BGR, H×W×3)
    │
    ▼
Detector.run(frame) → List[Detection]
    │  each Detection: { track_id, bbox, confidence, class_name, keypoints }
    ▼
BehaviorEngine.update(detections, frame_index) → List[PersonState]
    │  each PersonState: { track_id, bbox, risk_score, signals }
    ▼
AlertEngine.classify(person_states) → ZoneState
    │  ZoneState: { label: "CALM"|"WATCH"|"WARNING"|"CRITICAL", zone_score: float }
    ▼
Renderer.draw(frame, person_states, zone_state) → np.ndarray (annotated frame)
    │
    ▼
Output (cv2.imshow / cv2.VideoWriter / RTSP push)
```

---

## Shared data types (define in `pipeline/__init__.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.height / max(self.width, 1)


@dataclass
class Detection:
    track_id: int
    bbox: BBox
    confidence: float
    class_name: str                        # "person", "baseball bat", "knife", etc.
    keypoints: Optional[np.ndarray]        # shape (17, 3) — x, y, conf per keypoint. None if pose not run.


@dataclass
class SignalVector:
    velocity: float          # 0–1 normalized
    acceleration: float      # 0–1 normalized
    proximity_count: int     # raw count of neighbors within radius
    pose_anomaly: float      # 0–1 arm-raise score
    object_flag: bool        # True if dangerous object detected in/near bbox
    object_type: str         # e.g. "baseball bat", "knife", "thin_rod", "" if none


@dataclass
class PersonState:
    track_id: int
    bbox: BBox
    risk_score: float           # 0.0 – 1.0
    risk_tier: str              # "low" | "medium" | "high" | "critical"
    signals: SignalVector


@dataclass
class ZoneState:
    label: str              # "CALM" | "WATCH" | "WARNING" | "CRITICAL"
    zone_score: float       # 0.0 – 1.0
    high_risk_count: int    # number of persons with risk_score >= THRESH_HIGH_RISK
    frame_index: int
```

---

## Dependencies

```
# requirements.txt
ultralytics>=8.2.0
opencv-python>=4.9.0
numpy>=1.26.0
torch>=2.2.0
torchvision>=0.17.0
```

Install: `pip install -r requirements.txt`

Model weights are downloaded automatically by Ultralytics on first run.
Use `yolov8m.pt` (detection) and `yolov8m-pose.pt` (pose) as defaults.
For CPU-only: switch to `yolov8n.pt` and `yolov8n-pose.pt`.

---

## Performance targets

| Metric | Target |
|---|---|
| Inference fps (GPU) | ≥ 20 fps on 640×640 input |
| Inference fps (CPU only) | ≥ 10 fps using nano weights |
| Detection precision | Pretrained COCO — no fine-tuning needed for MVP |
| Max persons tracked simultaneously | 50 per zone frame |
| Latency (frame in → annotated frame out) | < 100ms on GPU |

---

## Acceptance criteria (system-level)

1. Given a video with a person running and raising both arms, their risk score must be ≥ 0.60.
2. Given a video with a person holding a baseball-bat-shaped object, `object_flag` must be `True`.
3. Given a calm crowd video (people sitting), zone state must be `CALM` or `WATCH`.
4. The HUD band must be visible on the top of every output frame.
5. Each tracked person must retain the same `track_id` across consecutive frames (no ID flicker).
6. Pipeline must not crash on frames with zero detections.
