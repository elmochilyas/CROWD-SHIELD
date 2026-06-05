# CROWD SHIELD 🏟️
> Early riot and crowd disturbance detection for stadium zones
> Built for the Football Tech Morocco Hackathon

## What it does
CrowdGuard analyzes a video feed from a single stadium stand zone and detects
abnormal crowd behavior in real time — before a situation escalates.

**Output:**
- Colored bounding boxes per person (green → yellow → orange → red by risk level)
- Risk score label on each person: `ID:N  0.XX`
- HUD band at top of frame: `CALM / WATCH / WARNING / CRITICAL`

## How it works
Each tracked person is scored across 5 behavioral signals:
| Signal | Description |
|---|---|
| Velocity | Abnormal movement speed |
| Acceleration | Sudden speed changes |
| Proximity | Crowd clustering density |
| Pose anomaly | Arm raise detection via keypoints |
| Object flag | Detection of dangerous objects (bat, rod, knife) |

Scores are aggregated into a zone-level threat state with temporal smoothing
to avoid false alarms.

## Tech stack
- Python 3.11
- YOLOv8 (Ultralytics) — person detection + pose estimation
- ByteTrack — multi-object tracking
- OpenCV — video I/O and overlay rendering
- PyTorch — inference backend

## Installation
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```
YOLOv8 weights (~6MB) are downloaded automatically on first run.

## Usage
```bash
# Display only
python main.py --source videos/your_video.mp4 --device cpu

# Save annotated output
python main.py --source videos/your_video.mp4 --output videos/result.mp4 --device cpu

# GPU (if available)
python main.py --source videos/your_video.mp4 --device cuda:0

# Headless (no display window)
python main.py --source videos/your_video.mp4 --output videos/result.mp4 --device cpu --no-display
```

## Project structure
```
crowdguard/
├── main.py              # Entry point
├── config.py            # All thresholds and constants
├── requirements.txt     # Dependencies
├── pipeline/
│   ├── __init__.py      # Shared data types
│   ├── frame_reader.py  # Video capture loop
│   ├── detector.py      # YOLOv8 + ByteTrack
│   ├── behavior.py      # Signal extraction + risk scoring
│   ├── alert.py         # Zone state classifier
│   └── renderer.py      # Overlay drawing
├── specs/               # Technical specification files
└── .gitignore
```

## Team
Built at the Football Tech Morocco Hackathon 🇲🇦
