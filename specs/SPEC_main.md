# SPEC_main.md — `main.py`

## Purpose

Entry point. Wires all pipeline components together.
Handles CLI args, the main processing loop, output (display / write), and optional score graph.
Must be runnable with a single command: `python main.py --source video.mp4`

---

## CLI arguments (use `argparse`)

| Argument        | Type   | Default          | Description                                      |
|-----------------|--------|------------------|--------------------------------------------------|
| `--source`      | str    | required         | Video file path or RTSP URL                      |
| `--output`      | str    | `None`           | Output video file path (.mp4). If not set, display only. |
| `--fps`         | int    | `20`             | Target processing fps                            |
| `--device`      | str    | `"cuda:0"`       | Torch device ("cuda:0" or "cpu")                 |
| `--no-display`  | flag   | False            | Suppress cv2.imshow (useful for headless mode)   |
| `--graph`       | flag   | False            | Show live zone score graph in a second window    |
| `--det-model`   | str    | from config      | Override detection model path                    |
| `--pose-model`  | str    | from config      | Override pose model path                         |

---

## Main loop pseudocode

```python
def main():
    args = parse_args()

    # Override config if CLI args provided
    if args.device:    config.DEVICE = args.device
    if args.det_model: config.DETECTION_MODEL = args.det_model
    if args.pose_model: config.POSE_MODEL = args.pose_model

    # Initialize components
    reader   = FrameReader(args.source, target_fps=args.fps)
    detector = Detector()
    behavior = BehaviorEngine()
    alert    = AlertEngine()
    renderer = Renderer()

    # Output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, args.fps,
            (reader.frame_width, reader.frame_height)
        )

    print(f"[CrowdGuard] Starting on: {args.source}")
    print(f"[CrowdGuard] Device: {config.DEVICE}")

    try:
        with reader:
            while True:
                ok, frame, idx = reader.read()
                if not ok:
                    break

                # Core pipeline
                detections    = detector.run(frame, idx)
                person_states = behavior.update(detections, idx, reader.frame_height)
                zone_state    = alert.classify(person_states, idx)
                output_frame  = renderer.draw(
                    frame,
                    person_states,
                    zone_state,
                    detector.last_objects,
                )

                # Display
                if not args.no_display:
                    cv2.imshow("CrowdGuard", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Write
                if writer:
                    writer.write(output_frame)

                # Optional graph update
                if args.graph:
                    _update_graph(alert.get_score_log())

                # Console log every 30 frames
                if idx % 30 == 0:
                    print(
                        f"[frame {idx:05d}]  zone={zone_state.label:<8}  "
                        f"score={zone_state.zone_score:.3f}  "
                        f"risk_persons={zone_state.high_risk_count}"
                    )

    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[CrowdGuard] Done.")
```

---

## Optional: `_update_graph(score_log)`

```python
def _update_graph(score_log: list[float]) -> None:
    """
    Renders a live zone score timeline using OpenCV drawing (not matplotlib).
    Draws into a separate 640×120 window named "Zone Score".

    X axis: last N scores (N = len(score_log), max 300 displayed)
    Y axis: 0.0 to 1.0

    Draw:
      - Background: dark gray (30, 30, 30)
      - Grid lines at y=0.25 (green), y=0.45 (yellow), y=0.65 (orange), y=1.0 (red)
        with dashed horizontal lines and threshold labels on left edge
      - Score line: white polyline connecting score points
      - Current score: filled circle at the last point, color = zone tier color
      - Title text: "Zone threat score" top-left

    Must not block the main loop — use cv2.waitKey(1) only.
    """
```

Using pure OpenCV for the graph (no matplotlib) avoids threading issues in the main loop.

---

## Acceptance criteria

1. `python main.py --source test.mp4 --no-display` processes the full video and exits cleanly.
2. `python main.py --source test.mp4 --output out.mp4` writes an annotated video to `out.mp4`.
3. Pressing `q` during display exits cleanly without hanging.
4. With `--device cpu`, the pipeline runs on CPU (slower but functional).
5. Console log prints zone state every 30 frames.
6. No frame is dropped silently — frame_index in the log must be monotonically increasing.
7. KeyboardInterrupt (Ctrl+C) is handled gracefully: writer is released before exit.

---

## Example usage

```bash
# Run on a video file, display only
python main.py --source stadium_zone_a.mp4

# Run and save output
python main.py --source stadium_zone_a.mp4 --output result.mp4

# CPU mode with graph
python main.py --source stadium_zone_a.mp4 --device cpu --graph

# Headless (server / pipeline mode)
python main.py --source rtsp://192.168.1.10/stream1 --no-display --output output.mp4
```
