import argparse

import cv2
import numpy as np

import config
from pipeline.frame_reader import FrameReader
from pipeline.detector import Detector
from pipeline.behavior import BehaviorEngine
from pipeline.alert import AlertEngine
from pipeline.renderer import Renderer


def _update_graph(score_log: list[float]) -> None:
    graph_w, graph_h = 640, 120
    graph = np.full((graph_h, graph_w, 3), 30, dtype=np.uint8)

    margin_left = 40
    margin_right = 10
    margin_top = 20
    margin_bottom = 10
    plot_x1 = margin_left
    plot_x2 = graph_w - margin_right
    plot_y1 = margin_top
    plot_y2 = graph_h - margin_bottom
    plot_w = plot_x2 - plot_x1
    plot_h = plot_y2 - plot_y1

    def score_to_y(score: float) -> int:
        return plot_y2 - int(score / 1.0 * plot_h)

    def draw_dashed_line(y: int, color: tuple) -> None:
        dash_len = 8
        gap_len = 4
        x = plot_x1
        while x < plot_x2:
            x2 = min(x + dash_len, plot_x2)
            cv2.line(graph, (x, y), (x2, y), color, 1)
            x = x2 + gap_len

    # Draw grid lines and labels
    thresholds = [
        (config.ZONE_CALM,    config.ZONE_DISPLAY["CALM"]["color"]),
        (config.ZONE_WATCH,   config.ZONE_DISPLAY["WATCH"]["color"]),
        (config.ZONE_WARNING, config.ZONE_DISPLAY["WARNING"]["color"]),
        (1.0, (60, 60, 60)),
    ]
    for val, color in thresholds:
        y = score_to_y(val)
        if val < 1.0:
            draw_dashed_line(y, color)
        else:
            cv2.line(graph, (plot_x1, y), (plot_x2, y), color, 1)
        label = f"{val:.2f}"
        cv2.putText(graph, label, (2, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw score polyline
    if len(score_log) >= 2:
        pts = []
        n = len(score_log)
        for i, score in enumerate(score_log):
            if i > 300:
                offset = i - 300
                break
            x = plot_x1 + int(i / max(n - 1, 1) * plot_w)
            y = score_to_y(score)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(graph, pts[i - 1], pts[i], (200, 200, 200), 1)

    # Draw current score circle
    if score_log:
        last_score = score_log[-1]
        last_x = plot_x1 + plot_w
        last_y = score_to_y(last_score)
        if last_score < config.ZONE_CALM:
            tier_color = (60, 200, 60)
        elif last_score < config.ZONE_WATCH:
            tier_color = (0, 210, 255)
        elif last_score < config.ZONE_WARNING:
            tier_color = (0, 140, 255)
        else:
            tier_color = (0, 40, 220)
        cv2.circle(graph, (last_x, last_y), 5, tier_color, -1)

    # Title
    cv2.putText(graph, "Zone threat score", (plot_x1, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("Zone Score", graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrowdGuard — Early crowd riot detection for stadium zones"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Video file path or RTSP URL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video file path (.mp4)")
    parser.add_argument("--fps", type=int, default=config.TARGET_FPS,
                        help=f"Target processing fps (default: {config.TARGET_FPS})")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help='Torch device: "cuda:0" or "cpu" (default: cuda:0)')
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress cv2.imshow (headless mode)")
    parser.add_argument("--graph", action="store_true",
                        help="Show live zone score graph in a second window")
    parser.add_argument("--det-model", type=str, default=None,
                        help="Override detection model path")
    parser.add_argument("--pose-model", type=str, default=None,
                        help="Override pose model path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        config.DEVICE = args.device
    if args.det_model:
        config.DETECTION_MODEL = args.det_model
    if args.pose_model:
        config.POSE_MODEL = args.pose_model

    source: str | int = args.source
    if source.isdigit():
        source = int(source)

    reader = FrameReader(source, target_fps=args.fps)
    detector = Detector()
    behavior = BehaviorEngine()
    alert = AlertEngine()
    renderer = Renderer()

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.output, fourcc, args.fps,
            (reader.frame_width, reader.frame_height),
        )

    print(f"[CrowdGuard] Starting on: {args.source}")
    print(f"[CrowdGuard] Device: {config.DEVICE}")

    try:
        with reader:
            while True:
                ok, frame, idx = reader.read()
                if not ok:
                    break

                detections = detector.run(frame, idx)
                person_states = behavior.update(
                    detections, idx, reader.frame_height,
                )
                zone_state = alert.classify(person_states, idx)
                output_frame = renderer.draw(
                    frame,
                    person_states,
                    zone_state,
                    detector.last_objects,
                )

                if not args.no_display:
                    display_frame = cv2.resize(output_frame, (960, 540))
                    cv2.imshow("CrowdGuard", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if writer is not None:
                    writer.write(output_frame)

                if args.graph:
                    _update_graph(alert.get_score_log())

                if idx % 30 == 0:
                    print(
                        f"[frame {idx:05d}]  zone={zone_state.label:<8}  "
                        f"score={zone_state.zone_score:.3f}  "
                        f"risk_persons={zone_state.high_risk_count}"
                    )
    except KeyboardInterrupt:
        print("\n[CrowdGuard] Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[CrowdGuard] Done.")


if __name__ == "__main__":
    main()
