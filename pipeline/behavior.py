from collections import deque, defaultdict
from dataclasses import dataclass, field
import math

import numpy as np

import config
from pipeline import BBox, Detection, SignalVector, PersonState


@dataclass
class TrackRecord:
    track_id: int
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=config.TRACK_HISTORY_LEN))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=config.ACCEL_WINDOW))
    last_seen_frame: int = -1


class BehaviorEngine:
    def __init__(self) -> None:
        self._tracks: dict[int, TrackRecord] = {}

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame_height: int,
    ) -> list[PersonState]:
        if not detections:
            return []

        # Step 1 — Update centroid history
        for det in detections:
            tid = det.track_id
            if tid not in self._tracks:
                self._tracks[tid] = TrackRecord(track_id=tid)
            record = self._tracks[tid]
            record.centroid_history.append((det.bbox.cx, det.bbox.cy))
            record.last_seen_frame = frame_index

        # Step 2 — Purge stale tracks
        stale_ids = [
            tid for tid, rec in self._tracks.items()
            if frame_index - rec.last_seen_frame > 30
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        # Step 3-6 — Compute signals, risk score, tier, and build PersonState list
        person_states: list[PersonState] = []
        for det in detections:
            velocity = self._compute_velocity(det.track_id, frame_height)
            acceleration = self._compute_acceleration(det.track_id, frame_height)
            proximity = self._compute_proximity(det, detections)
            pose = self._compute_pose_anomaly(det.keypoints)
            obj_flag, obj_type = self._compute_object_flag(det)

            signals = SignalVector(
                velocity=velocity,
                acceleration=acceleration,
                proximity_count=proximity,
                pose_anomaly=pose,
                object_flag=obj_flag,
                object_type=obj_type,
            )

            risk_score = (
                config.WEIGHT_VELOCITY * signals.velocity
                + config.WEIGHT_ACCEL * signals.acceleration
                + config.WEIGHT_PROXIMITY * min(signals.proximity_count / config.PROXIMITY_HIGH, 1.0)
                + config.WEIGHT_POSE * signals.pose_anomaly
                + config.WEIGHT_OBJECT * (1.0 if signals.object_flag else 0.0)
            )
            risk_score = float(np.clip(risk_score, 0.0, 1.0))

            if risk_score < config.THRESH_LOW:
                tier = "low"
            elif risk_score < config.THRESH_MEDIUM:
                tier = "medium"
            elif risk_score < config.THRESH_HIGH:
                tier = "high"
            else:
                tier = "critical"

            person_states.append(PersonState(
                track_id=det.track_id,
                bbox=det.bbox,
                risk_score=risk_score,
                risk_tier=tier,
                signals=signals,
            ))

        return person_states

    def _compute_velocity(self, track_id: int, frame_height: int) -> float:
        record = self._tracks.get(track_id)
        if record is None:
            return 0.0
        history = record.centroid_history
        if len(history) < 2:
            return 0.0
        steps = min(3, len(history) - 1)
        dx = history[-1][0] - history[-(steps + 1)][0]
        dy = history[-1][1] - history[-(steps + 1)][1]
        raw_speed = math.sqrt(dx * dx + dy * dy) / steps
        normalized = raw_speed / config.VELOCITY_NORM_FACTOR
        return float(min(normalized, 1.0))

    def _compute_acceleration(self, track_id: int, frame_height: int) -> float:
        record = self._tracks.get(track_id)
        if record is None:
            return 0.0

        current_velocity = self._compute_velocity(track_id, frame_height)
        record.velocity_history.append(current_velocity)

        if len(record.velocity_history) < 2:
            return 0.0
        accel = abs(record.velocity_history[-1] - record.velocity_history[-2])
        normalized = accel / (config.ACCEL_NORM_FACTOR / 100)
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _compute_proximity(detection: Detection, all_detections: list[Detection]) -> int:
        count = 0
        for other in all_detections:
            if other.track_id == detection.track_id:
                continue
            dist = math.hypot(
                detection.bbox.cx - other.bbox.cx,
                detection.bbox.cy - other.bbox.cy,
            )
            if dist <= config.PROXIMITY_RADIUS_PX:
                count += 1
        return count

    @staticmethod
    def _compute_pose_anomaly(keypoints: np.ndarray | None) -> float:
        if keypoints is None:
            return 0.0
        raised = 0
        pairs = [
            (9, 5),   # left wrist, left shoulder
            (10, 6),  # right wrist, right shoulder
        ]
        for wrist_idx, shoulder_idx in pairs:
            wrist = keypoints[wrist_idx]
            shoulder = keypoints[shoulder_idx]
            if wrist[2] >= config.POSE_CONF_MIN and shoulder[2] >= config.POSE_CONF_MIN:
                if wrist[1] < (shoulder[1] - config.ARM_RAISE_ANGLE_THRESH):
                    raised += 1
        if raised == 0:
            return 0.0
        elif raised == 1:
            return 0.5
        else:
            return 1.0

    @staticmethod
    def _compute_object_flag(detection: Detection) -> tuple[bool, str]:
        if detection.signals_hint_object:
            return True, detection.signals_hint_object_type
        return False, ""
