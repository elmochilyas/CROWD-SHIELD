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
    class_name: str
    keypoints: Optional[np.ndarray] = None
    signals_hint_object: bool = False
    signals_hint_object_type: str = ""


@dataclass
class SignalVector:
    velocity: float
    acceleration: float
    proximity_count: int
    pose_anomaly: float
    object_flag: bool
    object_type: str


@dataclass
class PersonState:
    track_id: int
    bbox: BBox
    risk_score: float
    risk_tier: str
    signals: SignalVector


@dataclass
class ZoneState:
    label: str
    zone_score: float
    high_risk_count: int
    frame_index: int
