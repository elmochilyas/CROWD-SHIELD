from collections import deque, Counter

import numpy as np

import config
from pipeline import PersonState, ZoneState


class AlertEngine:
    def __init__(self) -> None:
        self._score_history: deque = deque(maxlen=6)
        self._label_history: deque = deque(maxlen=3)
        self._current_label: str = "CALM"
        self.score_log: list[float] = []

    def classify(
        self,
        person_states: list[PersonState],
        frame_index: int,
    ) -> ZoneState:
        # Step 1 — Handle empty frame
        if not person_states:
            zone_score = 0.0
            smoothed_score = zone_score
            raw_label = "CALM"
        else:
            # Step 2 — Compute raw zone score
            scores = [p.risk_score for p in person_states]
            top_k = sorted(scores, reverse=True)[:config.ZONE_TOP_K]
            top_k_mean = sum(top_k) / len(top_k)
            pop_mean = sum(scores) / len(scores)

            zone_score = (
                config.ZONE_WEIGHT_MAX * top_k_mean
                + config.ZONE_WEIGHT_MEAN * pop_mean
            )
            zone_score = float(np.clip(zone_score, 0.0, 1.0))

            # Step 3 — Smooth with rolling average
            self._score_history.append(zone_score)
            smoothed_score = sum(self._score_history) / len(self._score_history)

            # Step 4 — Map to label
            if smoothed_score < config.ZONE_CALM:
                raw_label = "CALM"
            elif smoothed_score < config.ZONE_WATCH:
                raw_label = "WATCH"
            elif smoothed_score < config.ZONE_WARNING:
                raw_label = "WARNING"
            else:
                raw_label = "CRITICAL"

        # Step 5 — Hysteresis
        self._label_history.append(raw_label)
        counts = Counter(self._label_history)
        top_label, top_count = counts.most_common(1)[0]
        if top_count >= 3:
            self._current_label = top_label

        # Step 6 — Log and return
        self.score_log.append(smoothed_score)
        high_risk_count = sum(
            1 for p in person_states if p.risk_score >= config.THRESH_HIGH
        )

        return ZoneState(
            label=self._current_label,
            zone_score=smoothed_score,
            high_risk_count=high_risk_count,
            frame_index=frame_index,
        )

    def get_score_log(self) -> list[float]:
        return list(self.score_log)

    def reset(self) -> None:
        self._score_history.clear()
        self._label_history.clear()
        self._current_label = "CALM"
        self.score_log.clear()
