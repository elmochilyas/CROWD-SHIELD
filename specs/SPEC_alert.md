# SPEC_alert.md — `pipeline/alert.py`

## Purpose

Aggregates individual risk scores into a single zone-level threat state.
Applies temporal smoothing to avoid flickering between states.
Maintains a rolling history of zone scores for graph output.

---

## Class: `AlertEngine`

### Constructor

```python
def __init__(self) -> None:
```

**Internal state:**
```python
self._score_history: deque  # deque of float, maxlen=30 (rolling window ~1.5s at 20fps)
self._label_history: deque  # deque of str, maxlen=5  (for hysteresis/smoothing)
self._current_label: str    # last committed zone label, starts as "CALM"
self.score_log: list[float] # append-only list of zone scores (for graph)
```

### Method: `classify(person_states, frame_index)`

```python
def classify(
    self,
    person_states: list[PersonState],
    frame_index: int,
) -> ZoneState:
    """
    Parameters:
        person_states: output of BehaviorEngine.update()
        frame_index:   current frame index

    Returns:
        ZoneState with label, zone_score, high_risk_count, frame_index
    """
```

**Step 1 — Handle empty frame**
```
If person_states is empty:
    zone_score = 0.0
    Go to Step 4 directly.
```

**Step 2 — Compute raw zone score**
```python
scores = [p.risk_score for p in person_states]
top_k  = sorted(scores, reverse=True)[:config.ZONE_TOP_K]
top_k_mean = sum(top_k) / len(top_k)
pop_mean   = sum(scores) / len(scores)

zone_score = (
    config.ZONE_WEIGHT_MAX  * top_k_mean +
    config.ZONE_WEIGHT_MEAN * pop_mean
)
zone_score = float(np.clip(zone_score, 0.0, 1.0))
```

**Step 3 — Smooth with rolling average**
```python
self._score_history.append(zone_score)
smoothed_score = sum(self._score_history) / len(self._score_history)
```
Use `smoothed_score` for label classification (avoids 1-frame spikes triggering CRITICAL).

**Step 4 — Map to label (raw thresholds)**
```python
if smoothed_score < config.ZONE_CALM:    raw_label = "CALM"
elif smoothed_score < config.ZONE_WATCH: raw_label = "WATCH"
elif smoothed_score < config.ZONE_WARNING: raw_label = "WARNING"
else:                                    raw_label = "CRITICAL"
```

**Step 5 — Hysteresis (avoid flickering)**
```python
self._label_history.append(raw_label)
# Commit to a new label only if it appears in 3 of the last 5 frames
from collections import Counter
counts = Counter(self._label_history)
committed_label = counts.most_common(1)[0][0]
self._current_label = committed_label
```

**Step 6 — Log and return**
```python
self.score_log.append(smoothed_score)
high_risk_count = sum(1 for p in person_states if p.risk_score >= config.THRESH_HIGH)

return ZoneState(
    label=self._current_label,
    zone_score=smoothed_score,
    high_risk_count=high_risk_count,
    frame_index=frame_index,
)
```

### Method: `get_score_log()`

```python
def get_score_log(self) -> list[float]:
    """Returns a copy of the full score history for graphing."""
    return list(self.score_log)
```

### Method: `reset()`

```python
def reset(self) -> None:
    """Clears all history. Use when switching video sources."""
```

---

## Acceptance criteria

1. An empty frame list must return `ZoneState(label="CALM", zone_score=0.0, high_risk_count=0)`.
2. 3 consecutive frames with 3 persons all at risk_score=0.9 must result in label="CRITICAL".
3. A single spike frame (1 frame CRITICAL surrounded by CALM) must not change the committed label (hysteresis).
4. `zone_score` is always in [0.0, 1.0].
5. `score_log` grows by 1 entry per call to `classify()`.
6. `reset()` clears `score_log`, `_score_history`, and `_label_history`, and resets `_current_label` to "CALM".
