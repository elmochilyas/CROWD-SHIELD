# SPEC_renderer.md — `pipeline/renderer.py`

## Purpose

All drawing happens here. Takes a raw frame + analysis results → returns an annotated frame.
No logic, no scoring, no state. Pure visual output.

---

## Class: `Renderer`

### Constructor

```python
def __init__(self) -> None:
```

No state. All methods are stateless transforms.

---

### Method: `draw(frame, person_states, zone_state, danger_objects)`

```python
def draw(
    self,
    frame: np.ndarray,
    person_states: list[PersonState],
    zone_state: ZoneState,
    danger_objects: list[Detection],
) -> np.ndarray:
    """
    Returns a new annotated frame (copy of input, not in-place).

    Drawing order (bottom to top in z-stack):
        1. Person bounding boxes + labels
        2. Danger object bounding boxes
        3. HUD bar (top of frame, drawn last so always on top)
    """
```

---

## Drawing spec — person bounding boxes

For each `PersonState` in `person_states`:

**Box:**
```
color  = config.TIER_COLORS[person.risk_tier]
cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, config.BBOX_THICKNESS)
```

**Label background pill:**
- Position: just above the top-left corner of the bbox (y = bbox.y1 - 22)
- Text: `f"ID:{person.track_id}  {person.risk_score:.2f}"`
- Draw a filled rounded rectangle behind the text (same color as box, alpha 0.85)
- Text color: white `(255, 255, 255)`
- If the label would go above y=0, reposition it inside the top of the bbox instead

**Object hint indicator:**
- If `person.signals.object_flag is True`:
  - Draw a small warning icon to the right of the label: `"⚠"` character or a filled orange triangle (3 points)
  - Or add `" [OBJ]"` suffix to the label text if Unicode rendering is unreliable

---

## Drawing spec — danger object boxes

For each Detection in `danger_objects`:
```
color = (0, 0, 255)  # bright red in BGR
cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
label = detection.class_name.upper()
# draw label above box in same style as person label
```

---

## Drawing spec — HUD bar

The HUD is a semi-transparent band at the very top of the frame.

**Structure:**
```
┌──────────────────────────────────────────────────────────────┐  ← y=0
│  ● ZONE CALM          Score: 0.18     Active risk: 2    🕐 00:32  │  ← HUD content
└──────────────────────────────────────────────────────────────┘  ← y=HUD_HEIGHT_PX
```

**Implementation:**
```python
# 1. Create overlay copy
overlay = frame.copy()

# 2. Draw filled rectangle for HUD background
cv2.rectangle(overlay, (0, 0), (frame.shape[1], config.HUD_HEIGHT_PX),
              config.HUD_BG_COLOR, -1)

# 3. Blend with original
cv2.addWeighted(overlay, config.HUD_BG_ALPHA, frame,
                1 - config.HUD_BG_ALPHA, 0, frame)

# 4. Draw zone state text (centered horizontally)
zone_cfg = config.ZONE_DISPLAY[zone_state.label]
text     = zone_cfg["text"]
color    = zone_cfg["color"]
# compute text size, center it
text_x, text_y = _compute_centered_text_pos(frame.shape[1], config.HUD_HEIGHT_PX, text, ...)
cv2.putText(frame, text, (text_x, text_y), config.HUD_FONT,
            config.HUD_FONT_SCALE, color, config.HUD_FONT_THICKNESS, cv2.LINE_AA)

# 5. Draw secondary info (right-aligned)
info = f"Score: {zone_state.zone_score:.2f}   Risk persons: {zone_state.high_risk_count}"
cv2.putText(frame, info, (right_x, text_y), config.LABEL_FONT,
            0.7, (200, 200, 200), 1, cv2.LINE_AA)
```

**Zone state color rules:**
| State    | Text color (BGR)  | Example display          |
|----------|-------------------|--------------------------|
| CALM     | (60, 200, 60)     | `● ZONE CALM`            |
| WATCH    | (0, 210, 255)     | `◉ ZONE WATCH`           |
| WARNING  | (0, 140, 255)     | `▲ ZONE WARNING`         |
| CRITICAL | (0, 40, 220)      | `⬛ ZONE CRITICAL`        |

---

## Private helper: `_draw_filled_rounded_rect`

```python
def _draw_filled_rounded_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple, alpha: float = 0.85, radius: int = 6,
) -> None:
```

Draws a rounded rectangle with alpha blending using the overlay/addWeighted pattern.
Used for label backgrounds.

---

## Acceptance criteria

1. Output frame has identical shape and dtype as input frame.
2. HUD bar always occupies y=0 to y=HUD_HEIGHT_PX across full frame width.
3. A `PersonState` with tier="critical" has a red bbox.
4. A `PersonState` with tier="low" has a green bbox.
5. A person with `signals.object_flag=True` has a visible indicator on their label.
6. A frame with `danger_objects=[...]` shows red boxes around those objects.
7. Calling `draw()` with empty `person_states` and `danger_objects` returns a valid frame with only the HUD drawn.
8. The method never modifies the input frame in-place — always operates on a copy.
