import cv2
import numpy as np

import config
from pipeline import Detection, PersonState, ZoneState


class Renderer:
    def __init__(self) -> None:
        pass

    def draw(
        self,
        frame: np.ndarray,
        person_states: list[PersonState],
        zone_state: ZoneState,
        danger_objects: list[Detection],
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Layer 1 — Person bounding boxes + labels
        for p in person_states:
            color = config.TIER_COLORS[p.risk_tier]
            bbox = p.bbox
            cv2.rectangle(out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                          color, config.BBOX_THICKNESS)

            label_text = f"ID:{p.track_id}  {p.risk_score:.2f}"
            if p.signals.object_flag:
                label_text += " [OBJ]"

            (label_w, label_h), baseline = cv2.getTextSize(
                label_text, config.LABEL_FONT, config.LABEL_FONT_SCALE,
                config.LABEL_FONT_THICK
            )
            label_y1 = bbox.y1 - label_h - 6
            label_x1 = bbox.x1
            if label_y1 < 0:
                label_y1 = bbox.y1 + 2
            label_x2 = label_x1 + label_w + 8
            label_y2 = label_y1 + label_h + 6

            self._draw_filled_rounded_rect(
                out, label_x1, label_y1, label_x2, label_y2,
                color
            )
            cv2.putText(
                out, label_text,
                (label_x1 + 4, label_y1 + label_h + 2),
                config.LABEL_FONT, config.LABEL_FONT_SCALE,
                (255, 255, 255), config.LABEL_FONT_THICK, cv2.LINE_AA,
            )

        # Layer 2 — Danger object boxes
        for obj in danger_objects:
            color = (0, 0, 255)
            bbox = obj.bbox
            cv2.rectangle(out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                          color, 2)
            label_text = obj.class_name.upper()
            (label_w, label_h), baseline = cv2.getTextSize(
                label_text, config.LABEL_FONT, config.LABEL_FONT_SCALE,
                config.LABEL_FONT_THICK
            )
            label_y1 = bbox.y1 - label_h - 6
            label_x1 = bbox.x1
            if label_y1 < 0:
                label_y1 = bbox.y1 + 2
            label_x2 = label_x1 + label_w + 8
            label_y2 = label_y1 + label_h + 6

            self._draw_filled_rounded_rect(
                out, label_x1, label_y1, label_x2, label_y2,
                color
            )
            cv2.putText(
                out, label_text,
                (label_x1 + 4, label_y1 + label_h + 2),
                config.LABEL_FONT, config.LABEL_FONT_SCALE,
                (255, 255, 255), config.LABEL_FONT_THICK, cv2.LINE_AA,
            )

        # Layer 3 — HUD bar (always on top)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, config.HUD_HEIGHT_PX),
                      config.HUD_BG_COLOR, -1)
        cv2.addWeighted(overlay, config.HUD_BG_ALPHA, out,
                        1 - config.HUD_BG_ALPHA, 0, out)

        zone_cfg = config.ZONE_DISPLAY[zone_state.label]
        zone_text = zone_cfg["text"]
        zone_color = zone_cfg["color"]

        (tw, th), _ = cv2.getTextSize(zone_text, config.HUD_FONT,
                                       config.HUD_FONT_SCALE,
                                       config.HUD_FONT_THICKNESS)
        text_x = (w - tw) // 2
        text_y = (config.HUD_HEIGHT_PX + th) // 2
        cv2.putText(out, zone_text, (text_x, text_y),
                    config.HUD_FONT, config.HUD_FONT_SCALE,
                    zone_color, config.HUD_FONT_THICKNESS, cv2.LINE_AA)

        info_text = (
            f"Score: {zone_state.zone_score:.2f}   "
            f"Risk persons: {zone_state.high_risk_count}"
        )
        (iw, ih), _ = cv2.getTextSize(info_text, config.LABEL_FONT,
                                       0.7, 1)
        info_x = w - iw - 16
        info_y = text_y
        cv2.putText(out, info_text, (info_x, info_y),
                    config.LABEL_FONT, 0.7,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return out

    @staticmethod
    def _draw_filled_rounded_rect(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple, alpha: float = 0.85, radius: int = 6,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        if radius > 0:
            cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
            cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
