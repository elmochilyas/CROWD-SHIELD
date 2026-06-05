import numpy as np
from ultralytics import YOLO

import config
from pipeline import BBox, Detection


def compute_iou(a: BBox, b: BBox) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter_area / float(area_a + area_b - inter_area)


class Detector:
    def __init__(
        self,
        det_model_path: str = config.DETECTION_MODEL,
        pose_model_path: str = config.POSE_MODEL,
        device: str = config.DEVICE,
    ) -> None:
        self._device = device
        self._det = YOLO(det_model_path)
        self._det.eval()
        self._pose = YOLO(pose_model_path)
        self._pose.eval()
        self._last_objects: list[Detection] = []
        print(f"[Detector] Detection model loaded: {det_model_path}")
        print(f"[Detector] Pose model loaded: {pose_model_path}")

    @property
    def last_objects(self) -> list[Detection]:
        return self._last_objects

    def run(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        # Step 1 — Object detection with tracking
        det_results = self._det.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INPUT_SIZE,
            verbose=False,
            device=self._device,
        )
        boxes_data = det_results[0].boxes
        if boxes_data is None:
            self._last_objects = []
            return []

        xyxy = boxes_data.xyxy
        confs = boxes_data.conf
        cls_ids = boxes_data.cls
        track_ids = boxes_data.id

        person_detections: list[Detection] = []
        raw_person_boxes: list[tuple] = []
        object_detections: list[Detection] = []
        raw_object_boxes: list[tuple] = []

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
            conf = float(confs[i])
            cls_id = int(cls_ids[i])
            bbox = BBox(x1_i, y1_i, x2_i, y2_i)
            track_id: int | None = None
            if track_ids is not None:
                track_id = int(track_ids[i].item())

            if cls_id == 0:
                if track_id is None or track_id < 0:
                    track_id = - (i + 1)
                det = Detection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_name="person",
                )
                person_detections.append(det)
                raw_person_boxes.append((x1_i, y1_i, x2_i, y2_i))
            elif cls_id in config.DANGEROUS_CLASS_IDS:
                class_name = self._map_class_id(cls_id)
                det = Detection(
                    track_id=- (i + 1000),
                    bbox=bbox,
                    confidence=conf,
                    class_name=class_name,
                )
                object_detections.append(det)
                raw_object_boxes.append((x1_i, y1_i, x2_i, y2_i, class_name))

            # Step 2 — Thin-rod heuristic (check all boxes)
            if bbox.aspect_ratio >= config.THIN_ROD_ASPECT_RATIO and bbox.width <= config.THIN_ROD_MAX_WIDTH_PX:
                existing = False
                for obj_det in object_detections:
                    if obj_det.bbox == bbox:
                        obj_det.class_name = "thin_rod"
                        existing = True
                        break
                if not existing:
                    det = Detection(
                        track_id=- (i + 1000),
                        bbox=bbox,
                        confidence=conf,
                        class_name="thin_rod",
                    )
                    object_detections.append(det)
                    raw_object_boxes.append((x1_i, y1_i, x2_i, y2_i, "thin_rod"))

        # Step 2b — Fallback inference if tracking detected no persons
        if not person_detections:
            fallback = self._det(
                frame,
                conf=0.15,
                iou=0.30,
                imgsz=config.INPUT_SIZE,
                device=self._device,
                verbose=False,
            )
            fb_boxes = fallback[0].boxes
            if fb_boxes is not None and len(fb_boxes) > 0:
                for i in range(len(fb_boxes)):
                    x1, y1, x2, y2 = fb_boxes.xyxy[i].tolist()
                    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                    conf = float(fb_boxes.conf[i])
                    cls_id = int(fb_boxes.cls[i])
                    bbox = BBox(x1_i, y1_i, x2_i, y2_i)
                    if cls_id == 0:
                        det = Detection(
                            track_id=-1,
                            bbox=bbox,
                            confidence=conf,
                            class_name="person",
                        )
                        person_detections.append(det)
                        raw_person_boxes.append((x1_i, y1_i, x2_i, y2_i))
                    elif cls_id in config.DANGEROUS_CLASS_IDS:
                        class_name = self._map_class_id(cls_id)
                        det = Detection(
                            track_id=-(i + 2000),
                            bbox=bbox,
                            confidence=conf,
                            class_name=class_name,
                        )
                        object_detections.append(det)
                        raw_object_boxes.append((x1_i, y1_i, x2_i, y2_i, class_name))
                    if bbox.aspect_ratio >= config.THIN_ROD_ASPECT_RATIO and bbox.width <= config.THIN_ROD_MAX_WIDTH_PX:
                        existing = False
                        for obj_det in object_detections:
                            if obj_det.bbox == bbox:
                                obj_det.class_name = "thin_rod"
                                existing = True
                                break
                        if not existing:
                            det = Detection(
                                track_id=-(i + 2000),
                                bbox=bbox,
                                confidence=conf,
                                class_name="thin_rod",
                            )
                            object_detections.append(det)
                            raw_object_boxes.append((x1_i, y1_i, x2_i, y2_i, "thin_rod"))

        # Step 3 — Pose estimation (persons only)
        if person_detections:
            pose_results = self._pose(
                frame,
                conf=config.CONF_THRESHOLD,
                imgsz=config.INPUT_SIZE,
                verbose=False,
                device=self._device,
            )
            pose_boxes = pose_results[0].boxes
            if pose_boxes is not None:
                pose_keypoints = pose_results[0].keypoints
                if pose_keypoints is not None and pose_keypoints.data is not None:
                    kp_data = pose_keypoints.data
                    for j in range(len(person_detections)):
                        person_det = person_detections[j]
                        p_box = raw_person_boxes[j]
                        best_iou = 0.50
                        best_kp = None
                        for k in range(len(kp_data)):
                            pk_box = pose_boxes.xyxy[k].tolist()
                            pk_bbox = BBox(int(pk_box[0]), int(pk_box[1]), int(pk_box[2]), int(pk_box[3]))
                            iou_val = compute_iou(person_det.bbox, pk_bbox)
                            if iou_val >= best_iou:
                                best_iou = iou_val
                                kp_np = kp_data[k].cpu().numpy()
                                best_kp = kp_np
                        if best_kp is not None:
                            person_det.keypoints = best_kp

        # Step 4 — Object-person association
        for obj_det in object_detections:
            best_iou = 0.0
            best_person = None
            for person_det in person_detections:
                iou_val = compute_iou(obj_det.bbox, person_det.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_person = person_det
            if best_person is not None:
                best_person.signals_hint_object = True
                best_person.signals_hint_object_type = obj_det.class_name

        self._last_objects = object_detections

        # Step 5 — Return only person detections
        return person_detections

    @staticmethod
    def _map_class_id(cls_id: int) -> str:
        mapping = {
            38: "baseball bat",
            49: "knife",
        }
        return mapping.get(cls_id, f"object_{cls_id}")
