"""
Traffic Sign Detector — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Detects and classifies regulatory and warning traffic signs:
  - STOP (octagonal red field)
  - SPEED_LIMIT (circular disc with red border)
  - NO_PARKING (circular blue disc with red diagonal slash)
  - SCHOOL_ZONE (warning triangle/pentagon)
  - ONE_WAY (directional arrow rectangle)

Uses a hybrid perception model:
1. Deep learning YOLO model for recognized object proposals (e.g. stop signs, road signs).
2. Geometric and chromatic morphological verification for regulatory signs.
Outputs standardized Detection objects conforming to EventType.TRAFFIC_SIGN.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sixth_sense.schemas.urban_event import (
    ClassificationSource,
    Detection,
    EventType,
    GPSPoint,
    QualityScore,
)

logger = logging.getLogger(__name__)

SUPPORTED_SIGN_CLASSES = (
    "STOP",
    "SPEED_LIMIT",
    "NO_PARKING",
    "SCHOOL_ZONE",
    "ONE_WAY",
)


class TrafficSignDetector:
    """
    Traffic sign detector and classifier.
    Combines YOLO object recognition with chromatic/morphological verification.
    """

    def __init__(
        self,
        yolo_model: Optional[Any] = None,
        conf_threshold: float = 0.30,
        device: str = "cuda:0",
        imgsz: int = 640,
    ) -> None:
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        self.model_version = "traffic_sign_v1_hybrid"

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float = 0.0,
        quality: Optional[QualityScore] = None,
        gps: Optional[GPSPoint] = None,
    ) -> List[Detection]:
        """
        Scan frame for traffic signs.
        
        Args:
            frame: BGR numpy image
            frame_idx: Video frame index
            timestamp: Frame timestamp in seconds
            quality: Optional frame quality score
            gps: Optional GPS telemetry
            
        Returns:
            List of Detection objects with event_type=EventType.TRAFFIC_SIGN.
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        conf_mult = quality.conf_multiplier if quality else 1.0

        detections: List[Detection] = []
        seen_boxes: List[Tuple[int, int, int, int]] = []

        # ── 1. YOLO Ingestion (Stop signs and standard classes) ─────────── #
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(
                    frame,
                    imgsz=self.imgsz,
                    verbose=False,
                    device=self.device,
                )
                for res in results:
                    names = res.names
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = names.get(cls_id, str(cls_id)).lower()
                        raw_conf = float(box.conf[0])

                        # YOLO COCO class 11 is 'stop sign'
                        if "stop" in cls_name and raw_conf >= self.conf_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            bbox_area = max(0, (x2 - x1) * (y2 - y1))
                            rel_area = bbox_area / frame_area if frame_area > 0 else 0.0
                            adj_conf = min(1.0, raw_conf * conf_mult)

                            det = Detection(
                                det_id=Detection.make_id(),
                                frame_idx=frame_idx,
                                timestamp=timestamp,
                                event_type=EventType.TRAFFIC_SIGN,
                                class_name="STOP",
                                classification_source=ClassificationSource.DETECTED,
                                raw_confidence=round(raw_conf, 4),
                                confidence=round(adj_conf, 4),
                                bbox=(x1, y1, x2, y2),
                                bbox_area_px=bbox_area,
                                relative_area=round(rel_area, 6),
                                frame_width=w,
                                frame_height=h,
                                gps=gps,
                                quality=quality,
                                model_name=self.model_version,
                            )
                            detections.append(det)
                            seen_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                logger.debug("YOLO sign detection pass skipped: %s", e)

        # ── 2. Chromatic and Geometric Sign Analysis ────────────────────── #
        # Signs typically appear in the upper 75% of frame and along sides/shoulders
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red mask (Stop, Speed Limit, No Parking borders)
        r_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        r_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r_mask1, r_mask2)

        # Blue mask (No Parking inner disc, One Way)
        blue_mask = cv2.inRange(hsv, np.array([100, 80, 50]), np.array([130, 255, 255]))

        # Yellow/Amber mask (School Zone)
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))

        combined_masks = [
            ("SPEED_LIMIT", red_mask, "circular"),
            ("NO_PARKING", cv2.bitwise_and(red_mask, cv2.dilate(blue_mask, np.ones((5, 5)))), "circular"),
            ("SCHOOL_ZONE", yellow_mask, "triangular"),
            ("ONE_WAY", blue_mask, "rectangular"),
        ]

        # Scan for salient sign contours
        for sign_cls, mask, target_shape in combined_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter out tiny noise and gigantic areas
                if area < 400 or area > 0.15 * frame_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / float(bh) if bh > 0 else 0.0

                # Must be in upper/middle vertical region
                if y > 0.85 * h:
                    continue

                # Check IoU overlap with already detected signs
                box_overlap = False
                for sx1, sy1, sx2, sy2 in seen_boxes:
                    inter_x1, inter_y1 = max(x, sx1), max(y, sy1)
                    inter_x2, inter_y2 = min(x + bw, sx2), min(y + bh, sy2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        box_overlap = True
                        break
                if box_overlap:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_vertices = len(approx)

                matched = False
                conf = 0.0

                if target_shape == "circular" and 0.75 <= aspect <= 1.30:
                    circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
                    if circularity > 0.60 or num_vertices >= 6:
                        matched = True
                        conf = min(0.92, 0.55 + 0.35 * circularity)

                elif target_shape == "triangular" and 0.70 <= aspect <= 1.35:
                    if num_vertices in (3, 4, 5):
                        matched = True
                        conf = 0.72

                elif target_shape == "rectangular" and (aspect > 1.3 or aspect < 0.75):
                    rect_extent = float(area) / (bw * bh) if (bw * bh) > 0 else 0
                    if rect_extent > 0.65:
                        matched = True
                        conf = min(0.88, 0.50 + 0.38 * rect_extent)

                if matched and conf >= self.conf_threshold:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + bw), min(h, y + bh)
                    bbox_area = max(0, (x2 - x1) * (y2 - y1))
                    rel_area = bbox_area / frame_area if frame_area > 0 else 0.0
                    adj_conf = min(1.0, conf * conf_mult)

                    det = Detection(
                        det_id=Detection.make_id(),
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        event_type=EventType.TRAFFIC_SIGN,
                        class_name=sign_cls,
                        classification_source=ClassificationSource.DETECTED,
                        raw_confidence=round(conf, 4),
                        confidence=round(adj_conf, 4),
                        bbox=(x1, y1, x2, y2),
                        bbox_area_px=bbox_area,
                        relative_area=round(rel_area, 6),
                        frame_width=w,
                        frame_height=h,
                        gps=gps,
                        quality=quality,
                        model_name=self.model_version,
                    )
                    detections.append(det)
                    seen_boxes.append((x1, y1, x2, y2))

        return detections

    def annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and class labels on detected traffic signs."""
        out = frame.copy()
        for det in detections:
            if det.event_type != EventType.TRAFFIC_SIGN:
                continue
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"{det.class_name} ({det.confidence:.2f})"
            cv2.putText(
                out,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        return out
