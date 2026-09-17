"""
Road Marking & Median Detector — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Detects and assesses road surface markings and medians:
  1. Zebra Crossings:
     - Detects periodic transverse stripe patterns on carriageway.
     - Condition grading:
       - ZEBRA_CROSSING_PRESENT: High contrast, regular stripes.
       - ZEBRA_CROSSING_FADED_CANDIDATE: Degraded contrast or worn paint.
       - ZEBRA_CROSSING_MISSING_CANDIDATE: Expected crossing zone with absent markings.
  2. Road Dividers / Medians:
     - Identifies physical median barriers and road dividers along central corridor.
     - Condition grading:
       - DIVIDER_PRESENT: Continuous structural median.
       - DIVIDER_DAMAGED_CANDIDATE: Structural break, displacement, or broken marker.
       - DIVIDER_MISSING_CANDIDATE: Prolonged gap across divided road segment.

Guarantees:
- Never declares legal non-compliance or missing marking from a single isolated frame.
- Adheres to standard Detection schema with EventType.ZEBRA_CROSSING and EventType.ROAD_DIVIDER.
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


class RoadMarkingDetector:
    """
    Detector for zebra crossings and road dividers/medians with condition assessment.
    """

    def __init__(
        self,
        min_crossing_stripes: int = 3,
        min_contrast_faded: float = 0.15,
        min_contrast_good: float = 0.40,
        model_version: str = "road_marking_v1",
    ) -> None:
        self.min_stripes = min_crossing_stripes
        self.min_contrast_faded = min_contrast_faded
        self.min_contrast_good = min_contrast_good
        self.model_version = model_version

    def detect_zebra_crossings(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float = 0.0,
        quality: Optional[QualityScore] = None,
        gps: Optional[GPSPoint] = None,
        is_crossing_zone: bool = False,
    ) -> List[Detection]:
        """
        Scan lower perspective road corridor for zebra crossing markings.
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        conf_mult = quality.conf_multiplier if quality else 1.0

        # Focus on lower 60% of frame where road surface resides
        road_y_start = int(0.40 * h)
        road_roi = frame[road_y_start:, :]
        roi_h, roi_w = road_roi.shape[:2]

        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding to cleanly isolate white markings from darker road surface
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological horizontal structuring to isolate transverse zebra bands
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stripe_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh) if bh > 0 else 0
            # Zebra stripes are wider than they are tall in front-camera perspective
            if 1.5 <= aspect <= 25.0 and bh >= 8:
                stripe_boxes.append((x, y + road_y_start, bw, bh))

        detections: List[Detection] = []

        if len(stripe_boxes) >= self.min_stripes:
            # Cluster bounding box encompassing all stripes
            min_x = min(b[0] for b in stripe_boxes)
            min_y = min(b[1] for b in stripe_boxes)
            max_x = max(b[0] + b[2] for b in stripe_boxes)
            max_y = max(b[1] + b[3] for b in stripe_boxes)

            # Compute contrast ratio between stripe pixels and surrounding asphalt
            mask_stripes = opened > 0
            asphalt_mask = opened == 0
            mean_stripe = float(np.mean(gray[mask_stripes])) if np.any(mask_stripes) else 128.0
            mean_asphalt = float(np.mean(gray[asphalt_mask])) if np.any(asphalt_mask) else 64.0

            contrast = (mean_stripe - mean_asphalt) / (mean_stripe + mean_asphalt + 1e-6)

            if contrast >= self.min_contrast_good:
                condition_class = "ZEBRA_CROSSING_PRESENT"
                raw_conf = min(0.95, 0.60 + 0.35 * contrast)
            elif contrast >= self.min_contrast_faded:
                condition_class = "ZEBRA_CROSSING_FADED_CANDIDATE"
                raw_conf = min(0.85, 0.50 + 0.35 * contrast)
            else:
                condition_class = "ZEBRA_CROSSING_MISSING_CANDIDATE"
                raw_conf = 0.55

            bbox_area = (max_x - min_x) * (max_y - min_y)
            rel_area = bbox_area / frame_area if frame_area > 0 else 0.0

            det = Detection(
                det_id=Detection.make_id(),
                frame_idx=frame_idx,
                timestamp=timestamp,
                event_type=EventType.ZEBRA_CROSSING,
                class_name=condition_class,
                classification_source=ClassificationSource.DETECTED,
                raw_confidence=round(raw_conf, 4),
                confidence=round(min(1.0, raw_conf * conf_mult), 4),
                bbox=(min_x, min_y, max_x, max_y),
                bbox_area_px=bbox_area,
                relative_area=round(rel_area, 6),
                frame_width=w,
                frame_height=h,
                gps=gps,
                quality=quality,
                model_name=self.model_version,
            )
            detections.append(det)

        elif is_crossing_zone and len(stripe_boxes) < self.min_stripes:
            # Designated crossing zone with absent stripes
            det = Detection(
                det_id=Detection.make_id(),
                frame_idx=frame_idx,
                timestamp=timestamp,
                event_type=EventType.ZEBRA_CROSSING,
                class_name="ZEBRA_CROSSING_MISSING_CANDIDATE",
                classification_source=ClassificationSource.INFERRED,
                raw_confidence=0.60,
                confidence=round(min(1.0, 0.60 * conf_mult), 4),
                bbox=(int(0.2 * w), int(0.6 * h), int(0.8 * w), int(0.9 * h)),
                bbox_area_px=int(0.18 * frame_area),
                relative_area=0.18,
                frame_width=w,
                frame_height=h,
                gps=gps,
                quality=quality,
                model_name=self.model_version,
            )
            detections.append(det)

        return detections

    def detect_dividers(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float = 0.0,
        quality: Optional[QualityScore] = None,
        gps: Optional[GPSPoint] = None,
        expected_divided_corridor: bool = False,
    ) -> List[Detection]:
        """
        Scan central road corridor for road divider/median barriers and condition.
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        conf_mult = quality.conf_multiplier if quality else 1.0

        # Central vertical corridor where dividers/medians reside
        cx_start, cx_end = int(0.25 * w), int(0.75 * w)
        cy_start, cy_end = int(0.35 * h), int(0.95 * h)
        center_roi = frame[cy_start:cy_end, cx_start:cx_end]
        roi_h, roi_w = center_roi.shape[:2]

        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        # Yellow curb or concrete/metal barrier detection
        y_mask = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([35, 255, 255]))
        # High-contrast edge detection for concrete curbs/guardrails
        gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.bitwise_or(y_mask, edges)

        lines = cv2.HoughLinesP(combined, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=15)

        detections: List[Detection] = []

        if lines is not None and len(lines) >= 2:
            # Calculate divider bounding box from detected line segments
            all_pts = []
            for line in lines:
                coords = line[0] if len(line) == 1 and hasattr(line[0], '__len__') else line
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                all_pts.extend([(x1 + cx_start, y1 + cy_start), (x2 + cx_start, y2 + cy_start)])

            min_x = max(0, min(p[0] for p in all_pts) - 10)
            min_y = max(0, min(p[1] for p in all_pts) - 5)
            max_x = min(w, max(p[0] for p in all_pts) + 10)
            max_y = min(h, max(p[1] for p in all_pts) + 5)

            # Measure line continuity (detect breaks / damage)
            y_span = max_y - min_y
            expected_span = cy_end - cy_start

            if y_span > 0.40 * expected_span:
                condition = "DIVIDER_PRESENT"
                raw_conf = 0.82
            else:
                condition = "DIVIDER_DAMAGED_CANDIDATE"
                raw_conf = 0.68

            bbox_area = (max_x - min_x) * (max_y - min_y)
            rel_area = bbox_area / frame_area if frame_area > 0 else 0.0

            det = Detection(
                det_id=Detection.make_id(),
                frame_idx=frame_idx,
                timestamp=timestamp,
                event_type=EventType.ROAD_DIVIDER,
                class_name=condition,
                classification_source=ClassificationSource.DETECTED,
                raw_confidence=raw_conf,
                confidence=round(min(1.0, raw_conf * conf_mult), 4),
                bbox=(min_x, min_y, max_x, max_y),
                bbox_area_px=bbox_area,
                relative_area=round(rel_area, 6),
                frame_width=w,
                frame_height=h,
                gps=gps,
                quality=quality,
                model_name=self.model_version,
            )
            detections.append(det)

        elif expected_divided_corridor:
            # Expected median corridor with complete absence
            det = Detection(
                det_id=Detection.make_id(),
                frame_idx=frame_idx,
                timestamp=timestamp,
                event_type=EventType.ROAD_DIVIDER,
                class_name="DIVIDER_MISSING_CANDIDATE",
                classification_source=ClassificationSource.INFERRED,
                raw_confidence=0.60,
                confidence=round(min(1.0, 0.60 * conf_mult), 4),
                bbox=(int(0.45 * w), int(0.40 * h), int(0.55 * w), int(0.90 * h)),
                bbox_area_px=int(0.05 * frame_area),
                relative_area=0.05,
                frame_width=w,
                frame_height=h,
                gps=gps,
                quality=quality,
                model_name=self.model_version,
            )
            detections.append(det)

        return detections

    def annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and condition labels for zebra crossings and dividers."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if det.event_type == EventType.ZEBRA_CROSSING:
                color = (255, 255, 0) if "PRESENT" in det.class_name else (0, 165, 255)
            elif det.event_type == EventType.ROAD_DIVIDER:
                color = (0, 255, 128) if "PRESENT" in det.class_name else (0, 0, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"{det.class_name} ({det.confidence:.2f})",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return out
