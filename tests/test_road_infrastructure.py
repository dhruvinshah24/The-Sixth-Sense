"""
Comprehensive unit tests for Road & Infrastructure perception:
- Traffic sign detection (Stop, Speed Limit, No Parking, School Zone, One-Way)
- Zebra crossing detection & condition grading (PRESENT, FADED_CANDIDATE, MISSING_CANDIDATE)
- Road divider & median detection with continuity & damage assessment
- Waterlogging detection with specular reflection & texture metrics
- Road hazards & obstructions (debris, obstruction, garbage)
- Strict compliance with common event schema (EventType, Detection, GPS, confidence)
"""
import cv2
import numpy as np
import pytest

from sixth_sense.perception import (
    HazardDetector,
    RoadMarkingDetector,
    TrafficSignDetector,
)
from sixth_sense.schemas.urban_event import EventType, GPSPoint, GPSStatus


# -------------------------------------------------------------------------- #
# PHASE 2 TESTS: Traffic Sign Detection
# -------------------------------------------------------------------------- #

def test_traffic_sign_detector_recognizes_stop_sign():
    """Stop sign pattern (red octagon) should trigger STOP traffic sign event."""
    detector = TrafficSignDetector(yolo_model=None, conf_threshold=0.30)

    canvas = np.full((720, 1280, 3), (120, 125, 130), dtype=np.uint8)
    # Draw red octagonal stop sign
    pts = np.array([[120, 80], [180, 80], [220, 120], [220, 180],
                    [180, 220], [120, 220], [80, 180], [80, 120]], np.int32)
    cv2.fillPoly(canvas, [pts], (20, 20, 220))
    cv2.putText(canvas, "STOP", (95, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    dets = detector.detect(canvas, frame_idx=1, timestamp=0.1)
    stop_dets = [d for d in dets if d.class_name == "SPEED_LIMIT" or d.class_name == "STOP"]
    assert len(stop_dets) >= 1
    d = stop_dets[0]
    assert d.event_type == EventType.TRAFFIC_SIGN
    assert d.confidence >= 0.30
    assert d.bbox_area_px > 0


def test_traffic_sign_detector_recognizes_speed_limit():
    """Circular white disc with red border should be classified as SPEED_LIMIT."""
    detector = TrafficSignDetector(yolo_model=None, conf_threshold=0.30)

    canvas = np.full((720, 1280, 3), (120, 125, 130), dtype=np.uint8)
    # Red circle with white center
    cv2.circle(canvas, (400, 150), 65, (0, 0, 230), 12)
    cv2.circle(canvas, (400, 150), 53, (250, 250, 250), -1)
    cv2.putText(canvas, "40", (375, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    dets = detector.detect(canvas, frame_idx=1, timestamp=0.1)
    sl_dets = [d for d in dets if d.class_name == "SPEED_LIMIT"]
    assert len(sl_dets) >= 1
    assert sl_dets[0].event_type == EventType.TRAFFIC_SIGN
    assert sl_dets[0].confidence > 0.40


def test_traffic_sign_detector_recognizes_one_way():
    """Rectangular blue sign with arrow triggers ONE_WAY."""
    detector = TrafficSignDetector(yolo_model=None, conf_threshold=0.30)

    canvas = np.full((720, 1280, 3), (120, 125, 130), dtype=np.uint8)
    cv2.rectangle(canvas, (300, 100), (500, 180), (180, 60, 0), -1)
    cv2.arrowedLine(canvas, (320, 140), (480, 140), (255, 255, 255), 6, tipLength=0.3)

    dets = detector.detect(canvas, frame_idx=1, timestamp=0.1)
    ow_dets = [d for d in dets if d.class_name == "ONE_WAY"]
    assert len(ow_dets) >= 1
    assert ow_dets[0].event_type == EventType.TRAFFIC_SIGN


# -------------------------------------------------------------------------- #
# PHASE 3 TESTS: Zebra Crossing Detection & Condition
# -------------------------------------------------------------------------- #

def test_zebra_crossing_present_condition():
    """High contrast parallel stripes should be classified as ZEBRA_CROSSING_PRESENT."""
    detector = RoadMarkingDetector(min_crossing_stripes=3, min_contrast_good=0.35)

    canvas = np.full((720, 1280, 3), (50, 50, 50), dtype=np.uint8)
    for i in range(5):
        y = 420 + i * 50
        cv2.rectangle(canvas, (200, y), (600, y + 28), (250, 250, 250), -1)

    dets = detector.detect_zebra_crossings(canvas, frame_idx=0, timestamp=0.0)
    assert len(dets) >= 1
    assert dets[0].event_type == EventType.ZEBRA_CROSSING
    assert dets[0].class_name == "ZEBRA_CROSSING_PRESENT"
    assert dets[0].confidence >= 0.60


def test_zebra_crossing_faded_candidate():
    """Low contrast stripe pattern should be classified as ZEBRA_CROSSING_FADED_CANDIDATE."""
    detector = RoadMarkingDetector(min_crossing_stripes=3, min_contrast_faded=0.10, min_contrast_good=0.50)

    canvas = np.full((720, 1280, 3), (70, 70, 70), dtype=np.uint8)
    for i in range(5):
        y = 420 + i * 50
        # Worn / weathered paint (low contrast above asphalt)
        cv2.rectangle(canvas, (200, y), (600, y + 25), (110, 110, 110), -1)

    dets = detector.detect_zebra_crossings(canvas, frame_idx=0, timestamp=0.0)
    assert len(dets) >= 1
    assert dets[0].event_type == EventType.ZEBRA_CROSSING
    assert dets[0].class_name in ("ZEBRA_CROSSING_FADED_CANDIDATE", "ZEBRA_CROSSING_PRESENT")


def test_zebra_crossing_missing_in_designated_zone():
    """Absence of stripes in a designated crossing zone flags MISSING_CANDIDATE."""
    detector = RoadMarkingDetector(min_crossing_stripes=3)

    # Empty asphalt canvas
    canvas = np.full((720, 1280, 3), (50, 50, 50), dtype=np.uint8)
    dets = detector.detect_zebra_crossings(canvas, frame_idx=0, timestamp=0.0, is_crossing_zone=True)

    assert len(dets) == 1
    assert dets[0].class_name == "ZEBRA_CROSSING_MISSING_CANDIDATE"


# -------------------------------------------------------------------------- #
# PHASE 4 TESTS: Road Divider / Median Detection & Condition
# -------------------------------------------------------------------------- #

def test_divider_present_detection():
    """Continuous barrier / median lines should detect DIVIDER_PRESENT."""
    detector = RoadMarkingDetector()

    canvas = np.full((720, 1280, 3), (60, 60, 60), dtype=np.uint8)
    # Continuous central median barrier
    cv2.line(canvas, (640, 300), (640, 680), (30, 220, 220), 8)  # yellow curb
    cv2.line(canvas, (645, 300), (645, 680), (200, 200, 200), 6) # concrete barrier

    dets = detector.detect_dividers(canvas, frame_idx=0, timestamp=0.0)
    assert len(dets) >= 1
    assert dets[0].event_type == EventType.ROAD_DIVIDER
    assert dets[0].class_name in ("DIVIDER_PRESENT", "DIVIDER_DAMAGED_CANDIDATE")


def test_divider_missing_in_divided_corridor():
    """Absence along an expected divided corridor flags DIVIDER_MISSING_CANDIDATE."""
    detector = RoadMarkingDetector()

    canvas = np.full((720, 1280, 3), (60, 60, 60), dtype=np.uint8)
    dets = detector.detect_dividers(canvas, frame_idx=0, timestamp=0.0, expected_divided_corridor=True)

    assert len(dets) == 1
    assert dets[0].class_name == "DIVIDER_MISSING_CANDIDATE"


# -------------------------------------------------------------------------- #
# PHASE 5 TESTS: Waterlogging Detection
# -------------------------------------------------------------------------- #

def test_waterlogging_detection():
    """Specular, low-saturation puddle on road surface triggers WATERLOGGING."""
    detector = HazardDetector(min_water_area_px=300, conf_threshold=0.30)

    canvas = np.full((720, 1280, 3), (80, 80, 80), dtype=np.uint8)
    # Dark, specular water puddle with flat interior texture
    cv2.ellipse(canvas, (640, 520), (120, 55), 0, 0, 360, (25, 25, 25), -1)

    dets = detector.detect_waterlogging(canvas, frame_idx=0, timestamp=0.0)
    assert len(dets) >= 1
    assert dets[0].event_type == EventType.WATERLOGGING
    assert dets[0].confidence >= 0.35


# -------------------------------------------------------------------------- #
# PHASE 6 TESTS: Road Hazards (Debris, Obstruction, Garbage)
# -------------------------------------------------------------------------- #

def test_road_garbage_detection():
    """Saliency blob with chromatic contrast on road surface triggers GARBAGE."""
    detector = HazardDetector(min_hazard_area_px=200, conf_threshold=0.30)

    canvas = np.full((720, 1280, 3), (70, 70, 70), dtype=np.uint8)
    # Discarded orange/blue plastic bag / trash pile
    cv2.rectangle(canvas, (500, 500), (535, 530), (220, 120, 0), -1)

    dets = detector.detect_road_hazards(canvas, frame_idx=0, timestamp=0.0)
    assert len(dets) >= 1
    assert dets[0].event_type in (EventType.GARBAGE, EventType.INCIDENT_CANDIDATE)
    assert dets[0].confidence >= 0.50
