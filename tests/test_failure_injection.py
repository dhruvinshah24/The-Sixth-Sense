from __future__ import annotations
import pytest
from sixth_sense.association.gps_associator import GPSAssociator
from sixth_sense.schemas.urban_event import (
    Detection,
    GPSPoint,
    GPSStatus,
    EventType,
    SeverityTier,
    Observation,
    PersistentIssue,
    ClassificationSource,
)
from sixth_sense.schemas.unified_event import UnifiedObservation, DomainType
from sixth_sense.events.observation_builder import ObservationBuilder
from sixth_sense.intelligence.cross_domain_fusion import CrossDomainFusionEngine
from sixth_sense.actionable.priority_engine_v2 import PriorityEngineV2
from sixth_sense.intelligence.road_health import RoadHealthEngine

def _make_det(frame_idx: int, timestamp: float, confidence: float = 0.85):
    bbox = (100.0, 100.0, 200.0, 200.0)
    area = 10000.0
    return Detection(
        det_id=Detection.make_id(),
        frame_idx=frame_idx,
        timestamp=timestamp,
        event_type=EventType.POTHOLE,
        class_name='D40',
        classification_source=ClassificationSource.DETECTED,
        raw_confidence=confidence,
        confidence=confidence,
        bbox=bbox,
        bbox_area_px=area,
        relative_area=area / (1920 * 1080),
        frame_width=1920,
        frame_height=1080,
        gps=None,
        quality=None,
        model_name='yolo12s-rdd2022',
    )

def test_gps_out_of_bounds_before_start():
    assoc = GPSAssociator(gps_source=None, video_start_unix=0.0, max_interpolation_gap_sec=10.0)
    assoc._samples = [
        {'timestamp': 100.0, 'lat': 19.0760, 'lon': 72.8777, 'heading': 90.0},
        {'timestamp': 105.0, 'lat': 19.0765, 'lon': 72.8780, 'heading': 90.0},
    ]
    pt = assoc.get_location(50.0)
    assert pt.status == GPSStatus.UNAVAILABLE
    assert pt.uncertainty_m == 9999.0
    assert pt.lat == 0.0

def test_gps_out_of_bounds_after_end():
    assoc = GPSAssociator(gps_source=None, video_start_unix=0.0, max_interpolation_gap_sec=10.0)
    assoc._samples = [
        {'timestamp': 100.0, 'lat': 19.0760, 'lon': 72.8777, 'heading': 90.0},
        {'timestamp': 105.0, 'lat': 19.0765, 'lon': 72.8780, 'heading': 90.0},
    ]
    pt = assoc.get_location(130.0)
    assert pt.status == GPSStatus.UNAVAILABLE
    assert pt.uncertainty_m == 9999.0

def test_gps_strictly_bounded_interpolation():
    assoc = GPSAssociator(gps_source=None, video_start_unix=0.0, max_interpolation_gap_sec=15.0)
    assoc._samples = [
        {'timestamp': 100.0, 'lat': 19.0000, 'lon': 72.0000, 'heading': None},
        {'timestamp': 110.0, 'lat': 19.1000, 'lon': 72.1000, 'heading': None},
    ]
    mid = assoc.get_location(105.0)
    assert mid.status == GPSStatus.INTERPOLATED
    assert pytest.approx(mid.lat, abs=1e-4) == 19.0500
    assert pytest.approx(mid.lon, abs=1e-4) == 72.0500

def test_observation_builder_rejects_single_frame_flicker():
    cfg = {'observation': {'min_detections': 3, 'max_gap_frames': 10, 'min_spatial_overlap_iou': 0.3}}
    builder = ObservationBuilder(profile_cfg=cfg, bus_id='BUS_01', camera_id='CAM_FRONT', run_id='RUN_TEST')
    d1 = _make_det(frame_idx=10, timestamp=1.0, confidence=0.85)
    builder.ingest([d1])
    observations = builder.finalise()
    assert len(observations) == 0

def test_observation_builder_accepts_persistent_detection():
    cfg = {'observation': {'min_detections': 3, 'max_gap_frames': 10, 'min_spatial_overlap_iou': 0.3}}
    builder = ObservationBuilder(profile_cfg=cfg, bus_id='BUS_01', camera_id='CAM_FRONT', run_id='RUN_TEST')
    for f_idx in [10, 11, 12]:
        d = _make_det(frame_idx=f_idx, timestamp=f_idx * 0.1, confidence=0.80)
        builder.ingest([d])
    observations = builder.finalise()
    assert len(observations) == 1
    assert observations[0].detection_count == 3
    assert observations[0].event_type == EventType.POTHOLE

def test_cross_domain_isolation_traffic_only():
    engine = CrossDomainFusionEngine()
    res = engine.fuse_segment_events(segment_id='SEG_PURE_TRAFFIC', unified_events=[], traffic_exposure=0.9, traffic_congestion_state='SEVERE')
    assert res.cross_domain_synergy == 'STANDARD_MONITORING'
    assert res.urgency_multiplier == 1.0

def test_priority_engine_score_bounds():
    engine = PriorityEngineV2()
    issue = PersistentIssue(
        issue_id='iss_test', event_type=EventType.POTHOLE, class_name='D40',
        severity=SeverityTier.CRITICAL, confidence=1.0, observation_count=10,
        bus_count=5, bus_ids=['BUS_1', 'BUS_2'], first_seen_ts=0.0, last_seen_ts=100.0,
        center_gps=GPSPoint(lat=19.0, lon=72.0, timestamp=0.0, uncertainty_m=5.0, heading=None, status=GPSStatus.DIRECT)
    )
    score_res = engine.score(issue, traffic_exposure_score=1.0, traffic_congestion_state='SEVERE', safety_risk_exposure=1.0)
    assert 0.0 <= score_res.priority_score <= 100.0

def test_road_health_bounds_and_monotonicity():
    engine = RoadHealthEngine()
    clean_res = engine.evaluate_segment('SEG_PRISTINE', observations=[])
    assert clean_res.health_score == 100.0
    obs_list = [
        Observation(
            obs_id=f'obs_{i}', bus_id=f'BUS_{i}', camera_id='CAM_FRONT', run_id='RUN_1',
            event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=1, last_seen_frame=5,
            first_seen_ts=1.0, last_seen_ts=1.5, representative_frame=3, confidence=0.95,
            severity=SeverityTier.HIGH, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
            relative_area=0.05, gps=None, evidence_ref=None, detection_count=5, model_name='yolo12s-rdd2022'
        ) for i in range(10)
    ]
    damaged_res = engine.evaluate_segment('SEG_DAMAGED', observations=obs_list)
    assert 0.0 <= damaged_res.health_score <= 100.0
    assert damaged_res.health_score < clean_res.health_score
