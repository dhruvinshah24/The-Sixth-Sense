from __future__ import annotations
from sixth_sense.events.observation_builder import ObservationBuilder
import math
import numpy as np
import pytest
from typing import List, Dict, Any

from sixth_sense.association.gps_associator import GPSAssociator
from sixth_sense.schemas.urban_event import (
    Detection, GPSPoint, GPSStatus, EventType, SeverityTier,
    Observation, PersistentIssue, ClassificationSource, Track,
)
from sixth_sense.schemas.unified_event import UnifiedObservation, DomainType
from sixth_sense.tracking.urban_tracker import UrbianTracker
from sixth_sense.events.city_memory import CityMemory
from sixth_sense.intelligence.cross_domain_fusion import CrossDomainFusionEngine
from sixth_sense.actionable.priority_engine_v2 import PriorityEngineV2
from sixth_sense.intelligence.road_health import RoadHealthEngine
from sixth_sense.closure.verification_engine import (
    VerificationEngine, FollowUpPass, VerificationOutcome
)
from sixth_sense.closure.repair_claim import RepairClaim, RepairClaimStatus
from sixth_sense.privacy.anonymiser import Anonymiser

def _make_det(
    frame_idx: int,
    bbox: tuple = (100.0, 100.0, 200.0, 200.0),
    confidence: float = 0.85,
    class_name: str = 'car',
    event_type: EventType = EventType.VEHICLE,
    lat: float = 19.0760,
    lon: float = 72.8777,
) -> Detection:
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    gps = GPSPoint(lat=lat, lon=lon, timestamp=frame_idx * 0.1, uncertainty_m=5.0, heading=90.0, status=GPSStatus.INTERPOLATED)
    return Detection(
        det_id=Detection.make_id(),
        frame_idx=frame_idx,
        timestamp=frame_idx * 0.1,
        event_type=event_type,
        class_name=class_name,
        classification_source=ClassificationSource.DETECTED,
        raw_confidence=confidence,
        confidence=confidence,
        bbox=bbox,
        bbox_area_px=area,
        relative_area=area / (1920 * 1080),
        frame_width=1920,
        frame_height=1080,
        gps=gps,
        quality=None,
        model_name='yolo11n',
    )

# ---------------------------------------------------------------------------
# SECTION 3: TRACKING ADVERSARIAL TESTS
# ---------------------------------------------------------------------------

def test_tracking_line_crossing_idempotence():
    tracker = UrbianTracker(confirm_frames=2, max_lost_frames=5)
    counted_track_ids = set()
    line_y = 550.0

    # Realistic smooth continuous motion with IoU overlap > 0.6
    y_positions = [440.0, 460.0, 480.0, 510.0, 510.0, 480.0, 515.0]
    prev_positions = {}

    for f_idx, y in enumerate(y_positions, start=1):
        det = _make_det(frame_idx=f_idx, bbox=(200.0, y, 300.0, y + 100.0), class_name='car')
        active_tracks = tracker.update([det], frame_idx=f_idx)
        for trk in active_tracks:
            tid = trk.track_id
            curr_cy = (trk.latest_bbox[1] + trk.latest_bbox[3]) / 2.0
            if tid in prev_positions:
                prev_cy = prev_positions[tid]
                if prev_cy < line_y <= curr_cy:
                    counted_track_ids.add(tid)
            prev_positions[tid] = curr_cy

    # Assert exactly ONE unique count despite oscillation/reversal
    assert len(counted_track_ids) == 1


def test_tracking_two_simultaneous_vehicles():
    tracker = UrbianTracker(confirm_frames=2, max_lost_frames=5)
    # Two vehicles in distinct lanes (lane 1: x=100..200, lane 2: x=500..600)
    for f_idx in [1, 2, 3]:
        d1 = _make_det(frame_idx=f_idx, bbox=(100.0, 300.0 + f_idx * 20, 200.0, 400.0 + f_idx * 20), class_name='car')
        d2 = _make_det(frame_idx=f_idx, bbox=(500.0, 300.0 + f_idx * 20, 600.0, 400.0 + f_idx * 20), class_name='bus')
        active = tracker.update([d1, d2], frame_idx=f_idx)
        if f_idx >= 2:
            assert len(active) == 2
            tids = {t.track_id for t in active}
            assert len(tids) == 2


def test_tracking_temporary_occlusion_recovery():
    tracker = UrbianTracker(confirm_frames=2, max_lost_frames=5)
    # Frame 1, 2: detected
    tracker.update([_make_det(1, (100.0, 100.0, 200.0, 200.0))], 1)
    tracks = tracker.update([_make_det(2, (100.0, 105.0, 200.0, 205.0))], 2)
    initial_id = tracks[0].track_id

    # Frame 3, 4: occluded (no detections)
    tracker.update([], 3)
    tracker.update([], 4)

    # Frame 5: reappears nearby
    recovered = tracker.update([_make_det(5, (100.0, 110.0, 200.0, 210.0))], 5)
    assert len(recovered) == 1
    assert recovered[0].track_id == initial_id


# ---------------------------------------------------------------------------
# SECTION 5: GPS ADVERSARIAL MATRIX
# ---------------------------------------------------------------------------

def test_gps_adversarial_scenarios():
    samples = [
        {'timestamp': 100.0, 'lat': 19.0000, 'lon': 72.0000, 'heading': 90.0},
        {'timestamp': 105.0, 'lat': 19.0500, 'lon': 72.0500, 'heading': 90.0},
        {'timestamp': 115.0, 'lat': 19.1500, 'lon': 72.1500, 'heading': 90.0},
    ]
    assoc = GPSAssociator(gps_source=None, max_interpolation_gap_sec=10.0)
    assoc._samples = sorted(samples, key=lambda s: s['timestamp'])

    # 1. Exact match
    p_exact = assoc.get_location(100.0)
    assert p_exact.status == GPSStatus.DIRECT
    assert pytest.approx(p_exact.lat, 1e-4) == 19.0000

    # 2. 1 sec offset
    p_1s = assoc.get_location(101.0)
    assert p_1s.status == GPSStatus.INTERPOLATED
    assert pytest.approx(p_1s.lat, 1e-4) == 19.0100

    # 3. Exactly 10s gap (boundary)
    p_10s = assoc.get_location(110.0)
    assert p_10s.status == GPSStatus.INTERPOLATED

    # 4. Out-of-bounds (> 10s before start)
    p_early = assoc.get_location(80.0)
    assert p_early.status == GPSStatus.UNAVAILABLE
    assert p_early.uncertainty_m == 9999.0

    # 5. Out-of-bounds (> 10s after end)
    p_late = assoc.get_location(130.0)
    assert p_late.status == GPSStatus.UNAVAILABLE
    assert p_late.uncertainty_m == 9999.0


# ---------------------------------------------------------------------------
# SECTION 6: CITY MEMORY BOUNDARY TESTS
# ---------------------------------------------------------------------------

def test_city_memory_same_defect_different_buses():
    mem = CityMemory(dedup_radius_m=15.0)
    # Bus 1 observes pothole at (19.0760, 72.8777)
    d1 = _make_det(1, class_name='D40', event_type=EventType.POTHOLE, lat=19.07600, lon=72.87770)
    obs1 = Observation(
        obs_id='obs_1', bus_id='BUS_A', camera_id='CAM_1', run_id='RUN_1',
        event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=1, last_seen_frame=5,
        first_seen_ts=1.0, last_seen_ts=1.5, representative_frame=3, confidence=0.85,
        severity=SeverityTier.HIGH, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
        relative_area=0.01, gps=d1.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    issue1 = mem.ingest_observation(obs1)
    assert issue1.bus_count == 1
    assert issue1.observation_count == 1

    # Bus 2 observes the same pothole 2 meters away
    d2 = _make_det(10, class_name='D40', event_type=EventType.POTHOLE, lat=19.07601, lon=72.87771)
    obs2 = Observation(
        obs_id='obs_2', bus_id='BUS_B', camera_id='CAM_1', run_id='RUN_2',
        event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=10, last_seen_frame=15,
        first_seen_ts=100.0, last_seen_ts=100.5, representative_frame=12, confidence=0.88,
        severity=SeverityTier.HIGH, bbox=(105, 100, 205, 200), bbox_area_px=10000.0,
        relative_area=0.01, gps=d2.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    issue2 = mem.ingest_observation(obs2)
    # Merges into existing issue
    assert issue2.issue_id == issue1.issue_id
    assert issue2.bus_count == 2
    assert issue2.observation_count == 2
    assert set(issue2.bus_ids) == {'BUS_A', 'BUS_B'}


def test_city_memory_different_defect_classes_isolation():
    mem = CityMemory(dedup_radius_m=15.0)
    d = _make_det(1, lat=19.07600, lon=72.87770)
    obs_pothole = Observation(
        obs_id='obs_ph', bus_id='BUS_A', camera_id='CAM_1', run_id='RUN_1',
        event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=1, last_seen_frame=5,
        first_seen_ts=1.0, last_seen_ts=1.5, representative_frame=3, confidence=0.85,
        severity=SeverityTier.HIGH, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
        relative_area=0.01, gps=d.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    obs_crack = Observation(
        obs_id='obs_cr', bus_id='BUS_A', camera_id='CAM_1', run_id='RUN_1',
        event_type=EventType.ROAD_CRACK, class_name='D00', first_seen_frame=1, last_seen_frame=5,
        first_seen_ts=1.0, last_seen_ts=1.5, representative_frame=3, confidence=0.80,
        severity=SeverityTier.MEDIUM, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
        relative_area=0.01, gps=d.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    i1 = mem.ingest_observation(obs_pothole)
    i2 = mem.ingest_observation(obs_crack)
    assert i1.issue_id != i2.issue_id
    assert len(mem.get_all_issues()) == 2


# ---------------------------------------------------------------------------
# SECTION 7: CROSS-DOMAIN FUSION ADVERSARIAL TESTS
# ---------------------------------------------------------------------------

def test_cross_domain_fusion_matrix_isolation():
    engine = CrossDomainFusionEngine()

    # 1. Congestion only
    r1 = engine.fuse_segment_events('SEG_1', unified_events=[], traffic_exposure=0.8, traffic_congestion_state='SEVERE')
    assert r1.cross_domain_synergy == 'STANDARD_MONITORING'
    assert r1.urgency_multiplier == 1.0

    # 2. Road defect + congestion
    ev_defect = UnifiedObservation(
        observation_id='e1', bus_id='BUS_1', timestamp=10.0, domain=DomainType.ROAD,
        event_type='POTHOLE', confidence=0.85, severity='HIGH',
        location={'lat': 19.0, 'lon': 72.0, 'road_segment_id': 'SEG_2'}
    )
    r2 = engine.fuse_segment_events('SEG_2', unified_events=[ev_defect], traffic_exposure=0.8, traffic_congestion_state='SEVERE')
    assert r2.cross_domain_synergy == 'COMPOUND_INFRASTRUCTURE_TRAFFIC_STRESS'
    assert r2.urgency_multiplier == 1.40

    # 3. Road defect + Pedestrian safety
    ev_safety = UnifiedObservation(
        observation_id='e2', bus_id='BUS_1', timestamp=10.0, domain=DomainType.SAFETY,
        event_type='PEDESTRIAN', confidence=0.88, severity='HIGH',
        location={'lat': 19.0, 'lon': 72.0, 'road_segment_id': 'SEG_3'}
    )
    r3 = engine.fuse_segment_events('SEG_3', unified_events=[ev_defect, ev_safety], traffic_exposure=0.1, traffic_congestion_state='NORMAL_FLOW')
    assert r3.cross_domain_synergy == 'SAFETY_CRITICAL_CORRIDOR'
    assert r3.urgency_multiplier == 1.35

    # 4. Incident + Congestion
    ev_incident = UnifiedObservation(
        observation_id='e3', bus_id='BUS_1', timestamp=10.0, domain=DomainType.INCIDENT,
        event_type='ILLEGAL_PARKING', confidence=0.90, severity='HIGH',
        location={'lat': 19.0, 'lon': 72.0, 'road_segment_id': 'SEG_4'}
    )
    r4 = engine.fuse_segment_events('SEG_4', unified_events=[ev_incident], traffic_exposure=0.8, traffic_congestion_state='SEVERE')
    assert r4.cross_domain_synergy == 'INCIDENT_CONGESTION_COMPOUND'
    assert r4.urgency_multiplier == 1.45


# ---------------------------------------------------------------------------
# SECTION 8: PRIORITY ENGINE SENSITIVITY
# ---------------------------------------------------------------------------

def test_priority_engine_monotonicity():
    engine = PriorityEngineV2()
    gps = GPSPoint(lat=19.0, lon=72.0, timestamp=0.0, uncertainty_m=5.0, heading=None, status=GPSStatus.DIRECT)

    def _make_issue(severity: SeverityTier, confidence: float, bus_count: int, obs_count: int):
        return PersistentIssue(
            issue_id='iss_test', event_type=EventType.POTHOLE, class_name='D40',
            severity=severity, confidence=confidence, observation_count=obs_count,
            bus_count=bus_count, bus_ids=[f'B_{i}' for i in range(bus_count)],
            first_seen_ts=0.0, last_seen_ts=10.0, center_gps=gps
        )

    # 1. Severity monotonicity: LOW < MEDIUM < HIGH < CRITICAL
    s_low = engine.score(_make_issue(SeverityTier.LOW, 0.8, 1, 1)).priority_score
    s_med = engine.score(_make_issue(SeverityTier.MEDIUM, 0.8, 1, 1)).priority_score
    s_high = engine.score(_make_issue(SeverityTier.HIGH, 0.8, 1, 1)).priority_score
    s_crit = engine.score(_make_issue(SeverityTier.CRITICAL, 0.8, 1, 1)).priority_score
    assert s_low < s_med < s_high < s_crit

    # 2. Confidence monotonicity
    c_low = engine.score(_make_issue(SeverityTier.HIGH, 0.3, 1, 1)).priority_score
    c_high = engine.score(_make_issue(SeverityTier.HIGH, 0.9, 1, 1)).priority_score
    assert c_low < c_high

    # 3. Traffic exposure monotonicity
    t_normal = engine.score(_make_issue(SeverityTier.HIGH, 0.8, 1, 1), traffic_congestion_state='NORMAL_FLOW').priority_score
    t_severe = engine.score(_make_issue(SeverityTier.HIGH, 0.8, 1, 1), traffic_congestion_state='SEVERE').priority_score
    assert t_normal < t_severe

    # 4. Fleet corroboration monotonicity
    f_single = engine.score(_make_issue(SeverityTier.HIGH, 0.8, 1, 1)).priority_score
    f_multi = engine.score(_make_issue(SeverityTier.HIGH, 0.8, 3, 3)).priority_score
    assert f_single < f_multi


# ---------------------------------------------------------------------------
# SECTION 9: ROAD HEALTH PROPERTY TESTS
# ---------------------------------------------------------------------------

def test_road_health_properties():
    engine = RoadHealthEngine()

    # Clean road = 100.0
    h_clean = engine.evaluate_segment('SEG_CLEAN', observations=[])
    assert h_clean.health_score == 100.0

    # Adding a defect decreases score
    d = _make_det(1)
    obs = Observation(
        obs_id='obs_d', bus_id='BUS_1', camera_id='CAM_1', run_id='R1',
        event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=1, last_seen_frame=5,
        first_seen_ts=1.0, last_seen_ts=1.5, representative_frame=3, confidence=0.85,
        severity=SeverityTier.HIGH, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
        relative_area=0.02, gps=d.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    h_defect = engine.evaluate_segment('SEG_DEFECT', observations=[obs])
    assert h_defect.health_score < h_clean.health_score
    assert 0.0 <= h_defect.health_score <= 100.0


# ---------------------------------------------------------------------------
# SECTION 10: PROOF-OF-CLOSURE ADVERSARIAL TESTS
# ---------------------------------------------------------------------------

def test_proof_of_closure_scenarios():
    engine = VerificationEngine()
    gps = GPSPoint(lat=19.0760, lon=72.8777, timestamp=10.0, uncertainty_m=5.0, heading=None, status=GPSStatus.DIRECT)
    d = _make_det(1, lat=19.0760, lon=72.8777)
    obs_original = Observation(
        obs_id='obs_orig', bus_id='BUS_1', camera_id='CAM_1', run_id='R1',
        event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=1, last_seen_frame=5,
        first_seen_ts=1.0, last_seen_ts=1.5, representative_frame=3, confidence=0.85,
        severity=SeverityTier.HIGH, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
        relative_area=0.02, gps=d.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    issue = PersistentIssue(
        issue_id='ISS_VERIF', event_type=EventType.POTHOLE, class_name='D40',
        severity=SeverityTier.HIGH, confidence=0.85, observation_count=1,
        bus_count=1, bus_ids=['BUS_1'], first_seen_ts=1.0, last_seen_ts=1.5,
        center_gps=gps, observations=[obs_original], status='OPEN'
    )
    claim = RepairClaim(
        claim_id='CLM_001', issue_id='ISS_VERIF', work_item_id='WORK_001',
        claimed_at='2026-09-18T10:00:00Z', claimed_by='MUNICIPAL_CONTRACTOR_01',
        claimed_status=RepairClaimStatus.REPAIR_CLAIMED
    )

    # 1. Clean follow-up pass (covers location with no defect detected)
    pass_clean = FollowUpPass(bus_id='BUS_2', pass_timestamp=100.0, pass_gps=gps, observations=[])
    res_clean = engine.verify(issue, claim, pass_clean)
    assert res_clean.verification_result == VerificationOutcome.VERIFIED_REPAIRED

    # 2. Defect still present
    obs_still = Observation(
        obs_id='obs_still', bus_id='BUS_2', camera_id='CAM_1', run_id='R2',
        event_type=EventType.POTHOLE, class_name='D40', first_seen_frame=100, last_seen_frame=105,
        first_seen_ts=100.0, last_seen_ts=100.5, representative_frame=103, confidence=0.82,
        severity=SeverityTier.HIGH, bbox=(100, 100, 200, 200), bbox_area_px=10000.0,
        relative_area=0.02, gps=d.gps, evidence_ref=None, detection_count=5, model_name='yolo12s'
    )
    pass_defect = FollowUpPass(bus_id='BUS_2', pass_timestamp=100.0, pass_gps=gps, observations=[obs_still])
    res_defect = engine.verify(issue, claim, pass_defect)
    assert res_defect.verification_result in (VerificationOutcome.STILL_PRESENT, VerificationOutcome.REOPENED)


# ---------------------------------------------------------------------------
# SECTION 12: PRIVACY ADVERSARIAL CHECK
# ---------------------------------------------------------------------------

def test_privacy_anonymisation():
    anonymiser = Anonymiser(blur_faces=True, blur_plates=False)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    # Add high-frequency checkerboard pattern in person face region
    img[20:60, 100:150] = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)

    det_person = _make_det(1, bbox=(100.0, 20.0, 150.0, 120.0), class_name='person', event_type=EventType.PEDESTRIAN)
    masked = anonymiser.anonymise(img, [det_person])

    # Original must not be modified
    assert not np.shares_memory(img, masked)

    # Face region should be blurred (variance of Laplacian dramatically reduced)
    import cv2
    orig_var = cv2.Laplacian(img[20:60, 100:150], cv2.CV_64F).var()
    blur_var = cv2.Laplacian(masked[20:60, 100:150], cv2.CV_64F).var()
    assert blur_var < orig_var

# ---------------------------------------------------------------------------
# SECTION 11: FAILURE SAFETY & INPUT RESILIENCE
# ---------------------------------------------------------------------------

def test_failure_safety_corrupted_and_extreme_inputs():
    from sixth_sense.core.quality_gate import QualityGate
    gate = QualityGate()

    # 1. Pitch black / degenerate frame
    black_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    q_black = gate.assess(black_frame)
    assert not q_black.is_usable, 'Empty frame must fail quality assessment'

    # 2. Unknown class name in observation builder
    cfg = {'observation': {'min_detections': 3, 'max_gap_frames': 10, 'min_spatial_overlap_iou': 0.3}}
    builder = ObservationBuilder(profile_cfg=cfg)
    det_unknown = Detection(
        det_id='d_unk', frame_idx=1, timestamp=0.1, event_type=EventType.UNKNOWN,
        class_name='UNKNOWN_CLASS_XYZ', classification_source=ClassificationSource.DETECTED,
        raw_confidence=0.8, confidence=0.8, bbox=(100, 100, 200, 200),
        bbox_area_px=10000.0, relative_area=0.01, frame_width=1920, frame_height=1080,
        gps=None, quality=None, model_name='test'
    )
    # Must not throw an unhandled crash
    builder.ingest([det_unknown])
    obs = builder.finalise()
    assert isinstance(obs, list)

    # 3. Missing GPS fix -> Observation safely retains gps=None or UNAVAILABLE
    for f in range(2, 5):
        d_no_gps = Detection(
            det_id=f'd_{f}', frame_idx=f, timestamp=f*0.1, event_type=EventType.POTHOLE,
            class_name='D40', classification_source=ClassificationSource.DETECTED,
            raw_confidence=0.85, confidence=0.85, bbox=(100, 100, 200, 200),
            bbox_area_px=10000.0, relative_area=0.01, frame_width=1920, frame_height=1080,
            gps=None, quality=None, model_name='test'
        )
        builder.ingest([d_no_gps])
    confirmed = builder.finalise()
    if confirmed:
        assert confirmed[0].gps is None

    # 4. Priority engine resilience with negative/NaN-like extreme inputs
    priority_engine = PriorityEngineV2()
    issue = PersistentIssue(
        issue_id='iss_edge', event_type=EventType.POTHOLE, class_name='D40',
        severity=SeverityTier.UNKNOWN, confidence=0.0, observation_count=0,
        bus_count=0, bus_ids=[], first_seen_ts=0.0, last_seen_ts=0.0
    )
    score_edge = priority_engine.score(issue, traffic_exposure_score=0.0, traffic_congestion_state='UNKNOWN', safety_risk_exposure=0.0)
    assert 0.0 <= score_edge.priority_score <= 100.0
