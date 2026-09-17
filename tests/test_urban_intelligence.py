"""
Comprehensive Unit & Integration Tests for Urban Intelligence Layer
The Sixth Sense — SIH 2026 PS 26124 / PS 26125
Person 1 (Road Infrastructure) + Person 2 (Traffic Mobility) Intelligence.
"""
import os
import pytest
from typing import Optional

from sixth_sense.schemas.urban_event import (
    EventType,
    Observation,
    SeverityTier,
    GPSPoint,
    GPSStatus,
    PersistentIssue,
    IssueTrend,
)
from sixth_sense.schemas.unified_event import DomainType, UnifiedObservation
from sixth_sense.intelligence import (
    RoadHealthEngine,
    RoadSegmentHealth,
    RoadHealthState,
    RoadTrend,
    apply_closure_to_road_health,
    CrossDomainFusionEngine,
    FusedSegmentContext,
    ConfidenceAutomationGovernor,
    AutomationTier,
    GovernanceAction,
    AutomationDecision,
)
from sixth_sense.events.city_memory import CityMemory
from sixth_sense.traffic.traffic_state_engine import (
    TrafficStateEngine,
    SegmentTrafficState,
    CongestionClassification,
)
from sixth_sense.actionable.priority_engine_v2 import PriorityEngineV2
from sixth_sense.closure.verification_engine import VerificationResult, VerificationOutcome


def _make_gps(
    lat: float = 19.0760,
    lon: float = 72.8777,
    ts: float = 1000.0,
    uncertainty: float = 5.0,
    heading: Optional[float] = 90.0,
    status: GPSStatus = GPSStatus.INTERPOLATED,
) -> GPSPoint:
    return GPSPoint(
        lat=lat,
        lon=lon,
        timestamp=ts,
        uncertainty_m=uncertainty,
        heading=heading,
        status=status,
    )


def _create_test_obs(
    obs_id: str,
    bus_id: str = "BUS_01",
    event_type: EventType = EventType.POTHOLE,
    class_name: str = "D40",
    confidence: float = 0.85,
    severity: SeverityTier = SeverityTier.HIGH,
    lat: float = 19.0760,
    lon: float = 72.8777,
    first_seen_ts: float = 10.0,
) -> Observation:
    """Helper to generate a valid typed Observation for tests."""
    return Observation(
        obs_id=obs_id,
        bus_id=bus_id,
        camera_id="CAM_FRONT",
        run_id=f"run_{bus_id}",
        event_type=event_type,
        class_name=class_name,
        first_seen_frame=100,
        last_seen_frame=115,
        first_seen_ts=first_seen_ts,
        last_seen_ts=first_seen_ts + 0.5,
        representative_frame=108,
        confidence=confidence,
        severity=severity,
        bbox=(200, 300, 400, 500),
        bbox_area_px=40000,
        relative_area=0.02,
        gps=_make_gps(lat=lat, lon=lon, ts=first_seen_ts),
        evidence_ref=None,
        detection_count=10,
        model_name="test_model",
    )


# -------------------------------------------------------------------------- #
# 1. Unified Event Schema & Conversion Tests
# -------------------------------------------------------------------------- #

def test_unified_observation_conversion_road():
    """Verify legacy road Observation converts cleanly to UnifiedObservation."""
    obs = _create_test_obs(
        obs_id="obs_test_road_01",
        bus_id="BUS_101",
        event_type=EventType.POTHOLE,
        class_name="D40",
        confidence=0.88,
        severity=SeverityTier.HIGH,
        lat=19.0760,
        lon=72.8777,
    )

    unified = UnifiedObservation.from_observation(obs, road_segment_id="SEG_BANDRA_01")
    assert unified.domain == DomainType.ROAD
    assert unified.observation_id == "obs_test_road_01"
    assert unified.bus_id == "BUS_101"
    assert unified.event_type == "D40"
    assert unified.confidence == 0.88
    assert unified.severity == SeverityTier.HIGH.value
    assert unified.location["road_segment_id"] == "SEG_BANDRA_01"
    assert unified.location["lat"] == 19.0760


def test_unified_observation_conversion_traffic():
    """Verify legacy traffic Observation maps to DomainType.TRAFFIC."""
    obs = _create_test_obs(
        obs_id="obs_test_traffic_01",
        bus_id="BUS_202",
        event_type=EventType.VEHICLE,
        class_name="bus",
        confidence=0.92,
        severity=SeverityTier.MEDIUM,
    )

    unified = UnifiedObservation.from_observation(obs, road_segment_id="SEG_BANDRA_01")
    assert unified.domain == DomainType.TRAFFIC
    assert unified.event_type == "bus"
    assert unified.confidence == 0.92


# -------------------------------------------------------------------------- #
# 2. Road Health Intelligence Engine Tests
# -------------------------------------------------------------------------- #

def test_road_health_healthy_segment():
    """Pristine segment with zero defects has 100 score, HEALTHY state."""
    engine = RoadHealthEngine()
    result = engine.evaluate_segment("SEG_CLEAN_01", observations=[])
    assert result.health_score == 100.0
    assert result.health_state == RoadHealthState.HEALTHY
    assert result.trend == RoadTrend.INSUFFICIENT_HISTORY
    assert result.active_issues_count == 0


def test_road_health_defect_penalties_and_degradation():
    """Severe defects deduct points and degrade road state."""
    engine = RoadHealthEngine()

    obs1 = _create_test_obs("obs_ph_01", bus_id="BUS_1", event_type=EventType.POTHOLE, class_name="D40", severity=SeverityTier.CRITICAL)
    obs2 = _create_test_obs("obs_cr_01", bus_id="BUS_2", event_type=EventType.ROAD_CRACK, class_name="D20", severity=SeverityTier.HIGH)

    result = engine.evaluate_segment("SEG_ROUGH_01", observations=[obs1, obs2])
    assert result.health_score < 70.0
    assert result.health_state in (RoadHealthState.WATCH, RoadHealthState.DEGRADED)
    assert result.bus_count == 2
    assert "D40" in result.defect_breakdown or "POTHOLE" in result.defect_breakdown


def test_road_health_traffic_stress_interaction():
    """Heavy traffic and persistent bottleneck exacerbate road health deduction."""
    engine = RoadHealthEngine()

    obs = _create_test_obs("obs_ph_02", bus_id="BUS_1", event_type=EventType.POTHOLE, class_name="D40", severity=SeverityTier.HIGH)

    eval_normal = engine.evaluate_segment("SEG_BUSY_01", [obs], traffic_congestion_state="NORMAL_FLOW")
    eval_congested = engine.evaluate_segment("SEG_BUSY_01", [obs], traffic_congestion_state="CONGESTION")

    assert eval_congested.health_score < eval_normal.health_score
    assert "traffic_stress_penalty" in eval_congested.scoring_factors


def test_road_health_trend_requires_history():
    """Empirical trend rule: single bus pass or sparse data returns INSUFFICIENT_HISTORY."""
    engine = RoadHealthEngine()

    obs_single = [_create_test_obs("obs_01", bus_id="BUS_1", event_type=EventType.POTHOLE, class_name="D40", confidence=0.90)]

    res = engine.evaluate_segment("SEG_NEW_01", obs_single)
    assert res.trend == RoadTrend.INSUFFICIENT_HISTORY


# -------------------------------------------------------------------------- #
# 3. City Memory & Multi-Bus Deduplication Tests
# -------------------------------------------------------------------------- #

def test_city_memory_spatial_fusion_and_deduplication():
    """Two buses observing same defect at same GPS should update existing issue without duplicates."""
    memory = CityMemory(dedup_radius_m=30.0)

    obs_bus1 = _create_test_obs(
        obs_id="obs_b1_01",
        bus_id="BUS_A",
        event_type=EventType.POTHOLE,
        class_name="D40",
        confidence=0.75,
        lat=19.07600,
        lon=72.87770,
        first_seen_ts=1000.0,
    )

    # Bus 2 passes 20 minutes later, GPS coordinates within 5 meters
    obs_bus2 = _create_test_obs(
        obs_id="obs_b2_01",
        bus_id="BUS_B",
        event_type=EventType.POTHOLE,
        class_name="D40",
        confidence=0.88,
        severity=SeverityTier.CRITICAL,
        lat=19.07604,
        lon=72.87773,
        first_seen_ts=2200.0,
    )

    issue1 = memory.ingest_observation(obs_bus1, road_segment_id="SEG_LINK_ROAD")
    assert len(memory.get_all_issues()) == 1
    assert issue1.bus_count == 1

    issue2 = memory.ingest_observation(obs_bus2, road_segment_id="SEG_LINK_ROAD")
    # Deduplication check: should NOT create a second issue
    assert len(memory.get_all_issues()) == 1
    assert issue2.issue_id == issue1.issue_id
    assert issue2.bus_count == 2
    assert "BUS_A" in issue2.bus_ids and "BUS_B" in issue2.bus_ids
    assert issue2.confidence >= 0.88


# -------------------------------------------------------------------------- #
# 4. Traffic State Engine Tests (Temporary vs. Bottleneck)
# -------------------------------------------------------------------------- #

def test_traffic_state_engine_temporary_vs_recurring_vs_bottleneck():
    """Engine classifies temporary delay vs. recurring congestion vs. persistent bottleneck."""
    engine = TrafficStateEngine(window_seconds=60.0)
    seg_id = "SEG_JVLR_01"

    # 1. Single window with 9 vehicles (HIGH density, CONGESTION) -> TEMPORARY_CONGESTION
    obs_single_window = [
        {"obs_id": f"v_w0_{i}", "event_type": "VEHICLE", "class_name": "car", "first_seen_ts": 5.0 + i}
        for i in range(9)
    ]
    st1 = engine.evaluate_segment_traffic(seg_id, obs_single_window)
    assert st1.classification == CongestionClassification.TEMPORARY_CONGESTION

    # 2. Non-consecutive windows with HIGH density -> RECURRING_CONGESTION
    obs_non_consecutive = list(obs_single_window)
    # Window 2 is at ts 120s..130s (skipping Window 1 at 60s)
    obs_non_consecutive.extend([
        {"obs_id": f"v_w2_{i}", "event_type": "VEHICLE", "class_name": "car", "first_seen_ts": 125.0 + i}
        for i in range(9)
    ])
    st2 = engine.evaluate_segment_traffic(seg_id, obs_non_consecutive)
    assert st2.classification == CongestionClassification.RECURRING_CONGESTION

    # 3. Consecutive windows with HIGH density -> PERSISTENT_BOTTLENECK
    obs_consecutive = list(obs_single_window)
    # Window 1 is at ts 65s..75s (immediately consecutive to Window 0)
    obs_consecutive.extend([
        {"obs_id": f"v_w1_{i}", "event_type": "VEHICLE", "class_name": "bus", "first_seen_ts": 65.0 + i}
        for i in range(9)
    ])
    st3 = engine.evaluate_segment_traffic(seg_id, obs_consecutive)
    assert st3.classification == CongestionClassification.PERSISTENT_BOTTLENECK


# -------------------------------------------------------------------------- #
# 5. Cross-Domain Fusion Engine Tests
# -------------------------------------------------------------------------- #

def test_cross_domain_fusion_compound_hazard_and_safety_risk():
    """Synergy detection: Pothole + Congestion, Defect + Pedestrian risk."""
    fusion_engine = CrossDomainFusionEngine()

    pothole_ev = UnifiedObservation(
        observation_id="ev_01",
        bus_id="BUS_5",
        timestamp=100.0,
        location={"lat": 19.01, "lon": 72.85, "road_segment_id": "SEG_MARKET_ROAD"},
        domain=DomainType.ROAD,
        event_type="POTHOLE",
        confidence=0.85,
        severity="CRITICAL",
    )

    pedestrian_ev = UnifiedObservation(
        observation_id="ev_02",
        bus_id="BUS_5",
        timestamp=102.0,
        location={"lat": 19.01, "lon": 72.85, "road_segment_id": "SEG_MARKET_ROAD"},
        domain=DomainType.SAFETY,
        event_type="PEDESTRIAN_DENSITY",
        confidence=0.82,
        severity="HIGH",
    )

    # Cross-domain fusion with heavy traffic
    context = fusion_engine.fuse_segment_events(
        segment_id="SEG_MARKET_ROAD",
        unified_events=[pothole_ev, pedestrian_ev],
        traffic_congestion_state="PERSISTENT_BOTTLENECK",
        traffic_exposure=0.85,
    )

    assert context.urgency_multiplier > 1.30
    assert len(context.explainable_synthesis) >= 1
    assert any("Compound" in s or "Safety" in s for s in context.explainable_synthesis)


# -------------------------------------------------------------------------- #
# 6. Priority Engine V2 Tests (Explainability & Multi-Factor)
# -------------------------------------------------------------------------- #

def test_priority_engine_v2_factor_breakdown():
    """Priority Engine V2 calculates explainable score with clear component weights."""
    engine = PriorityEngineV2()

    issue = PersistentIssue(
        issue_id="ISSUE_CRITICAL_01",
        event_type=EventType.POTHOLE,
        class_name="D40",
        first_seen_ts=100.0,
        last_seen_ts=1500.0,
        observation_count=5,
        bus_count=3,
        bus_ids=["B1", "B2", "B3"],
        severity=SeverityTier.CRITICAL,
        confidence=0.92,
        center_gps=_make_gps(lat=19.0760, lon=72.8777),
        road_segment="SEG_HIGHWAY_01",
    )

    priority_result = engine.score(
        issue=issue,
        traffic_exposure_score=0.90,
        traffic_congestion_state="PERSISTENT_BOTTLENECK",
        safety_risk_exposure=0.80,
    )

    assert priority_result.priority_score > 75.0
    assert "severity_pts" in priority_result.score_breakdown
    assert "corroboration_pts" in priority_result.score_breakdown
    assert "traffic_exposure_pts" in priority_result.score_breakdown
    assert len(priority_result.reasons) >= 2


# -------------------------------------------------------------------------- #
# 7. Proof-of-Closure Road Health Feedback Tests
# -------------------------------------------------------------------------- #

def test_proof_of_closure_road_health_feedback():
    """Verified repaired issue restores health; reopened issue penalizes score."""
    engine = RoadHealthEngine()

    issue = PersistentIssue(
        issue_id="ISS_REPAIR_01",
        event_type=EventType.POTHOLE,
        class_name="D40",
        first_seen_ts=100.0,
        last_seen_ts=200.0,
        observation_count=3,
        bus_count=2,
        bus_ids=["BUS_1", "BUS_2"],
        severity=SeverityTier.HIGH,
        confidence=0.85,
        center_gps=_make_gps(lat=19.05, lon=72.84),
        road_segment="SEG_REPAIR_TEST",
    )

    prior_health = engine.evaluate_segment("SEG_REPAIR_TEST", observations=[], persistent_issues=[issue])
    initial_score = prior_health.health_score

    # Case A: Repaired verification outcome
    res_repaired = VerificationResult(
        verification_id="verif_01",
        issue_id="ISS_REPAIR_01",
        claim_id="claim_01",
        verification_result=VerificationOutcome.VERIFIED_REPAIRED,
        verification_confidence=0.95,
        reasons=["Repair patch observed, no remaining defect"],
        verification_basis="follow_up_pass",
    )

    improved_health = apply_closure_to_road_health(prior_health, issue, res_repaired)
    assert improved_health.health_score > initial_score
    assert improved_health.trend == RoadTrend.IMPROVING
    assert improved_health.active_issues_count == 0

    # Case B: Reopened verification outcome (failed repair)
    res_reopened = VerificationResult(
        verification_id="verif_02",
        issue_id="ISS_REPAIR_01",
        claim_id="claim_01",
        verification_result=VerificationOutcome.REOPENED,
        verification_confidence=0.88,
        reasons=["Defect still detected after claimed repair"],
        verification_basis="follow_up_pass",
    )

    worsened_health = apply_closure_to_road_health(prior_health, issue, res_reopened)
    assert worsened_health.health_score < initial_score
    assert worsened_health.trend == RoadTrend.WORSENING


# -------------------------------------------------------------------------- #
# 8. Confidence-Aware Automation & Governance Tests
# -------------------------------------------------------------------------- #

def test_confidence_automation_governor_tiers():
    """Tiers enforce automated vs human approval rules strictly."""
    gov = ConfidenceAutomationGovernor(low_threshold=0.40, high_threshold=0.65)

    base_issue = PersistentIssue(
        issue_id="ISS_01",
        event_type=EventType.ROAD_CRACK,
        class_name="D00",
        first_seen_ts=10.0,
        last_seen_ts=10.0,
        observation_count=1,
        bus_count=1,
        bus_ids=["BUS_1"],
        severity=SeverityTier.LOW,
        confidence=0.30,  # Low
        center_gps=_make_gps(lat=19.0, lon=72.8),
    )

    # 1. Low confidence -> LOG_AND_GROUP, no auto dispatch
    d1 = gov.evaluate_issue(base_issue)
    assert d1.tier == AutomationTier.LOG_AND_GROUP
    assert d1.auto_dispatch_permitted is False

    # 2. Medium confidence -> REVIEW_REQUIRED
    base_issue.confidence = 0.55
    d2 = gov.evaluate_issue(base_issue)
    assert d2.tier == AutomationTier.REVIEW_REQUIRED
    assert d2.requires_human_signoff is True

    # 3. High confidence, single pass -> AWAITING_FLEET_CORROBORATION
    base_issue.confidence = 0.85
    base_issue.bus_count = 1
    d3 = gov.evaluate_issue(base_issue)
    assert d3.tier == AutomationTier.AWAITING_FLEET_CORROBORATION
    assert d3.auto_dispatch_permitted is False

    # 4. High confidence, multi-bus corroborated -> ACTIONABLE_WORK_ORDER
    base_issue.bus_count = 2
    d4 = gov.evaluate_issue(base_issue)
    assert d4.tier == AutomationTier.ACTIONABLE_WORK_ORDER
    assert d4.auto_dispatch_permitted is True
    # 5. High consequence override -> mandatory human sign-off regardless of confidence/buses
    d5 = gov.evaluate_issue(base_issue, is_high_consequence=True)
    assert d5.requires_human_signoff is True
    assert d5.auto_dispatch_permitted is False
    assert d5.governance_action == GovernanceAction.HUMAN_APPROVAL_REQUIRED


# -------------------------------------------------------------------------- #
# 9. Hardening Regression Tests: Road Health Bounds & Strict Trend Rules
# -------------------------------------------------------------------------- #

def test_road_health_bounds_and_single_obs_trend():
    """Verify that score never drops below 0 even with overwhelming defects, and trend is strict."""
    engine = RoadHealthEngine()

    # 10 critical potholes
    critical_obs = [
        _create_test_obs(f"obs_crit_{i}", bus_id="BUS_1", event_type=EventType.POTHOLE, severity=SeverityTier.CRITICAL, confidence=0.99)
        for i in range(10)
    ]
    res = engine.evaluate_segment("SEG_DISASTER", observations=critical_obs, traffic_congestion_state="CONGESTION", safety_incident_count=5)
    # Must be clamped at 0.0, never negative
    assert res.health_score >= 0.0
    assert res.health_score <= 100.0
    assert res.health_state == RoadHealthState.CRITICAL
    # Single bus pass MUST report INSUFFICIENT_HISTORY, not WORSENING
    assert res.trend == RoadTrend.INSUFFICIENT_HISTORY


# -------------------------------------------------------------------------- #
# 10. Hardening Regression Tests: City Memory Defect Separation & Lifecycle
# -------------------------------------------------------------------------- #

def test_city_memory_separation_rules():
    """Verify different defect types or defects beyond 30m remain distinct issues."""
    memory = CityMemory(dedup_radius_m=30.0)

    # Defect 1: Pothole at lat: 19.0760, lon: 72.8777
    obs_pothole = _create_test_obs("obs_p1", bus_id="BUS_1", event_type=EventType.POTHOLE, class_name="D40", lat=19.0760, lon=72.8777)

    # Defect 2: Crack at SAME location
    obs_crack = _create_test_obs("obs_c1", bus_id="BUS_1", event_type=EventType.ROAD_CRACK, class_name="D00", lat=19.0760, lon=72.8777)

    # Defect 3: Another Pothole 60m away (lat +0.0006 degrees ~ 66 meters)
    obs_far_pothole = _create_test_obs("obs_p2", bus_id="BUS_1", event_type=EventType.POTHOLE, class_name="D40", lat=19.0766, lon=72.8777)

    iss1 = memory.ingest_observation(obs_pothole)
    iss2 = memory.ingest_observation(obs_crack)
    iss3 = memory.ingest_observation(obs_far_pothole)

    # Must produce 3 distinct PersistentIssues
    all_issues = memory.get_all_issues()
    assert len(all_issues) == 3
    assert len({iss1.issue_id, iss2.issue_id, iss3.issue_id}) == 3

    # Reopened issue test: set iss1 status to REOPENED and ingest new observation
    iss1.status = "REOPENED"
    obs_pothole_reopen = _create_test_obs("obs_p1_re", bus_id="BUS_2", event_type=EventType.POTHOLE, class_name="D40", lat=19.07601, lon=72.87771)
    iss_reopened_updated = memory.ingest_observation(obs_pothole_reopen)

    assert iss_reopened_updated.issue_id == iss1.issue_id
    assert iss_reopened_updated.bus_count == 2
    assert len(memory.get_all_issues()) == 3  # No duplicate created


# -------------------------------------------------------------------------- #
# 11. Hardening Regression Tests: Cross-Domain Fusion Teammate Ingestion
# -------------------------------------------------------------------------- #

def test_cross_domain_fusion_teammate_safety_and_incident():
    """Verify extensible ingestion for future Person 3 (Safety) and Person 4 (Incident)."""
    fusion = CrossDomainFusionEngine(
        safety_pedestrian_multiplier=0.40,
        incident_blockage_multiplier=0.50,
    )

    safety_ev = UnifiedObservation(
        observation_id="safe_01",
        bus_id="BUS_3",
        timestamp=200.0,
        location={"lat": 19.0, "lon": 72.8, "road_segment_id": "SEG_CORRIDOR"},
        domain=DomainType.SAFETY,
        event_type="PEDESTRIAN",
        confidence=0.89,
        severity="HIGH",
    )

    incident_ev = UnifiedObservation(
        observation_id="inc_01",
        bus_id="BUS_3",
        timestamp=205.0,
        location={"lat": 19.0, "lon": 72.8, "road_segment_id": "SEG_CORRIDOR"},
        domain=DomainType.INCIDENT,
        event_type="VEHICLE_COLLISION_CANDIDATE",
        confidence=0.91,
        severity="CRITICAL",
    )

    fused = fusion.fuse_segment_events(
        segment_id="SEG_CORRIDOR",
        unified_events=[safety_ev, incident_ev],
        traffic_exposure=0.85,
        traffic_congestion_state="CONGESTION",
    )

    assert fused.urgency_multiplier > 1.30
    assert len(fused.safety_risks) == 1
    assert len(fused.incidents) == 1
    assert any("Incident" in s for s in fused.explainable_synthesis)


# -------------------------------------------------------------------------- #
# 12. End-to-End Orchestration Pipeline Tests
# -------------------------------------------------------------------------- #

def test_urban_intelligence_pipeline_e2e(tmp_path):
    """Verify complete end-to-end UrbanIntelligencePipeline execution and artifact generation."""
    from sixth_sense.pipeline import UrbanIntelligencePipeline
    from sixth_sense.closure.repair_claim import RepairClaimBuilder
    from sixth_sense.closure.verification_engine import FollowUpPass

    pipeline = UrbanIntelligencePipeline(dedup_radius_m=30.0)

    # Ingest bus 1 pothole
    obs1 = _create_test_obs("obs_p1", bus_id="BUS_10", event_type=EventType.POTHOLE, class_name="D40", confidence=0.85)
    iss1 = pipeline.ingest_observation(obs1, road_segment_id="SEG_TEST_E2E")

    # Ingest bus 2 pothole (same location)
    obs2 = _create_test_obs("obs_p2", bus_id="BUS_20", event_type=EventType.POTHOLE, class_name="D40", confidence=0.90)
    pipeline.ingest_observation(obs2, road_segment_id="SEG_TEST_E2E")

    # Ingest traffic vehicles
    traffic_recs = [
        {"obs_id": f"v_{i}", "event_type": "VEHICLE", "class_name": "car", "first_seen_ts": 10.0 + i}
        for i in range(15)
    ]
    pipeline.ingest_traffic_records(traffic_recs, road_segment_id="SEG_TEST_E2E")

    # Process all intelligence
    summary = pipeline.process_all()
    assert summary["total_raw_observations"] == 2
    assert summary["total_persistent_issues"] == 1
    assert summary["multi_bus_corroborated_issues"] == 1
    assert "SEG_TEST_E2E" in summary["road_health_overview"]
    assert "SEG_TEST_E2E" in summary["traffic_state_overview"]

    # Test Proof-of-closure loop
    claim = RepairClaimBuilder.build(iss1.issue_id, "WO_01", "PWD_Contractor")
    follow_up = FollowUpPass(
        bus_id="BUS_30",
        pass_timestamp=500.0,
        pass_gps=_make_gps(19.0760, 72.8777),
        observations=[],
        data_provenance="SIMULATED_SCENARIO",
    )
    v_res, updated_health = pipeline.verify_repair(iss1.issue_id, claim, follow_up)
    assert v_res.verification_result == "VERIFIED_REPAIRED"
    assert updated_health is not None
    assert updated_health.trend == RoadTrend.IMPROVING

    # Test export artifacts
    out_dir = str(tmp_path / "urban_intel_export")
    paths = pipeline.export_artifacts(out_dir)
    assert len(paths) == 8
    for p in paths.values():
        assert os.path.exists(p)
