"""
Team Integration & Multi-Domain Pipeline Tests — The Sixth Sense
SIH 2026 PS 26124 / PS 26125

Verifies integration across:
  - Person 1 (Road & Infrastructure)
  - Person 2 (Traffic & Mobility)
  - Person 3 (Vulnerable Road Users & Safety)
  - Person 4 (Incident & Enforcement Safeguards)
  - Person 5 (Platform Telemetry, Anonymization, Audit & Work-Orders)
  - Person 6 (GIS & Command Center Server Endpoints)
"""
from __future__ import annotations

import os
import json
import pytest
from pathlib import Path

from sixth_sense.schemas.unified_event import (
    DomainType,
    UnifiedObservation,
    create_safety_observation,
    create_incident_observation,
)
from sixth_sense.schemas.urban_event import (
    EventType,
    GPSPoint,
    GPSStatus,
    Observation,
    SeverityTier,
)
from sixth_sense.intelligence.cross_domain_fusion import (
    CrossDomainFusionEngine,
    FusedSegmentContext,
)
from sixth_sense.pipeline.urban_intelligence_pipeline import UrbanIntelligencePipeline
from sixth_sense.actionable.priority_engine_v2 import PriorityEngineV2
from run_full_team_integration_demo import run_full_integration
from serve_command_center import CommandCenterHandler, PROJECT_ROOT, URBAN_INTEL_ROOT


def test_person_3_safety_observation_factory():
    """Verify Person 3 safety observation factory creates compliant schema."""
    obs = create_safety_observation(
        bus_id="BUS_VRU_01",
        timestamp=1000.0,
        lat=19.0760,
        lon=72.8777,
        event_type="PEDESTRIAN",
        confidence=0.88,
        severity="HIGH",
        road_segment_id="SEG_TEST_VRU",
    )
    assert obs.domain == DomainType.SAFETY
    assert obs.event_type == "PEDESTRIAN"
    assert obs.confidence == 0.88
    assert obs.location["road_segment_id"] == "SEG_TEST_VRU"
    assert obs.privacy_status == "ANONYMIZED"
    assert obs.provenance["source"] == "PERSON_3_SAFETY_MODULE"


def test_person_4_incident_observation_factory():
    """Verify Person 4 incident observation factory sets mandatory human review safeguard."""
    obs = create_incident_observation(
        bus_id="BUS_INC_01",
        timestamp=1005.0,
        lat=19.0765,
        lon=72.8780,
        event_type="VEHICLE_COLLISION_CANDIDATE",
        confidence=0.92,
        severity="CRITICAL",
        road_segment_id="SEG_TEST_INC",
    )
    assert obs.domain == DomainType.INCIDENT
    assert obs.event_type == "VEHICLE_COLLISION_CANDIDATE"
    assert obs.confidence == 0.92
    assert obs.evidence.get("requires_human_review") is True
    assert obs.provenance.get("enforcement_safeguard") == "MANDATORY_OFFICER_REVIEW"


def test_cross_domain_synergy_with_all_four_domains():
    """Verify CrossDomainFusionEngine correctly combines Road, Traffic, Safety, and Incident."""
    fusion = CrossDomainFusionEngine(
        heavy_traffic_defect_multiplier=0.40,
        safety_pedestrian_multiplier=0.35,
        incident_blockage_multiplier=0.45,
    )

    road_ev = UnifiedObservation(
        observation_id="u_road_01",
        bus_id="BUS_01",
        timestamp=100.0,
        location={"lat": 19.076, "lon": 72.877, "road_segment_id": "SEG_CORRIDOR_MULTI"},
        domain=DomainType.ROAD,
        event_type="POTHOLE",
        confidence=0.90,
        severity="HIGH",
    )

    traffic_ev = UnifiedObservation(
        observation_id="u_traf_01",
        bus_id="BUS_01",
        timestamp=102.0,
        location={"lat": 19.076, "lon": 72.877, "road_segment_id": "SEG_CORRIDOR_MULTI"},
        domain=DomainType.TRAFFIC,
        event_type="vehicle",
        confidence=0.88,
        severity="UNKNOWN",
    )

    safety_ev = create_safety_observation(
        bus_id="BUS_01",
        timestamp=104.0,
        lat=19.076,
        lon=72.877,
        event_type="PEDESTRIAN",
        road_segment_id="SEG_CORRIDOR_MULTI",
    )

    incident_ev = create_incident_observation(
        bus_id="BUS_01",
        timestamp=105.0,
        lat=19.076,
        lon=72.877,
        event_type="OBSTRUCTION_CANDIDATE",
        road_segment_id="SEG_CORRIDOR_MULTI",
    )

    fused = fusion.fuse_segment_events(
        segment_id="SEG_CORRIDOR_MULTI",
        unified_events=[road_ev, traffic_ev, safety_ev, incident_ev],
        traffic_exposure=0.80,
        traffic_congestion_state="CONGESTION",
    )

    assert fused.urgency_multiplier >= 1.70
    assert len(fused.road_defects) == 1
    assert len(fused.safety_risks) == 1
    assert len(fused.incidents) == 1
    assert len(fused.explainable_synthesis) >= 2


def test_command_center_urban_intelligence_path_translation():
    """Verify serve_command_center.py translates /urban-intelligence/ paths accurately."""
    handler = CommandCenterHandler.__new__(CommandCenterHandler)
    translated = handler.translate_path("/urban-intelligence/final_summary.json")
    expected = (URBAN_INTEL_ROOT / "final_summary.json").resolve()
    assert Path(translated).resolve() == expected


def test_end_to_end_full_team_integration_scenario(tmp_path):
    """Verify run_full_integration executes all 14 steps and produces valid artifacts."""
    out_dir = str(tmp_path / "test_urban_intel")
    summary = run_full_integration(output_dir=out_dir)

    assert summary["integration_status"] == "COMPLETE_AND_VERIFIED"
    assert len(summary["workstreams_integrated"]) == 6
    assert summary["metrics"]["total_raw_observations"] == 2
    assert summary["metrics"]["total_persistent_issues"] == 1
    assert summary["metrics"]["multi_bus_corroborated_issues"] == 1
    assert summary["metrics"]["verification_result"] == "VERIFIED_REPAIRED"

    for artifact_name, artifact_path in summary["artifacts"].items():
        assert os.path.exists(artifact_path)
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data is not None
