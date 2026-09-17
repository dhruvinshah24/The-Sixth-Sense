"""
Cross-Domain Fusion Engine — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Multi-domain urban intelligence fusion bridging:
  - ROAD: Potholes, Cracks, Waterlogging, Hazards (Person 1)
  - TRAFFIC: Congestion, Bottlenecks, Flow Density (Person 2)
  - SAFETY: VRU Conflict, Pedestrian Density, School Zones (Future Person 3)
  - INCIDENT: Collision Candidates, Blockages (Future Person 4)

Provides an extensible plug-and-play event ingestion interface so teammate
modules can connect directly without altering the core pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sixth_sense.schemas.unified_event import DomainType, UnifiedObservation

logger = logging.getLogger(__name__)


@dataclass
class FusedSegmentContext:
    """Multi-domain fused situational context for a specific road corridor."""
    segment_id: str
    time_window: Dict[str, float]
    primary_domain: DomainType
    cross_domain_synergy: str          # e.g., 'COMPOUND_HAZARD', 'SAFETY_CRITICAL'
    road_defects: List[Dict[str, Any]]
    traffic_state: Dict[str, Any]
    safety_risks: List[Dict[str, Any]]
    incidents: List[Dict[str, Any]]
    urgency_multiplier: float          # 1.0 – 2.0 (boosts maintenance priority)
    explainable_synthesis: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "time_window": self.time_window,
            "primary_domain": self.primary_domain.value,
            "cross_domain_synergy": self.cross_domain_synergy,
            "road_defects_count": len(self.road_defects),
            "traffic_state": self.traffic_state,
            "safety_risks_count": len(self.safety_risks),
            "incidents_count": len(self.incidents),
            "urgency_multiplier": round(self.urgency_multiplier, 2),
            "explainable_synthesis": self.explainable_synthesis,
        }


class CrossDomainFusionEngine:
    """
    Fuses multi-modal observations along common spatial segments.
    """

    def fuse_segment_events(
        self,
        segment_id: str,
        unified_events: List[UnifiedObservation],
        traffic_exposure: float = 0.0,
        traffic_congestion_state: str = "NORMAL_FLOW",
    ) -> FusedSegmentContext:
        """
        Synthesize cross-domain events for a specific road segment.
        """
        synthesis: List[str] = []
        multiplier = 1.0

        road_obs: List[Dict[str, Any]] = []
        traffic_obs: List[Dict[str, Any]] = []
        safety_obs: List[Dict[str, Any]] = []
        incident_obs: List[Dict[str, Any]] = []

        timestamps: List[float] = []

        for ev in unified_events:
            timestamps.append(ev.timestamp)
            ev_dict = ev.to_dict()
            if ev.domain == DomainType.ROAD:
                road_obs.append(ev_dict)
            elif ev.domain == DomainType.TRAFFIC:
                traffic_obs.append(ev_dict)
            elif ev.domain == DomainType.SAFETY:
                safety_obs.append(ev_dict)
            elif ev.domain == DomainType.INCIDENT:
                incident_obs.append(ev_dict)

        t_min = min(timestamps) if timestamps else 0.0
        t_max = max(timestamps) if timestamps else 0.0

        # Cross-Domain Rules & Compound Multipliers
        has_critical_road_defect = any(
            r.get("severity") in ("CRITICAL", "HIGH") or "POTHOLE" in r.get("event_type", "").upper()
            for r in road_obs
        )
        is_heavy_traffic = (
            traffic_congestion_state in ("CONGESTION", "PERSISTENT_BOTTLENECK", "SEVERE")
            or traffic_exposure > 0.6
        )
        has_waterlogging = any("WATERLOGGING" in r.get("event_type", "").upper() for r in road_obs)
        has_pedestrian_exposure = len(safety_obs) > 0 or any("PEDESTRIAN" in str(r.get("event_type", "")).upper() for r in road_obs)

        synergy = "STANDARD_MONITORING"

        # Synergy 1: Critical Road Defect + Heavy Traffic
        if has_critical_road_defect and is_heavy_traffic:
            multiplier += 0.40
            synergy = "COMPOUND_INFRASTRUCTURE_TRAFFIC_STRESS"
            synthesis.append(
                "Compound Risk: Critical road defect in congested transit corridor; "
                "causes vehicle slowdowns, rapid pothole expansion, and high repair urgency."
            )

        # Synergy 2: Road Defect / Waterlogging + Pedestrian / Safety Risk
        if (has_critical_road_defect or has_waterlogging) and has_pedestrian_exposure:
            multiplier += 0.35
            synergy = "SAFETY_CRITICAL_CORRIDOR"
            synthesis.append(
                "Safety Alert: Surface defect/waterlogging in pedestrian zone creates "
                "immediate vehicle avoidance swerving hazard near vulnerable road users."
            )

        # Synergy 3: Waterlogging + Traffic Bottleneck
        if has_waterlogging and is_heavy_traffic:
            multiplier += 0.30
            synthesis.append(
                "Drainage/Flow Interaction: Standing water accumulation reducing carriageway capacity, "
                "causing localized bottlenecks."
            )

        # Standalone observations
        if not synthesis:
            if road_obs:
                synthesis.append(f"{len(road_obs)} road infrastructure observations recorded.")
            if is_heavy_traffic:
                synthesis.append(f"Elevated traffic density ({traffic_congestion_state}).")
            if not road_obs and not is_heavy_traffic:
                synthesis.append("Normal road and traffic conditions.")

        primary_domain = DomainType.ROAD if len(road_obs) >= len(traffic_obs) else DomainType.TRAFFIC

        return FusedSegmentContext(
            segment_id=segment_id,
            time_window={"start": round(t_min, 2), "end": round(t_max, 2), "duration": round(t_max - t_min, 2)},
            primary_domain=primary_domain,
            cross_domain_synergy=synergy,
            road_defects=road_obs,
            traffic_state={"congestion_state": traffic_congestion_state, "exposure_score": traffic_exposure},
            safety_risks=safety_obs,
            incidents=incident_obs,
            urgency_multiplier=min(2.0, round(multiplier, 2)),
            explainable_synthesis=synthesis,
        )
