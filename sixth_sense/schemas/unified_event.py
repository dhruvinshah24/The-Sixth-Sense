"""
Unified Event Schema — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Standardized, domain-agnostic observation and event schema bridging:
  - PERSON 1 (Road & Infrastructure)
  - PERSON 2 (Traffic & Mobility)
  - Future PERSON 3 (Vulnerable Road User / Safety)
  - Future PERSON 4 (Incident & Enforcement)

Full backward compatibility with sixth_sense.schemas.urban_event.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sixth_sense.schemas.urban_event import (
    EventType,
    GPSPoint,
    Observation,
    PersistentIssue,
    SeverityTier,
)


class DomainType(str, Enum):
    ROAD = "ROAD"
    TRAFFIC = "TRAFFIC"
    SAFETY = "SAFETY"
    INCIDENT = "INCIDENT"


@dataclass
class UnifiedObservation:
    """
    Common observation schema ingested across all urban sensing domains.
    Designed so future Persons 3 and 4 can plug into our intelligence layer.
    """
    observation_id: str
    bus_id: str
    timestamp: float
    location: Dict[str, Any]
    domain: DomainType
    event_type: str
    confidence: float
    severity: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    privacy_status: str = "ANONYMIZED"

    @staticmethod
    def make_id() -> str:
        return f"uobs_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "bus_id": self.bus_id,
            "timestamp": round(self.timestamp, 3),
            "location": self.location,
            "domain": self.domain.value,
            "event_type": self.event_type,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "privacy_status": self.privacy_status,
        }

    @classmethod
    def from_observation(
        cls,
        obs: Observation,
        road_segment_id: Optional[str] = None,
        domain: Optional[DomainType] = None,
    ) -> UnifiedObservation:
        """Convert an existing Observation object to UnifiedObservation."""
        # Determine domain if not explicitly provided
        if domain is None:
            et = obs.event_type.value if hasattr(obs.event_type, "value") else str(obs.event_type)
            if et in ("VEHICLE", "CONGESTION"):
                domain = DomainType.TRAFFIC
            elif et in ("PEDESTRIAN", "CYCLIST"):
                domain = DomainType.SAFETY
            elif et in ("INCIDENT_CANDIDATE",):
                domain = DomainType.INCIDENT
            else:
                domain = DomainType.ROAD

        loc = {
            "lat": obs.gps.lat if obs.gps else None,
            "lon": obs.gps.lon if obs.gps else None,
            "road_segment_id": road_segment_id,
            "uncertainty_m": obs.gps.uncertainty_m if obs.gps else None,
            "status": obs.gps.status.value if obs.gps else "UNAVAILABLE",
        }

        ev = {
            "evidence_ref": obs.evidence_ref,
            "bbox": list(obs.bbox),
            "bbox_area_px": obs.bbox_area_px,
            "relative_area": obs.relative_area,
            "representative_frame": obs.representative_frame,
            "detection_count": obs.detection_count,
        }

        prov = {
            "model_name": obs.model_name,
            "camera_id": obs.camera_id,
            "run_id": obs.run_id,
            "first_seen_frame": obs.first_seen_frame,
            "last_seen_frame": obs.last_seen_frame,
        }

        return cls(
            observation_id=obs.obs_id,
            bus_id=obs.bus_id,
            timestamp=obs.first_seen_ts,
            location=loc,
            domain=domain,
            event_type=obs.class_name or obs.event_type.value,
            confidence=obs.confidence,
            severity=obs.severity.value if hasattr(obs.severity, "value") else str(obs.severity),
            evidence=ev,
            provenance=prov,
            privacy_status=obs.privacy_status,
        )


def create_safety_observation(
    bus_id: str,
    timestamp: float,
    lat: float,
    lon: float,
    event_type: str = "PEDESTRIAN",
    confidence: float = 0.85,
    severity: str = "HIGH",
    road_segment_id: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    obs_id: Optional[str] = None,
) -> UnifiedObservation:
    """
    Standard factory for Person 3 (Vulnerable Road User / Safety) observations.
    Bridges pedestrian/cyclist detections into the Unified Intelligence pipeline.
    """
    return UnifiedObservation(
        observation_id=obs_id or UnifiedObservation.make_id(),
        bus_id=bus_id,
        timestamp=timestamp,
        location={
            "lat": lat,
            "lon": lon,
            "road_segment_id": road_segment_id,
            "status": "VALID",
        },
        domain=DomainType.SAFETY,
        event_type=event_type,
        confidence=confidence,
        severity=severity,
        evidence=evidence or {"vru_type": event_type, "corridor": road_segment_id},
        provenance=provenance or {"source": "PERSON_3_SAFETY_MODULE", "model": "yolov8_vru_detector"},
        privacy_status="ANONYMIZED",
    )


def create_incident_observation(
    bus_id: str,
    timestamp: float,
    lat: float,
    lon: float,
    event_type: str = "INCIDENT_CANDIDATE",
    confidence: float = 0.88,
    severity: str = "CRITICAL",
    road_segment_id: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    obs_id: Optional[str] = None,
) -> UnifiedObservation:
    """
    Standard factory for Person 4 (Incident & Enforcement) candidate observations.
    Bridges collisions, breakdowns, and blockages with mandatory human review provenance.
    """
    return UnifiedObservation(
        observation_id=obs_id or UnifiedObservation.make_id(),
        bus_id=bus_id,
        timestamp=timestamp,
        location={
            "lat": lat,
            "lon": lon,
            "road_segment_id": road_segment_id,
            "status": "VALID",
        },
        domain=DomainType.INCIDENT,
        event_type=event_type,
        confidence=confidence,
        severity=severity,
        evidence=evidence or {
            "incident_type": event_type,
            "requires_human_review": True,
            "corridor": road_segment_id,
        },
        provenance=provenance or {
            "source": "PERSON_4_INCIDENT_MODULE",
            "model": "incident_detector_v1",
            "enforcement_safeguard": "MANDATORY_OFFICER_REVIEW",
        },
        privacy_status="ANONYMIZED",
    )

