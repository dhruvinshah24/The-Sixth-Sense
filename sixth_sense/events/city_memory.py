"""
City Memory — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Temporal and spatial knowledge store maintaining persistent urban state.

Pipeline:
  Observation
  ↓
  Spatial + Temporal Fusion
  ↓
  Persistent Issue
  ↓
  Issue History & Corroboration
  ↓
  Recurrence & Trend
  ↓
  Road Segment Health

Guarantees:
- Deduplication: Multiple observations of the same physical defect do NOT create duplicate issues.
- Fleet Corroboration: Honest accounting of multiple bus passes / multi-vehicle corroboration.
- Traceability: Full history of confidence, location, severity, and status changes.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from sixth_sense.events.issue_manager import IssueManager, _haversine_m
from sixth_sense.intelligence.road_health import (
    RoadHealthEngine,
    RoadSegmentHealth,
)
from sixth_sense.schemas.urban_event import (
    GPSPoint,
    GPSStatus,
    IssueTrend,
    Observation,
    PersistentIssue,
    SeverityTier,
)

logger = logging.getLogger(__name__)


class CityMemory:
    """
    Continuous urban spatial-temporal memory.
    Ingests bus observations, updates persistent issues, tracks historical transitions,
    and maps defects to road segment health representations.
    """

    def __init__(
        self,
        dedup_radius_m: float = 20.0,
        health_engine: Optional[RoadHealthEngine] = None,
    ) -> None:
        self.issue_manager = IssueManager(dedup_radius_m=dedup_radius_m)
        self.health_engine = health_engine or RoadHealthEngine()

        # Segment associations
        self._segment_observations: Dict[str, List[Observation]] = defaultdict(list)
        self._segment_issues: Dict[str, List[str]] = defaultdict(list)  # seg_id -> [issue_ids]

        # Rich issue historical logs (issue_id -> history dict)
        self._issue_audit_history: Dict[str, Dict[str, Any]] = {}

    def ingest_observation(
        self,
        obs: Observation,
        road_segment_id: Optional[str] = None,
    ) -> PersistentIssue:
        """
        Ingest a single bus observation into City Memory.
        Fuses spatially/temporally with existing issues or spawns a new persistent issue.
        """
        # Assign default segment ID if none given
        seg_id = road_segment_id or self._infer_segment(obs)
        obs_k = obs.obs_id

        # Check duplicate observation ID within segment
        if not any(o.obs_id == obs_k for o in self._segment_observations[seg_id]):
            self._segment_observations[seg_id].append(obs)

        issue = self.issue_manager.ingest(obs)
        issue.road_segment = seg_id

        if issue.issue_id not in self._segment_issues[seg_id]:
            self._segment_issues[seg_id].append(issue.issue_id)

        # Update rich audit history for this issue
        self._record_issue_history(issue, obs)

        return issue

    def ingest_batch(
        self,
        observations: List[Observation],
        road_segment_id: Optional[str] = None,
    ) -> List[PersistentIssue]:
        """Ingest a batch of observations across one or more bus runs."""
        updated_issues = {}
        for obs in observations:
            iss = self.ingest_observation(obs, road_segment_id)
            updated_issues[iss.issue_id] = iss
        return list(updated_issues.values())

    def get_all_issues(self) -> List[PersistentIssue]:
        """Retrieve all current persistent issues."""
        return self.issue_manager.get_all_issues()

    def get_issue(self, issue_id: str) -> Optional[PersistentIssue]:
        """Retrieve a specific issue by ID."""
        for iss in self.issue_manager.get_all_issues():
            if iss.issue_id == issue_id:
                return iss
        return None

    def get_issue_history(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the complete audit history log for an issue."""
        return self._issue_audit_history.get(issue_id)

    def evaluate_segment_health(
        self,
        road_segment_id: str,
        traffic_congestion_state: str = "NORMAL_FLOW",
        safety_incident_count: int = 0,
    ) -> RoadSegmentHealth:
        """
        Compute road health and trend for a specific road segment.
        """
        obs_list = self._segment_observations.get(road_segment_id, [])
        issue_ids = self._segment_issues.get(road_segment_id, [])
        issues_list = [self.get_issue(iid) for iid in issue_ids if self.get_issue(iid) is not None]

        return self.health_engine.evaluate_segment(
            road_segment_id=road_segment_id,
            observations=obs_list,
            persistent_issues=issues_list,
            traffic_congestion_state=traffic_congestion_state,
            safety_incident_count=safety_incident_count,
        )

    def evaluate_all_segments(self) -> Dict[str, RoadSegmentHealth]:
        """Evaluate health across all monitored road segments in memory."""
        all_segments = set(self._segment_observations.keys()) | set(self._segment_issues.keys())
        results = {}
        for seg in sorted(all_segments):
            results[seg] = self.evaluate_segment_health(seg)
        return results

    # ── Internal Helpers ────────────────────────────────────────────────── #

    def _infer_segment(self, obs: Observation) -> str:
        """Infer or assign a segment based on spatial coordinates or fallback."""
        if obs.gps and obs.gps.lat is not None and obs.gps.lon is not None:
            # Hash latitude/longitude into a 200m spatial segment identifier
            lat_bin = int(round(obs.gps.lat / 0.002))
            lon_bin = int(round(obs.gps.lon / 0.002))
            return f"SEG_{abs(lat_bin % 1000):03d}_{abs(lon_bin % 1000):03d}"
        return "SEG_CORRIDOR_MAIN"

    def _record_issue_history(self, issue: PersistentIssue, obs: Observation) -> None:
        """Record an immutable transition step in the issue's audit log."""
        hist = self._issue_audit_history.setdefault(
            issue.issue_id,
            {
                "issue_id": issue.issue_id,
                "event_type": issue.event_type.value,
                "first_seen": issue.first_seen_ts,
                "created_at_frame": obs.first_seen_frame,
                "observations": [],
                "confidence_log": [],
                "severity_log": [],
                "bus_passes": [],
                "status_log": [],
                "verification_events": [],
            },
        )

        hist["observations"].append(obs.obs_id)
        hist["confidence_log"].append(round(obs.confidence, 4))
        hist["severity_log"].append(obs.severity.value)
        if obs.bus_id not in hist["bus_passes"]:
            hist["bus_passes"].append(obs.bus_id)
        hist["status_log"].append(issue.status)
        hist["last_seen"] = issue.last_seen_ts
        hist["fleet_corroborated"] = len(hist["bus_passes"]) >= 2
