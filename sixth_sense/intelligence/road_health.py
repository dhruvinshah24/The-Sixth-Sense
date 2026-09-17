"""
Road Health Intelligence Engine — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Computes deterministic, auditable Road Health Scores (0–100) at segment level.

Moves the system from:
  DETECT → ALERT
towards:
  DETECT → REMEMBER → CORROBORATE → UNDERSTAND → PRIORITIZE → ACT → RECHECK

Core Metric:
  Health Score (100 = Pristine road, 0 = Impassable/critical failure)
  Health State: HEALTHY, WATCH, DEGRADED, CRITICAL
  Condition Trend: NEW, STABLE, PERSISTENT, WORSENING, IMPROVING, INSUFFICIENT_HISTORY
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sixth_sense.schemas.urban_event import (
    EventType,
    Observation,
    PersistentIssue,
    SeverityTier,
)

logger = logging.getLogger(__name__)


class RoadHealthState(str, Enum):
    HEALTHY = "HEALTHY"    # 85 – 100
    WATCH = "WATCH"        # 65 – 84
    DEGRADED = "DEGRADED"  # 40 – 64
    CRITICAL = "CRITICAL"  # 0 – 39


class RoadTrend(str, Enum):
    NEW = "NEW"
    STABLE = "STABLE"
    PERSISTENT = "PERSISTENT"
    WORSENING = "WORSENING"
    IMPROVING = "IMPROVING"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


_SEVERITY_DEDUCTIONS = {
    SeverityTier.CRITICAL: 25.0,
    SeverityTier.HIGH: 15.0,
    SeverityTier.MEDIUM: 8.0,
    SeverityTier.LOW: 3.0,
    SeverityTier.UNKNOWN: 5.0,
}

_DEFECT_WEIGHTS = {
    EventType.POTHOLE: 1.30,          # Potholes cause severe structural & vehicle damage
    EventType.ROAD_DAMAGE: 1.20,
    EventType.ROAD_CRACK: 1.00,       # Longitudinal/transverse/alligator fatigue
    EventType.WATERLOGGING: 1.15,     # Drainage failure / hydroplaning risk
    EventType.ROAD_DIVIDER: 1.10,     # Median / physical barrier hazard
    EventType.ROAD_REPAIR: 0.50,      # Prior repair patches (mild roughness)
}


@dataclass
class RoadSegmentHealth:
    """Segment-level health evaluation result."""
    road_segment_id: str
    health_score: float                # 0.0 – 100.0 (higher is better)
    health_state: RoadHealthState
    trend: RoadTrend
    observation_count: int
    bus_count: int
    active_issues_count: int
    defect_breakdown: Dict[str, int]
    reasons: List[str]
    scoring_factors: Dict[str, Any]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_segment_id": self.road_segment_id,
            "health_score": round(self.health_score, 1),
            "health_state": self.health_state.value,
            "trend": self.trend.value,
            "observation_count": self.observation_count,
            "bus_count": self.bus_count,
            "active_issues_count": self.active_issues_count,
            "defect_breakdown": self.defect_breakdown,
            "reasons": self.reasons,
            "scoring_factors": self.scoring_factors,
            "recommendation": self.recommendation,
        }


class RoadHealthEngine:
    """
    Deterministic road health assessment engine.
    Calculates segment-level index from verified observations and persistent issues.
    """

    def __init__(
        self,
        base_score: float = 100.0,
        traffic_penalty_factor: float = 1.25,
    ) -> None:
        self.base_score = base_score
        self.traffic_penalty_factor = traffic_penalty_factor

    def evaluate_segment(
        self,
        road_segment_id: str,
        observations: List[Observation],
        persistent_issues: Optional[List[PersistentIssue]] = None,
        traffic_congestion_state: str = "NORMAL_FLOW",
        safety_incident_count: int = 0,
    ) -> RoadSegmentHealth:
        """
        Evaluate health score and trend for a specific road segment.
        """
        reasons: List[str] = []
        factors: Dict[str, Any] = {}

        if not observations and not persistent_issues:
            return RoadSegmentHealth(
                road_segment_id=road_segment_id,
                health_score=100.0,
                health_state=RoadHealthState.HEALTHY,
                trend=RoadTrend.INSUFFICIENT_HISTORY,
                observation_count=0,
                bus_count=0,
                active_issues_count=0,
                defect_breakdown={},
                reasons=["No structural defects or hazards observed on this segment"],
                scoring_factors={"base_score": 100.0, "total_deductions": 0.0},
                recommendation="Routine maintenance monitoring; road in good standing.",
            )

        current_score = self.base_score
        total_deduction = 0.0

        # Aggregate unique buses and defect types
        buses = set()
        defect_counts: Dict[str, int] = {}
        for obs in observations:
            buses.add(obs.bus_id)
            cls_k = obs.class_name or (obs.event_type.value if hasattr(obs.event_type, "value") else str(obs.event_type))
            defect_counts[cls_k] = defect_counts.get(cls_k, 0) + 1

        active_issues = persistent_issues or []
        for issue in active_issues:
            for b in issue.bus_ids:
                buses.add(b)
            cls_k = issue.class_name or (issue.event_type.value if hasattr(issue.event_type, "value") else str(issue.event_type))
            defect_counts[cls_k] = defect_counts.get(cls_k, 0) + issue.observation_count

        bus_count = len(buses)
        obs_count = len(observations) + sum(i.observation_count for i in active_issues)

        # ── 1. Defect Severity Deductions ───────────────────────────────── #
        defect_deductions = 0.0
        # Evaluate persistent issues first (deduplicated real defects)
        evaluated_items = active_issues if active_issues else observations
        for item in evaluated_items:
            sev = item.severity if hasattr(item, "severity") else SeverityTier.UNKNOWN
            et = item.event_type if hasattr(item, "event_type") else EventType.UNKNOWN
            conf = getattr(item, "confidence", 0.70)

            base_ded = _SEVERITY_DEDUCTIONS.get(sev, 5.0)
            multiplier = _DEFECT_WEIGHTS.get(et, 1.0)

            item_ded = base_ded * multiplier * max(0.5, conf)
            defect_deductions += item_ded

        # Dampen cumulative deductions to avoid negative over-penalization
        scaled_defect_deduction = min(70.0, defect_deductions)
        total_deduction += scaled_defect_deduction
        factors["defect_severity_deduction"] = round(scaled_defect_deduction, 1)

        if scaled_defect_deduction >= 30.0:
            reasons.append(f"Severe/High structural road damage present (-{scaled_defect_deduction:.1f}pts)")
        elif scaled_defect_deduction > 0.0:
            reasons.append(f"Surface defects observed (-{scaled_defect_deduction:.1f}pts)")

        # ── 2. Fleet Corroboration & Recurrence Impact ──────────────────── #
        if bus_count >= 2:
            corrob_factor = min(10.0, (bus_count - 1) * 3.5)
            total_deduction += corrob_factor
            factors["fleet_corroboration_deduction"] = round(corrob_factor, 1)
            reasons.append(f"Corroborated across {bus_count} separate bus passes (-{corrob_factor:.1f}pts)")

        # ── 3. Traffic Exposure Stress Multiplier ───────────────────────── #
        if traffic_congestion_state in ("CONGESTION", "PERSISTENT_BOTTLENECK", "SEVERE"):
            traffic_penalty = 12.0
            total_deduction += traffic_penalty
            factors["traffic_stress_penalty"] = traffic_penalty
            reasons.append(f"Heavy traffic stress / congestion accelerates deterioration (-{traffic_penalty:.1f}pts)")
        elif traffic_congestion_state == "SLOW_FLOW":
            traffic_penalty = 5.0
            total_deduction += traffic_penalty
            factors["traffic_stress_penalty"] = traffic_penalty
            reasons.append(f"Moderate traffic exposure (-{traffic_penalty:.1f}pts)")

        # ── 4. Safety & Hazard Exposure ─────────────────────────────────── #
        if safety_incident_count > 0:
            safety_penalty = min(15.0, safety_incident_count * 5.0)
            total_deduction += safety_penalty
            factors["safety_risk_penalty"] = safety_penalty
            reasons.append(f"Safety risk exposure / nearby VRU conflict (-{safety_penalty:.1f}pts)")

        # Compute final health score
        health_score = max(0.0, round(self.base_score - total_deduction, 1))
        factors["total_deductions"] = round(total_deduction, 1)

        # ── 5. Health State Classification ──────────────────────────────── #
        if health_score >= 85.0:
            state = RoadHealthState.HEALTHY
            recommendation = "Normal periodic inspection. No immediate engineering intervention needed."
        elif health_score >= 65.0:
            state = RoadHealthState.WATCH
            recommendation = "Place segment on active watch list. Schedule routine surface maintenance."
        elif health_score >= 40.0:
            state = RoadHealthState.DEGRADED
            recommendation = "Priority maintenance required: asphalt resurfacing / pothole patching."
        else:
            state = RoadHealthState.CRITICAL
            recommendation = "Urgent public works dispatch: severe pavement failure endangering transit."

        # ── 6. Trend Determination ──────────────────────────────────────── #
        trend = self._determine_trend(active_issues, observations, bus_count)

        return RoadSegmentHealth(
            road_segment_id=road_segment_id,
            health_score=health_score,
            health_state=state,
            trend=trend,
            observation_count=obs_count,
            bus_count=bus_count,
            active_issues_count=len(active_issues),
            defect_breakdown=defect_counts,
            reasons=reasons,
            scoring_factors=factors,
            recommendation=recommendation,
        )

    def _determine_trend(
        self,
        issues: List[PersistentIssue],
        observations: List[Observation],
        bus_count: int,
    ) -> RoadTrend:
        """
        Derive empirical trend strictly from available observation history.
        Never fabricates historical baselines.
        """
        total_evidence = len(observations) + sum(i.observation_count for i in issues)
        if total_evidence < 2 or bus_count < 2:
            return RoadTrend.INSUFFICIENT_HISTORY

        # Check issue trends if issues exist
        if issues:
            trends = [i.trend.value for i in issues if hasattr(i, "trend")]
            if any(t in ("DETERIORATING", "WORSENING") for t in trends):
                return RoadTrend.WORSENING
            if all(t in ("IMPROVING", "RESOLVED") for t in trends):
                return RoadTrend.IMPROVING
            if any(t == "PERSISTENT" for t in trends) or len(issues) >= 2:
                return RoadTrend.PERSISTENT
            return RoadTrend.STABLE

        return RoadTrend.STABLE


def apply_closure_to_road_health(
    prior_health: RoadSegmentHealth,
    issue: PersistentIssue,
    verification_result: Any,
) -> RoadSegmentHealth:
    """
    Adjust segment road health following deterministic closure verification.
    If VERIFIED_REPAIRED: improves health score and marks trend as IMPROVING.
    If REOPENED or STILL_PRESENT: applies failed-repair penalty and marks trend as WORSENING.
    If REVIEW_REQUIRED: keeps health guarded without automatic recovery.
    """
    res_str = getattr(verification_result, "verification_result", str(verification_result))

    updated_score = prior_health.health_score
    updated_trend = prior_health.trend
    updated_reasons = list(prior_health.reasons)
    updated_factors = dict(prior_health.scoring_factors)
    active_count = prior_health.active_issues_count

    if res_str == "VERIFIED_REPAIRED":
        sev = getattr(issue, "severity", SeverityTier.MEDIUM)
        base_ded = _SEVERITY_DEDUCTIONS.get(sev, 8.0)
        mult = _DEFECT_WEIGHTS.get(getattr(issue, "event_type", EventType.ROAD_DAMAGE), 1.0)
        recovered_pts = round(base_ded * mult * 0.85, 1)

        updated_score = min(100.0, round(updated_score + recovered_pts, 1))
        updated_trend = RoadTrend.IMPROVING
        active_count = max(0, active_count - 1)
        updated_factors["closure_recovery_bonus"] = recovered_pts
        updated_reasons.append(f"Issue {issue.issue_id} verified repaired: +{recovered_pts}pts recovered.")

    elif res_str in ("REOPENED", "STILL_PRESENT"):
        penalty = 10.0
        updated_score = max(0.0, round(updated_score - penalty, 1))
        updated_trend = RoadTrend.WORSENING
        updated_factors["closure_failure_penalty"] = penalty
        updated_reasons.append(f"Issue {issue.issue_id} reopened/failed verification: -{penalty}pts recurrence penalty.")

    elif res_str == "REVIEW_REQUIRED":
        updated_reasons.append(f"Issue {issue.issue_id} closure requires human review; score remains guarded.")

    if updated_score >= 85.0:
        new_state = RoadHealthState.HEALTHY
        rec = "Normal periodic inspection. Road in good standing."
    elif updated_score >= 65.0:
        new_state = RoadHealthState.WATCH
        rec = "Place segment on active watch list. Schedule routine surface maintenance."
    elif updated_score >= 40.0:
        new_state = RoadHealthState.DEGRADED
        rec = "Priority maintenance required: asphalt resurfacing / pothole patching."
    else:
        new_state = RoadHealthState.CRITICAL
        rec = "Urgent public works dispatch: severe pavement failure endangering transit."

    return RoadSegmentHealth(
        road_segment_id=prior_health.road_segment_id,
        health_score=updated_score,
        health_state=new_state,
        trend=updated_trend,
        observation_count=prior_health.observation_count + 1,
        bus_count=prior_health.bus_count,
        active_issues_count=active_count,
        defect_breakdown=prior_health.defect_breakdown,
        reasons=updated_reasons,
        scoring_factors=updated_factors,
        recommendation=rec,
    )
