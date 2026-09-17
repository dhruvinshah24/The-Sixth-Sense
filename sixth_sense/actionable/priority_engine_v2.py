"""
Priority Engine V2 — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Upgraded deterministic, multi-factor priority scoring for PersistentIssues.

Incorporates:
  1. Defect Severity (0–35 pts)
  2. Detection Confidence (0–15 pts)
  3. Fleet Corroboration / Bus Passes (0–15 pts)
  4. Persistence / Observation Count (0–10 pts)
  5. Traffic Exposure Stress (0–15 pts) [From Traffic Intelligence]
  6. Safety Exposure (0–10 pts) [From Cross-Domain Fusion]

Zero Black-Box Math:
  Every single point added has a named, human-readable reason in the output.
  Never outputs a bare number without auditable factor provenance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sixth_sense.actionable.priority_engine import (
    PriorityBand,
    PriorityResult,
    _ROAD_DEFECT_TYPES,
)
from sixth_sense.schemas.urban_event import (
    GPSStatus,
    PersistentIssue,
    SeverityTier,
)

logger = logging.getLogger(__name__)

SCORE_VERSION_V2 = "priority_v2.0_multimodal"

_SEVERITY_POINTS_V2 = {
    SeverityTier.CRITICAL: 35.0,
    SeverityTier.HIGH: 25.0,
    SeverityTier.MEDIUM: 15.0,
    SeverityTier.LOW: 8.0,
    SeverityTier.UNKNOWN: 5.0,
}


class PriorityEngineV2:
    """
    Multi-domain priority scoring engine with cross-domain traffic and safety weighting.
    """

    def __init__(
        self,
        band_critical: float = 75.0,
        band_high: float = 50.0,
        band_medium: float = 25.0,
    ) -> None:
        self.band_critical = band_critical
        self.band_high = band_high
        self.band_medium = band_medium

    def score(
        self,
        issue: PersistentIssue,
        traffic_exposure_score: float = 0.0,
        traffic_congestion_state: str = "NORMAL_FLOW",
        safety_risk_exposure: float = 0.0,
    ) -> PriorityResult:
        """
        Compute explainable priority score (0–100) for a PersistentIssue.
        """
        reasons: List[str] = []
        breakdown: Dict[str, Any] = {}
        total = 0.0

        # ── 1. Severity (0–35 pts) ──────────────────────────────────── #
        sev_pts = _SEVERITY_POINTS_V2.get(issue.severity, 5.0)
        total += sev_pts
        breakdown["severity_pts"] = sev_pts
        reasons.append(f"{issue.severity.value} defect severity (+{sev_pts:.0f}pts)")

        # ── 2. Detection Confidence (0–15 pts) ──────────────────────── #
        conf_pts = min(15.0, round(issue.confidence * 15.0, 1))
        total += conf_pts
        breakdown["confidence_pts"] = conf_pts
        reasons.append(f"Model confidence {issue.confidence:.3f} (+{conf_pts:.1f}pts)")

        # ── 3. Fleet Corroboration / Bus Passes (0–15 pts) ──────────── #
        corr_pts = min(15.0, max(0.0, (issue.bus_count - 1) * 7.5))
        total += corr_pts
        breakdown["corroboration_pts"] = corr_pts
        if issue.bus_count >= 2:
            reasons.append(f"Fleet corroboration across {issue.bus_count} bus passes (+{corr_pts:.1f}pts)")
        else:
            reasons.append("Single bus pass — awaiting fleet corroboration (+0pts)")

        # ── 4. Persistence / Observation Count (0–10 pts) ───────────── #
        obs_pts = min(10.0, issue.observation_count * 2.0)
        total += obs_pts
        breakdown["persistence_pts"] = obs_pts
        reasons.append(f"{issue.observation_count} repeated observations (+{obs_pts:.0f}pts)")

        # ── 5. Traffic Exposure Stress (0–15 pts) ───────────────────── #
        traffic_pts = 0.0
        if traffic_congestion_state in ("CONGESTION", "PERSISTENT_BOTTLENECK", "SEVERE"):
            traffic_pts = 15.0
            reasons.append(f"Heavy traffic corridor / congestion stress (+{traffic_pts:.0f}pts)")
        elif traffic_congestion_state == "SLOW_FLOW" or traffic_exposure_score > 0.4:
            traffic_pts = min(10.0, round(traffic_exposure_score * 12.0, 1))
            reasons.append(f"Moderate traffic volume stress (+{traffic_pts:.1f}pts)")
        else:
            reasons.append("Normal traffic flow — baseline exposure (+0pts)")
        total += traffic_pts
        breakdown["traffic_exposure_pts"] = traffic_pts

        # ── 6. Safety & Vulnerable Road User Exposure (0–10 pts) ────── #
        safety_pts = min(10.0, round(safety_risk_exposure * 10.0, 1))
        total += safety_pts
        breakdown["safety_exposure_pts"] = safety_pts
        if safety_pts > 0:
            reasons.append(f"Safety exposure / pedestrian proximity (+{safety_pts:.1f}pts)")

        # Cap total score at 100.0
        total = min(100.0, round(total, 2))
        breakdown["total"] = total

        band = self._band(total)

        return PriorityResult(
            priority_score=total,
            priority_band=band,
            reasons=reasons,
            score_breakdown=breakdown,
            score_version=SCORE_VERSION_V2,
        )

    def _band(self, score: float) -> str:
        if score >= self.band_critical:
            return PriorityBand.CRITICAL
        if score >= self.band_high:
            return PriorityBand.HIGH
        if score >= self.band_medium:
            return PriorityBand.MEDIUM
        return PriorityBand.LOW
