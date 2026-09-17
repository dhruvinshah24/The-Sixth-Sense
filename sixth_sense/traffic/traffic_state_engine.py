"""
Traffic State Engine — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Computes segment-level and corridor-level traffic mobility states.

Key Capabilities:
- Aggregates unique vehicle counts and discrete class breakdowns.
- Distinguishes TEMPORARY_CONGESTION vs. RECURRING_CONGESTION vs. PERSISTENT_BOTTLENECK.
- Measures traffic exposure stress used by the Road Health & Priority Engines.
- Enforces strict anti-hallucination: uncalibrated speed and lane occupancy stay UNAVAILABLE.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sixth_sense.traffic.intelligence import (
    SUPPORTED_VEHICLE_CLASSES,
    WINDOW_SECONDS,
    congestion_for_density,
    density_level,
)

logger = logging.getLogger(__name__)


class CongestionClassification(str, Enum):
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    TEMPORARY_CONGESTION = "TEMPORARY_CONGESTION"
    RECURRING_CONGESTION = "RECURRING_CONGESTION"
    PERSISTENT_BOTTLENECK = "PERSISTENT_BOTTLENECK"


@dataclass
class SegmentTrafficState:
    """Consolidated traffic mobility intelligence for a road segment / corridor."""
    segment_id: str
    time_window: Dict[str, float]
    total_vehicle_count: int
    vehicle_classes: Dict[str, int]
    density_state: str                 # LOW, MODERATE, HIGH, SEVERE
    congestion_state: str              # NORMAL_FLOW, SLOW_FLOW, CONGESTION
    classification: CongestionClassification
    recurrence_count: int              # Number of supporting observation windows
    traffic_exposure_score: float      # 0.0 – 1.0 (stress metric for road durability)
    direction: Optional[str] = None
    speed_kmh: str = "UNAVAILABLE"
    lane_occupancy: str = "UNAVAILABLE"
    evidence_basis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "time_window": self.time_window,
            "total_vehicle_count": self.total_vehicle_count,
            "vehicle_classes": self.vehicle_classes,
            "density_state": self.density_state,
            "congestion_state": self.congestion_state,
            "classification": self.classification.value,
            "recurrence_count": self.recurrence_count,
            "traffic_exposure_score": round(self.traffic_exposure_score, 3),
            "direction": self.direction,
            "speed_kmh": self.speed_kmh,
            "lane_occupancy": self.lane_occupancy,
            "evidence_basis": self.evidence_basis,
        }


class TrafficStateEngine:
    """
    Evaluates traffic mobility dynamics across road segments and temporal windows.
    """

    def __init__(self, window_seconds: float = WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds

    def evaluate_segment_traffic(
        self,
        segment_id: str,
        vehicle_observations: List[Dict[str, Any]],
        window_seconds: Optional[float] = None,
    ) -> SegmentTrafficState:
        """
        Evaluate traffic state across observations for a specific segment.
        """
        w_sec = window_seconds or self.window_seconds

        # Filter valid vehicle observations
        eligible = [
            obs for obs in vehicle_observations
            if obs.get("event_type") == "VEHICLE"
            and obs.get("class_name") in SUPPORTED_VEHICLE_CLASSES
        ]

        if not eligible:
            return SegmentTrafficState(
                segment_id=segment_id,
                time_window={"start": 0.0, "end": 0.0, "duration": 0.0},
                total_vehicle_count=0,
                vehicle_classes={},
                density_state="LOW",
                congestion_state="NORMAL_FLOW",
                classification=CongestionClassification.NORMAL,
                recurrence_count=0,
                traffic_exposure_score=0.0,
                evidence_basis="No vehicle traffic observed on this segment.",
            )

        # Time bounds
        timestamps = [float(o.get("first_seen_ts", 0.0)) for o in eligible]
        t_start = min(timestamps)
        t_end = max(timestamps)
        duration = max(w_sec, round(t_end - t_start, 2))

        # Class breakdown and deduplicated vehicle count
        unique_obs = {o.get("obs_id", str(i)): o for i, o in enumerate(eligible)}
        class_counts = Counter(o.get("class_name") for o in unique_obs.values())
        count = len(unique_obs)

        # Bucket into temporal sub-windows to check recurrence
        window_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for o in unique_obs.values():
            ts = float(o.get("first_seen_ts", 0.0))
            w_idx = int((ts - t_start) // w_sec)
            window_buckets[w_idx].append(o)

        num_windows = len(window_buckets)
        density = density_level(count)
        base_congestion = congestion_for_density(density)

        # Analyze congestion recurrence across windows
        congested_windows = 0
        for w_recs in window_buckets.values():
            w_count = len(w_recs)
            if density_level(w_count) in ("HIGH", "SEVERE"):
                congested_windows += 1

        # Determine Congestion Classification
        if congested_windows >= 2:
            # Check if windows are consecutive
            consecutive = False
            sorted_w_indices = sorted(window_buckets.keys())
            for i in range(1, len(sorted_w_indices)):
                if sorted_w_indices[i] == sorted_w_indices[i - 1] + 1:
                    w1_cnt = len(window_buckets[sorted_w_indices[i - 1]])
                    w2_cnt = len(window_buckets[sorted_w_indices[i]])
                    if density_level(w1_cnt) in ("HIGH", "SEVERE") and density_level(w2_cnt) in ("HIGH", "SEVERE"):
                        consecutive = True
                        break

            if consecutive:
                classification = CongestionClassification.PERSISTENT_BOTTLENECK
            else:
                classification = CongestionClassification.RECURRING_CONGESTION
        elif base_congestion == "CONGESTION":
            classification = CongestionClassification.TEMPORARY_CONGESTION
        elif base_congestion == "SLOW_FLOW":
            classification = CongestionClassification.SLOW
        else:
            classification = CongestionClassification.NORMAL

        # Traffic exposure stress score (0.0 to 1.0)
        exposure = min(1.0, count / 20.0)

        # Evidence description
        evidence = (
            f"{count} unique vehicles observed across {num_windows} time window(s). "
            f"Peak density: {density}. Congestion classification: {classification.value}."
        )

        return SegmentTrafficState(
            segment_id=segment_id,
            time_window={"start": round(t_start, 2), "end": round(t_end, 2), "duration": duration},
            total_vehicle_count=count,
            vehicle_classes=dict(sorted(class_counts.items())),
            density_state=density,
            congestion_state=base_congestion,
            classification=classification,
            recurrence_count=num_windows,
            traffic_exposure_score=exposure,
            evidence_basis=evidence,
        )
