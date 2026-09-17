"""
Urban Intelligence Pipeline — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
End-to-End Orchestration Layer uniting Person 1 (Road) & Person 2 (Traffic).

Executes the complete closed-loop architecture:
  DETECT → REMEMBER → CORROBORATE → UNDERSTAND → PRIORITIZE → ACT → RECHECK
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from sixth_sense.actionable.priority_engine_v2 import PriorityEngineV2
from sixth_sense.closure.verification_engine import (
    VerificationEngine,
    VerificationOutcome,
    VerificationResult,
)
from sixth_sense.events.city_memory import CityMemory
from sixth_sense.intelligence.confidence_automation import (
    AutomationDecision,
    ConfidenceAutomationGovernor,
)
from sixth_sense.intelligence.cross_domain_fusion import (
    CrossDomainFusionEngine,
    FusedSegmentContext,
)
from sixth_sense.intelligence.road_health import (
    RoadHealthEngine,
    RoadSegmentHealth,
    apply_closure_to_road_health,
)
from sixth_sense.schemas.unified_event import DomainType, UnifiedObservation
from sixth_sense.schemas.urban_event import (
    Observation,
    PersistentIssue,
)
from sixth_sense.traffic.traffic_state_engine import (
    SegmentTrafficState,
    TrafficStateEngine,
)

logger = logging.getLogger(__name__)


class UrbanIntelligencePipeline:
    """
    Unified Orchestrator executing the complete continuous urban intelligence loop.
    Reuses existing components without duplicating logic or altering underlying models.
    """

    def __init__(
        self,
        dedup_radius_m: float = 30.0,
        city_memory: Optional[CityMemory] = None,
        road_health_engine: Optional[RoadHealthEngine] = None,
        traffic_state_engine: Optional[TrafficStateEngine] = None,
        fusion_engine: Optional[CrossDomainFusionEngine] = None,
        priority_engine: Optional[PriorityEngineV2] = None,
        confidence_governor: Optional[ConfidenceAutomationGovernor] = None,
        verification_engine: Optional[VerificationEngine] = None,
    ) -> None:
        self.road_health_engine = road_health_engine or RoadHealthEngine()
        self.city_memory = city_memory or CityMemory(
            dedup_radius_m=dedup_radius_m,
            health_engine=self.road_health_engine,
        )
        self.traffic_state_engine = traffic_state_engine or TrafficStateEngine()
        self.fusion_engine = fusion_engine or CrossDomainFusionEngine()
        self.priority_engine = priority_engine or PriorityEngineV2()
        self.confidence_governor = confidence_governor or ConfidenceAutomationGovernor()
        self.verification_engine = verification_engine or VerificationEngine()

        # Ingested stores
        self._raw_observations: List[Observation] = []
        self._traffic_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._unified_events: Dict[str, List[UnifiedObservation]] = defaultdict(list)
        self._verification_results: Dict[str, VerificationResult] = {}

        # Evaluated states
        self._evaluated_road_health: Dict[str, RoadSegmentHealth] = {}
        self._evaluated_traffic_state: Dict[str, SegmentTrafficState] = {}
        self._evaluated_fused_contexts: Dict[str, FusedSegmentContext] = {}
        self._evaluated_priorities: Dict[str, Any] = {}
        self._evaluated_governance: Dict[str, AutomationDecision] = {}

    # ── 1. Ingestion Phase (DETECT → REMEMBER) ──────────────────────────── #

    def ingest_observation(
        self,
        obs: Observation,
        road_segment_id: Optional[str] = None,
    ) -> PersistentIssue:
        """Ingest single observation into City Memory."""
        self._raw_observations.append(obs)
        issue = self.city_memory.ingest_observation(obs, road_segment_id=road_segment_id)

        # Convert to UnifiedObservation for cross-domain representation
        unified = UnifiedObservation.from_observation(obs, road_segment_id=issue.road_segment)
        seg = issue.road_segment or "SEG_CORRIDOR_MAIN"
        self._unified_events[seg].append(unified)

        return issue

    def ingest_observations(
        self,
        observations: List[Observation],
        road_segment_id: Optional[str] = None,
    ) -> List[PersistentIssue]:
        """Ingest batch of observations."""
        updated = []
        for o in observations:
            iss = self.ingest_observation(o, road_segment_id=road_segment_id)
            if iss not in updated:
                updated.append(iss)
        return updated

    def ingest_traffic_records(
        self,
        traffic_records: List[Dict[str, Any]],
        road_segment_id: str,
    ) -> None:
        """Ingest vehicle observations for traffic density and corridor analysis."""
        self._traffic_records[road_segment_id].extend(traffic_records)
        # Store as unified traffic events
        for r in traffic_records:
            u_ev = UnifiedObservation(
                observation_id=r.get("obs_id", f"v_{time.time()}"),
                bus_id=r.get("bus_id", "BUS_TRANSIT"),
                timestamp=float(r.get("first_seen_ts", time.time())),
                location={"road_segment_id": road_segment_id},
                domain=DomainType.TRAFFIC,
                event_type=r.get("class_name", "vehicle"),
                confidence=float(r.get("confidence", 0.90)),
                severity="UNKNOWN",
            )
            self._unified_events[road_segment_id].append(u_ev)

    def ingest_teammate_event(self, event: UnifiedObservation) -> None:
        """Extensible ingestion point for teammate Safety (Person 3) or Incident (Person 4) events."""
        seg = event.location.get("road_segment_id", "SEG_CORRIDOR_MAIN")
        self._unified_events[seg].append(event)

    # ── 2. Evaluation Phase (CORROBORATE → UNDERSTAND) ───────────────────── #

    def process_all(self) -> Dict[str, Any]:
        """
        Execute full intelligence evaluation across all monitored segments and issues.
        """
        all_segments = set(self._traffic_records.keys()) | set(self._unified_events.keys())
        for iss in self.city_memory.get_all_issues():
            if iss.road_segment:
                all_segments.add(iss.road_segment)

        if not all_segments:
            all_segments.add("SEG_CORRIDOR_MAIN")

        # Step A: Evaluate Traffic State per Segment
        for seg in all_segments:
            t_records = self._traffic_records.get(seg, [])
            t_state = self.traffic_state_engine.evaluate_segment_traffic(seg, t_records)
            self._evaluated_traffic_state[seg] = t_state

        # Step B: Evaluate Road Health per Segment
        for seg in all_segments:
            t_state = self._evaluated_traffic_state.get(seg)
            cong_str = t_state.congestion_state if t_state else "NORMAL_FLOW"
            if t_state and t_state.classification.value == "PERSISTENT_BOTTLENECK":
                cong_str = "PERSISTENT_BOTTLENECK"

            health = self.city_memory.evaluate_segment_health(
                road_segment_id=seg,
                traffic_congestion_state=cong_str,
            )
            self._evaluated_road_health[seg] = health

        # Step C: Cross-Domain Fusion per Segment
        for seg in all_segments:
            u_events = self._unified_events.get(seg, [])
            t_state = self._evaluated_traffic_state.get(seg)
            t_exp = t_state.traffic_exposure_score if t_state else 0.0
            cong_str = t_state.congestion_state if t_state else "NORMAL_FLOW"
            if t_state and t_state.classification.value == "PERSISTENT_BOTTLENECK":
                cong_str = "PERSISTENT_BOTTLENECK"

            fused_ctx = self.fusion_engine.fuse_segment_events(
                segment_id=seg,
                unified_events=u_events,
                traffic_exposure=t_exp,
                traffic_congestion_state=cong_str,
            )
            self._evaluated_fused_contexts[seg] = fused_ctx

        # Step D: Prioritize Issues (PRIORITIZE)
        for iss in self.city_memory.get_all_issues():
            seg = iss.road_segment or "SEG_CORRIDOR_MAIN"
            t_state = self._evaluated_traffic_state.get(seg)
            t_exp = t_state.traffic_exposure_score if t_state else 0.0
            cong_str = t_state.congestion_state if t_state else "NORMAL_FLOW"
            if t_state and t_state.classification.value == "PERSISTENT_BOTTLENECK":
                cong_str = "PERSISTENT_BOTTLENECK"

            fused_ctx = self._evaluated_fused_contexts.get(seg)
            safety_risk = min(1.0, len(fused_ctx.safety_risks) * 0.3) if fused_ctx else 0.0

            p_result = self.priority_engine.score(
                issue=iss,
                traffic_exposure_score=t_exp,
                traffic_congestion_state=cong_str,
                safety_risk_exposure=safety_risk,
            )
            self._evaluated_priorities[iss.issue_id] = p_result

        # Step E: Confidence & Governance Gating (ACT)
        for iss in self.city_memory.get_all_issues():
            decision = self.confidence_governor.evaluate_issue(iss)
            self._evaluated_governance[iss.issue_id] = decision

        return self.get_summary()

    # ── 3. Proof-of-Closure Phase (RECHECK) ──────────────────────────────── #

    def verify_repair(
        self,
        issue_id: str,
        claim: Any,
        follow_up_pass: Any,
    ) -> Tuple[VerificationResult, Optional[RoadSegmentHealth]]:
        """
        Execute deterministic closure verification and apply feedback to road health.
        """
        issue = self.city_memory.get_issue(issue_id)
        if not issue:
            raise ValueError(f"Issue {issue_id} not found in City Memory.")

        verif_result = self.verification_engine.verify(issue, claim, follow_up_pass)
        self.verification_engine.apply_result(issue, claim, verif_result)
        self._verification_results[issue_id] = verif_result

        # Update Road Health with feedback loop
        seg = issue.road_segment or "SEG_CORRIDOR_MAIN"
        prior_health = self._evaluated_road_health.get(seg)
        updated_health = None
        if prior_health:
            updated_health = apply_closure_to_road_health(prior_health, issue, verif_result)
            self._evaluated_road_health[seg] = updated_health

        return verif_result, updated_health

    # ── 4. Export & Summary ─────────────────────────────────────────────── #

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level status summary."""
        issues = self.city_memory.get_all_issues()
        multi_bus_corroborated = sum(1 for i in issues if i.bus_count >= 2)

        return {
            "total_raw_observations": len(self._raw_observations),
            "total_persistent_issues": len(issues),
            "multi_bus_corroborated_issues": multi_bus_corroborated,
            "monitored_segments": len(self._evaluated_road_health),
            "road_health_overview": {
                seg: {
                    "score": h.health_score,
                    "state": h.health_state.value,
                    "trend": h.trend.value,
                    "active_issues": h.active_issues_count,
                }
                for seg, h in self._evaluated_road_health.items()
            },
            "traffic_state_overview": {
                seg: {
                    "classification": t.classification.value,
                    "density": t.density_state,
                    "vehicles": t.total_vehicle_count,
                    "speed": t.speed_kmh,
                    "lane_occupancy": t.lane_occupancy,
                }
                for seg, t in self._evaluated_traffic_state.items()
            },
            "priority_overview": {
                iid: {
                    "priority_score": p.priority_score,
                    "priority_band": p.priority_band,
                    "top_reasons": p.reasons[:2],
                }
                for iid, p in self._evaluated_priorities.items()
            },
            "governance_overview": {
                iid: {
                    "tier": g.tier.value,
                    "action": g.governance_action.value,
                    "auto_dispatch_permitted": g.auto_dispatch_permitted,
                }
                for iid, g in self._evaluated_governance.items()
            },
            "verifications_completed": len(self._verification_results),
        }

    def export_artifacts(self, output_dir: str = "outputs/urban_intelligence") -> Dict[str, str]:
        """
        Consolidate and write all structured pipeline outputs to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # 1. observations.json
        obs_data = [o.to_dict() for o in self._raw_observations]
        p_obs = os.path.join(output_dir, "observations.json")
        with open(p_obs, "w") as f:
            json.dump(obs_data, f, indent=2)
        paths["observations"] = p_obs

        # 2. persistent_issues.json
        issues_data = [i.to_dict() for i in self.city_memory.get_all_issues()]
        p_iss = os.path.join(output_dir, "persistent_issues.json")
        with open(p_iss, "w") as f:
            json.dump(issues_data, f, indent=2)
        paths["persistent_issues"] = p_iss

        # 3. road_health.json
        health_data = {seg: h.to_dict() for seg, h in self._evaluated_road_health.items()}
        p_health = os.path.join(output_dir, "road_health.json")
        with open(p_health, "w") as f:
            json.dump(health_data, f, indent=2)
        paths["road_health"] = p_health

        # 4. traffic_state.json
        traffic_data = {seg: t.to_dict() for seg, t in self._evaluated_traffic_state.items()}
        p_traffic = os.path.join(output_dir, "traffic_state.json")
        with open(p_traffic, "w") as f:
            json.dump(traffic_data, f, indent=2)
        paths["traffic_state"] = p_traffic

        # 5. fused_context.json
        fused_data = {seg: f.to_dict() for seg, f in self._evaluated_fused_contexts.items()}
        p_fused = os.path.join(output_dir, "fused_context.json")
        with open(p_fused, "w") as f:
            json.dump(fused_data, f, indent=2)
        paths["fused_context"] = p_fused

        # 6. priority_queue.json
        priority_data = [
            {
                "issue_id": iid,
                "priority_score": p.priority_score,
                "priority_band": p.priority_band,
                "reasons": p.reasons,
                "factors": p.score_breakdown,
            }
            for iid, p in sorted(
                self._evaluated_priorities.items(),
                key=lambda x: x[1].priority_score,
                reverse=True,
            )
        ]
        p_prio = os.path.join(output_dir, "priority_queue.json")
        with open(p_prio, "w") as f:
            json.dump(priority_data, f, indent=2)
        paths["priority_queue"] = p_prio

        # 7. governance_decisions.json
        gov_data = {iid: g.to_dict() for iid, g in self._evaluated_governance.items()}
        p_gov = os.path.join(output_dir, "governance_decisions.json")
        with open(p_gov, "w") as f:
            json.dump(gov_data, f, indent=2)
        paths["governance_decisions"] = p_gov

        # 8. final_summary.json
        summary = self.get_summary()
        summary["export_timestamp"] = time.time()
        summary["output_paths"] = paths
        p_sum = os.path.join(output_dir, "final_summary.json")
        with open(p_sum, "w") as f:
            json.dump(summary, f, indent=2)
        paths["final_summary"] = p_sum

        logger.info("Successfully exported all 8 urban intelligence artifacts to %s", output_dir)
        return paths
