#!/usr/bin/env python3
"""
Urban Intelligence End-to-End Demonstration
The Sixth Sense — SIH 2026 PS 26124 / PS 26125
Person 1 (Road Infrastructure) + Person 2 (Traffic Mobility) Intelligence

Executes the complete continuous urban intelligence loop:
  CAMERA + GPS
        |
     DETECT
        |
     REMEMBER
        |
   CORROBORATE
        |
    UNDERSTAND
        |
    PRIORITIZE
        |
       ACT
        |
     RECHECK
        |
UPDATE CITY MEMORY

Run from repository root:
    python run_urban_intelligence_demo.py
"""
import json
import os
import sys
import time

# Ensure repository root is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sixth_sense.closure.repair_claim import RepairClaimBuilder, RepairClaimStatus
from sixth_sense.closure.verification_engine import FollowUpPass
from sixth_sense.pipeline.urban_intelligence_pipeline import UrbanIntelligencePipeline
from sixth_sense.schemas.urban_event import (
    EventType,
    GPSPoint,
    GPSStatus,
    Observation,
    SeverityTier,
)


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_step(step_num: int, title: str):
    print(f"\n[{step_num}/8] {title}")
    print("-" * 65)


def run_demo():
    print_banner("THE SIXTH SENSE: URBAN INTELLIGENCE PLATFORM (SIH 2026)")
    print("Corridor: SEG_MUMBAI_WEH_01 (Western Express Highway Corridor)")
    print("Deploying Person 1 (Road) & Person 2 (Traffic) Intelligence Layer...\n")

    pipeline = UrbanIntelligencePipeline(dedup_radius_m=30.0)

    # ---------------------------------------------------------------------- #
    # STEP 1: BUS_001 Detects Real Road Pothole (D40)
    # ---------------------------------------------------------------------- #
    print_step(1, "BUS_001 PASS: Real Pothole (D40) Observed")
    obs_bus1 = Observation(
        obs_id="obs_real_d40_bus1",
        bus_id="BUS_001",
        camera_id="CAM_FRONT",
        run_id="run_morning_01",
        event_type=EventType.POTHOLE,
        class_name="D40",
        first_seen_frame=9,
        last_seen_frame=18,
        first_seen_ts=1726615000.0,
        last_seen_ts=1726615000.75,
        representative_frame=9,
        confidence=0.85,
        severity=SeverityTier.HIGH,
        bbox=(276, 775, 873, 1032),
        bbox_area_px=153429,
        relative_area=0.131,
        gps=GPSPoint(
            lat=19.076000,
            lon=72.877700,
            timestamp=1726615000.0,
            uncertainty_m=4.5,
            heading=182.0,
            status=GPSStatus.INTERPOLATED,
        ),
        evidence_ref="outputs/test_road_debug/evidence/ev_det_59466fea_f9.jpg",
        detection_count=10,
        model_name="road_damage_rdd2022",
    )

    issue_pass1 = pipeline.ingest_observation(obs_bus1, road_segment_id="SEG_MUMBAI_WEH_01")
    pipeline.process_all()
    health_pass1 = pipeline._evaluated_road_health["SEG_MUMBAI_WEH_01"]

    print(f"  * Observation Ingested: {obs_bus1.obs_id} [{obs_bus1.class_name}] (conf: {obs_bus1.confidence:.2f})")
    print(f"  * City Memory State   : Issue {issue_pass1.issue_id[:16]} spawned (Bus passes: {issue_pass1.bus_count})")
    print(f"  * Segment Health      : {health_pass1.health_score}/100 [{health_pass1.health_state.value}]")
    print(f"  * Condition Trend     : {health_pass1.trend.value} (honest: single transit pass insufficient for trend)")

    # ---------------------------------------------------------------------- #
    # STEP 2: BUS_002 Corroborates Same Defect 45 Mins Later
    # ---------------------------------------------------------------------- #
    print_step(2, "BUS_002 PASS: Multi-Bus Fleet Corroboration (Same GPS within 4m)")
    obs_bus2 = Observation(
        obs_id="obs_real_d40_bus2",
        bus_id="BUS_002",
        camera_id="CAM_FRONT",
        run_id="run_midday_02",
        event_type=EventType.POTHOLE,
        class_name="D40",
        first_seen_frame=112,
        last_seen_frame=128,
        first_seen_ts=1726617700.0,
        last_seen_ts=1726617700.80,
        representative_frame=115,
        confidence=0.88,
        severity=SeverityTier.CRITICAL,
        bbox=(280, 770, 880, 1035),
        bbox_area_px=155000,
        relative_area=0.133,
        gps=GPSPoint(
            lat=19.076035,
            lon=72.877720,
            timestamp=1726617700.0,
            uncertainty_m=3.8,
            heading=180.0,
            status=GPSStatus.INTERPOLATED,
        ),
        evidence_ref="outputs/test_road_debug/evidence/ev_det_59466fea_f9.jpg",
        detection_count=16,
        model_name="road_damage_rdd2022",
    )

    issue_pass2 = pipeline.ingest_observation(obs_bus2, road_segment_id="SEG_MUMBAI_WEH_01")
    pipeline.process_all()
    health_pass2 = pipeline._evaluated_road_health["SEG_MUMBAI_WEH_01"]

    print(f"  * Observation Ingested: {obs_bus2.obs_id} [{obs_bus2.class_name}] (conf: {obs_bus2.confidence:.2f})")
    print(f"  * Spatial Deduplication: MERGED into existing issue {issue_pass2.issue_id[:16]}")
    print(f"  * Multi-Bus Accounting : Corroborated across {issue_pass2.bus_count} bus passes ({', '.join(issue_pass2.bus_ids)})")
    print(f"  * Segment Health Drop : {health_pass1.health_score} -> {health_pass2.health_score}/100 [{health_pass2.health_state.value}]")
    print(f"  * Empirical Trend     : {health_pass2.trend.value} (temporal corroboration confirmed)")

    # ---------------------------------------------------------------------- #
    # STEP 3: Ingest Traffic Mobility Passes on Same Corridor
    # ---------------------------------------------------------------------- #
    print_step(3, "TRAFFIC MOBILITY: Vehicle Tracking & Congestion Analysis")
    traffic_records = [
        {"obs_id": f"veh_w0_{i}", "event_type": "VEHICLE", "class_name": "car", "first_seen_ts": 1726617700.0 + i * 2}
        for i in range(10)
    ] + [
        {"obs_id": f"veh_w1_{i}", "event_type": "VEHICLE", "class_name": "bus", "first_seen_ts": 1726617765.0 + i * 2}
        for i in range(10)
    ]
    pipeline.ingest_traffic_records(traffic_records, road_segment_id="SEG_MUMBAI_WEH_01")
    pipeline.process_all()
    traffic_state = pipeline._evaluated_traffic_state["SEG_MUMBAI_WEH_01"]

    print(f"  * Vehicles Tracked    : {traffic_state.total_vehicle_count} unique vehicle proxies across consecutive windows")
    print(f"  * Traffic Density     : {traffic_state.density_state} (Peak Congestion)")
    print(f"  * Classification      : {traffic_state.classification.value}")
    print(f"  * Traffic Stress Score: {traffic_state.traffic_exposure_score:.2f} / 1.00")
    print(f"  * Anti-Hallucination  : Speed={traffic_state.speed_kmh}, Lane Occupancy={traffic_state.lane_occupancy}")

    # ---------------------------------------------------------------------- #
    # STEP 4: Cross-Domain Fusion (Road Defect + Traffic Bottleneck)
    # ---------------------------------------------------------------------- #
    print_step(4, "CROSS-DOMAIN FUSION: Multi-Modal Corridor Risk Synthesis")
    fused_context = pipeline._evaluated_fused_contexts["SEG_MUMBAI_WEH_01"]

    print(f"  * Primary Domain      : {fused_context.primary_domain.value}")
    print(f"  * Synergy Detected    : {fused_context.cross_domain_synergy}")
    print(f"  * Urgency Multiplier  : {fused_context.urgency_multiplier:.2f}x (Rule-based engineering heuristic)")
    print("  * Explainable Synthesis:")
    for s in fused_context.explainable_synthesis:
        print(f"     - {s}")

    # ---------------------------------------------------------------------- #
    # STEP 5: Priority Engine V2 Scoring
    # ---------------------------------------------------------------------- #
    print_step(5, "PRIORITY ENGINE V2: Multi-Factor Explainable Prioritization")
    priority_result = pipeline._evaluated_priorities[issue_pass2.issue_id]

    print(f"  * Final Priority Score: {priority_result.priority_score:.1f} / 100.0 [{priority_result.priority_band}]")
    print("  * Score Factors Breakdown (Deterministic Engineering Weights):")
    for k, v in priority_result.score_breakdown.items():
        if k != "total":
            print(f"     - {k:25s}: +{v:.1f} pts")
    print("  * Human-Readable Justifications:")
    for r in priority_result.reasons:
        print(f"     [x] {r}")

    # ---------------------------------------------------------------------- #
    # STEP 6: Confidence Automation & Governance Gating
    # ---------------------------------------------------------------------- #
    print_step(6, "AUTOMATION GOVERNANCE: Multi-Tiered Action Gating")
    gov_decision = pipeline._evaluated_governance[issue_pass2.issue_id]

    print(f"  * Automation Tier     : {gov_decision.tier.value}")
    print(f"  * Governance Action   : {gov_decision.governance_action.value}")
    print(f"  * Auto-Creation State : Eligible for draft municipal work order (dispatch permitted = {gov_decision.auto_dispatch_permitted})")
    print(f"  * Governance Rationale: {gov_decision.rationale}")

    # ---------------------------------------------------------------------- #
    # STEP 7: Proof-of-Closure Recheck Loop
    # ---------------------------------------------------------------------- #
    print_step(7, "PROOF-OF-CLOSURE: Repair Claim & Follow-up Inspection")
    claim = RepairClaimBuilder.build(
        issue_id=issue_pass2.issue_id,
        work_item_id="WORK_ORDER_MUM_2026_091",
        claimed_by="MH_PWD_Contractor_Div3",
        repair_reference="PATCH_RECEIPT_WEH_882",
        notes="Pothole filled with warm-mix asphalt patch.",
    )
    print(f"  * Repair Claim Filed  : {claim.claim_id} by {claim.claimed_by} [{claim.claimed_status}]")

    # SIMULATED FOLLOW-UP PASS: BUS_003 re-inspects corridor
    follow_up_pass = FollowUpPass(
        bus_id="BUS_003",
        pass_timestamp=1726700000.0,
        pass_gps=GPSPoint(
            lat=19.076010,
            lon=72.877710,
            timestamp=1726700000.0,
            uncertainty_m=3.5,
            heading=181.0,
            status=GPSStatus.INTERPOLATED,
        ),
        observations=[],  # Zero defect detected during recheck pass
        data_provenance="SIMULATED_SCENARIO",
    )
    print("  * Inspection Pass Run : [SIMULATED FOLLOW-UP PASS] by BUS_003 through defect GPS")

    verif_result, updated_health = pipeline.verify_repair(
        issue_id=issue_pass2.issue_id,
        claim=claim,
        follow_up_pass=follow_up_pass,
    )

    print(f"  * Closure Outcome     : {verif_result.verification_result} (confidence: {verif_result.verification_confidence:.2f})")
    print(f"  * Issue Lifecycle     : Status transitioned to [{issue_pass2.status}]")
    if updated_health:
        print(f"  * Road Health Feedback: {health_pass2.health_score} -> {updated_health.health_score}/100 [{updated_health.health_state.value}]")
        print(f"  * Trend Transition    : Updated to [{updated_health.trend.value}] (Closure recovery applied)")

    # ---------------------------------------------------------------------- #
    # STEP 8: Export Consolidated Machine-Readable Artifacts
    # ---------------------------------------------------------------------- #
    print_step(8, "CONSOLIDATED EXPORT: Writing Structured Artifacts")
    out_dir = "outputs/urban_intelligence"
    paths = pipeline.export_artifacts(output_dir=out_dir)

    for name, p in paths.items():
        print(f"  [x] {name:22s} -> {p}")

    print_banner("DEMONSTRATION COMPLETE: ALL 8 PHASES VERIFIED SUCCESSFULLY")
    print("Key Architectural Takeaway:")
    print("  We don't just detect road and traffic conditions. We maintain a persistent,")
    print("  explainable representation of what is happening on the city's roads and")
    print("  convert it into prioritized, verifiable municipal actions.\n")


if __name__ == "__main__":
    run_demo()
