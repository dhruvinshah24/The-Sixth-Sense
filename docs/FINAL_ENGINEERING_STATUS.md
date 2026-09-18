# Final Engineering Status & SIH Defense Dossier
**The Sixth Sense — AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet**  
**SIH 2026 Problem Statements: PS 26124 / PS 26125**  
**Audit Date:** 2026-09-18  
**Baseline Commit:** `730bc1e` | **Test Suite:** 165 / 165 Passing  

---

## 1. System Architecture

The Sixth Sense transforms ordinary public transit bus fleets into continuous urban monitoring nodes. It organizes processing into a unified 14-stage closed-loop architecture:

```
[BUS CAMERAS + TELEMETRY] (Person 5: Quality Gate & GPS Associator)
           │
  ┌────────┴───────────────────────────────────────────────────────┐
  │                        PERCEPTION LAYER                        │
  │  Road Defects (Person 1: RDD2022) │ Traffic Mobility (Person 2)   │
  │  VRU / Pedestrians (Person 3)     │ Incident Candidate (Person 4) │
  └────────┬───────────────────────────────────────────────────────┘
           │
[UNIFIED OBSERVATION SCHEMA] (`DomainType`: ROAD, TRAFFIC, SAFETY, INCIDENT)
           │
[CITY MEMORY & MULTI-BUS CORROBORATION] (30m spatial clustering across fleet)
           │
[CROSS-DOMAIN FUSION ENGINE] (Synergies: Compound Stress, Safety Critical, Incident Delay)
           │
[ROAD HEALTH ENGINE] (IRC:SP:20 standard structural condition scoring)
           │
[PRIORITY ENGINE V2] (Explainable 0–100 multi-factor priority ranking)
           │
[CONFIDENCE GOVERNANCE] (3-tier automation gating: Autonomous vs Human Review)
           │
[MUNICIPAL ACTION & AUDIT TRAIL] (WorkItem with SLA, DLP, and EvidenceChain)
           │
[PROOF-OF-CLOSURE VERIFICATION] (Coverage checking, defect absence, health recovery)
           │
[COMMAND CENTER & GIS] (Person 6: Interactive Leaflet map, priority queue, evidence inspector)
```

---

## 2. Person 1–6 Workstream Integration

| Workstream | Modules | Key Technical Capabilities | Integration Interface |
|---|---|---|---|
| **Person 1 (Road & Infra)** | `road_damage_detector.py`, `hazard_detector.py`, `road_marking_detector.py`, `road_health.py` | RDD2022 D00/D10/D20/D40 perception, waterlogging, divider tracking, zebra crossing contrast, IRC:SP:20 health scoring | Ingests `Observation` into `CityMemory` and `RoadHealthEngine` |
| **Person 2 (Traffic & Mobility)** | `vehicle_detector.py`, `urban_tracker.py`, `vehicle_counter.py`, `traffic_state_engine.py` | Line-crossing unique vehicle counter, 6-class separation, density tiers, persistent bottlenecks, route delay, OD flow, heatmap GeoJSON | `ingest_traffic_records(...)` feeds `TrafficStateEngine` & `CrossDomainFusionEngine` |
| **Person 3 (Safety & VRU)** | `vehicle_detector.py`, `department_router.py`, `cross_domain_fusion.py` | Pedestrian & cyclist detection, `TRAFFIC_PEDESTRIAN_SAFETY` routing, `SAFETY_CRITICAL_CORRIDOR` synergy boost | `create_safety_observation(...)` ingested into `UrbanIntelligencePipeline` |
| **Person 4 (Incident & Enforcement)** | `schemas/urban_event.py`, `department_router.py`, `frame_scheduler.py` | Incident candidate schema, mandatory officer review disclaimer, dynamic high-FPS burst capture trigger | `create_incident_observation(...)` ingested into `UrbanIntelligencePipeline` |
| **Person 5 (Platform Core & Audit)** | `video_reader.py`, `quality_gate.py`, `gps_associator.py`, `anonymiser.py`, `city_memory.py`, `work_item.py`, `evidence_chain.py`, `verification_engine.py` | Video ingestion, quality gate, GPS interpolation, in-memory face blur, 30m deduplication, WorkItem, DLP lookup, EvidenceChain, proof-of-closure verification | Core pipeline foundation & data contracts across all domains |
| **Person 6 (Command Center & GIS)** | `serve_command_center.py`, `command_center/index.html`, `command_center/js/app.js` | Threading HTTP server, Leaflet map, defect pins, priority bands, traffic intensity panel, evidence modal | Consumes `/data/`, `/traffic-data/`, and `/urban-intelligence/` |

---

## 3. Test Suite Execution & Verification

The repository contains 11 dedicated test suites running on `pytest 9.1.1` under Python 3.14.5:

```
collected 165 items
tests\test_multipass_corroboration.py .........                          [  5%]
tests\test_phase_a.py ..........................                         [ 21%]
tests\test_phase_c.py ........................................           [ 45%]
tests\test_phase_d.py ...................                                [ 56%]
tests\test_phase_e.py .........                                          [ 62%]
tests\test_phase_f.py ........                                           [ 67%]
tests\test_phase_h.py ...........                                        [ 73%]
tests\test_road_infrastructure.py ..........                             [ 80%]
tests\test_team_integration.py .....                                     [ 83%]
tests\test_traffic_mobility.py ............                              [ 90%]
tests\test_urban_intelligence.py ................                        [100%]

============================= 165 passed in 0.85s =============================
```

- **Pass Rate:** 100% (165 passed, 0 failed, 0 skipped).
- **Execution Time:** 0.85 seconds.
- **Zero-Regression Guarantee:** No tests were relaxed, modified, or disabled.

---

## 4. Real Validation Summary

- **Road Damage Perception:**
  - Validated on genuine Indian road dashcam video (`test_road.mp4`, `test_road1.mp4`) and authentic RDD2022 India asphalt pavement stills (`India_000004` to `India_008899`).
  - D00 (longitudinal crack): 4 confirmed observations in video, peak conf 0.82.
  - D20 (alligator crack): 10 detections across 5 stills, peak conf 0.78.
  - D40 (pothole): Confirmed in 2 video streams (conf 0.70) and static image (conf 0.49).
- **Traffic & Mobility:**
  - Real video counting validated on `test_road1.mp4` using 2D line-crossing with zero duplicate counting.
  - 6-class vehicle separation verified.
  - Congestion heatmap GeoJSON, route delay, and corridor OD flow verified on 69 synchronized vehicle/GPS observations (`outputs/night_drive/`).
- **Telemetry & Privacy:**
  - GPS linear interpolation and distance uncertainty verified on 61 waypoints (`data/demo_gps.csv`).
  - In-memory face blurring validated on detected person bounding boxes before disk persistence.
- **Exclusion of Indoor Lobby Footage:**
  - An early test video recorded inside an indoor tiled lobby produced crack false-positives due to tile grout lines. This was diagnosed as domain shift and **completely excluded** from all validation claims.

---

## 5. Simulated Validation Summary

The following capabilities are implemented with complete algorithmic logic but validated via simulated scenarios:
1. **Proof-of-Closure Verification:** Validated using simulated contractor repair claims and simulated subsequent fleet passes (`FollowUpPass`). We demonstrate the closed-loop verification and health recovery logic, but do not claim that actual municipal physical repairs were conducted during testing.
2. **School-Zone Geofencing:** Geofence hazard multiplier logic is verified via synthetic scenario bounding boxes; live municipal GIS boundary shapefiles are not yet wired.
3. **14-Step Master Demonstration (`run_full_team_integration_demo.py`):** Uses deterministic scenario observations to verify integration across all 6 teammates.

---

## 6. Not Implemented Features (Honest Disclosure)

1. **ANPR (Automatic Number Plate Recognition):** Deliberately deactivated; high legal liability and not required for pavement/traffic monitoring.
2. **Automated e-Challan Issuance:** Barred by architectural governance policy (requires human officer review).
3. **Vehicle Speed & Lane Occupancy:** Camera optics and elevation are uncalibrated; explicitly marked `UNAVAILABLE` in UI.
4. **Rash Driving / Hit-and-Run:** Requires 3D collision physics and metric velocity calibration; not implemented.
5. **Offending Vehicle Tracking:** Cross-camera re-identification across city fleets is not implemented.

---

## 7. Known Limitations

1. **Adverse Weather & Lighting:** High rain/glare and unlit nighttime roads degrade confidence scores; handled gracefully by the Quality Gate (`quality_gate.py`).
2. **GPS Multi-Path Drift:** In dense high-rise corridors (urban canyons), GPS uncertainty can reach 15–20m; mitigated by 30m deduplication radius and multi-bus corroboration.
3. **Dashcam Elevation & Field of View:** Fixed forward-facing cameras have blind spots directly adjacent to the bus wheels.
4. **Lack of Ground-Truth Annotations:** The RDD2022 test set lacks XML bounding box ground truth; results are visually confirmed rather than scored on formal mAP.

---

## 8. Runnable Commands & Demo Instructions

### A. Run Genuine GPU Road Damage Inference
```bash
python run_urban_ai.py --video tests/fixtures/test_road1.mp4 --process-all
```

### B. Run Real Traffic Counting & Mobility Analytics
```bash
python run_traffic_demo.py
```

### C. Run Full 14-Step Deterministic Team Integration Master Demo
```bash
python run_full_team_integration_demo.py
```

### D. Serve the Visual Command Center Dashboard
```bash
python serve_command_center.py
# Open in browser: http://127.0.0.1:8765
```

---

## 9. Performance & Latency Measurements

**Hardware Environment:** NVIDIA GeForce RTX 5050 Laptop GPU (`cuda:0`), CUDA 13.2, PyTorch 2.6.0.

| Measurement Domain | Metric | Value | Technical Context |
|---|---|---|---|
| **Model Inference Latency** | YOLO12s RDD2022 @ $1080 \times 1080$ | **20.13 ms / frame** (~49 FPS) | Raw forward-pass GPU tensor inference |
| **Model Inference Latency** | YOLO12s RDD2022 @ $848 \times 392$ | **13.38 ms / frame** (~75 FPS) | Raw forward-pass GPU tensor inference |
| **Intelligence Layer Latency** | City Memory + Fusion + Health + Priority | **< 1.5 ms / observation** | Python memory & math operations |
| **End-to-End Pipeline Throughput** | Decode + Quality Gate + Dual Inference + Tracker + Render + MP4 Encode | **3.1 – 8.5 FPS** | Full pipeline with disk writes and annotated video encoding |
| **Camera Ingestion Rate** | Frame Scheduler Baseline | **3.0 FPS** (Burst: 10.0 FPS) | Edge-optimized line-rate processing |

> **Honest Performance Disclosure:** We do NOT claim 60 FPS end-to-end processing with video writing. Raw model execution is ~49 FPS, and end-to-end pipeline execution runs at ~3.1–8.5 FPS, perfectly matching our 3.0 FPS camera ingestion policy.

---

## 10. Privacy & Governance Safeguards

1. **Privacy-by-Design:** In-memory face blurring (`anonymiser.py`) before disk persistence; zero raw facial footage retained; no MAC address or Wi-Fi packet sniffing.
2. **Consequential Decision Safeguards:** All incident proposals carry `requires_human_review: True`. Automated penalization without sworn officer review is structurally barred.
3. **Traceable Evidence Chain:** Every work order bundles a complete `EvidenceChain` preserving detector checkpoints, observation timestamps, GPS uncertainty, and bus IDs.

---

## 11. Problem Statement Requirement Coverage Summary

```
Total Requirements Evaluated: 47
├── IMPLEMENTED + REAL VALIDATED             : 31 (66.0%)
├── IMPLEMENTED + PARTIALLY VALIDATED        :  7 (14.9%)
├── IMPLEMENTED + SIMULATED/DETERMINISTIC    :  3 ( 6.4%)
└── NOT IMPLEMENTED                          :  6 (12.8%)
```

---

## 12. Recommended 2–3 Minute Live SIH Presentation Script

1. **Minute 0:00–0:45 (The Problem & Mobile Bus Concept):**
   - *"Existing smart city initiatives rely on costly fixed CCTV cameras that leave 90% of suburban roads unmonitored. The Sixth Sense turns existing public transit buses into mobile edge perception nodes."*
   - Show live video inference on `test_road1.mp4`: point out real-time D00/D40 defect bounding boxes and vehicle tracking.
2. **Minute 0:45–1:45 (City Memory & Cross-Domain Intelligence):**
   - *"A single bus detection is an observation; repeated sightings across independent buses within 30 meters create a corroborated Persistent Issue in City Memory."*
   - *"Our Cross-Domain Fusion Engine connects road defects with traffic congestion and pedestrian safety, calculating an explainable 0–100 priority score."*
   - Run `python run_full_team_integration_demo.py` and show the 14-stage execution output.
3. **Minute 1:45–2:30 (Command Center & Closed-Loop Verification):**
   - Open `http://127.0.0.1:8765`: show the interactive Leaflet map, prioritized action queue, evidence frame popup, and traffic congestion overlay.
   - Conclude with the closed-loop governance story: *"We generate draft municipal work orders with contractor warranty tracking, and verify repair closure when subsequent fleet buses re-observe the repaired location."*
