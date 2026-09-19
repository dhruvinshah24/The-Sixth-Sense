# The Sixth Sense — Model, Metrics & Architectural Truth Audit
**SIH 2026 Problem Statement PS 26124 & PS 26125**  
*AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet*  
*Document Status:* Comprehensive Pre-Demo System Verification & Forensic Integrity Audit  
*Core System Freeze:* Core codebase frozen. Zero speculative modifications.

---

## 1. Actual Model Identity (Section A)

### Exact In-Code & Disk Verification:
- **Exact Weight Filename:** `models/yolo12s_RDD2022_best.pt` (18.1 MB)
- **Local Disk Verification:** Present on disk in `models/yolo12s_RDD2022_best.pt`. (There is NO `models/best.pt` on disk).
- **HuggingFace Repository:** `rezzzq/yolo12s-road-damage-rdd2022` (checkpoint `yolo12s_RDD2022_best.pt`)
- **Model Architecture:** `YOLOv12s` (`DetectionModel` with `A2C2f` attention layers in backbone and head, scale `'s'` [0.5, 0.5, 1024], configuration `yolov12s.yaml`).
- **Actual Model Output Classes (5 total):**
  ```python
  {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40', 4: 'Repair'}
  ```
- **Taxonomy Mapping (`_RDD_CLASS_MAP` in `road_damage_detector.py`):**
  - `D00` (Longitudinal Crack) -> `EventType.ROAD_CRACK`
  - `D10` (Transverse Crack) -> `EventType.ROAD_CRACK`
  - `D20` (Alligator/Fatigue Crack) -> `EventType.ROAD_CRACK`
  - `D40` (Pothole) -> `EventType.POTHOLE`
  - `Repair` (Pavement Patch/Repair) -> `EventType.ROAD_REPAIR`
- **Actual Inference Configuration (`config/profiles.yaml` -> `urban_mvp`):**
  - `road_damage_imgsz`: 640
  - `road_damage_fp16`: true (CUDA half-precision)
  - `target_fps`: 3 (baseline fleet dashcam sampling policy)
- **Actual Production Thresholds (`config/profiles.yaml`):**
  - `D00`: 0.28 (calibrated from continuous longitudinal crack runs)
  - `D10`: 0.25 (calibrated to recover transient transverse cracks)
  - `D20`: 0.25 (alligator cracks)
  - `D40`: 0.25 (calibrated to recover water-filled potholes with specular reflections)
  - Semantic Aliases: `pothole`: 0.25, `crack`: 0.28, `road_damage`: 0.30
  - Vehicle Thresholds: `car`: 0.45, `bus`: 0.45, `truck`: 0.45, `motorcycle`: 0.40, `bicycle`: 0.40, `person`: 0.40

### Contradictory Claims Found Across Repository:
1. `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md` (earlier draft) referred to the model as `best.pt (RDD2022 YOLOv8n)` and `YOLOv8`.
2. `sixth_sense/perception/road_damage_detector.py` docstrings referred to "YOLOv8 model (keremberke/yolov8-road-damage-detection)".
3. Historical development logs referred to `yolo12s_RDD2022_best.pt` and `best.pt` interchangeably.

### Claims Fixed:
- `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md` corrected to identify `models/yolo12s_RDD2022_best.pt` as YOLO12s architecture.
- Documented that `yolo12s_RDD2022_best.pt` utilizes the Ultralytics PyTorch wrapper with YOLO12s attention backbone.

### Claims That MUST NOT Be Used:
- **DO NOT SAY:** "We trained YOLOv8n from scratch."
- **DO NOT SAY:** "Our model file is `best.pt`."
- **DO NOT SAY:** "We use YOLO COCO weights to detect potholes." (COCO has no pothole class; specialized RDD2022 weights are mandatory).

---

## 2. Performance Truth Audit (Section B)

### Measured Benchmark Numbers (NVIDIA GeForce RTX 5050 Laptop GPU / AMD Ryzen):
1. **Raw YOLO12s Road Damage Forward-Pass:**
   - 1080p frame ($1080 \times 1080$ scaled to imgsz 640): **20.13 ms – 31.83 ms** (~31.4 to 49.7 raw GPU tensor FPS).
2. **Raw General Vehicle/Pedestrian Detector Forward-Pass:**
   - YOLO11x / YOLOv8n: **16.8 ms – 35.0 ms** depending on co-loaded profile.
3. **Tracking Overhead (`UrbianTracker`):**
   - **4.5 ms – 6.2 ms** per frame (IoU computation, trajectory maintenance, track confirmation).
4. **Observation Building & Spatial ROI Filtering:**
   - **1.8 ms – 2.5 ms** per frame.
5. **City Memory & Cross-Domain Fusion:**
   - **< 1.5 ms** per event observation.
6. **Annotation Rendering & Video Encoding:**
   - OpenCV bounding box HUD drawing + H.264 MP4 disk encoding: **45 ms – 90 ms** per frame.
7. **Full End-to-End Pipeline Throughput:**
   - **3.1 – 8.5 FPS** (End-to-end: Video decode + quality gate + dual AI inference + tracker + observation builder + visual HUD rendering + MP4 file writing).
8. **Fleet Camera Capture Policy:**
   - **3.0 FPS baseline** (adaptive burst trigger to 10 FPS during incident candidates).
   - **Conclusion:** 3.1–8.5 FPS full pipeline comfortably satisfies and exceeds the 3.0 FPS capture rate in real time.

### Contradictory Claims Found & Corrected:
- **Contradiction:** An earlier section stated "Total Pipeline Throughput: > 45 FPS sustained on GPU".
- **Correction:** >45 FPS represents isolated raw GPU tensor forward pass (`1 / 20.13ms = 49.6 FPS`). When disk I/O, video decoding, quality gating, tracking, and video encoding are active, end-to-end throughput is **3.1 – 8.5 FPS**.
- **Rule:** Never combine raw model FPS with end-to-end pipeline FPS.

---

## 3. Road Damage Accuracy Claim Audit (Section C)

### Empirical Validation Breakdown:
1. **Real Dashcam Video (`test_road.mp4` — 48 frames, wet road):**
   - At threshold $0.35$: 0 detections (false negative due to water reflection specular glint).
   - At threshold $0.25$: 2 verified detections of D40 pothole (Frame 9 conf 0.329, Frame 18 conf 0.271).
   - *Truth Claim:* "Recovered 2 previously missed wet-pothole detections on genuine dashcam footage."
   - *Prohibited Claim:* NEVER call this "100% recall" or "100% accuracy improvement".
2. **Real Dashcam Video (`test_road1.mp4` — 413 frames, arterial corridor):**
   - D00 (Longitudinal Cracks): 19 detections across frames (average conf 0.44, peak 0.68).
   - D10 (Transverse Cracks): 2 detections (conf 0.413). Transverse fissures move out of dashcam FOV in 2 frames.
   - D40 (Potholes): 8 detections across 3 spatial defect clusters.
3. **RDD2022 Still-Image Evaluation Samples:**
   - Evaluated across RDD2022 India evaluation imagery to confirm class parsing and bbox localization.
4. **Accuracy vs Detection Count Disclosure:**
   - Raw detection count $\neq$ accuracy. Without full dense per-pixel ground-truth annotations across 10,000 continuous frames, reporting detection count as "accuracy" is prohibited.
   - *Allowed Terminology:* "Empirically validated on 461 real road frames and RDD2022 test samples."

---

## 4. Temporal Confirmation Truth (Section D)

### Production vs Evaluation Configuration:
- **Production `urban_mvp` Profile (`config/profiles.yaml`):**
  - `min_detections: 3`
  - *Engineering Rationale:* Maximizes precision in production fleet operations by rejecting single-frame and two-frame optical flickers (tree shadows, overhead cables, wiper sweeps, road glare).
- **Evaluation Profiles (`indian_road_eval` / Empirical Sweeps):**
  - `min_detections: 2`
  - *Engineering Rationale:* Captures fast-moving transverse cracks (D10) that pass through forward dashcam FOV in only 2 frames at vehicle speeds $> 40\text{ km/h}$.
- **Decision:** Production code retains `min_detections: 3` in `urban_mvp` to preserve strict noise immunity, with evaluation documentation explicitly noting the speed-dependent trade-off.

---

## 5. City Memory / Deduplication Audit (Section E)

### Production Parameters:
- `dedup_radius_m`: `20.0` meters (calibrated from older 30.0 m default).
- `heading guard threshold`: `110.0` degrees difference ($|\Delta\theta| > 110^\circ$ prevents merging).
- `heading availability behavior`: If either observation lacks heading, system falls back to distance-only check ($dist \le 20\text{ m}$).
- `class matching`: Strict type equality (`issue.event_type == obs.event_type`). Different defect classes never merge.
- `GPS uncertainty behavior`: Observations with `GPSStatus.UNAVAILABLE` are never merged spatially into existing issues.

### Tested Adversarial Scenario Matrix (7 Scenarios):
1. Same defect exact coordinates -> Merged (Correct)
2. Same defect 3m GPS drift -> Merged (Correct)
3. Same defect 10m GPS drift -> Merged (Correct)
4. Distinct defect 25m away -> Separated (Correct, $25\text{m} > 20\text{m}$)
5. Distinct defect 45m away -> Separated (Correct, $45\text{m} > 20\text{m}$)
6. Opposing carriageway defect 12m away, heading $180^\circ$ opposite -> Separated (Correct, heading guard triggered)
7. Different defect class at exact same location -> Separated (Correct, class guard triggered)

### Truth Phrasing:
- **Permitted Claim:** "100% correct outcomes on the tested 7-scenario adversarial deduplication matrix."
- **Prohibited Claim:** NEVER claim "100% real-world deduplication accuracy". Real-world GPS multipath reflection in urban canyons can exceed 20m.

---

## 6. Tracker Truth (Section F)

### In-Code Tracker Identity:
- **Actual Tracker:** `UrbianTracker` (`sixth_sense/tracking/urban_tracker.py`).
- **Algorithm:** Bounding box IoU-based bipartite association + class constraints + temporal track confirmation (`confirm_frames=3`) + coasting memory (`max_lost_frames=10`).
- **What It Is NOT:** It is NOT Kalman-filter ByteTrack, SORT, or Deep appearance ReID (DeepSORT).
- **Tracker Limitations:**
  - Fast-moving vehicles with sudden erratic acceleration or zero IoU frame-to-frame can cause track reassignment.
  - Prolonged total occlusion (>10 frames) drops the track and initializes a new ID upon re-emergence.
- **Audit Action:** Removed historical mentions of "ByteTrack" in `docs/FINAL_ACCURACY_AUDIT.md`. All documentation now accurately reflects `UrbianTracker`.

---

## 7. Vehicle Counting Truth (Section G)

### Four-Level Operational Distinction:
1. **Raw Detections:** Every bounding box predicted in every single video frame (e.g., 547 boxes in `test_road1.mp4`).
2. **Confirmed Tracks:** Continuous temporal tracks meeting confirmation criteria (`confirm_frames >= 3`, e.g., 17 distinct tracks).
3. **Observations:** High-level urban intelligence events emitted by `ObservationBuilder` after temporal confirmation (`min_detections`).
4. **Unique Line-Crossing Counts:** Tracks whose trajectory centroids cross a defined virtual counting line or enter an ROI.
- **Duplicate Prevention Guarantee:** Maintained via `self.counted_track_ids: Set[int]`. Once a track ID is counted, it is permanently locked out from re-counting.
- **Truth Phrasing:** "Validated counting invariant on tested footage with zero duplicate counting of tracked IDs." (Do not claim generalized open-world counting accuracy without ground-truth multi-camera benchmarks).

---

## 8. GPS & Telemetry Truth (Section H)

### GPS Ingestion Safeguards (`GPSAssociator`):
- **Status Enum:** Strictly assigned as `DIRECT`, `INTERPOLATED`, or `UNAVAILABLE`. Coordinates are NEVER fabricated.
- **Interpolation:** Linear interpolation between adjacent GPS fixes with alpha parameter $\alpha \in [0, 1]$.
- **Stale Gap Handling:** Any gap $> 10.0$ seconds emits `lat=0.0, lon=0.0, uncertainty_m=9999.0, status=GPSStatus.UNAVAILABLE`.
- **Dynamic Uncertainty Propagation:**
  $$\text{uncertainty} = \text{base\_uncertainty} (8.0\text{ m}) + \text{speed\_mps} \times 0.5$$
- **Truth Phrasing:** Typical GPS uncertainty is **8.0 to 15.0+ meters** on moving transit vehicles. We do NOT claim blanket sub-meter GPS accuracy without RTK differential hardware.

---

## 9. Traffic Intelligence Truth (Section I)

### Capability Audit Matrix:
| Capability | Implementation Status | Method / Description | Limitations Disclosed |
| :--- | :--- | :--- | :--- |
| **Vehicle Detection** | `REAL VALIDATED` | YOLOv8 / YOLO11 on real video | Detects car, bus, truck, motorcycle, bicycle |
| **Vehicle Classification** | `REAL VALIDATED` | Multi-class bounding boxes | Limited to 5 COCO transit categories |
| **Vehicle Counting** | `REAL VALIDATED` | Virtual line crossing & ROI entering | Requires stable track; occlusion can split tracks |
| **Traffic Density** | `PARTIALLY VALIDATED` | Observation-level proxy (count / window & bbox area) | Not calibrated vehicles/km without camera calibration |
| **Congestion State** | `PARTIALLY VALIDATED` | State machine: NORMAL_FLOW, SLOW_FLOW, CONGESTION | Mapped from proxy density; not physical queue length |
| **Persistent Bottleneck** | `PARTIALLY VALIDATED` | Temporal recurrence counter across observation windows | Based on recurring window counts |
| **Route Delay** | `SIMULATED` | Bus travel time delta against schedule baseline | No external Google Maps / TomTom live API integration |
| **Origin-Destination (OD)** | `SIMULATED` | Corridor transit flow checkpoints | No individual vehicle license-plate OD tracking |
| **Congestion Heatmap** | `REAL VALIDATED` | GeoJSON export rendered dynamically on Leaflet GIS | Intensity weighted by vehicle concentration |

---

## 10. Incident & Enforcement Truth (Section J)

### Unimplemented Features Disclosed:
- **ANPR (Automatic Number Plate Recognition):** **NOT IMPLEMENTED.** (`_PLATE_BLUR_CLASSES = set()`, no CRNN/PaddleOCR plate reading engine).
- **Rash Driving Detection:** **NOT IMPLEMENTED.** (Requires multi-sensor IMU/telematics and calibrated metric speed).
- **Hit-and-Run Detection:** **NOT IMPLEMENTED.** (High legal liability, requires wide-angle 360 multi-camera tracking).
- **Offending Vehicle Tracking:** **NOT IMPLEMENTED.**
- **Automated e-Challan / Fines:** **NOT IMPLEMENTED & EXPLICITLY FORBIDDEN BY POLICY.**

### Implemented Incident Features:
- **Incident Candidate Event Schema:** `create_incident_observation(...)` defines non-enforcement urban candidates (`OBSTRUCTION_BREAKDOWN_CANDIDATE`, `WATERLOGGING_HAZARD`).
- **Enforcement Safeguard:** Mandatory officer review flag (`enforcement_safeguard = "MANDATORY_OFFICER_REVIEW"`, `requires_human_review = True`). Zero automated penalization is executed.

---

## 11. Proof-of-Closure Truth (Section K)

### Implementation vs Operational Reality:
- **System Logic Status:** `IMPLEMENTED + UNIT TESTED`.
  - Workflow: Contractor Repair Claim filed -> Secondary bus pass observation -> Spatial/temporal matching -> Outcome classified (`VERIFIED_REPAIRED`, `REOPENED`, `REVIEW_REQUIRED`) -> Pavement health score updated.
- **Physical Road Verification Status:** `SIMULATED OPERATIONAL VALIDATION`.
  - Evaluated on deterministic multi-bus scenarios and synthetic post-repair passes.
  - We do NOT have access to before-and-after physical road re-paving passes from Indian municipal contractors.
  - *Truth Rule:* Never claim "real municipal repair independently verified on live city streets". Present as "verified algorithmic proof-of-closure workflow".

---

## 12. Privacy Truth (Section L)

### Prototype Safeguards Implemented:
- **Person Head/Face Masking:** Detected `person` bounding boxes have their upper 20% region blurred with Gaussian filter before snapshot export.
- **Float Bounding Box Gating:** Bounding box coordinates are strictly clamped to image boundary integers, preventing OpenCV crop crashes on boundary edge cases.
- **Raw Image Persistence:** No unmasked raw frames showing human faces are written to municipal incident export stores.
- **Plate Masking Status:** Inactive pending dedicated ANPR integration.
- **Legal Compliance Phrasing:** Use **"privacy-by-design prototype safeguards"**. Do NOT claim full legal certification under the Digital Personal Data Protection (DPDP) Act without a formal statutory legal audit.

---

## 13. Road Health Index Truth (Section M)

### Index Formulation:
- Formulated as an **IRC:SP:20-aligned prototype road-health index**.
- Segment score begins at $100.0$ and applies standardized deductions:
  - Critical Pothole (D40): $-15.0$ to $-25.0$
  - Alligator Fatigue Crack (D20): $-10.0$
  - Linear Crack (D00/D10): $-4.0$
- Output Score: strictly bounded $RHI \in [0.0, 100.0]$.
- **Truth Phrasing:** This is an engineering prototype metric aligned with Indian Road Congress distress principles. It is NOT an official IRC-certified or certified laboratory pavement assessment.

---

## 14. Governance & Automation Truth (Section N)

### Production Governance Tiers (`confidence_automation.py`):
1. **High Consequence Events:** (`is_high_consequence=True`):
   - Tier: `REVIEW_REQUIRED`
   - Governance Action: `HUMAN_APPROVAL_REQUIRED`
   - Auto-dispatch: Strictly `False`, Human Signoff: `True`.
2. **Low Confidence (< 0.40):**
   - Tier: `LOG_AND_GROUP`
   - Governance Action: `INTERNAL_MONITORING`
   - Auto-dispatch: `False`.
3. **Moderate Confidence (0.40 – 0.65):**
   - Tier: `REVIEW_REQUIRED`
   - Governance Action: `SUPERVISOR_REVIEW_CANDIDATE`
   - Auto-dispatch: `False`, Human Signoff: `True`.
4. **High Confidence (>= 0.65) with Multi-Bus Corroboration (bus_count >= 2):**
   - Tier: `ACTIONABLE_WORK_ORDER`
   - Governance Action: `DISPATCHABLE_TASK`
   - Auto-dispatch: `True` (eligible for automated municipal draft work order).
5. **High Confidence (>= 0.65) with Single Bus (bus_count < 2):**
   - Tier: `AWAITING_FLEET_CORROBORATION`
   - Governance Action: `INTERNAL_MONITORING`
   - Auto-dispatch: `False`.

- **Truth Phrasing:** The platform provides **confidence-aware decision support for municipal work orders**. It does NOT claim fully autonomous governmental legal or financial enforcement.

---

## 15. Final Document Consistency Audit Table (Section O)

| Target Concept | Inconsistent Historical Claim | Actual Code & Empirical Evidence | Correct Truthful Wording | Files Audited & Fixed |
| :--- | :--- | :--- | :--- | :--- |
| **Model Identity** | "YOLOv8n `best.pt`" | Checkpoint is `yolo12s_RDD2022_best.pt` (YOLO12s architecture with A2C2f attention) | `YOLO12s RDD2022 (models/yolo12s_RDD2022_best.pt)` | `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md`, `profiles.yaml` |
| **Tracker** | "ByteTrack 15-frame coasting" | Tracker is `UrbianTracker` (IoU-based matching + temporal confirmation) | `UrbianTracker (IoU-based tracker; not ByteTrack)` | `docs/FINAL_ACCURACY_AUDIT.md`, `urban_tracker.py` |
| **Pipeline FPS** | "> 45 FPS sustained throughput" | Raw GPU forward pass is ~31–49 FPS; full pipeline with video decode/write is 3.1–8.5 FPS | `3.1–8.5 FPS end-to-end (exceeds 3.0 FPS dashcam capture target)` | `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md`, `docs/FINAL_ROBUSTNESS_AUDIT.md` |
| **Dedup Precision** | "100% real-world deduplication accuracy" | Tested on a 7-scenario adversarial stress matrix (same defect, drift, opposing lane, class mismatch) | `100% correct outcomes on the tested 7-scenario adversarial matrix` | `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md`, `issue_manager.py` |
| **Defect Recall** | "100% recall / 100% accuracy improvement" | Lowering D40 threshold from 0.35 to 0.25 recovered 2 missed wet-pothole detections in `test_road.mp4` | `Recovered 2 previously missed wet-pothole detections on test footage` | `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md` |
| **Governance Thresholds**| "Conf >= 0.85 for auto-dispatch" | Production code uses `high_threshold = 0.65` and `b_count >= 2` | `High confidence (>= 0.65) with multi-bus corroboration (>= 2)` | `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md`, `confidence_automation.py` |
| **Privacy Redaction** | "Facial and license plate regions in FOV are masked" | Person face/head upper region masked; plate mask inactive pending ANPR | `Upper-body/head masking for persons; plate masking inactive` | `docs/FINAL_ACCURACY_MAXIMIZATION_REPORT.md`, `anonymiser.py` |
| **Proof-of-Closure** | "Real municipal repair independently verified" | Logic implemented and verified on deterministic multi-pass test scenarios | `Simulated operational validation of proof-of-closure workflow` | `run_full_team_integration_demo.py`, `verification_engine.py` |
| **ANPR / Speed** | Unqualified enforcement mentions | ANPR, calibrated metric speed, and automated fines are NOT implemented | `ANPR and automated enforcement: NOT IMPLEMENTED` | All documentation & demo scripts |
