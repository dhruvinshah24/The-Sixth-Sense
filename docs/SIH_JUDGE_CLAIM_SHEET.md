# The Sixth Sense — Official SIH 2026 Judge Claim Sheet
**SIH 2026 Problem Statement PS 26124 & PS 26125**  
*AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet*  
*Target Audience:* Grand Finale Technical Evaluation Jury  
*Policy:* Absolute forensic integrity, zero unsubstantiated claims, zero fabricated benchmarks.

---

## 1. CLAIMS WE CAN SAY CONFIDENTLY (Defensible & Proven)

1. **Automated Test Coverage:**
   - *"Our complete platform test suite achieves 185/185 passing tests across 13 test suites with zero regressions."*
2. **Real-World Perception Model:**
   - *"We deploy a specialized YOLO12s model (`yolo12s_RDD2022_best.pt`) trained on RDD2022 road damage data, capable of distinguishing longitudinal cracks (D00), transverse cracks (D10), alligator cracks (D20), and potholes (D40) on asphalt roads."*
3. **Hardware Acceleration:**
   - *"The perception pipeline runs on genuine GPU edge hardware (tested on an NVIDIA GeForce RTX 5050 Laptop GPU with PyTorch CUDA FP16 acceleration)."*
4. **Authentic Road Video Validation:**
   - *"The platform has been validated on genuine Indian road dashcam footage (`test_road.mp4` and `test_road1.mp4`), successfully capturing real road fissures, potholes, and moving vehicle fleets."*
5. **Calibrated Defect Thresholding:**
   - *"We empirically calibrated per-class confidence thresholds across 461 real road frames. Lowering the pothole threshold from 0.35 to 0.25 recovered water-filled potholes that were previously missed due to specular surface reflections."*
6. **Heading-Aware Carriageway Deduplication:**
   - *"City Memory integrates a directional heading guard ($|\Delta\theta| > 110^\circ$). In our 7-scenario adversarial stress test, this eliminated false issue merging between opposing carriageways of divided arterials."*
7. **Multi-Bus Corroboration:**
   - *"City Memory algorithmically promotes single-vehicle sightings into confirmed municipal issues only after multi-bus fleet corroboration."*
8. **Explainable Priority Scoring:**
   - *"The Priority Engine V2 deterministically ranks issues (0–100) by combining defect severity, model confidence, multi-bus corroboration, traffic density, and pedestrian safety exposure."*
9. **Full Cross-Domain Integration:**
   - *"The platform successfully unifies Persons 1 through 6 into a cohesive event pipeline with an interactive Leaflet GIS Command Center."*
10. **Deterministic Reproducibility:**
    - *"All algorithmic scoring functions, Bayesian confidence updates, and governance decision gates are 100% mathematically deterministic and bounded."*

---

## 2. CLAIMS WE MUST QUALIFY (Accurate Context Required)

1. **Pipeline Throughput:**
   - *State:* *"The end-to-end Python pipeline processes video at **3.1 to 8.5 FPS** (which comfortably satisfies our 3.0 FPS camera capture policy). Raw GPU tensor forward-pass executes at **31.4 to 49.7 FPS**, but full end-to-end processing with quality gating, tracking, and video disk writes runs at 3.1–8.5 FPS."*
2. **Vehicle Tracking:**
   - *State:* *"Vehicle tracking is executed by `UrbianTracker`, an IoU-based tracker with class constraints and temporal persistence. It is not Kalman-filter ByteTrack or deep appearance ReID."*
3. **Vehicle Counting:**
   - *State:* *"Line-crossing vehicle counting guarantees zero duplicate counting for tracked vehicles on tested footage via persistent track ID locking. We do not claim generalized open-world counting accuracy without multi-camera ground truth."*
4. **Traffic Mobility Metrics:**
   - *State:* *"Traffic density and congestion levels are computed from observation-level proxy metrics (vehicle count per window and bounding box area occupancy). Calibrated metric speed (km/h) and precise lane occupancy remain marked as UNAVAILABLE without metric survey calibration."*
5. **Road Health Index:**
   - *State:* *"Our segment road-health index (0–100) is an engineering prototype formulated in alignment with IRC:SP:20 distress deduction principles. It is not an officially certified Indian Road Congress laboratory assessment."*
6. **Proof-of-Closure Workflow:**
   - *State:* *"The closed-loop repair verification state machine (Claim -> Secondary Fleet Pass -> Verified/Reopened -> Health Update) is fully implemented and validated through deterministic simulation. We do not claim real-world municipal contractor repairs were verified in physical field trials."*
7. **Privacy Masking:**
   - *State:* *"The prototype includes privacy-by-design safeguards that apply Gaussian blur to detected human upper-body/head regions. License plate blurring is inactive pending dedicated ANPR hardware/model integration."*
8. **Governance & Automation:**
   - *State:* *"The governance engine provides decision support and generates draft municipal work orders for high-confidence corroborated defects. All high-consequence legal, enforcement, or financial actions strictly require mandatory human signoff."*

---

## 3. CLAIMS WE MUST NOT MAKE (Strictly Prohibited)

1. **DO NOT CLAIM 100% AI ACCURACY:**
   - *Never say:* *"Our AI detects road damage with 100% accuracy / 100% precision / 100% recall."* (No real computer vision model achieves 100% across arbitrary real-world weather, lighting, and occlusions).
2. **DO NOT CLAIM 45+ FPS END-TO-END PIPELINE:**
   - *Never say:* *"Our whole video platform runs at 45+ or 60 FPS end-to-end."* (45+ FPS is isolated GPU forward pass; full pipeline with disk I/O runs at 3.1–8.5 FPS).
3. **DO NOT CLAIM BYTETRACK OR DEEPSORT:**
   - *Never say:* *"We implemented ByteTrack with Kalman filters and deep appearance embedding."* (We implemented `UrbianTracker`).
4. **DO NOT CLAIM ANPR OR E-CHALLAN ENFORCEMENT:**
   - *Never say:* *"We automatically read license plates, detect rash driving, and issue automated e-challan fines."* (ANPR and automated penalization are NOT implemented; non-enforcement is a core governance safeguard).
5. **DO NOT CLAIM SUB-METER GPS ACCURACY:**
   - *Never say:* *"Our system pinpoints defects with sub-meter or centimeter GPS accuracy."* (Standard transit bus GPS uncertainty is 8 to 15+ meters; coordinates are interpolated with dynamic speed-dependent uncertainty).
6. **DO NOT CLAIM REAL CONTRACTOR REPAIR VERIFICATION:**
   - *Never say:* *"We verified that municipal contractors repaired real potholes on Indian city streets."* (Proof-of-closure was evaluated on simulated operational fleet passes).
7. **DO NOT CLAIM STATUTORY DPDP CERTIFICATION:**
   - *Never say:* *"The software is certified fully compliant with the Digital Personal Data Protection Act."* (We implement prototype privacy-by-design safeguards, not a statutory legal audit).

---

## 4. BEST VERIFIED NUMBERS (Forensic Reference Table)

| Metric | Verified Value | Benchmark Condition | Hardware / Data | Classification | Limitation Disclosed |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Pytest Pass Count** | **185 / 185** | Full test suite execution (`pytest tests/`) | CPU / Local Workspace | `REAL VERIFIED` | Unit & integration test coverage |
| **Test Suites** | **13 / 13** | 100% green pass rate | Python 3.14.5 | `REAL VERIFIED` | All suites passing |
| **Raw Model Latency (1080p)** | **20.13 – 31.83 ms** | Single-frame GPU forward pass (imgsz 640) | NVIDIA RTX 5050 Laptop GPU (CUDA FP16) | `REAL BENCHMARK` | Raw tensor inference only; excludes decode/render/encode |
| **Raw Model Throughput** | **31.4 – 49.7 FPS** | Inverted forward-pass latency ($1 / \text{latency}$) | NVIDIA RTX 5050 Laptop GPU | `REAL BENCHMARK` | GPU compute headroom; not full video pipeline |
| **End-to-End Pipeline FPS** | **3.1 – 8.5 FPS** | Full video pipeline with decode, tracking & disk encoding | RTX 5050 + SSD disk writes | `REAL BENCHMARK` | Meets 3.0 FPS capture rate; slower than raw model |
| **Dashcam Capture Baseline**| **3.0 FPS** | Phase A frame scheduler baseline policy | Edge Camera Simulator | `REAL BENCHMARK` | Burst triggers to 10 FPS for incidents |
| **Adversarial Dedup Matrix**| **7 / 7 (100%)** | Correct outcome across 7 adversarial scenarios | Synthetic adversarial coordinates | `DETERMINISTIC TEST`| Validated on tested matrix; urban multipath may exceed 20m |
| **Wet Pothole Recovery** | **2 of 2 frames** | Recovered D40 potholes at conf 0.25 (missed at 0.35) | `test_road.mp4` (48 frames) | `REAL BENCHMARK` | Specific to tested wet-road video snippet |
| **Arterial Track Count** | **17 tracks** | Confirmed unique vehicle tracks | `test_road1.mp4` (413 frames) | `REAL BENCHMARK` | UrbianTracker performance on active arterial |
| **GPS Base Uncertainty** | **8.0 meters** | Baseline GNSS fix uncertainty on moving fleet | Dynamic equation ($8.0 + 0.5 v$) | `REAL PROTOCOL` | Standard commercial transit GPS accuracy |
| **Road Health Deductions** | **0 – 100 scale** | Segment score deduction based on defect clusters | `RoadHealthEngine` (IRC:SP:20 aligned) | `DETERMINISTIC LOGIC`| Engineering prototype, not certified pavement lab test |
| **Work Order Auto-Dispatch**| **$\ge 0.65$ conf** | Eligible for draft work order when bus_count $\ge 2$ | `ConfidenceAutomation` | `DETERMINISTIC LOGIC`| Generates draft work order; human signoff for legal items |
