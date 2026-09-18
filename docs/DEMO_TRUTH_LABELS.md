# Demonstration Truth Labels & Evaluation Guide
**The Sixth Sense — AI-Powered Mobile Urban Intelligence Platform**  
**SIH 2026 Problem Statement: PS 26124 / PS 26125**  
**Guideline:** Strict Evaluation Transparency  

---

## 1. Core Evaluation Taxonomies

To prevent any confusion among judges, evaluators, and examiners, every runnable command and output artifact in The Sixth Sense is explicitly assigned to one of three categories:

| Taxonomy Category | Technical Meaning | Codebase Artifacts & Scripts | What Judges Should Know |
|---|---|---|---|
| **REAL PERCEPTION VALIDATION** | Deep learning and computer vision inference executed directly on real-world Indian road dashcam video, authentic RDD2022 pavement imagery, and actual vehicle traffic streams. | • `run_urban_ai.py`<br>• `run_traffic_demo.py`<br>• `models/yolo12s_RDD2022_best.pt`<br>• `outputs/demo/`<br>• `outputs/traffic_demo/`<br>• `outputs/road_infrastructure_validation/` | Genuine model inference outputs. Bounding boxes, timestamps, and confidence scores reflect real sensor data on asphalt roads. (Excludes early indoor lobby video). |
| **DETERMINISTIC SYSTEM INTEGRATION DEMONSTRATION** | End-to-end multi-domain architectural pipelines executed using deterministic scenario observations to verify that all 14 stages across Persons 1–6 communicate and function cohesively. | • `run_full_team_integration_demo.py`<br>• `run_urban_intelligence_demo.py`<br>• `outputs/urban_intelligence/`<br>• `outputs/final_integration_summary.json` | Used to demonstrate cross-domain fusion, priority scoring, confidence governance, and data schemas in a controlled, reproducible manner. It does NOT claim to be a single live video stream. |
| **SIMULATED OPERATIONAL WORKFLOW** | External municipal workflows (e.g. PWD contractor repair claims, secondary follow-up bus passes) generated to test and prove closed-loop verification logic. | • `run_sih_demo.py`<br>• `generate_verification_demo.py`<br>• `outputs/sih_demo/06_verification/`<br>• Step 14 in `run_full_team_integration_demo.py` | Demonstrates verification engine logic and health recovery. It does NOT claim that physical municipal repairs were performed on real streets during testing. |

---

## 2. Command-by-Command Evaluation Guide

### 2.1 Real Perception Runs

#### A. Real Road Damage Perception
```bash
python run_urban_ai.py --video tests/fixtures/test_road1.mp4 --process-all
```
- **What it runs:** YOLO12s RDD2022 inference on 413 frames of real Indian urban carriageway video.
- **What it produces:** 4 confirmed D00 crack observations, 1 D40 pothole observation, 32 vehicle detections.
- **Truth Label:** `REAL PERCEPTION VALIDATION`.

#### B. Real Traffic Perception & Unique Counting
```bash
python run_traffic_demo.py
```
- **What it runs:** YOLO vehicle detection + UrbianTracker + line-crossing counting on real traffic video (`test_road1.mp4`).
- **What it produces:** Class-specific vehicle counts, congestion heatmap GeoJSON, route delay analysis, and corridor OD transition matrix.
- **Truth Label:** `REAL PERCEPTION VALIDATION`.

---

### 2.2 Deterministic Integration Runs

#### A. Full Team 14-Step Master Demonstration
```bash
python run_full_team_integration_demo.py
```
- **What it runs:** Executes all 14 pipeline stages: Camera Ingestion $\to$ Road Defect $\to$ Traffic Mobility $\to$ VRU Safety $\to$ Incident Candidate $\to$ GPS $\to$ Unified Observation $\to$ City Memory $\to$ Cross-Domain Fusion $\to$ Road Health (IRC:SP:20) $\to$ Priority V2 $\to$ Governance $\to$ Municipal Work-Order $\to$ Proof-of-Closure Recheck.
- **What it produces:** Standardized JSON exports in `outputs/urban_intelligence/` and `outputs/final_integration_summary.json`.
- **Truth Label:** `DETERMINISTIC SYSTEM INTEGRATION DEMONSTRATION` (Steps 1–13 are deterministic integration tests; Step 14 is a simulated operational workflow).

---

### 2.3 Command Center UI & Visualization

```bash
python serve_command_center.py
```
- **URL:** `http://127.0.0.1:8765`
- **What it serves:**
  - `/data/*` $\to$ Phase B/C/D cached pipeline artifacts (`outputs/sih_demo/`).
  - `/traffic-data/*` $\to$ Real traffic analytics (`outputs/traffic_demo/`).
  - `/urban-intelligence/*` $\to$ Unified multi-domain intelligence artifacts (`outputs/urban_intelligence/`).
- **Truth Label on UI:**
  - Pavement issues: Verified pipeline outputs.
  - Traffic panel: Relative vehicle intensity; speed and lane occupancy are explicitly marked `unavailable`.
  - Verification badges: Reflects verification engine outcome on simulated follow-up pass.

---

## 3. Strict Prohibitions (What We Never Claim)

1. **Never claim mAP without ground truth XMLs:** We state visual and operational validation on RDD2022 test images.
2. **Never claim indoor footage as road damage:** The early lobby video is explicitly documented as domain shift.
3. **Never claim real municipal repairs:** All proof-of-closure follow-up passes are explicitly labeled simulated.
4. **Never claim ANPR or automated e-challans:** ANPR is documented as NOT IMPLEMENTED; automated fines are prohibited by governance safeguards.
5. **Never claim calibrated speed or lane occupancy:** Both are clearly marked UNAVAILABLE.
