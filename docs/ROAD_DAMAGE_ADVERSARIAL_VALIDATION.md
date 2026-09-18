# Road Damage Adversarial Validation Report — The Sixth Sense
**SIH 2026 PS 26124 / PS 26125**  
*Model Checkpoint: models/yolo12s_RDD2022_best.pt*  
*Validation Target: 16 Real-World Adversarial Visual Conditions*  
*Methodology: Visual and forensic tracking across test_road.mp4, test_road1.mp4, and authentic RDD2022 India imagery*

---

## 1. Adversarial Testing Summary Across 16 Difficult Conditions

| Condition | Visual Scenario Description | TP-like Visual Confirmation | FP-like Error Observed | FN-like Missed Case | Confidence Range | Persistence (Frames) | Observation Formed? | PersistentIssue Formed? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Shadows** | Tree canopy / flyover shadows across road | Cracks visible within penumbra detected | Hard rectilinear shadow edge triggers single-frame flicker | Hairline crack in pitch-black shadow trough | 0.25 – 0.38 | 1 frame | NO (rejected by min_detections=3) | NO |
| **2. Lane Markings** | Thermoplastic stripes & broken dashes | Cracks propagating across white stripes | Chipped paint edge mimicking transverse crack | Hairline crack under reflective paint | 0.30 – 0.42 | 1 – 2 frames | NO (insufficient persistence) | NO |
| **3. Tar Patches** | Dark bitumen sealant on old repairs | Cracking along patch joint interface | Serpentine sealant band mistaken for D00/D20 | Shallow settlement inside patch | 0.35 – 0.52 | 2 – 4 frames | Sometimes (low-severity crack) | Monitored / Review |
| **4. Road Seams** | Longitudinal paving cold joints | Raveling & joint failure detected (D00) | Pristine rolled joint detected as crack | Tight hairline seam before spalling | 0.32 – 0.48 | 3 – 6 frames | YES (observed in test_road1.mp4) | YES (WATCH state) |
| **5. Drainage Lines** | Curbside grates and gutter slots | Asphalt subsidence at curb interface | Metal drainage slot mimicking D10 | Deeply recessed curb inlets | 0.28 – 0.40 | 1 – 2 frames | NO (lateral ROI & short lifetime) | NO |
| **6. Manholes** | Circular iron utility covers & collars | Cracks around sunken manhole collar | Flush circular iron plate mistaken for D40 | Adjacent edge merged into collar | 0.25 – 0.34 | 1 – 2 frames | NO (perspective shifts drop IoU) | NO |
| **7. Speed Breakers** | Painted asphalt speed humps | Crater downstream of breaker ramp (D40) | Yellow chevron paint mistaken for D10 | Crown surface raveling | 0.22 – 0.31 | 1 frame | NO (below confidence/persistence) | NO |
| **8. Reflections** | Wet road sheen and sunlight glare | Water-filled pothole crater (test_road.mp4) | Sun glare off oil film mistaken for void | Submerged crack in turbid water | 0.27 – 0.33 | 2 – 3 frames | YES (pothole candidate confirmed) | Awaits 2nd bus |
| **9. Puddles** | Standing pooling water on carriageway | Standing water body on road shoulder | Dark pristine asphalt mistaken for sheen | Thin surface sheet flow | 0.45 – 0.60 | Persistent | YES (waterlogging event) | YES |
| **10. Dark Asphalt** | Freshly resurfaced dark mastic asphalt | Severe aggregate raveling on fresh binder | Aggressive aggregate texture flicker | Low-contrast hairline crack | 0.20 – 0.26 | 1 frame | NO (filtered by conf floor 0.25) | NO |
| **11. Camera Vibration**| Engine idle / rough road chassis shudder | Defect tracked across vertical jitter | Blurred vertical scanline smearing | Edge loss drops conf below floor | 0.24 – 0.32 | Intermittent | Filtered if frame gap > 15 | NO |
| **12. Motion Blur** | High speed or sudden transit deceleration | Elongated crack followed along axis | Smudged litter streak mistaken for crack | Pothole rim smeared into road | 0.20 – 0.28 | 1 frame | NO (Frame Quality Gate rejects blur) | NO |
| **13. Occlusions** | Preceding car bumper / shadow overlap | Partially visible pothole tracked | Exhaust shadow misclassified | >60% occluded defect missed | 0.30 – 0.45 | 3 – 8 frames | YES (uncovers as vehicle leaves) | YES |
| **14. Small Defects** | Minor spall / chip (<0.5% relative area) | Early-stage isolated pothole crater | Small flattened dark leaf or rubbish | Stone chip ravelling missed | 0.25 – 0.35 | 2 – 3 frames | Filtered if area < min_area | LOW / WATCH |
| **15. Large Defects** | Deep pothole cluster / massive alligator | Severe pothole crater (test_road1.mp4) | Rare (massive contrast & structure) | None observed | 0.65 – 0.83 | 8 – 25 frames | YES (unanimous confirmation) | YES (CRITICAL) |
| **16. Crossed Cracks**| Intersection of D00 & D10 fractures | Distinct D00 & D10 bboxes at intersection | Label oscillation (D00 vs D10 vs D20) | Intersection treated as alligator | 0.40 – 0.62 | 4 – 10 frames | YES (both classes grouped) | YES (Compound) |

---

## 2. Road Context Gating Analysis

An engineering investigation was conducted to determine whether an explicit optical carriageway context gate (e.g. road surface classifier or heuristic segmentation mask) should be inserted prior to road damage detection.

### Findings:
1. **Brittle Heuristic Failure:** Simple color, brightness, or texture thresholding (e.g. rejecting images based on darkness or tile-like gradients) leads to catastrophic false negatives under legitimate road variations (such as dark wet bitumen, shade from trees, or concrete flyover decks).
2. **Sufficiency of Multi-Signal Pipeline:**
   - **Quality Gate:** Rejects blurred frames (Laplacian variance < 100) and severe under/over-exposure.
   - **Temporal Grouping:** Requires at least 3 consecutive frame detections (min_detections >= 3) with spatial overlap (IoU >= 0.3) within a 15-frame window.
   - **Geometry Verification:** Rejects aspect ratios or relative bounding box areas outside feasible road distress profiles.
   - **Multi-Bus Corroboration:** Upgrades an issue from UNCONFIRMED to CONFIRMED only after observation by $\\ge 2$ distinct transit vehicles.

### Conclusion & Verdict:
**Current temporal + spatial safeguards retained; no additional context gate justified by available evidence.**

---

## 3. Ground Truth & Metrics Disclaimer

In accordance with strict forensic integrity standards:
- Precision, recall, and mAP are **NOT** calculated for 	est_road.mp4 or 	est_road1.mp4 because dense pixel-level bounding-box ground-truth annotations are not officially provided for these arbitrary dashcam video clips.
- Real road validation is reported via direct visual and forensic tracking of frame indices, bounding box coordinates, and model confidence scores.
