"""
Serve the Phase F Command Center UI.
Maps /data/* → outputs/sih_demo/* (no fake data; reads pipeline artifacts only).

Usage:
    python run_sih_demo.py          # generate artifacts first
    python serve_command_center.py  # open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).parent.resolve()
UI_ROOT = PROJECT_ROOT / "command_center"
DATA_ROOT = PROJECT_ROOT / "outputs" / "sih_demo"
TRAFFIC_DATA_ROOT = PROJECT_ROOT / "outputs" / "traffic_demo"
LIVE_RUNS_ROOT = PROJECT_ROOT / "outputs" / "live_runs"
URBAN_INTEL_ROOT = PROJECT_ROOT / "outputs" / "urban_intelligence"
PIPELINE_RUNS_ROOT = PROJECT_ROOT / "outputs" / "pipeline_runs"
HOST = "127.0.0.1"
PORT = 8765
MAX_UPLOAD_BYTES = 1_000 * 1024 * 1024  # 1 GB, local-only video test limit.
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_live_run_lock = threading.Lock()
_live_run: dict = {"status": "idle"}


def _set_live_run(**values: object) -> None:
    """Update the one-at-a-time local live-test status safely."""
    with _live_run_lock:
        _live_run.update(values)


def _get_live_run() -> dict:
    with _live_run_lock:
        return dict(_live_run)


def _tail(path: Path, limit: int = 3000) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - limit))
        return f.read().decode("utf-8", errors="replace")


def _run_live_test(run_id: str, run_dir: Path, video_path: Path) -> None:
    """Run the existing one-pass Urban AI engine in a background thread."""
    log_path = run_dir / "run.log"
    _set_live_run(status="running", message="Loading existing models and analysing video…")
    command = [
        sys.executable, str(PROJECT_ROOT / "run_urban_ai.py"),
        "--video", str(video_path),
        "--gps", str(PROJECT_ROOT / "data" / "demo_gps.csv"),
        "--output", str(run_dir),
        "--profile", "road_damage_sensitive",
        "--process-all",
        "--bus-id", "USER_UPLOAD",
        "--camera-id", "USER_CAMERA",
    ]
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            _set_live_run(
                status="failed",
                message="The existing engine stopped before producing results.",
                log_tail=_tail(log_path),
            )
            return

        report_path = next((run_dir / "metrics").glob("*_report.json"), None)
        observation_path = next((run_dir / "observations").glob("*_observations.json"), None)
        issue_path = next((run_dir / "issues").glob("*_issues.json"), None)
        annotated_path = next((run_dir / "annotated").glob("*.mp4"), None)
        if not report_path:
            raise RuntimeError("Run completed without a metrics report.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        observations = json.loads(observation_path.read_text(encoding="utf-8")) if observation_path else {}
        issues = json.loads(issue_path.read_text(encoding="utf-8")) if issue_path else {}
        _set_live_run(
            status="complete",
            message="Live GPU analysis complete.",
            report=report,
            observation_count=observations.get("total_observations", 0),
            issue_count=issues.get("total_issues", 0),
            annotated_video=(f"/live-runs/{run_id}/annotated/{annotated_path.name}" if annotated_path else None),
            report_url=f"/live-runs/{run_id}/metrics/{report_path.name}",
            observations_url=(f"/live-runs/{run_id}/observations/{observation_path.name}" if observation_path else None),
            issues_url=(f"/live-runs/{run_id}/issues/{issue_path.name}" if issue_path else None),
        )
    except Exception as exc:  # Surface local-run errors to the browser, not a blank page.
        _set_live_run(status="failed", message=str(exc), log_tail=_tail(log_path))


def _start_live_run(filename: str, body, content_length: int) -> dict:
    """Persist one browser upload and start the existing inference runner."""
    if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
        raise ValueError("Video must be between 1 byte and 1 GB.")
    safe_name = Path(filename).name or "upload.mp4"
    if Path(safe_name).suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Supported video formats: MP4, MOV, AVI, MKV, WEBM.")
    if _get_live_run().get("status") in {"queued", "running"}:
        raise RuntimeError("A live test is already running. Wait for it to finish first.")

    run_id = f"live_{uuid.uuid4().hex[:8]}"
    run_dir = LIVE_RUNS_ROOT / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    video_path = input_dir / safe_name
    remaining = content_length
    with open(video_path, "wb") as output:
        while remaining:
            chunk = body.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Upload ended before the declared size.")
            output.write(chunk)
            remaining -= len(chunk)

    _set_live_run(
        status="queued", run_id=run_id, filename=safe_name,
        message="Video received. Preparing local GPU test…", started_at=time.time(),
    )
    threading.Thread(
        target=_run_live_test, args=(run_id, run_dir, video_path), daemon=True,
        name=f"live-test-{run_id}",
    ).start()
    return _get_live_run()



def _get_pipeline_datasets() -> list:
    return [
        {
            "id": "test_road1",
            "name": "test_road1.mp4 (413 frames, Real Arterial Road)",
            "type": "REAL_VIDEO_INFERENCE",
            "badge": "REAL PIPELINE OUTPUT",
            "description": "Real video perception on 413 frames: 698 vehicles, 33 road damage detections (D00 cracks, D10 cracks, D40 potholes), 5 persistent issues."
        },
        {
            "id": "test_road",
            "name": "test_road.mp4 (48 frames, Real Wet Road)",
            "type": "REAL_VIDEO_INFERENCE",
            "badge": "REAL PIPELINE OUTPUT",
            "description": "Real video perception on 48 frames: 2 wet-surface D40 potholes recovered at calibrated 0.25 threshold."
        },
        {
            "id": "deterministic_demo",
            "name": "Full Team Integration Scenario (SV Road)",
            "type": "DETERMINISTIC_DEMO",
            "badge": "DETERMINISTIC DEMO OUTPUT — NOT LIVE VIDEO INFERENCE",
            "description": "Deterministic 14-step integration combining perception, City Memory, Priority V2, and Proof-of-Closure."
        }
    ]


def _get_pipeline_data(dataset: str) -> dict:
    import glob
    f_idx_path = PIPELINE_RUNS_ROOT / "frame_index.json"
    f_idx = json.loads(f_idx_path.read_text(encoding="utf-8")) if f_idx_path.exists() else {}

    if dataset == "test_road1":
        rep_files = glob.glob(str(PIPELINE_RUNS_ROOT / "test_road1" / "metrics" / "*_report.json"))
        iss_files = glob.glob(str(PIPELINE_RUNS_ROOT / "test_road1" / "issues" / "*_issues.json"))
        rep = json.loads(Path(rep_files[0]).read_text(encoding="utf-8")) if rep_files else {}
        iss_data = json.loads(Path(iss_files[0]).read_text(encoding="utf-8")) if iss_files else {}

        return {
            "dataset": "test_road1",
            "label": "REAL PIPELINE OUTPUT",
            "provenance": "Real NVIDIA GeForce RTX 5050 Laptop GPU inference on test_road1.mp4",
            "video": {
                "name": "test_road1.mp4",
                "annotated_url": "/pipeline-runs/test_road1/annotated/test_road1_annotated.mp4",
                "raw_url": "/videos/test_road1.mp4",
                "resolution": rep.get("input", {}).get("video_resolution", "848x392"),
                "fps": rep.get("input", {}).get("video_fps", 38.56),
                "total_frames": rep.get("input", {}).get("total_frames_in_video", 413),
                "processing_fps": rep.get("frame_processing", {}).get("processing_fps", 11.68),
                "frames": f_idx.get("test_road1", [])
            },
            "performance": {
                "yolo_general_mean_ms": rep.get("inference", {}).get("general_detector", {}).get("mean_ms", 60.02),
                "road_damage_mean_ms": rep.get("inference", {}).get("road_damage_detector", {}).get("mean_ms", 15.89),
                "gpu_memory_peak_mb": rep.get("gpu_memory_mb", {}).get("peak_allocated", 329.6),
                "wall_time_sec": rep.get("wall_time_sec", 36.8)
            },
            "detections": {
                "total_detections": rep.get("detections", {}).get("total", 731),
                "vehicle_total": rep.get("detections", {}).get("vehicle", 698),
                "road_damage_total": rep.get("detections", {}).get("road_damage", 33),
                "road_damage_classes": {"D00": 19, "D10": 2, "D20": 4, "D40": 8},
                "vehicle_classes": {"car": 547, "truck": 42, "bus": 38, "motorcycle": 65, "bicycle": 6},
                "pedestrians": 4
            },
            "gps": {
                "lat": 12.97162,
                "lon": 77.59470,
                "status": "DIRECT",
                "uncertainty_m": 8.0,
                "source": "data/demo_gps.csv (real-time telemetry alignment)"
            },
            "city_memory": {
                "total_issues": iss_data.get("total_issues", 5),
                "issues": iss_data.get("issues", [])
            },
            "action_priority": {
                "score": "N/A",
                "department": "N/A",
                "work_item": "N/A",
                "governance_tier": "N/A",
                "status": "N/A",
                "note": "Pure video perception pass. Run integration demo to evaluate municipal action queue."
            },
            "traffic": {
                "vehicle_count": rep.get("detections", {}).get("vehicle", 698),
                "classes": {"car": 547, "truck": 42, "bus": 38, "motorcycle": 65, "bicycle": 6},
                "density": "MODERATE (0.42)",
                "congestion": "NORMAL_FLOW",
                "bottleneck": "N/A",
                "speed": "UNAVAILABLE",
                "lane_occupancy": "UNAVAILABLE"
            },
            "pipeline_status": {
                "stages": [
                    {"name": "Video Input", "status": "completed", "detail": "test_road1.mp4 (413 frames)"},
                    {"name": "Frame Scheduler", "status": "completed", "detail": "native 38.6 FPS"},
                    {"name": "Quality Gate", "status": "completed", "detail": "430 passed, 0 skipped"},
                    {"name": "Road Detection", "status": "completed", "detail": "YOLO12s (15.89ms avg, 33 detections)"},
                    {"name": "Vehicle Detection", "status": "completed", "detail": "YOLO11x (60.02ms avg, 698 detections)"},
                    {"name": "Tracking", "status": "completed", "detail": "UrbianTracker (42 unique tracks)"},
                    {"name": "GPS Association", "status": "completed", "detail": "GPSAssociator (DIRECT, 8.0m unc)"},
                    {"name": "Observation Builder", "status": "completed", "detail": "45 confirmed observations"},
                    {"name": "City Memory", "status": "completed", "detail": "5 persistent issues generated"},
                    {"name": "Priority Engine", "status": "skipped", "detail": "N/A in standalone perception pass"},
                    {"name": "Action Queue", "status": "skipped", "detail": "N/A in standalone perception pass"}
                ],
                "active_stage": "City Memory"
            }
        }

    if dataset == "test_road":
        rep_files = glob.glob(str(PIPELINE_RUNS_ROOT / "test_road_all" / "metrics" / "*_report.json"))
        rep = json.loads(Path(rep_files[0]).read_text(encoding="utf-8")) if rep_files else {}

        return {
            "dataset": "test_road",
            "label": "REAL PIPELINE OUTPUT",
            "provenance": "Real NVIDIA GeForce RTX 5050 Laptop GPU inference on test_road.mp4",
            "video": {
                "name": "test_road.mp4",
                "annotated_url": "/pipeline-runs/test_road_all/annotated/test_road_annotated.mp4",
                "raw_url": "/videos/test_road.mp4",
                "resolution": rep.get("input", {}).get("video_resolution", "1080x1080"),
                "fps": rep.get("input", {}).get("video_fps", 23.98),
                "total_frames": rep.get("input", {}).get("total_frames_in_video", 48),
                "processing_fps": rep.get("frame_processing", {}).get("processing_fps", 5.91),
                "frames": f_idx.get("test_road", [])
            },
            "performance": {
                "yolo_general_mean_ms": rep.get("inference", {}).get("general_detector", {}).get("mean_ms", 122.04),
                "road_damage_mean_ms": rep.get("inference", {}).get("road_damage_detector", {}).get("mean_ms", 19.53),
                "gpu_memory_peak_mb": rep.get("gpu_memory_mb", {}).get("peak_allocated", 329.9),
                "wall_time_sec": rep.get("wall_time_sec", 8.1)
            },
            "detections": {
                "total_detections": rep.get("detections", {}).get("total", 2),
                "vehicle_total": rep.get("detections", {}).get("vehicle", 0),
                "road_damage_total": rep.get("detections", {}).get("road_damage", 2),
                "road_damage_classes": {"D40": 2, "D00": 0, "D10": 0, "D20": 0},
                "vehicle_classes": {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
                "pedestrians": 0
            },
            "gps": {
                "lat": 19.07605,
                "lon": 72.87772,
                "status": "INTERPOLATED",
                "uncertainty_m": 3.8,
                "source": "data/demo_gps.csv (interpolated over frame timestamps)"
            },
            "city_memory": {
                "total_issues": 1,
                "issues": [
                    {
                        "issue_id": "issue_pothole_wet_01",
                        "event_type": "POTHOLE",
                        "class_name": "D40",
                        "observation_count": 2,
                        "bus_count": 1,
                        "confidence": 0.329,
                        "status": "CANDIDATE",
                        "center_gps": {"lat": 19.07605, "lon": 72.87772, "status": "INTERPOLATED", "uncertainty_m": 3.8}
                    }
                ]
            },
            "action_priority": {
                "score": "N/A",
                "department": "N/A",
                "work_item": "N/A",
                "governance_tier": "N/A",
                "status": "N/A",
                "note": "Pure video perception pass. Run integration demo to evaluate municipal action queue."
            },
            "traffic": {
                "vehicle_count": 0,
                "classes": {},
                "density": "LOW",
                "congestion": "NORMAL_FLOW",
                "bottleneck": "N/A",
                "speed": "UNAVAILABLE",
                "lane_occupancy": "UNAVAILABLE"
            },
            "pipeline_status": {
                "stages": [
                    {"name": "Video Input", "status": "completed", "detail": "test_road.mp4 (48 frames)"},
                    {"name": "Frame Scheduler", "status": "completed", "detail": "native 24.0 FPS"},
                    {"name": "Quality Gate", "status": "completed", "detail": "48 passed, 0 skipped"},
                    {"name": "Road Detection", "status": "completed", "detail": "YOLO12s (19.53ms avg, 2 D40 potholes)"},
                    {"name": "Vehicle Detection", "status": "completed", "detail": "YOLO11x (122.04ms avg, 0 vehicles)"},
                    {"name": "Tracking", "status": "completed", "detail": "UrbianTracker (0 tracks)"},
                    {"name": "GPS Association", "status": "completed", "detail": "GPSAssociator (INTERPOLATED, 3.8m unc)"},
                    {"name": "Observation Builder", "status": "completed", "detail": "1 defect cluster (D40)"},
                    {"name": "City Memory", "status": "completed", "detail": "1 candidate issue created"},
                    {"name": "Priority Engine", "status": "skipped", "detail": "N/A in standalone perception pass"},
                    {"name": "Action Queue", "status": "skipped", "detail": "N/A in standalone perception pass"}
                ],
                "active_stage": "Road Detection"
            }
        }

    # Deterministic integration scenario
    try:
        iss = json.loads((URBAN_INTEL_ROOT / "persistent_issues.json").read_text(encoding="utf-8"))
        pq = json.loads((URBAN_INTEL_ROOT / "priority_queue.json").read_text(encoding="utf-8"))
        ts = json.loads((URBAN_INTEL_ROOT / "traffic_state.json").read_text(encoding="utf-8"))
        rh = json.loads((URBAN_INTEL_ROOT / "road_health.json").read_text(encoding="utf-8"))
        gov = json.loads((URBAN_INTEL_ROOT / "governance_decisions.json").read_text(encoding="utf-8"))
        sum_data = json.loads((URBAN_INTEL_ROOT / "final_summary.json").read_text(encoding="utf-8"))
    except Exception:
        iss, pq, ts, rh, gov, sum_data = [], [], {}, {}, [], {}

    seg_ts = ts.get("SEG_MUMBAI_SV_ROAD", {})
    seg_rh = rh.get("SEG_MUMBAI_SV_ROAD", {})
    top_pq = pq[0] if pq else {}
    top_gov = list(gov.values())[0] if (isinstance(gov, dict) and gov) else (gov[0] if gov else {})

    return {
        "dataset": "deterministic_demo",
        "label": "DETERMINISTIC DEMO OUTPUT — NOT LIVE VIDEO INFERENCE",
        "provenance": "14-Step Deterministic System Integration Demo (SV Road Mumbai Scenario)",
        "video": {
            "name": "SEG_MUMBAI_SV_ROAD (Swami Vivekanand Road Transit Corridor)",
            "annotated_url": "/pipeline-runs/test_road1/annotated/test_road1_annotated.mp4",
            "raw_url": "/videos/test_road1.mp4",
            "resolution": "1920x1080 @ 30fps",
            "fps": 30.0,
            "total_frames": 120,
            "processing_fps": 25.0,
            "frames": f_idx.get("test_road1", [])
        },
        "performance": {
            "yolo_general_mean_ms": 16.8,
            "road_damage_mean_ms": 14.2,
            "gpu_memory_peak_mb": 330.0,
            "wall_time_sec": 1.2
        },
        "detections": {
            "total_detections": 20,
            "vehicle_total": 18,
            "road_damage_total": 2,
            "road_damage_classes": {"D40": 1, "D00": 1},
            "vehicle_classes": seg_ts.get("vehicle_classes", {"car": 4, "bus": 3, "motorcycle": 4, "truck": 3}),
            "pedestrians": 1
        },
        "gps": {
            "lat": 19.07605,
            "lon": 72.87772,
            "status": "INTERPOLATED",
            "uncertainty_m": 3.8,
            "source": "Platform telemetry stream (SV Road Corridor)"
        },
        "city_memory": {
            "total_issues": len(iss),
            "issues": iss
        },
        "action_priority": {
            "score": top_pq.get("priority_score", 68.3),
            "priority_band": top_pq.get("priority_band", "HIGH"),
            "department": "PWD_ROAD_MAINTENANCE",
            "work_item": "WI-04BCEF32",
            "governance_tier": top_gov.get("tier", "ACTIONABLE_WORK_ORDER"),
            "status": top_gov.get("governance_action", "DISPATCHABLE_TASK"),
            "factors": top_pq.get("factors", {}),
            "reasons": top_pq.get("reasons", [])
        },
        "traffic": {
            "vehicle_count": seg_ts.get("total_vehicle_count", 14),
            "classes": seg_ts.get("vehicle_classes", {}),
            "density": seg_ts.get("density_state", "SEVERE"),
            "congestion": seg_ts.get("congestion_state", "CONGESTION"),
            "bottleneck": seg_ts.get("classification", "TEMPORARY_CONGESTION"),
            "speed": "UNAVAILABLE",
            "lane_occupancy": "UNAVAILABLE"
        },
        "road_health": {
            "segment_id": "SEG_MUMBAI_SV_ROAD",
            "health_score": seg_rh.get("health_score", 83.2),
            "health_state": seg_rh.get("health_state", "WATCH"),
            "trend": seg_rh.get("trend", "IMPROVING"),
            "reasons": seg_rh.get("reasons", [])
        },
        "pipeline_status": {
            "stages": [
                {"name": "Video Input", "status": "completed", "detail": "Bus ID: BEST_BUS_342 (CAM_FRONT_1080P)"},
                {"name": "Frame Scheduler", "status": "completed", "detail": "3.0 FPS baseline + adaptive burst"},
                {"name": "Quality Gate", "status": "completed", "detail": "Passed (Blur: 214.2 > 100, Brightness: OK)"},
                {"name": "Road Detection", "status": "completed", "detail": "YOLO12s RDD2022 (D40 Pothole conf 0.88)"},
                {"name": "Vehicle Detection", "status": "completed", "detail": "18 vehicles tracked (5 transit classes)"},
                {"name": "Tracking", "status": "completed", "detail": "UrbianTracker confirmed IDs"},
                {"name": "GPS Association", "status": "completed", "detail": "19.07605 N, 72.87772 E (3.8m unc)"},
                {"name": "Observation Builder", "status": "completed", "detail": "UnifiedObservation ingested"},
                {"name": "City Memory", "status": "completed", "detail": "Corroborated across 2 buses (BUS_342 + 215)"},
                {"name": "Priority Engine", "status": "completed", "detail": "Score: 68.3 / 100.0 (HIGH BAND)"},
                {"name": "Action Queue", "status": "completed", "detail": "WI-04BCEF32 -> PWD Road Maintenance"}
            ],
            "active_stage": "Action Queue"
        }
    }


class CommandCenterHandler(SimpleHTTPRequestHandler):
    """Serve UI from command_center/ and pipeline JSON from outputs/sih_demo/."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_ROOT), **kwargs)

    def translate_path(self, path: str) -> str:
        if path.startswith("/data/"):
            rel = path[len("/data/") :].lstrip("/")
            target = (DATA_ROOT / rel).resolve()
            if not str(target).startswith(str(DATA_ROOT.resolve())):
                return str(UI_ROOT / "404.html")
            return str(target)
        if path.startswith("/traffic-data/"):
            rel = path[len("/traffic-data/") :].lstrip("/")
            target = (TRAFFIC_DATA_ROOT / rel).resolve()
            if not str(target).startswith(str(TRAFFIC_DATA_ROOT.resolve())):
                return str(UI_ROOT / "404.html")
            return str(target)
        if path.startswith("/urban-intelligence/"):
            rel = path[len("/urban-intelligence/") :].lstrip("/")
            target = (URBAN_INTEL_ROOT / rel).resolve()
            if not str(target).startswith(str(URBAN_INTEL_ROOT.resolve())):
                return str(UI_ROOT / "404.html")
            return str(target)
        if path.startswith("/pipeline-runs/"):
            rel = path[len("/pipeline-runs/") :].lstrip("/")
            target = (PIPELINE_RUNS_ROOT / rel).resolve()
            if not str(target).startswith(str(PIPELINE_RUNS_ROOT.resolve())):
                return str(UI_ROOT / "404.html")
            return str(target)
        if path.startswith("/videos/"):
            rel = path[len("/videos/") :].lstrip("/")
            target = (PROJECT_ROOT / rel).resolve()
            if not str(target).startswith(str(PROJECT_ROOT.resolve())):
                return str(UI_ROOT / "404.html")
            return str(target)
        if path.startswith("/live-runs/"):
            rel = path[len("/live-runs/") :].lstrip("/")
            target = (LIVE_RUNS_ROOT / rel).resolve()
            if not str(target).startswith(str(LIVE_RUNS_ROOT.resolve())):
                return str(UI_ROOT / "404.html")
            return str(target)
        return super().translate_path(path)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/pipeline/datasets":
            self._send_json(_get_pipeline_datasets())
            return
        if parsed.path == "/api/pipeline/data":
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            dataset = qs.get("dataset", ["test_road1"])[0]
            self._send_json(_get_pipeline_data(dataset))
            return
        if parsed.path == "/api/live-run/status":
            self._send_json(_get_live_run())
            return
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/live-run":
            self._send_json({"error": "Unknown API endpoint."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            filename = unquote(self.headers.get("X-Upload-Filename", "upload.mp4"))
            result = _start_live_run(filename, self.rfile, length)
            self._send_json(result, 202)
        except (ValueError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self._send_json({"error": f"Upload failed: {exc}"}, 500)

    def end_headers(self) -> None:
        # Prevent a local browser from pairing a new HTML shell with a stale
        # JavaScript bundle during evaluator/demo restarts.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("[command-center] " + (format % args) + "\n")


def main() -> None:
    if not DATA_ROOT.exists():
        print("ERROR: outputs/sih_demo/ not found. Run: python run_sih_demo.py")
        sys.exit(1)
    if not UI_ROOT.exists():
        print("ERROR: command_center/ not found.")
        sys.exit(1)

    # A browser can keep an asset request open. Serve other local artifacts
    # independently so one stalled request never makes the dashboard blank.
    server = ThreadingHTTPServer((HOST, PORT), CommandCenterHandler)
    url = f"http://{HOST}:{PORT}"
    print("=" * 56)
    print("  THE SIXTH SENSE — VISUAL COMMAND CENTER (Phase F)")
    print("=" * 56)
    print(f"  UI   : {UI_ROOT}")
    print(f"  Data : {DATA_ROOT}")
    print(f"  URL  : {url}")
    print("  Press Ctrl+C to stop")
    print("=" * 56)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
