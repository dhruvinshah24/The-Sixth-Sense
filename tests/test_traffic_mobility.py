"""
Comprehensive unit and integration tests for Traffic & Mobility Intelligence:
- Line-crossing and ROI vehicle counting
- Duplicate count prevention on persistent track IDs
- Spatial congestion heatmap GeoJSON generation
- Route segment delay calculation with prototype baseline
- Multi-window persistent bottleneck detection
- Aggregate corridor-level Origin-Destination (OD) flow patterns
"""
import pytest
from sixth_sense.schemas.urban_event import ClassificationSource, Detection, Track
from sixth_sense.tracking.vehicle_counter import (
    CountingLine,
    CountingROI,
    VehicleCounter,
)
from sixth_sense.traffic.intelligence import (
    analyze_route_delays,
    build_od_patterns,
    build_traffic_heatmap,
    build_traffic_patterns,
    build_traffic_windows,
    detect_persistent_bottlenecks,
)


def make_track(
    track_id: int,
    class_name: str,
    trajectory_points: list[tuple[int, int]],
    first_seen_ts: float = 0.0,
    last_seen_ts: float = 1.0,
) -> Track:
    """Helper to synthesize a confirmed Track for counting tests."""
    dets = []
    for i, (cx, cy) in enumerate(trajectory_points):
        d = Detection(
            det_id=f"det_{track_id}_{i}",
            frame_idx=i,
            timestamp=first_seen_ts + i * 0.1,
            event_type="VEHICLE",
            class_name=class_name,
            classification_source=ClassificationSource.DETECTED,
            raw_confidence=0.90,
            confidence=0.90,
            bbox=(cx - 20, cy - 20, cx + 20, cy + 20),
            bbox_area_px=1600,
            relative_area=0.01,
            frame_width=848,
            frame_height=392,
            gps=None,
            quality=None,
            model_name="general_detector",
        )
        dets.append(d)

    return Track(
        track_id=track_id,
        class_name=class_name,
        classification_source=ClassificationSource.DETECTED,
        first_seen_frame=0,
        last_seen_frame=len(trajectory_points) - 1,
        first_seen_ts=first_seen_ts,
        last_seen_ts=last_seen_ts,
        confirmed=True,
        trajectory=trajectory_points,
        detections=dets,
    )


# -------------------------------------------------------------------------- #
# TASK 1 TESTS: Vehicle Counting & Duplicate Prevention
# -------------------------------------------------------------------------- #

def test_vehicle_counter_line_crossing():
    """Vehicle moving across counting line should trigger count."""
    line = CountingLine((0, 200), (500, 200), name="midline")
    counter = VehicleCounter(line=line, mode="line")

    # Track crosses from y=180 to y=220
    track1 = make_track(1, "car", [(100, 180), (100, 220)])
    res = counter.update([track1], frame_idx=1, timestamp=0.1)

    assert counter.total_counted == 1
    assert counter.counts_by_class == {"car": 1}
    assert 1 in counter.counted_track_ids
    assert len(res["newly_counted"]) == 1
    assert res["newly_counted"][0]["trigger_type"] == "LINE_CROSS"


def test_vehicle_counter_duplicate_prevention():
    """A vehicle remaining across line or continuing trajectory must NOT be counted twice."""
    line = CountingLine((0, 200), (500, 200), name="midline")
    counter = VehicleCounter(line=line, mode="line")

    # Frame 1: crosses line
    track1 = make_track(1, "truck", [(100, 180), (100, 220)])
    counter.update([track1], frame_idx=1, timestamp=0.1)
    assert counter.total_counted == 1

    # Frame 2: track continues moving down (220 -> 250)
    track1_updated = make_track(1, "truck", [(100, 180), (100, 220), (100, 250)])
    res2 = counter.update([track1_updated], frame_idx=2, timestamp=0.2)

    # Count must remain exactly 1
    assert counter.total_counted == 1
    assert counter.counts_by_class == {"truck": 1}
    assert len(res2["newly_counted"]) == 0


def test_vehicle_counter_roi_mode():
    """Vehicle entering ROI is counted once; vehicle outside is ignored."""
    roi = CountingROI(50, 50, 200, 200, name="toll_gate")
    counter = VehicleCounter(roi=roi, mode="roi")

    # Vehicle inside ROI
    track_in = make_track(10, "bus", [(100, 100)])
    # Vehicle outside ROI
    track_out = make_track(20, "car", [(300, 300)])

    counter.update([track_in, track_out], frame_idx=1, timestamp=0.0)

    assert counter.total_counted == 1
    assert counter.counts_by_class == {"bus": 1}
    assert 10 in counter.counted_track_ids
    assert 20 not in counter.counted_track_ids

    # Next frame: track_in moves within ROI -> must not recount
    track_in_moved = make_track(10, "bus", [(100, 100), (120, 120)])
    counter.update([track_in_moved], frame_idx=2, timestamp=0.1)
    assert counter.total_counted == 1


def test_vehicle_counter_multi_class_breakdown():
    """Counter maintains discrete counts per vehicle class."""
    line = CountingLine((0, 100), (500, 100))
    counter = VehicleCounter(line=line, mode="line")

    tracks = [
        make_track(1, "car", [(50, 80), (50, 120)]),
        make_track(2, "car", [(100, 80), (100, 120)]),
        make_track(3, "truck", [(150, 80), (150, 120)]),
        make_track(4, "motorcycle", [(200, 80), (200, 120)]),
        make_track(5, "bus", [(250, 80), (250, 120)]),
    ]

    counter.update(tracks, frame_idx=1, timestamp=0.1)
    assert counter.total_counted == 5
    assert counter.total_unique_tracks == 5
    assert counter.counts_by_class == {
        "bus": 1,
        "car": 2,
        "motorcycle": 1,
        "truck": 1,
    }


# -------------------------------------------------------------------------- #
# TASK 2 TESTS: Congestion Heatmap (GeoJSON)
# -------------------------------------------------------------------------- #

def test_traffic_heatmap_structure_and_properties():
    """Heatmap GeoJSON contains required spatial properties and no synthetic data."""
    obs = [
        {
            "obs_id": "v1",
            "event_type": "VEHICLE",
            "class_name": "car",
            "first_seen_ts": 10.0,
            "gps": {"lat": 12.9716, "lon": 77.5946},
        },
        {
            "obs_id": "v2",
            "event_type": "VEHICLE",
            "class_name": "truck",
            "first_seen_ts": 12.0,
            "gps": {"lat": 12.9717, "lon": 77.5947},
        },
    ]

    geojson = build_traffic_heatmap(obs, grid_deg=0.005)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1

    feature = geojson["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert len(feature["geometry"]["coordinates"]) == 2

    props = feature["properties"]
    assert "latitude" in props
    assert "longitude" in props
    assert "traffic_intensity" in props
    assert "congestion_state" in props
    assert "timestamp" in props
    assert "time_window" in props
    assert props["observation_count"] == 2
    assert props["vehicle_counts"] == {"car": 1, "truck": 1}


def test_traffic_heatmap_omits_missing_gps():
    """Observations without valid GPS must not create synthetic coordinates."""
    obs = [
        {"obs_id": "no_gps_1", "event_type": "VEHICLE", "class_name": "car", "gps": None},
        {"obs_id": "no_gps_2", "event_type": "VEHICLE", "class_name": "car", "gps": {"lat": None, "lon": None}},
    ]
    geojson = build_traffic_heatmap(obs)
    assert len(geojson["features"]) == 0


# -------------------------------------------------------------------------- #
# TASK 3 TESTS: Route Delay Analysis
# -------------------------------------------------------------------------- #

def test_route_delay_calculation():
    """Route delay compares observed time against nominal baseline."""
    # 2 points separated by ~222 metres (0.002 deg latitude)
    # Travel time: 100 seconds (observed speed ~ 2.22 m/s = 8 km/h -> delayed)
    gps_samples = [
        {"timestamp": 0.0, "lat": 12.9700, "lon": 77.5900},
        {"timestamp": 100.0, "lat": 12.9720, "lon": 77.5900},
    ]

    delays = analyze_route_delays(gps_samples, route_id="ROUTE_42", nominal_speed_kmh=30.0, segment_length_m=200.0)

    assert len(delays) >= 1
    seg = delays[0]
    assert seg["route_id"] == "ROUTE_42"
    assert seg["segment_id"] == "SEG_001"
    assert seg["observed_travel_time_sec"] == 100.0
    assert seg["baseline_travel_time_sec"] < 100.0
    assert seg["delay_sec"] > 0.0
    assert seg["severity"] in ("MODERATE", "HIGH")
    assert "Prototype nominal free-flow speed" in seg["baseline_provenance"]


def test_route_delay_empty_on_insufficient_gps():
    """Insufficient GPS samples return empty list rather than fabricating delays."""
    assert analyze_route_delays([]) == []
    assert analyze_route_delays([{"timestamp": 0.0, "lat": 12.0, "lon": 77.0}]) == []


# -------------------------------------------------------------------------- #
# TASK 4 TESTS: Persistent Bottleneck Detection
# -------------------------------------------------------------------------- #

def test_single_window_not_persistent_bottleneck():
    """A single congested window must NOT be declared a persistent bottleneck."""
    single_window = [{
        "traffic_window_id": "w1",
        "start_time": 0.0,
        "end_time": 60.0,
        "source_bus_id": "BUS_001",
        "road_segment_id": "SEG_MAIN",
        "density_level": "HIGH",
        "congestion_state": "CONGESTION",
        "unique_vehicle_count": 10,
    }]
    bottlenecks = detect_persistent_bottlenecks(single_window, min_supporting_windows=2)
    assert bottlenecks == []


def test_repeated_windows_become_persistent_bottleneck():
    """Recurring congested windows in same segment trigger persistent bottleneck."""
    windows = [
        {
            "traffic_window_id": "w1",
            "start_time": 0.0,
            "end_time": 60.0,
            "source_bus_id": "BUS_001",
            "road_segment_id": "SEG_CORRIDOR",
            "density_level": "SEVERE",
            "congestion_state": "CONGESTION",
            "unique_vehicle_count": 15,
        },
        {
            "traffic_window_id": "w2",
            "start_time": 60.0,
            "end_time": 120.0,
            "source_bus_id": "BUS_001",
            "road_segment_id": "SEG_CORRIDOR",
            "density_level": "HIGH",
            "congestion_state": "CONGESTION",
            "unique_vehicle_count": 11,
        },
    ]

    bottlenecks = detect_persistent_bottlenecks(windows, min_supporting_windows=2)
    assert len(bottlenecks) == 1
    b = bottlenecks[0]
    assert b["segment_or_location"] == "SEG_CORRIDOR"
    assert b["number_of_supporting_windows"] == 2
    assert b["supporting_window_ids"] == ["w1", "w2"]
    assert b["total_vehicles_observed"] == 26
    assert b["confidence"] >= 0.70
    assert "Corroborated across 2 consecutive/repeated time windows" in b["evidence"]


# -------------------------------------------------------------------------- #
# TASK 5 TESTS: Aggregate Origin-Destination Flow Patterns
# -------------------------------------------------------------------------- #

def test_od_patterns_aggregate_corridor_flow():
    """OD patterns correctly reflect macro corridor travel without passenger claims."""
    gps = [
        {"timestamp": 0.0, "lat": 12.9700, "lon": 77.5900},
        {"timestamp": 60.0, "lat": 12.9750, "lon": 77.5950},
    ]
    obs = [
        {"obs_id": "1", "event_type": "VEHICLE", "class_name": "car"},
        {"obs_id": "2", "event_type": "VEHICLE", "class_name": "bus"},
    ]

    od = build_od_patterns(gps, obs, route_id="ROUTE_101")
    assert od["status"] == "AVAILABLE"
    assert od["count"] == 1

    corridor = od["corridors"][0]
    assert corridor["origin"]["segment_id"] == "SEG_ORIGIN"
    assert corridor["destination"]["segment_id"] == "SEG_DESTINATION"
    assert corridor["flow_metrics"]["total_vehicle_observations"] == 2
    assert corridor["flow_metrics"]["vehicle_breakdown"] == {"bus": 1, "car": 1}
    assert "UNAVAILABLE" in corridor["passenger_level_od"]


def test_od_patterns_empty_when_no_gps():
    """Without GPS, OD patterns return LIMITED_DATA status honestly."""
    od = build_od_patterns([], [])
    assert od["status"] == "LIMITED_DATA"
    assert "UNAVAILABLE" in od["passenger_level_od"]
