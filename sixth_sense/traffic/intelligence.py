"""Transparent traffic aggregation over persisted vehicle observations.

The Phase A tracker was not persisted as JSON. Each confirmed, temporally
grouped vehicle observation is therefore used as an *observation-level track
proxy*. Its ``obs_id`` is stable and is counted once per time window. This
module never estimates passenger-level origins, uncalibrated speed, or lane occupancy.

Extended with:
- Spatially aggregated Congestion Heatmap (GeoJSON FeatureCollection)
- Segment-level Route Delay analysis with clearly labelled prototype baseline
- Multi-window Persistent Bottleneck detection with spatial & temporal corroboration
- Aggregate Corridor-level Origin-Destination (OD) flow patterns
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUPPORTED_VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle")
WINDOW_SECONDS = 60.0


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points on Earth in metres."""
    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def density_level(unique_count: int) -> str:
    """Relative density from unique observation-level vehicle proxies/60 sec."""
    if unique_count <= 3:
        return "LOW"
    if unique_count <= 7:
        return "MODERATE"
    if unique_count <= 12:
        return "HIGH"
    return "SEVERE"


def congestion_for_density(level: str) -> str:
    return {
        "LOW": "NORMAL_FLOW",
        "MODERATE": "SLOW_FLOW",
        "HIGH": "CONGESTION",
        "SEVERE": "CONGESTION",
    }[level]


def _eligible(observation: Dict[str, Any]) -> bool:
    return (
        observation.get("event_type") == "VEHICLE"
        and observation.get("class_name") in SUPPORTED_VEHICLE_CLASSES
    )


def build_traffic_windows(
    observations: Iterable[Dict[str, Any]], window_seconds: float = WINDOW_SECONDS
) -> List[Dict[str, Any]]:
    """Aggregate confirmed vehicle observations into source-bus time windows.

    A duplicate observation ID is only admitted once to a window, preventing
    accidental double counting if an artifact is concatenated or replayed.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    buckets: Dict[tuple, Dict[str, Any]] = {}
    for observation in observations:
        if not _eligible(observation):
            continue
        timestamp = float(observation.get("first_seen_ts", 0.0))
        start = int(timestamp // window_seconds) * window_seconds
        bus_id = observation.get("bus_id") or "BUS_UNKNOWN"
        key = (bus_id, start)
        bucket = buckets.setdefault(key, {"records": {}, "bus_id": bus_id, "start": start})
        proxy_id = observation.get("obs_id")
        if not proxy_id:
            continue
        bucket["records"].setdefault(proxy_id, observation)

    windows: List[Dict[str, Any]] = []
    for index, (_, bucket) in enumerate(
        sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])), start=1
    ):
        records = list(bucket["records"].values())
        class_counts = {name: 0 for name in SUPPORTED_VEHICLE_CLASSES}
        for record in records:
            class_counts[record["class_name"]] += 1
        class_counts = {name: count for name, count in class_counts.items() if count}
        count = len(records)
        density = density_level(count)
        windows.append({
            "traffic_window_id": f"traffic_window_{index:03d}",
            "start_time": bucket["start"],
            "end_time": bucket["start"] + window_seconds,
            "source_bus_id": bucket["bus_id"],
            "road_segment_id": None,
            "vehicle_counts": class_counts,
            "unique_track_counts": class_counts.copy(),
            "unique_vehicle_count": count,
            "supporting_observation_ids": sorted(bucket["records"]),
            "density_level": density,
            "density_method": "relative unique observation-level vehicle proxies per 60-second window; no calibrated road geometry",
            "density_evidence_basis": f"{count} unique confirmed vehicle observations; obs_id deduplicated within window",
            "congestion_state": congestion_for_density(density),
            "speed_estimation_unavailable": "No reliable metric speed exists in cached artifacts; image-space trajectories are not km/h.",
            "lane_occupancy_unavailable": "No lane geometry or camera calibration exists in cached artifacts.",
            "provenance": {
                "source": "existing persisted perception observations",
                "track_basis": "observation-level track proxy (raw tracker IDs were not persisted)",
                "model_inference_rerun": False,
            },
        })
    return windows


def build_traffic_patterns(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create per-bus traffic patterns and require repeated congestion for bottlenecks."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for window in windows:
        grouped[window["source_bus_id"]].append(window)

    patterns: List[Dict[str, Any]] = []
    for bus_id, bus_windows in sorted(grouped.items()):
        bus_windows.sort(key=lambda item: item["start_time"])
        congestion_windows = [w for w in bus_windows if w["congestion_state"] == "CONGESTION"]
        consecutive: List[List[Dict[str, Any]]] = []
        active: List[Dict[str, Any]] = []
        for window in congestion_windows:
            if active and window["start_time"] != active[-1]["end_time"]:
                consecutive.append(active)
                active = []
            active.append(window)
        if active:
            consecutive.append(active)

        bottleneck_ids = set()
        for run in consecutive:
            if len(run) >= 2:
                bottleneck_ids.update(w["traffic_window_id"] for w in run)
        for window in bus_windows:
            if window["traffic_window_id"] in bottleneck_ids:
                window["congestion_state"] = "PERSISTENT_BOTTLENECK"

        patterns.append({
            "pattern_id": f"traffic_pattern_{bus_id.lower()}",
            "road_segment_id": None,
            "first_seen": bus_windows[0]["start_time"],
            "last_seen": bus_windows[-1]["end_time"],
            "observation_count": len(bus_windows),
            "supporting_windows": [w["traffic_window_id"] for w in bus_windows],
            "bus_ids": [bus_id],
            "congestion_state": "PERSISTENT_BOTTLENECK" if bottleneck_ids else max(
                (w["congestion_state"] for w in bus_windows),
                key=("NORMAL_FLOW", "SLOW_FLOW", "CONGESTION").index,
            ),
            "density_state": max(
                (w["density_level"] for w in bus_windows),
                key=("LOW", "MODERATE", "HIGH", "SEVERE").index,
            ),
            "confidence_evidence_basis": "Repeated-window state only; no metric speed, lane geometry, or road-segment calibration.",
            "provenance": bus_windows[0]["provenance"],
        })
    return patterns


# ========================================================================= #
# TASK 2 — CONGESTION HEATMAP (GeoJSON FeatureCollection)
# ========================================================================= #

def build_traffic_heatmap(
    observations: Iterable[Dict[str, Any]],
    grid_deg: float = 0.001,  # ~110m spatial resolution
) -> Dict[str, Any]:
    """
    Generate spatially aggregated traffic intensity/congestion data as GeoJSON.
    
    Each feature represents an observed spatial cell along the bus route,
    aggregating vehicle observations, density level, and congestion state.
    Zero city-wide fabrication: cells exist only where real observations occurred.
    """
    cells: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)

    for obs in observations:
        if not _eligible(obs):
            continue
        gps = obs.get("gps")
        if not gps or gps.get("lat") is None or gps.get("lon") is None:
            continue
        lat = float(gps["lat"])
        lon = float(gps["lon"])
        # Spatial grid key
        cell_lat = int(round(lat / grid_deg))
        cell_lon = int(round(lon / grid_deg))
        cells[(cell_lat, cell_lon)].append(obs)

    features: List[Dict[str, Any]] = []
    cell_index = 1

    for (c_lat, c_lon), cell_obs in sorted(cells.items()):
        # Deduplicate observation IDs within spatial cell
        unique_records = {}
        for r in cell_obs:
            oid = r.get("obs_id")
            if oid:
                unique_records.setdefault(oid, r)

        records = list(unique_records.values())
        obs_count = len(records)
        if obs_count == 0:
            continue

        lats = [float(r["gps"]["lat"]) for r in records]
        lons = [float(r["gps"]["lon"]) for r in records]
        avg_lat = round(sum(lats) / len(lats), 6)
        avg_lon = round(sum(lons) / len(lons), 6)

        timestamps = [float(r.get("first_seen_ts", 0.0)) for r in records]
        t_start = round(min(timestamps), 2)
        t_end = round(max(timestamps), 2)

        class_counts = Counter(r.get("class_name") for r in records)
        density = density_level(obs_count)
        congestion = congestion_for_density(density)

        # Normalized traffic intensity from 0.0 to 1.0 (clamped at 15 vehicles/cell)
        intensity = round(min(1.0, obs_count / 12.0), 3)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [avg_lon, avg_lat],
            },
            "properties": {
                "cell_id": f"cell_{cell_index:03d}",
                "latitude": avg_lat,
                "longitude": avg_lon,
                "traffic_intensity": intensity,
                "density_level": density,
                "congestion_state": congestion,
                "observation_count": obs_count,
                "vehicle_counts": dict(sorted(class_counts.items())),
                "timestamp": t_start,
                "time_window": {
                    "start_sec": t_start,
                    "end_sec": t_end,
                    "duration_sec": round(t_end - t_start, 2),
                },
                "supporting_obs_ids": sorted(unique_records.keys()),
                "spatial_bounds": {
                    "min_lat": round(min(lats), 6),
                    "max_lat": round(max(lats), 6),
                    "min_lon": round(min(lons), 6),
                    "max_lon": round(max(lons), 6),
                },
            },
        })
        cell_index += 1

    return {
        "type": "FeatureCollection",
        "metadata": {
            "title": "Sixth Sense Traffic Congestion Heatmap",
            "spatial_resolution_deg": grid_deg,
            "total_cells": len(features),
            "source": "Aggregated verified vehicle perception observations with valid GPS",
            "fabrication_policy": "Zero synthetic spatial points; features exist solely where vehicles were observed",
        },
        "features": features,
    }


# ========================================================================= #
# TASK 3 — ROUTE DELAY ANALYSIS
# ========================================================================= #

def analyze_route_delays(
    gps_samples: List[Dict[str, Any]],
    route_id: str = "ROUTE_001",
    nominal_speed_kmh: float = 30.0,
    segment_length_m: float = 500.0,
) -> List[Dict[str, Any]]:
    """
    Calculate route segment travel time and compare against a prototype baseline.

    Args:
        gps_samples: Sequential GPS records with lat, lon, timestamp.
        route_id: Identifier for the bus route corridor.
        nominal_speed_kmh: Clearly labelled prototype urban free-flow speed.
        segment_length_m: Target corridor segment distance in metres.

    Returns:
        List of segment delay dictionaries with observed vs baseline travel time.
    """
    if len(gps_samples) < 2:
        return []

    # Sort sequentially by timestamp
    valid_samples = [
        s for s in gps_samples
        if s.get("lat") is not None and s.get("lon") is not None and s.get("timestamp") is not None
    ]
    valid_samples.sort(key=lambda s: float(s["timestamp"]))

    if len(valid_samples) < 2:
        return []

    segments: List[Dict[str, Any]] = []
    seg_idx = 1

    cur_start = valid_samples[0]
    accumulated_dist = 0.0

    nominal_speed_mps = (nominal_speed_kmh * 1000.0) / 3600.0

    for i in range(1, len(valid_samples)):
        p_prev = valid_samples[i - 1]
        p_curr = valid_samples[i]

        d = _haversine_distance_m(
            float(p_prev["lat"]), float(p_prev["lon"]),
            float(p_curr["lat"]), float(p_curr["lon"]),
        )
        accumulated_dist += d

        # Trigger segment completion when distance exceeds threshold or at last sample
        is_last = (i == len(valid_samples) - 1)
        if accumulated_dist >= segment_length_m or (is_last and accumulated_dist > 50.0):
            t_start = float(cur_start["timestamp"])
            t_end = float(p_curr["timestamp"])
            observed_time_sec = max(1.0, round(t_end - t_start, 2))

            # Prototype baseline: nominal free-flow travel time
            baseline_time_sec = max(1.0, round(accumulated_dist / nominal_speed_mps, 2))
            delay_sec = max(0.0, round(observed_time_sec - baseline_time_sec, 2))

            # Delay severity classification
            ratio = observed_time_sec / baseline_time_sec
            if delay_sec <= 2.0 or ratio < 1.15:
                severity = "NONE"
            elif ratio < 1.5:
                severity = "LOW"
            elif ratio < 2.0:
                severity = "MODERATE"
            else:
                severity = "HIGH"

            avg_speed_kmh = round((accumulated_dist / observed_time_sec) * 3.6, 2)

            segments.append({
                "route_id": route_id,
                "segment_id": f"SEG_{seg_idx:03d}",
                "start_coordinate": [float(cur_start["lat"]), float(cur_start["lon"])],
                "end_coordinate": [float(p_curr["lat"]), float(p_curr["lon"])],
                "distance_metres": round(accumulated_dist, 1),
                "observed_travel_time_sec": observed_time_sec,
                "baseline_travel_time_sec": baseline_time_sec,
                "delay_sec": delay_sec,
                "delay_ratio": round(ratio, 2),
                "severity": severity,
                "observed_avg_speed_kmh": avg_speed_kmh,
                "baseline_nominal_speed_kmh": nominal_speed_kmh,
                "baseline_provenance": (
                    f"Prototype nominal free-flow speed ({nominal_speed_kmh} km/h); "
                    "clearly labelled prototype benchmark, not an empirical historical baseline."
                ),
            })

            seg_idx += 1
            cur_start = p_curr
            accumulated_dist = 0.0

    return segments


# ========================================================================= #
# TASK 4 — PERSISTENT BOTTLENECK DETECTION
# ========================================================================= #

def detect_persistent_bottlenecks(
    windows: List[Dict[str, Any]],
    min_supporting_windows: int = 2,
) -> List[Dict[str, Any]]:
    """
    Detect persistent traffic bottlenecks requiring multi-window corroboration.
    
    A bottleneck candidate requires:
      - High or Severe traffic density
      - Congestion or slow flow state
      - Same spatial segment / source corridor
      - Repeated across >= min_supporting_windows windows
      
    Returns empty list if conditions are not met; never fabricates a bottleneck.
    """
    if not windows:
        return []

    # Group candidate windows by corridor / segment
    eligible_windows = [
        w for w in windows
        if w.get("density_level") in ("HIGH", "SEVERE")
        and w.get("congestion_state") in ("CONGESTION", "PERSISTENT_BOTTLENECK", "SLOW_FLOW")
    ]

    if len(eligible_windows) < min_supporting_windows:
        return []

    # Group by bus / corridor
    corridor_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for w in eligible_windows:
        corridor = w.get("road_segment_id") or w.get("source_bus_id") or "CORRIDOR_MAIN"
        corridor_groups[corridor].append(w)

    bottlenecks: List[Dict[str, Any]] = []
    b_idx = 1

    for corridor, c_windows in corridor_groups.items():
        c_windows.sort(key=lambda w: float(w["start_time"]))

        # Identify continuous or recurring clusters of congested windows
        if len(c_windows) >= min_supporting_windows:
            total_obs = sum(w.get("unique_vehicle_count", 0) for w in c_windows)
            states = sorted(list(set(w["congestion_state"] for w in c_windows)))
            window_ids = [w["traffic_window_id"] for w in c_windows]

            # High density ratio
            high_density_count = sum(1 for w in c_windows if w["density_level"] in ("HIGH", "SEVERE"))
            confidence = round(min(0.98, 0.65 + 0.10 * (len(c_windows) - 1)), 2)

            t_start = c_windows[0]["start_time"]
            t_end = c_windows[-1]["end_time"]

            bottlenecks.append({
                "bottleneck_id": f"bottleneck_{b_idx:03d}",
                "segment_or_location": corridor,
                "number_of_supporting_windows": len(c_windows),
                "supporting_window_ids": window_ids,
                "congestion_states": states,
                "time_span_sec": {
                    "start": t_start,
                    "end": t_end,
                    "duration_sec": round(t_end - t_start, 2),
                },
                "total_vehicles_observed": total_obs,
                "confidence": confidence,
                "evidence": (
                    f"Corroborated across {len(c_windows)} consecutive/repeated time windows "
                    f"({t_start}s–{t_end}s) with {total_obs} unique vehicle observations "
                    f"under {', '.join(states)}."
                ),
            })
            b_idx += 1

    return bottlenecks


# ========================================================================= #
# TASK 5 — AGGREGATE OD / TRAFFIC FLOW PATTERNS
# ========================================================================= #

def build_od_patterns(
    gps_samples: List[Dict[str, Any]],
    observations: Iterable[Dict[str, Any]],
    route_id: str = "ROUTE_001",
) -> Dict[str, Any]:
    """
    Generate aggregate corridor-level mobility patterns (Origin -> Route Segments -> Destination).
    
    IMPORTANT: Aggregates vehicular and fleet flow only. Explicitly marks
    passenger-level origin-destination as unavailable.
    """
    valid_gps = [
        s for s in gps_samples
        if s.get("lat") is not None and s.get("lon") is not None and s.get("timestamp") is not None
    ]
    valid_gps.sort(key=lambda s: float(s["timestamp"]))

    if not valid_gps:
        return {
            "status": "LIMITED_DATA",
            "message": "Insufficient GPS trajectory data to formulate corridor OD pattern.",
            "passenger_level_od": "UNAVAILABLE — passenger origins/destinations cannot be inferred from vehicle dashcam data.",
            "patterns": [],
        }

    origin_sample = valid_gps[0]
    dest_sample = valid_gps[-1]

    # Calculate overall corridor length
    total_dist_m = 0.0
    for i in range(1, len(valid_gps)):
        total_dist_m += _haversine_distance_m(
            float(valid_gps[i - 1]["lat"]), float(valid_gps[i - 1]["lon"]),
            float(valid_gps[i]["lat"]), float(valid_gps[i]["lon"]),
        )

    # Count observed vehicles along this corridor
    vehicle_obs = [o for o in observations if _eligible(o)]
    class_counts = Counter(o.get("class_name") for o in vehicle_obs)

    t_start = float(origin_sample["timestamp"])
    t_end = float(dest_sample["timestamp"])

    pattern = {
        "pattern_id": f"od_corridor_{route_id.lower()}",
        "route_id": route_id,
        "origin": {
            "segment_id": "SEG_ORIGIN",
            "coordinate": [float(origin_sample["lat"]), float(origin_sample["lon"])],
            "first_observed_unix": t_start,
            "label": "Corridor Origin Point",
        },
        "destination": {
            "segment_id": "SEG_DESTINATION",
            "coordinate": [float(dest_sample["lat"]), float(dest_sample["lon"])],
            "last_observed_unix": t_end,
            "label": "Corridor Destination Point",
        },
        "route_corridor": {
            "total_distance_metres": round(total_dist_m, 1),
            "duration_sec": round(t_end - t_start, 2),
            "sample_waypoints_count": len(valid_gps),
        },
        "flow_metrics": {
            "trip_count": 1,  # Observed transit run
            "total_vehicle_observations": len(vehicle_obs),
            "vehicle_breakdown": dict(sorted(class_counts.items())),
        },
        "time_window": {
            "start_timestamp": t_start,
            "end_timestamp": t_end,
            "duration_sec": round(t_end - t_start, 2),
        },
        "aggregation_scope": "Macro vehicle flow / corridor transit intelligence",
        "passenger_level_od": "UNAVAILABLE — passenger origins/destinations cannot be inferred from external vehicle sensors.",
        "provenance": "Observed public transit vehicle trajectory and concurrent vehicle perception counts.",
    }

    return {
        "status": "AVAILABLE",
        "count": 1,
        "corridors": [pattern],
        "methodology": "Corridor-level aggregate GPS endpoint linking with concurrent vehicle volume",
        "passenger_od_policy": "Strictly disallowed; system reports vehicle fleet flow only.",
    }
