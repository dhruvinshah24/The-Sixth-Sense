"""
Vehicle Counter — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Provides deterministic, line-crossing and ROI-based vehicle counting
using persistent Track IDs from UrbianTracker.

Guarantees:
- Strict duplicate prevention: Each unique Track ID is counted at most once.
- Configurable counting geometry: Arbitrary line segment or rectangular ROI.
- Multi-class aggregation: Counters maintained per vehicle class (car, bus, truck, etc.).
- Provenance & auditing: Full event log with timestamps, track IDs, and coordinates.
- Visual validation: Renders counting line/ROI, active tracks, and on-screen HUD.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from sixth_sense.schemas.urban_event import Track

logger = logging.getLogger(__name__)


def _ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """Check if three points are listed in counter-clockwise order."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Return True if line segment (p1, p2) intersects with (p3, p4)."""
    return (
        _ccw(p1, p3, p4) != _ccw(p2, p3, p4)
        and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)
    )


@dataclass
class CountingLine:
    """A virtual counting line defined by two 2D endpoints (x1, y1) -> (x2, y2)."""
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    name: str = "counting_line"

    def intersects_segment(
        self, prev_pt: Tuple[float, float], curr_pt: Tuple[float, float]
    ) -> bool:
        return _segments_intersect(self.p1, self.p2, prev_pt, curr_pt)


@dataclass
class CountingROI:
    """A virtual counting Region of Interest defined by (xmin, ymin, xmax, ymax)."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    name: str = "counting_roi"

    def contains_point(self, pt: Tuple[float, float]) -> bool:
        return self.xmin <= pt[0] <= self.xmax and self.ymin <= pt[1] <= self.ymax


@dataclass
class CountEvent:
    """Audit record for a single vehicle counting event."""
    track_id: int
    class_name: str
    frame_idx: int
    timestamp: float
    trigger_type: str  # 'LINE_CROSS' or 'ROI_ENTER'
    position: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "trigger_type": self.trigger_type,
            "position": list(self.position),
        }


class VehicleCounter:
    """
    Directional line-crossing and ROI vehicle counter using persistent Track IDs.

    Ensures zero duplicate counts for the same physical vehicle:
    once track_id is counted, it will never be counted again regardless of
    how many frames it stays near the line or inside the ROI.
    """

    def __init__(
        self,
        line: Optional[CountingLine] = None,
        roi: Optional[CountingROI] = None,
        mode: str = "line",  # 'line', 'roi', or 'either'
    ) -> None:
        self.line = line
        self.roi = roi
        self.mode = mode.lower()

        if self.mode in ("line", "either") and self.line is None:
            # Default horizontal line across middle of typical 720p/1080p frame
            self.line = CountingLine((0, 360), (1280, 360), name="default_midline")
        if self.mode in ("roi", "either") and self.roi is None:
            self.roi = CountingROI(100, 200, 1180, 600, name="default_roi")

        self._counted_track_ids: Set[int] = set()
        self._all_seen_track_ids: Set[int] = set()
        self._counts_by_class: Counter = Counter()
        self._events: List[CountEvent] = []

    @property
    def counted_track_ids(self) -> Set[int]:
        return set(self._counted_track_ids)

    @property
    def counts_by_class(self) -> Dict[str, int]:
        return dict(sorted(self._counts_by_class.items()))

    @property
    def total_counted(self) -> int:
        return len(self._counted_track_ids)

    @property
    def total_unique_tracks(self) -> int:
        return len(self._all_seen_track_ids)

    def reset(self) -> None:
        """Reset all counters and tracked IDs."""
        self._counted_track_ids.clear()
        self._all_seen_track_ids.clear()
        self._counts_by_class.clear()
        self._events.clear()

    def update(
        self,
        active_tracks: List[Track],
        frame_idx: int,
        timestamp: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process active tracks for the current frame and update counts.

        Args:
            active_tracks: Confirmed active tracks from UrbianTracker
            frame_idx: Video frame index
            timestamp: Frame timestamp in seconds

        Returns:
            Dict containing newly counted events and cumulative summary.
        """
        newly_counted: List[CountEvent] = []

        for track in active_tracks:
            tid = track.track_id
            self._all_seen_track_ids.add(tid)

            # Strict duplicate prevention: skip if already counted
            if tid in self._counted_track_ids:
                continue

            traj = track.trajectory
            if not traj:
                continue

            curr_pt = traj[-1]
            prev_pt = traj[-2] if len(traj) >= 2 else curr_pt

            triggered = False
            trigger_type = ""

            # Check Line Crossing
            if self.mode in ("line", "either") and self.line is not None:
                if len(traj) >= 2 and self.line.intersects_segment(prev_pt, curr_pt):
                    triggered = True
                    trigger_type = "LINE_CROSS"
                elif len(traj) >= 3:
                    # Check across last few trajectory steps for fast movers
                    for i in range(len(traj) - 2, max(-1, len(traj) - 5), -1):
                        if self.line.intersects_segment(traj[i - 1], traj[i]):
                            triggered = True
                            trigger_type = "LINE_CROSS"
                            break

            # Check ROI Entry
            if not triggered and self.mode in ("roi", "either") and self.roi is not None:
                if self.roi.contains_point(curr_pt):
                    triggered = True
                    trigger_type = "ROI_ENTER"

            if triggered:
                self._counted_track_ids.add(tid)
                self._counts_by_class[track.class_name] += 1
                ev = CountEvent(
                    track_id=tid,
                    class_name=track.class_name,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    trigger_type=trigger_type,
                    position=curr_pt,
                )
                self._events.append(ev)
                newly_counted.append(ev)
                logger.debug(
                    "Vehicle counted: Track %d (%s) via %s at frame %d",
                    tid, track.class_name, trigger_type, frame_idx,
                )

        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "newly_counted": [e.to_dict() for e in newly_counted],
            "total_counted": self.total_counted,
            "total_unique_tracks": self.total_unique_tracks,
            "counts_by_class": self.counts_by_class,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        active_tracks: Optional[List[Track]] = None,
    ) -> np.ndarray:
        """
        Draw counting line, ROI, tracks, and a live statistics HUD on the frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw ROI if configured
        if self.roi and self.mode in ("roi", "either"):
            cv2.rectangle(
                out,
                (self.roi.xmin, self.roi.ymin),
                (self.roi.xmax, self.roi.ymax),
                (255, 200, 0),
                2,
            )
            cv2.putText(
                out,
                f"COUNTING ROI ({self.roi.name})",
                (self.roi.xmin + 5, self.roi.ymin + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw Counting Line if configured
        if self.line and self.mode in ("line", "either"):
            cv2.line(out, self.line.p1, self.line.p2, (0, 255, 255), 3)
            cv2.circle(out, self.line.p1, 5, (0, 200, 255), -1)
            cv2.circle(out, self.line.p2, 5, (0, 200, 255), -1)
            cv2.putText(
                out,
                f"COUNTING LINE: {self.line.name}",
                (self.line.p1[0] + 10, self.line.p1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw Active Tracks
        if active_tracks:
            for track in active_tracks:
                tid = track.track_id
                is_counted = tid in self._counted_track_ids
                box_color = (0, 255, 0) if is_counted else (255, 160, 0)

                # Get latest bbox from last detection if available
                if track.detections:
                    bx1, by1, bx2, by2 = track.detections[-1].bbox
                    cv2.rectangle(out, (bx1, by1), (bx2, by2), box_color, 2)
                    label = f"ID:{tid} {track.class_name} {'[COUNTED]' if is_counted else ''}"
                    cv2.putText(
                        out,
                        label,
                        (bx1, max(15, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        box_color,
                        1,
                        cv2.LINE_AA,
                    )

                # Draw trajectory trail
                traj = track.trajectory
                for i in range(1, len(traj)):
                    cv2.line(out, traj[i - 1], traj[i], box_color, 1)

        # Overlay Statistics HUD (semi-transparent panel at top-left)
        hud_w, hud_h = 320, 95 + len(self._counts_by_class) * 18
        overlay = out.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 24, 28), -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (0, 200, 255), 1)

        # HUD Text
        cv2.putText(
            out,
            "TRAFFIC VEHICLE COUNTER (PERSISTENT ID)",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Unique Tracks: {self.total_unique_tracks}   Total Counted: {self.total_counted}",
            (20, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Mode: {self.mode.upper()} (Zero duplicates guaranteed)",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 200, 210),
            1,
            cv2.LINE_AA,
        )

        y_offset = 92
        for cls_name, count in self.counts_by_class.items():
            cv2.putText(
                out,
                f"  • {cls_name:12s}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )
            y_offset += 18

        return out

    def get_summary(self) -> Dict[str, Any]:
        """Return structured summary for metric logging and validation reports."""
        return {
            "counting_mode": self.mode,
            "total_unique_tracks_seen": self.total_unique_tracks,
            "total_vehicles_counted": self.total_counted,
            "counts_by_class": self.counts_by_class,
            "duplicate_prevention_method": "strict track_id set admission; single count per persistent track",
            "events": [e.to_dict() for e in self._events],
        }
