"""
glance.py
---------
Glance zone classification, transition tracking, mode correlation,
and statistical reporting.

Adapted for L2CS-Net:
- Removed Gaze H / Gaze V / use_gaze logic.
- Classification relies purely on the Head Pose (Yaw/Pitch) Ellipse.
"""

import collections
import numpy as np
from typing import Optional

from config import (
    GLANCE_MIN_DURATION,
    OFFROAD_THRESHOLD,
    NO_DETECTION_GRACE_PERIOD,
)
from state import TrackingState


class GlanceTracker:
    def __init__(self, min_duration: float = GLANCE_MIN_DURATION,
                 grace_period: float = NO_DETECTION_GRACE_PERIOD):
        self.min_duration = min_duration
        self.grace_period = grace_period
        
        # --- FIX 1: LIGHTNING-FAST ON-ROAD DEBOUNCE ---
        # 100ms to confirm they looked back at the road. This ensures quick 
        # "check-backs" break up massive Off-Road blocks into the proper Short glances.
        self.debounce_on_road = 0.10   
        self.debounce_off_road = 0.20  # 200ms to confirm they actually looked away
        
        # --- FIX 2: REWIND CAP ---
        # Maximum time (seconds) we are allowed to backdate a glance.
        self.max_rewind_time = 0.40 
        
        self.pending_zone: Optional[str] = None
        self.pending_time: Optional[float] = None
        self.pending_frame: Optional[int] = None

        self.movement_history = collections.deque(maxlen=60)

        self.glance_history: list =[]
        self.glance_durations: collections.defaultdict = collections.defaultdict(float)
        self.current_glance: dict = {
            "zone": None,
            "start_time": None,
            "start_frame": None,
        }

        self.last_valid_zone: Optional[str] = None
        self.last_valid_zone_time: Optional[float] = None
        self.last_valid_confidence: float = 0.0
        self._last_yaw_vel: float = 0.0

    def reset(self):
        self.glance_history.clear()
        self.glance_durations.clear()
        self.movement_history.clear()
        self.current_glance = {"zone": None, "start_time": None, "start_frame": None}
        self.pending_zone = None
        self.pending_time = None
        self.pending_frame = None
        self.last_valid_zone = None
        self.last_valid_zone_time = None
        self.last_valid_confidence = 0.0
        self._last_yaw_vel = 0.0

    def classify_zone(self, state: TrackingState, yaw_range: tuple, pitch_range: tuple) -> str:
        if state.smoothed_yaw is None or state.smoothed_pitch is None:
            return "Undetermined"

        self._last_yaw_vel = state.head_ayaw

        center_yaw = (yaw_range[0] + yaw_range[1]) / 2.0
        center_pitch = (pitch_range[0] + pitch_range[1]) / 2.0
        
        radius_yaw = (yaw_range[1] - yaw_range[0]) / 2.0
        radius_pitch = (pitch_range[1] - pitch_range[0]) / 2.0

        if radius_yaw <= 0 or radius_pitch <= 0: 
            return "Undetermined"

        dx = state.smoothed_yaw - center_yaw
        dy = state.smoothed_pitch - center_pitch
        
        elliptical_distance = (dx**2 / radius_yaw**2) + (dy**2 / radius_pitch**2)

        # 1.44 (20% expansion) allows slight wandering inside the windshield.
        threshold = 1.44 if self.current_glance["zone"] == "On-Road" else 1.0
        
        if elliptical_distance <= threshold:
            return "On-Road"
        else:
            return "Off-Road"

    def apply_grace_period(self, zone: str, current_time: float) -> tuple:
        if zone != "Undetermined":
            self.last_valid_zone = zone
            self.last_valid_zone_time = current_time
            return zone, None

        if self.last_valid_zone is not None and self.last_valid_zone_time is not None:
            elapsed = current_time - self.last_valid_zone_time
            
            # --- FIX 3: STRICT OFF-ROAD GRACE PERIOD CAPPING ---
            # If tracking drops while off-road, holding it for 2.0s guarantees a false "Long" glance.
            # We cut it off after 0.5s if they were looking away.
            active_grace_period = 0.5 if self.last_valid_zone == "Off-Road" else self.grace_period
            
            if elapsed <= active_grace_period:
                decay = 1.0 - (elapsed / active_grace_period)
                decayed_confidence = self.last_valid_confidence * decay
                
                # Saccade check (Driver whipped head away)
                if abs(self._last_yaw_vel) > 45.0:
                    return "Off-Road", decayed_confidence

                return self.last_valid_zone, decayed_confidence

        return "Undetermined", 0.0

    def update(self, zone: str, current_time: float, frame_count: int, confidence: float, state: TrackingState, mode_tracker=None):
        if state and state.head_ayaw is not None:
            self.movement_history.append((current_time, frame_count, state.head_ayaw))

        if self.current_glance["zone"] is None:
            self.current_glance = {"zone": zone, "start_time": current_time, "start_frame": frame_count}
            return

        if zone != self.current_glance["zone"]:
            if self.pending_zone != zone:
                self.pending_zone = zone
                self.pending_time = current_time
                self.pending_frame = frame_count

            # Apply tailored debouncing
            required_debounce = self.debounce_on_road if zone == "On-Road" else self.debounce_off_road

            if (current_time - self.pending_time) >= required_debounce:
                anchored_time = self.pending_time
                anchored_frame = self.pending_frame
                consecutive_rest = 0
                
                # Saccade Rewind with Hard Capping
                for t, f, vel in reversed(self.movement_history):
                    if t > self.pending_time:
                        continue
                    if self.current_glance["start_time"] is not None and t <= self.current_glance["start_time"]:
                        break
                    
                    # Prevent endless rewinding that inflates off-road durations
                    if (self.pending_time - t) > self.max_rewind_time:
                        break
                        
                    if abs(vel) < 15.0:  
                        consecutive_rest += 1
                        anchored_time = t
                        anchored_frame = f
                        if consecutive_rest >= 2:
                            break
                    else:
                        consecutive_rest = 0

                self._finalize_current_glance(anchored_time, anchored_frame, confidence, mode_tracker)
                
                self.current_glance = {
                    "zone": zone,
                    "start_time": anchored_time,
                    "start_frame": anchored_frame,
                }
                self.pending_zone = None
        else:
            self.pending_zone = None

    def _finalize_current_glance(self, end_time: float, end_frame: int, confidence: float, mode_tracker=None):
        if self.current_glance["zone"] is None or self.current_glance["start_time"] is None: return
        duration = end_time - self.current_glance["start_time"]

        if self.current_glance["zone"] == "Undetermined" or duration >= self.min_duration:
            mode = mode_tracker.get_mode_at_time((self.current_glance["start_time"] + end_time) / 2.0) if mode_tracker else None
            self.glance_history.append({
                "zone": self.current_glance["zone"],
                "start_time": self.current_glance["start_time"],
                "end_time": end_time,
                "duration": duration,
                "start_frame": self.current_glance["start_frame"],
                "end_frame": end_frame,
                "avg_confidence": confidence,
                "mode": mode,
            })
            self.glance_durations[self.current_glance["zone"]] += duration

    def flush(self, final_time: float, final_frame: int, confidence: float, mode_tracker=None):
        if self.current_glance["zone"] is not None and self.current_glance["start_time"] is not None:
            duration = final_time - self.current_glance["start_time"]
            if duration > 0:
                mode = mode_tracker.get_mode_at_time((self.current_glance["start_time"] + final_time) / 2.0) if mode_tracker else None
                self.glance_history.append({
                    "zone": self.current_glance["zone"],
                    "start_time": self.current_glance["start_time"],
                    "end_time": final_time,
                    "duration": duration,
                    "start_frame": self.current_glance["start_frame"],
                    "end_frame": final_frame,
                    "avg_confidence": confidence,
                    "mode": mode,
                })
                self.glance_durations[self.current_glance["zone"]] += duration

    def correlate_glances_with_modes(self) -> dict:
        correlation = {}
        for glance in self.glance_history:
            mode = glance.get("mode")
            zone = glance["zone"]
            if mode not in correlation:
                correlation[mode] = {}
            if zone not in correlation[mode]:
                correlation[mode][zone] = {"count": 0, "total_duration": 0.0, "glances":[]}
            correlation[mode][zone]["count"] += 1
            correlation[mode][zone]["total_duration"] += glance["duration"]
            correlation[mode][zone]["glances"].append(glance)
        return correlation

    def build_report(self, blink_summary: dict, segment_info: dict, mode_tracker=None) -> dict:
        total_on = self.glance_durations.get("On-Road", 0.0)
        total_off = self.glance_durations.get("Off-Road", 0.0)
        total_undet = self.glance_durations.get("Undetermined", 0.0)
        total_all = total_on + total_off + total_undet

        zone_pct = {}
        if total_all > 0:
            zone_pct = {
                "On-Road": round(total_on / total_all * 100, 1),
                "Off-Road": round(total_off / total_all * 100, 1),
                "Undetermined": round(total_undet / total_all * 100, 1),
            }

        off_road_glances =[g for g in self.glance_history if g["zone"] == "Off-Road"]
        short =[g for g in off_road_glances if g["duration"] <= OFFROAD_THRESHOLD]
        long =[g for g in off_road_glances if g["duration"] > OFFROAD_THRESHOLD]

        off_road_breakdown = {
            "total_count": len(off_road_glances),
            "short": {"count": len(short), "duration": round(sum(g["duration"] for g in short), 2)},
            "long": {"count": len(long), "duration": round(sum(g["duration"] for g in long), 2)},
        }

        on_road_glances =[g for g in self.glance_history if g["zone"] == "On-Road"]
        on_road_summary = {"count": len(on_road_glances)}
        if on_road_glances:
            durs = [g["duration"] for g in on_road_glances]
            on_road_summary.update({
                "avg_duration": round(np.mean(durs), 2),
                "min_duration": round(min(durs), 2),
                "max_duration": round(max(durs), 2),
            })
        else:
            on_road_summary.update({"avg_duration": 0.0, "min_duration": 0.0, "max_duration": 0.0})

        off_road_summary = {"count": len(off_road_glances)}
        if off_road_glances:
            durs = [g["duration"] for g in off_road_glances]
            off_road_summary.update({
                "avg_duration": round(np.mean(durs), 2),
                "min_duration": round(min(durs), 2),
                "max_duration": round(max(durs), 2),
            })
        else:
            off_road_summary.update({"avg_duration": 0.0, "min_duration": 0.0, "max_duration": 0.0})

        mode_data = None
        if mode_tracker is not None and len(mode_tracker.mode_intervals) > 0:
            correlation = self.correlate_glances_with_modes()
            mode_durations = mode_tracker.get_mode_durations()
            mode_total = sum(mode_durations.values())

            mode_data = {
                "durations": {mode: round(dur, 2) for mode, dur in mode_durations.items()},
                "percentages": {mode: round(dur / mode_total * 100, 1) if mode_total > 0 else 0.0 for mode, dur in mode_durations.items()},
                "total_tracked": round(mode_total, 2),
                "transitions": len(mode_tracker.mode_intervals),
                "zone_breakdown": {},
            }

            for mode_name in sorted(m for m in correlation if m is not None):
                zones = correlation[mode_name]
                mode_total_time = mode_durations.get(mode_name, sum(z["total_duration"] for z in zones.values()))
                mode_zones = {}
                for zone_name in ["On-Road", "Off-Road", "Undetermined"]:
                    if zone_name in zones:
                        z = zones[zone_name]
                        pct = (round(z["total_duration"] / mode_total_time * 100, 1) if mode_total_time > 0 else 0.0)
                        mode_zones[zone_name] = {"count": z["count"], "duration": round(z["total_duration"], 2), "percentage": pct}
                    else:
                        mode_zones[zone_name] = {"count": 0, "duration": 0.0, "percentage": 0.0}

                mode_off_road = zones.get("Off-Road", {}).get("glances",[])
                if mode_off_road:
                    m_short = [g for g in mode_off_road if g["duration"] <= OFFROAD_THRESHOLD]
                    m_long = [g for g in mode_off_road if g["duration"] > OFFROAD_THRESHOLD]
                    mode_zones["off_road_bins"] = {
                        "short": {"count": len(m_short), "duration": round(sum(g["duration"] for g in m_short), 2)},
                        "long": {"count": len(m_long), "duration": round(sum(g["duration"] for g in m_long), 2)},
                    }

                mode_data["zone_breakdown"][mode_name] = {
                    "total_time": round(mode_total_time, 2),
                    "zones": mode_zones,
                }

            if None in correlation:
                no_mode_zones = correlation[None]
                no_mode_total = sum(z["total_duration"] for z in no_mode_zones.values())
                no_mode_count = sum(z["count"] for z in no_mode_zones.values())
                mode_data["no_mode_data"] = {"count": no_mode_count, "duration": round(no_mode_total, 2)}

        report = {
            "video_file": segment_info.get("video_file", ""),
            "segment": {
                "start": round(segment_info.get("segment_start", 0.0), 2),
                "end": round(segment_info.get("segment_end", 0.0), 2),
                "duration": round(segment_info.get("duration", 0.0), 2),
            },
            "zone_durations": {
                "On-Road": round(total_on, 2),
                "Off-Road": round(total_off, 2),
                "Undetermined": round(total_undet, 2),
                "total": round(total_all, 2),
            },
            "zone_percentages": zone_pct,
            "off_road_breakdown": off_road_breakdown,
            "on_road_summary": on_road_summary,
            "off_road_summary": off_road_summary,
            "blink_summary": {
                "total": blink_summary.get("total", 0),
                "rate_per_min": round(blink_summary.get("rate_per_min", 0.0), 1),
            },
            "mode_data": mode_data,
        }
        return report

    @staticmethod
    def print_report(report: dict):
        seg = report["segment"]
        print(f"\nSegment analyzed: {seg['start']:.2f}s - {seg['end']:.2f}s")
        print(f"Segment duration: {seg['duration']:.2f} seconds")
        print("\n" + "=" * 60)
        print("              GLANCE ANALYSIS REPORT")
        print("=" * 60)

        zd = report["zone_durations"]
        print(f"\n--- Zone Durations ---")
        print(f"  On-Road total:      {zd['On-Road']:.2f} seconds")
        print(f"  Off-Road total:     {zd['Off-Road']:.2f} seconds")
        print(f"  Undetermined total: {zd['Undetermined']:.2f} seconds")
        print(f"  Combined total:     {zd['total']:.2f} seconds")

        zp = report.get("zone_percentages", {})
        if zp:
            print(f"\n--- Zone Percentages ---")
            print(f"  On-Road:      {zp.get('On-Road', 0):.1f}%")
            print(f"  Off-Road:     {zp.get('Off-Road', 0):.1f}%")
            print(f"  Undetermined: {zp.get('Undetermined', 0):.1f}%")

        orb = report["off_road_breakdown"]
        print(f"\n--- Off-Road Glance Breakdown ---")
        print(f"  Total off-road glances: {orb['total_count']}\n")
        print(f"  Acceptable (<= {OFFROAD_THRESHOLD:.1f}s):\n    Count:    {orb['short']['count']}\n    Duration: {orb['short']['duration']:.2f} seconds\n")
        print(f"  Prolonged (> {OFFROAD_THRESHOLD:.1f}s):\n    Count:    {orb['long']['count']}\n    Duration: {orb['long']['duration']:.2f} seconds")

        ors = report["on_road_summary"]
        print(f"\n--- On-Road Glance Summary ---\n  Total on-road glances: {ors['count']}")
        if ors['count'] > 0:
            print(f"  Avg duration: {ors['avg_duration']:.2f} seconds\n  Min duration: {ors['min_duration']:.2f} seconds\n  Max duration: {ors['max_duration']:.2f} seconds")

        ofs = report["off_road_summary"]
        if ofs['count'] > 0:
            print(f"\n--- Off-Road Glance Duration Stats ---\n  Avg duration: {ofs['avg_duration']:.2f} seconds\n  Min duration: {ofs['min_duration']:.2f} seconds\n  Max duration: {ofs['max_duration']:.2f} seconds")

        md = report.get("mode_data")
        if md is not None:
            print(f"\n--- Glance Statistics by Driving Mode ---")
            for mode_name, mode_info in sorted(md["zone_breakdown"].items()):
                print(f"\n  [{mode_name}]\n    Total tracked time: {mode_info['total_time']:.2f}s")
                zones = mode_info["zones"]
                for zone_name in ["On-Road", "Off-Road", "Undetermined"]:
                    z = zones.get(zone_name, {})
                    print(f"    {zone_name}: {z.get('count', 0)} glances, {z.get('duration', 0):.2f}s ({z.get('percentage', 0):.1f}%)")
                bins = zones.get("off_road_bins")
                if bins:
                    print(f"    Off-Road Bins:")
                    print(f"      Acceptable (<= {OFFROAD_THRESHOLD:.1f}s): {bins['short']['count']} glances, {bins['short']['duration']:.2f}s")
                    print(f"      Prolonged (> {OFFROAD_THRESHOLD:.1f}s): {bins['long']['count']} glances, {bins['long']['duration']:.2f}s")

            if "no_mode_data" in md:
                nmd = md["no_mode_data"]
                print(f"\n[No Mode Data]\n    {nmd['count']} glances, {nmd['duration']:.2f}s total")

            print(f"\n--- Mode Detection Summary ---")
            for mode_name, dur in sorted(md["durations"].items()):
                pct = md["percentages"].get(mode_name, 0)
                print(f"  {mode_name}: {dur:.2f}s ({pct:.1f}%)")
            print(f"  Total tracked: {md['total_tracked']:.2f}s\n  Mode transitions: {md['transitions']}")

        print(f"=" * 60)

    def __repr__(self) -> str:
        return (f"GlanceTracker(current_zone={self.current_glance['zone']}, history_count={len(self.glance_history)}, "
                f"on_road={self.glance_durations.get('On-Road', 0):.1f}s, off_road={self.glance_durations.get('Off-Road', 0):.1f}s)")