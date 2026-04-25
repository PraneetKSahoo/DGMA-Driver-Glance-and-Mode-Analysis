"""
io_utils.py
-----------
File I/O utilities: video selection, video opening, and CSV export.

Handles all file system interactions including user dialogs for video
selection and validated video capture initialization.

Fixes addressed:
    #13 - Unsafe FPS validation (now checks for 0, negative, and extreme values)
    #15 - CSV always overwrites (now uses timestamped filenames)

Additions:
    - Mode interval CSV export
    - Glance history CSV with mode field
    - Mode-glance correlation CSV export
"""

import os
import csv
import json
from datetime import datetime
from typing import Optional, List

import cv2
import tkinter as tk
from tkinter import filedialog


# =============================================================================
# Video File Selection
# =============================================================================

def select_video_file() -> Optional[str]:
    """
    Open a file dialog for the user to select a video file.

    Returns:
        Path to the selected video file, or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path if file_path else None


# =============================================================================
# Video Opening & Validation (#13)
# =============================================================================

_MIN_VALID_FPS = 0.1
_MAX_VALID_FPS = 240.0
_DEFAULT_FPS = 30.0


def open_video(video_path: str) -> tuple:
    """
    Open a video file and return a validated capture object with metadata.

    Performs the following validations:
    1. Verifies the file can be opened by OpenCV.
    2. Reads a test frame to confirm the video stream is functional.
    3. Validates the reported FPS, falling back to a default if invalid.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (cap, total_frames, fps).

    Raises:
        RuntimeError: If the video file cannot be opened or read.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    test_success, test_frame = cap.read()
    if not test_success or test_frame is None:
        cap.release()
        raise RuntimeError(
            f"Could not read frames from video file: {video_path}"
        )
    print(f"DEBUG: Test frame read: success={test_success}, "
          f"shape={test_frame.shape}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= _MIN_VALID_FPS or fps > _MAX_VALID_FPS:
        print(f"Warning: Invalid FPS reported ({fps}). "
              f"Using default: {_DEFAULT_FPS}")
        fps = _DEFAULT_FPS
    else:
        print(f"Video FPS: {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: Could not determine frame count "
              f"(reported {total_frames}).")
        total_frames = 0

    return cap, total_frames, fps


# =============================================================================
# Filename Generation (#15)
# =============================================================================

def generate_output_filename(base: str = "gaze_data",
                             extension: str = ".csv",
                             video_name: Optional[str] = None) -> str:
    """
    Generate a unique timestamped output filename inside the CSVs subfolder.

    Args:
        base: Base name for the file (without extension).
        extension: File extension including the dot.
        video_name: Optional video filename (with or without path/extension).
                    The stem is extracted and embedded in the output name.

    Returns:
        Timestamped filepath string inside the CSVs directory.
    """
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CSVs")
    os.makedirs(csv_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if video_name:
        stem = os.path.splitext(os.path.basename(video_name))[0]
        return os.path.join(csv_dir, f"{base}_{stem}_{timestamp}{extension}")
    return os.path.join(csv_dir, f"{base}_{timestamp}{extension}")


# =============================================================================
# CSV Export — Gaze Data (#15)
# =============================================================================

def save_gaze_csv(
    collected_data: List[list],
    filename: Optional[str] = None,
    headers: Optional[List[str]] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """
    Save collected gaze/pose data to a CSV file.

    Args:
        collected_data: List of data rows [timestamp, yaw, pitch, roll, confidence].
        filename: Output filename. If None, auto-generates timestamped name.
        headers: Column headers.
        video_name: Source video filename (used in auto-generated name).

    Returns:
        The filename written to, or None if no data.
    """
    if not collected_data:
        print("No data collected — CSV export skipped.")
        return None

    if filename is None:
        filename = generate_output_filename("gaze_data", ".csv", video_name)

    if headers is None:
        headers = ['timestamp', 'yaw', 'pitch', 'confidence']

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(collected_data)
        print(f"Saved {len(collected_data)} rows of data to '{filename}'.")
        return filename
    except IOError as e:
        print(f"Error saving CSV: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error saving CSV: {e}")
        return None


# =============================================================================
# CSV Export — Glance History (with mode field)
# =============================================================================

def save_glance_history_csv(
    glance_history: List[dict],
    filename: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """
    Save glance history to a CSV file.

    Each row represents one completed glance with zone, timing, confidence,
    and driving mode.

    Args:
        glance_history: List of glance dicts from GlanceTracker.glance_history.
        filename: Output filename. If None, auto-generates timestamped name.
        video_name: Source video filename (used in auto-generated name).

    Returns:
        The filename written to, or None if no data.
    """
    if not glance_history:
        print("No glance history — CSV export skipped.")
        return None

    if filename is None:
        filename = generate_output_filename("glance_history", ".csv", video_name)

    headers = [
        'zone', 'start_time', 'end_time', 'duration',
        'start_frame', 'end_frame', 'avg_confidence', 'mode',
    ]

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for glance in glance_history:
                row = {k: glance.get(k, '') for k in headers}
                # Replace None mode with 'Unknown'
                if row['mode'] is None:
                    row['mode'] = 'Unknown'
                writer.writerow(row)
        print(f"Saved {len(glance_history)} glance records to '{filename}'.")
        return filename
    except Exception as e:
        print(f"Error saving glance history CSV: {e}")
        return None


# =============================================================================
# CSV Export — Mode Intervals (NEW)
# =============================================================================

def save_mode_intervals_csv(
    mode_intervals: List[dict],
    filename: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """
    Save mode detection intervals to a CSV file.

    Each row represents one continuous period of a driving mode.

    Args:
        mode_intervals: List of interval dicts from ModeTracker.mode_intervals.
        filename: Output filename. If None, auto-generates timestamped name.
        video_name: Source video filename (used in auto-generated name).

    Returns:
        The filename written to, or None if no data.
    """
    if not mode_intervals:
        print("No mode intervals — CSV export skipped.")
        return None

    if filename is None:
        filename = generate_output_filename("mode_intervals", ".csv", video_name)

    headers = [
        'mode', 'start', 'end', 'duration',
        'start_frame', 'end_frame',
    ]

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for interval in mode_intervals:
                row = {k: interval.get(k, '') for k in headers}
                # Round floats for readability
                for float_key in ['start', 'end', 'duration']:
                    if isinstance(row[float_key], float):
                        row[float_key] = round(row[float_key], 3)
                writer.writerow(row)
        print(f"Saved {len(mode_intervals)} mode intervals to '{filename}'.")
        return filename
    except Exception as e:
        print(f"Error saving mode intervals CSV: {e}")
        return None


# =============================================================================
# CSV Export — Mode-Glance Correlation (NEW)
# =============================================================================

def save_mode_glance_correlation_csv(
    correlation: dict,
    filename: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """
    Save mode-glance correlation data to a CSV file.

    Flattens the nested correlation dict from
    GlanceTracker.correlate_glances_with_modes() into rows.

    Args:
        correlation: Nested dict {mode: {zone: {count, total_duration, ...}}}.
        filename: Output filename. If None, auto-generates timestamped name.
        video_name: Source video filename (used in auto-generated name).

    Returns:
        The filename written to, or None if no data.
    """
    if not correlation:
        print("No correlation data — CSV export skipped.")
        return None

    if filename is None:
        filename = generate_output_filename("mode_glance_correlation", ".csv", video_name)

    headers = [
        'mode', 'zone', 'glance_count', 'total_duration',
        'avg_duration', 'min_duration', 'max_duration',
    ]

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for mode, zones in sorted(
                correlation.items(),
                key=lambda x: (x[0] is None, str(x[0]))
            ):
                mode_label = mode if mode is not None else "Unknown"

                for zone_name in ["On-Road", "Off-Road", "Undetermined"]:
                    if zone_name in zones:
                        z = zones[zone_name]
                        glances = z["glances"]
                        durations = [g["duration"] for g in glances]

                        writer.writerow({
                            'mode': mode_label,
                            'zone': zone_name,
                            'glance_count': z["count"],
                            'total_duration': round(z["total_duration"], 3),
                            'avg_duration': round(
                                sum(durations) / len(durations), 3
                            ) if durations else 0.0,
                            'min_duration': round(min(durations), 3)
                            if durations else 0.0,
                            'max_duration': round(max(durations), 3)
                            if durations else 0.0,
                        })
                    else:
                        writer.writerow({
                            'mode': mode_label,
                            'zone': zone_name,
                            'glance_count': 0,
                            'total_duration': 0.0,
                            'avg_duration': 0.0,
                            'min_duration': 0.0,
                            'max_duration': 0.0,
                        })

        print(f"Saved mode-glance correlation to '{filename}'.")
        return filename
    except Exception as e:
        print(f"Error saving correlation CSV: {e}")
        return None

# =============================================================================
# Report Export — JSON
# =============================================================================

import json


def save_report_json(
    report: dict,
    filename: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Optional[str]:
    """
    Save the structured analysis report as a JSON file.

    Args:
        report: Report dict from GlanceTracker.build_report().
        filename: Output filename. If None, auto-generates timestamped name.
        video_name: Source video filename (used in auto-generated name).

    Returns:
        The filename written to, or None on failure.
    """
    if not report:
        print("No report data — JSON export skipped.")
        return None

    if filename is None:
        filename = generate_output_filename(
            "report", ".json", video_name
        )

    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved analysis report to '{filename}'.")
        return filename
    except Exception as e:
        print(f"Error saving report JSON: {e}")
        return None


# =============================================================================
# Report Export — Long Format CSV
# =============================================================================

def save_report_long_csv(
    report: dict,
    filename: Optional[str] = None,
    video_name: Optional[str] = None,
    append: bool = True,
) -> Optional[str]:
    """
    Save the analysis report as a long-format CSV.

    Each metric becomes its own row with columns:
        video_file, segment_start, segment_end, category, metric, value

    If append=True and the file already exists, new rows are appended
    without rewriting the header. This allows building a comparison
    table across multiple runs.

    Args:
        report: Report dict from GlanceTracker.build_report().
        filename: Output filename. If None, auto-generates timestamped name.
        video_name: Source video filename (used in auto-generated name).
        append: If True, append to existing file (no duplicate header).

    Returns:
        The filename written to, or None on failure.
    """
    if not report:
        print("No report data — long CSV export skipped.")
        return None

    if filename is None:
        filename = generate_output_filename(
            "report_long", ".csv", video_name
        )

    rows = _build_long_rows(report)
    if not rows:
        print("No rows generated — long CSV export skipped.")
        return None

    headers = [
        'video_file', 'segment_start', 'segment_end',
        'category', 'metric', 'value',
    ]

    try:
        file_exists = os.path.exists(filename)
        mode = 'a' if (append and file_exists) else 'w'
        write_header = not (append and file_exists)

        with open(filename, mode, newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerows(rows)

        action = "Appended" if (append and file_exists) else "Saved"
        print(f"{action} {len(rows)} report rows to '{filename}'.")
        return filename
    except Exception as e:
        print(f"Error saving long CSV: {e}")
        return None


def _build_long_rows(report: dict) -> List[list]:
    """
    Convert a structured report dict into long-format CSV rows.

    Each row: [video_file, segment_start, segment_end, category, metric, value]

    Args:
        report: Report dict from GlanceTracker.build_report().

    Returns:
        List of row lists.
    """
    rows = []

    video_file = report.get("video_file", "")
    seg = report.get("segment", {})
    seg_start = seg.get("start", 0.0)
    seg_end = seg.get("end", 0.0)

    def _add(category, metric, value):
        rows.append([
            video_file, seg_start, seg_end,
            category, metric, value,
        ])

    # Zone durations
    zd = report.get("zone_durations", {})
    for zone in ["On-Road", "Off-Road", "Undetermined"]:
        _add("zone_duration", zone, zd.get(zone, 0.0))
    _add("zone_duration", "total", zd.get("total", 0.0))

    # Zone percentages
    zp = report.get("zone_percentages", {})
    for zone in ["On-Road", "Off-Road", "Undetermined"]:
        _add("zone_percentage", zone, zp.get(zone, 0.0))

    # Off-road breakdown
    orb = report.get("off_road_breakdown", {})
    _add("off_road_breakdown", "total_count", orb.get("total_count", 0))
    for bin_name in ["short", "long"]:
        bin_data = orb.get(bin_name, {})
        _add("off_road_count", bin_name, bin_data.get("count", 0))
        _add("off_road_duration", bin_name, bin_data.get("duration", 0.0))

    # On-road summary
    ors = report.get("on_road_summary", {})
    _add("on_road_summary", "count", ors.get("count", 0))
    _add("on_road_summary", "avg_duration", ors.get("avg_duration", 0.0))
    _add("on_road_summary", "min_duration", ors.get("min_duration", 0.0))
    _add("on_road_summary", "max_duration", ors.get("max_duration", 0.0))

    # Off-road summary
    ofs = report.get("off_road_summary", {})
    _add("off_road_summary", "count", ofs.get("count", 0))
    _add("off_road_summary", "avg_duration", ofs.get("avg_duration", 0.0))
    _add("off_road_summary", "min_duration", ofs.get("min_duration", 0.0))
    _add("off_road_summary", "max_duration", ofs.get("max_duration", 0.0))

    # Blink summary
    bs = report.get("blink_summary", {})
    _add("blink", "total", bs.get("total", 0))
    _add("blink", "rate_per_min", bs.get("rate_per_min", 0.0))

    # Mode data
    md = report.get("mode_data")
    if md is not None:
        # Mode durations
        for mode_name, dur in md.get("durations", {}).items():
            _add("mode_duration", mode_name, dur)

        # Mode percentages
        for mode_name, pct in md.get("percentages", {}).items():
            _add("mode_percentage", mode_name, pct)

        _add("mode_summary", "total_tracked", md.get("total_tracked", 0.0))
        _add("mode_summary", "transitions", md.get("transitions", 0))

        # Per-mode zone breakdown
        for mode_name, mode_info in md.get(
            "zone_breakdown", {}
        ).items():
            zones = mode_info.get("zones", {})
            for zone_name in ["On-Road", "Off-Road", "Undetermined"]:
                z = zones.get(zone_name, {})
                prefix = f"{mode_name}_{zone_name}"
                _add("mode_zone_count", prefix, z.get("count", 0))
                _add("mode_zone_duration", prefix,
                     z.get("duration", 0.0))
                _add("mode_zone_pct", prefix,
                     z.get("percentage", 0.0))

            # Off-road bins per mode
            bins = zones.get("off_road_bins")
            if bins:
                for bin_name in ["short", "long"]:
                    b = bins.get(bin_name, {})
                    prefix = f"{mode_name}_{bin_name}"
                    _add("mode_off_road_count", prefix,
                         b.get("count", 0))
                    _add("mode_off_road_duration", prefix,
                         b.get("duration", 0.0))

        # No-mode-data
        nmd = md.get("no_mode_data")
        if nmd:
            _add("no_mode_data", "count", nmd.get("count", 0))
            _add("no_mode_data", "duration", nmd.get("duration", 0.0))

    return rows